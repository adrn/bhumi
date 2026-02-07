"""HDF5 I/O for Gaia DR3 data.

Handles source lookup by source_id, CMD neighbor queries, and RVS spectra
retrieval from the local HDF5 file store.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# Root path to the Gaia DR3 HDF5 data store
GAIA_DATA_ROOT = Path("/mnt/ceph/users/gaia/dr3/hdf5")
GAIA_SOURCE_DIR = GAIA_DATA_ROOT / "GaiaSource"
RVS_SPECTRUM_DIR = GAIA_DATA_ROOT / "RvsMeanSpectrum"

# Columns to read from GaiaSource for the source detail page
SOURCE_COLUMNS = [
    "source_id",
    "ra",
    "dec",
    "l",
    "b",
    "parallax",
    "parallax_error",
    "parallax_over_error",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "radial_velocity",
    "radial_velocity_error",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "ruwe",
    "has_rvs",
]

# Columns needed for the CMD neighbor plot
CMD_COLUMNS = [
    "source_id",
    "parallax",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
]

# Maximum number of CMD neighbor stars to return
MAX_CMD_NEIGHBORS = 10000

# Fixed RVS wavelength grid (nm)
RVS_WAVELENGTH_NM = np.linspace(846.0, 870.0, 2401)


def healpix_from_source_id(source_id: int) -> int:
    """Compute HEALPix level-8 pixel index from a Gaia source_id.

    Args:
        source_id: Gaia DR3 source identifier (64-bit integer).

    Returns:
        HEALPix level-8 pixel index (nested scheme).
    """
    return source_id >> 43


class FileIndex:
    """Index mapping HEALPix pixel ranges to HDF5 filenames.

    Built once at startup by scanning the filenames in a directory and
    parsing the pixel-range suffixes. Enables O(log N) lookup of the
    correct file for any HEALPix pixel.
    """

    def __init__(self, directory: Path, prefix: str) -> None:
        """Build the file index from a directory of HDF5 files.

        Args:
            directory: Path containing the HDF5 chunk files.
            prefix: Filename prefix (e.g. "GaiaSource").
        """
        self.directory = directory
        self.prefix = prefix

        # Parse filenames into sorted (lo, hi, path) tuples
        self._ranges: list[tuple[int, int, Path]] = []
        for p in sorted(directory.glob(f"{prefix}_*.hdf5")):
            stem = p.stem  # e.g. "GaiaSource_000000-003111"
            parts = stem.removeprefix(f"{prefix}_").split("-")
            lo, hi = int(parts[0]), int(parts[1])
            self._ranges.append((lo, hi, p))

        # Pre-extract the low bounds for binary search
        self._lo_bounds = np.array([r[0] for r in self._ranges], dtype=np.int64)

        logger.info("Indexed %d files in %s/%s_*", len(self._ranges), directory, prefix)

    def find_file(self, healpix_pixel: int) -> Path | None:
        """Find the HDF5 file containing a given HEALPix pixel.

        Uses binary search on the sorted range bounds.

        Args:
            healpix_pixel: HEALPix level-8 pixel index.

        Returns:
            Path to the matching HDF5 file, or None if not found.
        """
        idx = int(np.searchsorted(self._lo_bounds, healpix_pixel, side="right")) - 1
        if idx < 0:
            return None
        lo, hi, path = self._ranges[idx]
        if lo <= healpix_pixel <= hi:
            return path
        return None


# Module-level singletons, initialized lazily
_source_index: FileIndex | None = None
_rvs_index: FileIndex | None = None


def _get_source_index() -> FileIndex:
    """Return the GaiaSource file index, building it on first call."""
    global _source_index
    if _source_index is None:
        _source_index = FileIndex(GAIA_SOURCE_DIR, "GaiaSource")
    return _source_index


def _get_rvs_index() -> FileIndex:
    """Return the RvsMeanSpectrum file index, building it on first call."""
    global _rvs_index
    if _rvs_index is None:
        _rvs_index = FileIndex(RVS_SPECTRUM_DIR, "RvsMeanSpectrum")
    return _rvs_index


def get_source(source_id: int) -> dict[str, Any] | None:
    """Read a single Gaia source by source_id from the HDF5 store.

    Args:
        source_id: Gaia DR3 source identifier.

    Returns:
        Dictionary of column-name → value for the source, or None if not found.
    """
    healpix = healpix_from_source_id(source_id)
    filepath = _get_source_index().find_file(healpix)
    if filepath is None:
        logger.warning(
            "No file found for source_id=%d (healpix=%d)", source_id, healpix
        )
        return None

    with h5py.File(filepath, "r") as f:
        source_ids = f["source_id"][:]
        mask = source_ids == source_id
        if not np.any(mask):
            logger.warning("source_id=%d not found in %s", source_id, filepath.name)
            return None

        idx = int(np.flatnonzero(mask)[0])
        row: dict[str, Any] = {}
        for col in SOURCE_COLUMNS:
            if col in f:
                val = f[col][idx]
                # Convert numpy scalars to Python types for JSON serialization
                if isinstance(val, (np.integer,)):
                    row[col] = int(val)
                elif isinstance(val, (np.floating,)):
                    val_float = float(val)
                    row[col] = None if np.isnan(val_float) else val_float
                else:
                    row[col] = val
            else:
                row[col] = None

    return row


def get_cmd_neighbors(source_id: int) -> dict[str, Any] | None:
    """Read CMD data for stars in the same HDF5 chunk as the target.

    Returns arrays of BP−RP color and absolute G magnitude for nearby
    stars, plus the target star's values separately.

    Args:
        source_id: Gaia DR3 source identifier.

    Returns:
        Dictionary with keys: bp_rp, abs_g (arrays for neighbors),
        target_bp_rp, target_abs_g (scalars for the target), or None.
    """
    healpix = healpix_from_source_id(source_id)
    filepath = _get_source_index().find_file(healpix)
    if filepath is None:
        return None

    with h5py.File(filepath, "r") as f:
        all_source_ids = f["source_id"][:]
        parallax = f["parallax"][:]
        g_mag = f["phot_g_mean_mag"][:]
        bp_mag = f["phot_bp_mean_mag"][:]
        rp_mag = f["phot_rp_mean_mag"][:]

    # Find the target first
    target_mask = all_source_ids == source_id
    if not np.any(target_mask):
        logger.warning(
            "source_id=%d not found in CMD chunk file %s", source_id, filepath.name
        )
        return None

    target_idx = int(np.flatnonzero(target_mask)[0])

    # Check if target has valid data for CMD
    target_valid = (
        np.isfinite(parallax[target_idx])
        and parallax[target_idx] > 0
        and np.isfinite(g_mag[target_idx])
        and np.isfinite(bp_mag[target_idx])
        and np.isfinite(rp_mag[target_idx])
    )
    if not target_valid:
        logger.warning(
            "source_id=%d has invalid photometry/parallax for CMD", source_id
        )
        return None

    # Filter to stars with valid photometry and positive parallax
    valid = (
        np.isfinite(parallax)
        & (parallax > 0)
        & np.isfinite(g_mag)
        & np.isfinite(bp_mag)
        & np.isfinite(rp_mag)
    )
    all_source_ids = all_source_ids[valid]
    parallax = parallax[valid]
    g_mag = g_mag[valid]
    bp_mag = bp_mag[valid]
    rp_mag = rp_mag[valid]

    # Absolute magnitude: M_G = G + 5*log10(parallax_mas) - 10
    abs_g = g_mag + 5.0 * np.log10(parallax) - 10.0
    bp_rp = bp_mag - rp_mag

    # Find the target in the filtered arrays
    target_mask = all_source_ids == source_id
    target_idx = int(np.flatnonzero(target_mask)[0])

    # Subsample neighbors if too many
    n_stars = len(abs_g)
    if n_stars > MAX_CMD_NEIGHBORS:
        rng = np.random.default_rng(seed=source_id)
        keep = rng.choice(n_stars, size=MAX_CMD_NEIGHBORS, replace=False)
        # Always include the target
        if target_idx not in keep:
            keep[0] = target_idx
        keep.sort()
        abs_g = abs_g[keep]
        bp_rp = bp_rp[keep]
        all_source_ids = all_source_ids[keep]
        # Recompute target index after subsample
        target_mask = all_source_ids == source_id
        target_idx = int(np.flatnonzero(target_mask)[0])

    return {
        "bp_rp": bp_rp.tolist(),
        "abs_g": abs_g.tolist(),
        "target_bp_rp": float(bp_rp[target_idx]),
        "target_abs_g": float(abs_g[target_idx]),
        "n_stars": len(abs_g),
    }


def get_rvs_spectrum(
    source_id: int, has_rvs: bool | None = None
) -> dict[str, Any] | None:
    """Read the RVS mean spectrum for a source, if it exists.

    Args:
        source_id: Gaia DR3 source identifier.
        has_rvs: Pre-checked has_rvs flag from GaiaSource. If False, skip file I/O.

    Returns:
        Dictionary with wavelength, flux, flux_error arrays, or None.
    """
    # If we know the source doesn't have RVS, skip file I/O
    if has_rvs is False:
        return None

    healpix = healpix_from_source_id(source_id)
    filepath = _get_rvs_index().find_file(healpix)
    if filepath is None:
        return None

    try:
        with h5py.File(filepath, "r") as f:
            source_ids = f["source_id"][:]
            mask = source_ids == source_id
            if not np.any(mask):
                return None

            idx = int(np.flatnonzero(mask)[0])
            flux = f["flux"][idx]
            flux_error = f["flux_error"][idx]

        return {
            "wavelength": RVS_WAVELENGTH_NM.tolist(),
            "flux": flux.tolist(),
            "flux_error": flux_error.tolist(),
        }
    except Exception as e:
        logger.error("Failed to read RVS spectrum for source_id=%d: %s", source_id, e)
        return None
