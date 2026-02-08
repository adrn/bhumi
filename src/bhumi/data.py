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
XP_SPECTRUM_DIR = GAIA_DATA_ROOT / "XpSampledMeanSpectrum"
VARI_CLASS_DIR = GAIA_DATA_ROOT / "VariClassifierResult"
NSS_ORBIT_DIR = GAIA_DATA_ROOT / "NssTwoBodyOrbit"

# Value-added catalog paths
ANDRAE_CATALOG_PATH = Path(
    "/mnt/home/apricewhelan/data/Gaia/vac/Andrae2023/table_1_catwise.fits"
)
ZHANG_CATALOG_PATH = Path(
    "/mnt/home/apricewhelan/data/Gaia/vac/Zhang2023/stellar_params_catalog_joined.h5"
)

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
    "has_xp_continuous",
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

# Fixed XP sampled spectrum wavelength grid (nm)
XP_WAVELENGTH_NM = np.linspace(336.0, 1020.0, 343)


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
            if len(parts) != 2:
                logger.debug("Skipping unparseable file: %s", p.name)
                continue
            try:
                lo, hi = int(parts[0]), int(parts[1])
            except ValueError:
                logger.debug("Skipping non-numeric range file: %s", p.name)
                continue
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
_xp_index: FileIndex | None = None
_vari_index: FileIndex | None = None

# Value-added catalog indexes (loaded at first use)
_andrae_source_ids: np.ndarray | None = None
_andrae_sort_idx: np.ndarray | None = None
_zhang_source_ids: np.ndarray | None = None
_zhang_sort_idx: np.ndarray | None = None
_nss_source_ids: np.ndarray | None = None
_nss_sort_idx: np.ndarray | None = None


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


def _get_xp_index() -> FileIndex:
    """Return the XpSampledMeanSpectrum file index, building it on first call."""
    global _xp_index
    if _xp_index is None:
        _xp_index = FileIndex(XP_SPECTRUM_DIR, "XpSampledMeanSpectrum")
    return _xp_index


def _get_vari_index() -> FileIndex:
    """Return the VariClassifierResult file index, building it on first call."""
    global _vari_index
    if _vari_index is None:
        _vari_index = FileIndex(VARI_CLASS_DIR, "VariClassifierResult")
    return _vari_index


def get_random_source_id() -> int | None:
    """Pick a random source_id from a random HDF5 chunk file.

    Returns:
        A random Gaia DR3 source_id, or None on failure.
    """
    idx = _get_source_index()
    if not idx._ranges:
        return None

    rng = np.random.default_rng()
    _, _, filepath = idx._ranges[rng.integers(len(idx._ranges))]
    try:
        with h5py.File(filepath, "r") as f:
            source_ids = f["source_id"]
            n = len(source_ids)
            return int(source_ids[rng.integers(n)])
    except Exception as e:
        logger.error("Failed to pick random source: %s", e)
        return None


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


def get_xp_spectrum(
    source_id: int, has_xp_continuous: bool | None = None
) -> dict[str, Any] | None:
    """Read the XP sampled mean spectrum for a source, if it exists.

    Args:
        source_id: Gaia DR3 source identifier.
        has_xp_continuous: Pre-checked flag from GaiaSource. If False, skip file I/O.

    Returns:
        Dictionary with wavelength, flux, flux_error arrays, or None.
    """
    if has_xp_continuous is False:
        return None

    healpix = healpix_from_source_id(source_id)
    filepath = _get_xp_index().find_file(healpix)
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
            "wavelength": XP_WAVELENGTH_NM.tolist(),
            "flux": flux.tolist(),
            "flux_error": flux_error.tolist(),
        }
    except Exception as e:
        logger.error("Failed to read XP spectrum for source_id=%d: %s", source_id, e)
        return None


# ---------------------------------------------------------------------------
# Value-added catalogs
# ---------------------------------------------------------------------------


def _load_andrae_index() -> tuple[np.ndarray, np.ndarray]:
    """Load the Andrae+2023 source_id column and build a sorted index.

    Returns:
        Tuple of (sorted_source_ids, sort_indices) for searchsorted lookup.
    """
    global _andrae_source_ids, _andrae_sort_idx
    if _andrae_source_ids is not None:
        return _andrae_source_ids, _andrae_sort_idx

    from astropy.io import fits

    logger.info("Loading Andrae+2023 source_id index from %s...", ANDRAE_CATALOG_PATH)
    with fits.open(ANDRAE_CATALOG_PATH, memmap=True) as hdul:
        raw_ids = hdul[1].data["source_id"].copy()
    sort_idx = np.argsort(raw_ids)
    _andrae_source_ids = raw_ids[sort_idx]
    _andrae_sort_idx = sort_idx
    logger.info("Andrae+2023 index loaded: %d rows", len(_andrae_source_ids))
    return _andrae_source_ids, _andrae_sort_idx


def get_andrae_params(source_id: int) -> dict[str, Any] | None:
    """Look up stellar parameters from the Andrae+2023 XGBoost catalog.

    Args:
        source_id: Gaia DR3 source identifier.

    Returns:
        Dictionary with Teff, logg, [M/H], or None if not found.
    """
    if not ANDRAE_CATALOG_PATH.exists():
        return None

    try:
        from astropy.io import fits

        sorted_ids, sort_idx = _load_andrae_index()
        pos = int(np.searchsorted(sorted_ids, source_id))
        if pos >= len(sorted_ids) or sorted_ids[pos] != source_id:
            return None

        row_idx = int(sort_idx[pos])
        with fits.open(ANDRAE_CATALOG_PATH, memmap=True) as hdul:
            row = hdul[1].data[row_idx]

        result: dict[str, Any] = {}
        columns = [
            ("teff_xgboost", "teff_xgboost"),
            ("logg_xgboost", "logg_xgboost"),
            ("mh_xgboost", "mh_xgboost"),
        ]
        for key, col in columns:
            val = float(row[col])
            result[key] = None if np.isnan(val) else round(val, 4)

        # Boolean flag
        try:
            result["in_training_sample"] = bool(row["in_training_sample"])
        except (KeyError, ValueError):
            result["in_training_sample"] = None

        return result
    except Exception as e:
        logger.error("Failed to read Andrae+2023 for source_id=%d: %s", source_id, e)
        return None


def _load_zhang_index() -> tuple[np.ndarray, np.ndarray]:
    """Load the Zhang+2023 source_id column and build a sorted index.

    Returns:
        Tuple of (sorted_source_ids, sort_indices) for searchsorted lookup.
    """
    global _zhang_source_ids, _zhang_sort_idx
    if _zhang_source_ids is not None:
        return _zhang_source_ids, _zhang_sort_idx

    logger.info("Loading Zhang+2023 source_id index from %s...", ZHANG_CATALOG_PATH)
    with h5py.File(ZHANG_CATALOG_PATH, "r") as f:
        raw_ids = f["__astropy_table__"]["source_id"][:]
    sort_idx = np.argsort(raw_ids)
    _zhang_source_ids = raw_ids[sort_idx]
    _zhang_sort_idx = sort_idx
    logger.info("Zhang+2023 index loaded: %d rows", len(_zhang_source_ids))
    return _zhang_source_ids, _zhang_sort_idx


def get_zhang_params(source_id: int) -> dict[str, Any] | None:
    """Look up stellar parameters from the Zhang, Green & Rix 2023 catalog.

    Args:
        source_id: Gaia DR3 source identifier.

    Returns:
        Dictionary with Teff, [Fe/H], logg, extinction (with errors), or None.
    """
    if not ZHANG_CATALOG_PATH.exists():
        return None

    try:
        sorted_ids, sort_idx = _load_zhang_index()
        pos = int(np.searchsorted(sorted_ids, source_id))
        if pos >= len(sorted_ids) or sorted_ids[pos] != source_id:
            return None

        row_idx = int(sort_idx[pos])
        with h5py.File(ZHANG_CATALOG_PATH, "r") as f:
            row = f["__astropy_table__"][row_idx]

        columns = [
            "zhang_teff",
            "zhang_teff_err",
            "zhang_feh",
            "zhang_feh_err",
            "zhang_logg",
            "zhang_logg_err",
            "zhang_extinction",
            "zhang_extinction_err",
            "zhang_parallax",
            "zhang_parallax_err",
            "quality_flags",
            "feh_confidence",
            "teff_confidence",
            "logg_confidence",
        ]
        result: dict[str, Any] = {}
        for col in columns:
            try:
                val = float(row[col])
                if col == "quality_flags":
                    result[col] = int(row[col])
                elif np.isnan(val):
                    result[col] = None
                else:
                    # Teff is stored in units of 1e3 K — convert to K
                    if col in ("zhang_teff", "zhang_teff_err"):
                        result[col] = round(val * 1000.0, 1)
                    else:
                        result[col] = round(val, 4)
            except (KeyError, ValueError, IndexError):
                result[col] = None

        # Derive distance from spectrophotometric parallax
        plx = result.get("zhang_parallax")
        if plx is not None and plx > 0:
            result["zhang_distance_kpc"] = round(1.0 / plx, 4)
        else:
            result["zhang_distance_kpc"] = None

        return result
    except Exception as e:
        logger.error("Failed to read Zhang+2023 for source_id=%d: %s", source_id, e)
        return None


# ---------------------------------------------------------------------------
# Variability & multiplicity
# ---------------------------------------------------------------------------

# Mapping from int8 class index to (short_name, description) from
# VariClassifierClassDefinition (nTransits:5+ classifier, 24 classes).
_VARI_CLASS_NAMES: dict[int, tuple[str, str]] = {
    0: ("AGN", "Active Galactic Nuclei (including Quasars)"),
    1: ("DSCT|GDOR|SXPHE", "δ Scuti / γ Doradus / SX Phoenicis"),
    2: ("WD", "White Dwarf variable (ZZ Ceti types)"),
    3: ("LPV", "Long Period Variable (Mira / OGLE SRG / semiregular)"),
    4: ("ACV|CP|MCP|ROAM|ROAP|SXARI", "Magnetic / Chemical Peculiar star"),
    5: ("S", "Short timescale variability"),
    6: ("MICROLENSING", "Microlensing event"),
    7: ("CEP", "Cepheid (δ Cep / anomalous / type-II)"),
    8: ("YSO", "Young Stellar Object"),
    9: ("RS", "RS Canum Venaticorum"),
    10: ("ACYG", "α Cygni variable"),
    11: ("BCEP", "β Cephei variable"),
    12: ("BE|GCAS|SDOR|WR", "Eruptive (Be / γ Cas / S Dor / Wolf-Rayet)"),
    13: ("SN", "Supernova"),
    14: ("SPB", "Slowly Pulsating B star"),
    15: ("ECL", "Eclipsing Binary"),
    16: ("ELL", "Ellipsoidal variable"),
    17: ("SYST", "Symbiotic variable star"),
    18: ("SOLAR_LIKE", "Solar-like variability (flares / spots / rotation)"),
    19: ("CV", "Cataclysmic variable"),
    20: ("SDB", "Sub-dwarf B star"),
    21: ("RR", "RR Lyrae"),
    22: ("EP", "Exoplanet transit"),
    23: ("RCB", "R Coronae Borealis"),
}


def get_variability(source_id: int) -> dict[str, Any] | None:
    """Look up variability classification from VariClassifierResult.

    Args:
        source_id: Gaia DR3 source identifier.

    Returns:
        Dictionary with best_class_name, best_class_description,
        and best_class_score, or None.
    """
    healpix = healpix_from_source_id(source_id)
    filepath = _get_vari_index().find_file(healpix)
    if filepath is None:
        return None

    try:
        with h5py.File(filepath, "r") as f:
            source_ids = f["source_id"][:]
            mask = source_ids == source_id
            if not np.any(mask):
                return None

            idx = int(np.flatnonzero(mask)[0])
            result: dict[str, Any] = {}

            # Read class index (stored as int8) and decode to name
            if "best_class_name" in f:
                class_idx = int(f["best_class_name"][idx])
                if class_idx in _VARI_CLASS_NAMES:
                    name, desc = _VARI_CLASS_NAMES[class_idx]
                    result["best_class_name"] = name
                    result["best_class_description"] = desc
                else:
                    result["best_class_name"] = str(class_idx)
                    result["best_class_description"] = None

            # Read classifier score
            if "best_class_score" in f:
                score = float(f["best_class_score"][idx])
                result["best_class_score"] = (
                    None if np.isnan(score) else round(score, 4)
                )

        return result if result else None
    except Exception as e:
        logger.error(
            "Failed to read VariClassifierResult for source_id=%d: %s", source_id, e
        )
        return None


def _load_nss_index() -> tuple[np.ndarray, np.ndarray]:
    """Load the NssTwoBodyOrbit source_id column and build a sorted index.

    The NSS data lives in a single monolithic HDF5 file (not HEALPix-chunked).

    Returns:
        Tuple of (sorted_source_ids, sort_indices) for searchsorted lookup.
    """
    global _nss_source_ids, _nss_sort_idx
    if _nss_source_ids is not None:
        return _nss_source_ids, _nss_sort_idx

    nss_file = NSS_ORBIT_DIR / "NssTwoBodyOrbit_1.hdf5"
    logger.info("Loading NssTwoBodyOrbit source_id index from %s...", nss_file)
    with h5py.File(nss_file, "r") as f:
        raw_ids = f["source_id"][:]
    sort_idx = np.argsort(raw_ids)
    _nss_source_ids = raw_ids[sort_idx]
    _nss_sort_idx = sort_idx
    logger.info("NssTwoBodyOrbit index loaded: %d rows", len(_nss_source_ids))
    return _nss_source_ids, _nss_sort_idx


def get_nss_orbit(source_id: int) -> dict[str, Any] | None:
    """Look up non-single-star two-body orbit from NssTwoBodyOrbit.

    Args:
        source_id: Gaia DR3 source identifier.

    Returns:
        Dictionary with orbital parameters, or None if not found.
    """
    nss_file = NSS_ORBIT_DIR / "NssTwoBodyOrbit_1.hdf5"
    if not nss_file.exists():
        return None

    try:
        sorted_ids, sort_idx = _load_nss_index()
        pos = int(np.searchsorted(sorted_ids, source_id))
        if pos >= len(sorted_ids) or sorted_ids[pos] != source_id:
            return None

        row_idx = int(sort_idx[pos])
        with h5py.File(nss_file, "r") as f:
            result: dict[str, Any] = {}

            float_cols = ["period", "eccentricity", "semi_amplitude_primary"]
            for col in float_cols:
                if col in f:
                    val = float(f[col][row_idx])
                    result[col] = None if np.isnan(val) else round(val, 4)

            str_cols = ["nss_solution_type"]
            for col in str_cols:
                if col in f:
                    val = f[col][row_idx]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8")
                    result[col] = str(val)

        return result if result else None
    except Exception as e:
        logger.error(
            "Failed to read NssTwoBodyOrbit for source_id=%d: %s", source_id, e
        )
        return None
