#!/usr/bin/env python
"""Precompute 2D velocity density histograms for Toomre and sausage diagrams.

Iterates all GaiaSource HDF5 chunk files, selects sources with well-measured
parallaxes (S/N > 5, parallax > 0) and measured radial velocities, computes
Galactocentric velocities in bulk, and accumulates 2D log-density histograms.

Outputs two JSON files into src/bhumi/static/:
  - toomre_density.json   : -v_phi vs sqrt(v_R^2 + v_z^2)
  - sausage_density.json  : v_r (spherical) vs -v_phi

Usage:
    uv run python scripts/precompute_toomre_density.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import astropy.coordinates as coord
import astropy.units as u
import gala.dynamics as gd
import h5py
import numpy as np

from bhumi.data import GAIA_SOURCE_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "src" / "bhumi" / "static"

# Galactocentric frame (must match science.py)
coord.galactocentric_frame_defaults.set("v4.0")
GALCEN_FRAME = coord.Galactocentric(galcen_distance=8.275 * u.kpc)

# Histogram bin definitions (must match app.js axis ranges)
# Toomre: -v_phi on x, sqrt(v_R^2 + v_z^2) on y
TOOMRE_XRANGE = (-150, 450)  # -v_phi (km/s)
TOOMRE_YRANGE = (0, 400)  # v_perp (km/s)
TOOMRE_BIN_SIZE = 2  # km/s per bin

# Sausage: v_r (spherical) on x, -v_phi on y
SAUSAGE_XRANGE = (-400, 400)  # v_r (km/s)
SAUSAGE_YRANGE = (-400, 400)  # -v_phi (km/s)
SAUSAGE_BIN_SIZE = 2  # km/s per bin

# Parallax quality cut
MIN_PARALLAX_SNR = 5.0


def compute_velocities(
    ra: np.ndarray,
    dec: np.ndarray,
    parallax: np.ndarray,
    pmra: np.ndarray,
    pmdec: np.ndarray,
    rv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Galactocentric velocity components for an array of sources.

    Returns:
        (v_R, v_phi, v_z) in km/s (cylindrical) and v_r_sph (spherical radial).
    """
    sc = coord.SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        distance=(1.0 / parallax) * u.kpc,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        radial_velocity=rv * u.km / u.s,
    )
    galcen = sc.transform_to(GALCEN_FRAME)
    w = gd.PhaseSpacePosition(galcen.data)

    # Cartesian for spherical v_r
    x = galcen.x.to(u.kpc).value
    y = galcen.y.to(u.kpc).value
    z = galcen.z.to(u.kpc).value
    vx = galcen.v_x.to(u.km / u.s).value
    vy = galcen.v_y.to(u.km / u.s).value
    vz_cart = galcen.v_z.to(u.km / u.s).value

    r_sph = np.sqrt(x**2 + y**2 + z**2)
    v_r_sph = np.where(r_sph > 0, (x * vx + y * vy + z * vz_cart) / r_sph, 0.0)

    # Cylindrical
    cyl = w.cylindrical
    v_R = cyl.v_rho.to(u.km / u.s).value
    v_phi = (
        (cyl.rho * cyl.pm_phi)
        .to(u.km / u.s, equivalencies=u.dimensionless_angles())
        .value
    )
    v_z_cyl = cyl.v_z.to(u.km / u.s).value

    return v_R, v_phi, v_z_cyl, v_r_sph


def main() -> None:
    t0 = time.time()

    # Build file list
    files = sorted(GAIA_SOURCE_DIR.glob("GaiaSource_*.hdf5"))
    n_files = len(files)
    print(f"Found {n_files} GaiaSource files in {GAIA_SOURCE_DIR}")

    # Histogram accumulators
    toomre_xedges = np.arange(
        TOOMRE_XRANGE[0], TOOMRE_XRANGE[1] + TOOMRE_BIN_SIZE, TOOMRE_BIN_SIZE
    )
    toomre_yedges = np.arange(
        TOOMRE_YRANGE[0], TOOMRE_YRANGE[1] + TOOMRE_BIN_SIZE, TOOMRE_BIN_SIZE
    )
    toomre_hist = np.zeros(
        (len(toomre_yedges) - 1, len(toomre_xedges) - 1), dtype=np.float64
    )

    sausage_xedges = np.arange(
        SAUSAGE_XRANGE[0], SAUSAGE_XRANGE[1] + SAUSAGE_BIN_SIZE, SAUSAGE_BIN_SIZE
    )
    sausage_yedges = np.arange(
        SAUSAGE_YRANGE[0], SAUSAGE_YRANGE[1] + SAUSAGE_BIN_SIZE, SAUSAGE_BIN_SIZE
    )
    sausage_hist = np.zeros(
        (len(sausage_yedges) - 1, len(sausage_xedges) - 1), dtype=np.float64
    )

    n_total = 0
    n_selected = 0

    for i, filepath in enumerate(files):
        try:
            with h5py.File(filepath, "r") as f:
                n_rows = len(f["source_id"])

                # Read only what we need for filtering
                parallax = f["parallax"][:]
                parallax_error = f["parallax_error"][:]
                rv = f["radial_velocity"][:]
                pmra = f["pmra"][:]
                pmdec = f["pmdec"][:]

            n_total += n_rows

            # Quality filter: parallax > 0, S/N > 5, valid RV and PMs
            good = (
                np.isfinite(parallax)
                & (parallax > 0)
                & np.isfinite(parallax_error)
                & (parallax_error > 0)
                & ((parallax / parallax_error) > MIN_PARALLAX_SNR)
                & np.isfinite(rv)
                & np.isfinite(pmra)
                & np.isfinite(pmdec)
            )
            n_good = int(np.sum(good))
            if n_good == 0:
                continue

            n_selected += n_good

            # Read ra, dec for good sources only
            with h5py.File(filepath, "r") as f:
                ra = f["ra"][:][good]
                dec = f["dec"][:][good]

            parallax_g = parallax[good]
            pmra_g = pmra[good]
            pmdec_g = pmdec[good]
            rv_g = rv[good]

            # Compute Galactocentric velocities
            v_R, v_phi, v_z, v_r_sph = compute_velocities(
                ra, dec, parallax_g, pmra_g, pmdec_g, rv_g
            )

            neg_vphi = -v_phi
            v_perp = np.sqrt(v_R**2 + v_z**2)

            # Accumulate Toomre histogram
            h, _, _ = np.histogram2d(
                v_perp, neg_vphi, bins=[toomre_yedges, toomre_xedges]
            )
            toomre_hist += h

            # Accumulate sausage histogram
            h, _, _ = np.histogram2d(
                neg_vphi, v_r_sph, bins=[sausage_yedges, sausage_xedges]
            )
            sausage_hist += h

        except Exception as e:
            print(f"  ERROR processing {filepath.name}: {e}", file=sys.stderr)
            continue

        if (i + 1) % 100 == 0 or i == n_files - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_files - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i + 1}/{n_files}] {n_selected:,} selected of {n_total:,} "
                f"total ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
            )

    print(f"\nDone: {n_selected:,} sources selected from {n_total:,} total")

    # Convert to log10 density (set zero bins to NaN for transparency)
    toomre_log = np.where(toomre_hist > 0, np.log10(toomre_hist), np.nan)
    sausage_log = np.where(sausage_hist > 0, np.log10(sausage_hist), np.nan)

    # Save as JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def save_density(filename: str, z_data, xedges, yedges) -> None:
        # Convert bin edges to centers for Plotly heatmap
        xcenters = ((xedges[:-1] + xedges[1:]) / 2).tolist()
        ycenters = ((yedges[:-1] + yedges[1:]) / 2).tolist()
        # z_data is (ny, nx); convert to list-of-lists
        z_list = []
        for row in z_data:
            z_list.append([None if np.isnan(v) else round(float(v), 3) for v in row])

        payload = {"x": xcenters, "y": ycenters, "z": z_list}
        outpath = OUTPUT_DIR / filename
        with open(outpath, "w") as fout:
            json.dump(payload, fout, separators=(",", ":"))
        size_mb = outpath.stat().st_size / 1024 / 1024
        print(f"Wrote {outpath} ({size_mb:.1f} MB)")

    save_density("toomre_density.json", toomre_log, toomre_xedges, toomre_yedges)
    save_density("sausage_density.json", sausage_log, sausage_xedges, sausage_yedges)

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
