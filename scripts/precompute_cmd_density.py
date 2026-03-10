#!/usr/bin/env python
"""Precompute a 2D color–absolute-magnitude density histogram for all Gaia sources.

Iterates all GaiaSource HDF5 chunk files, selects sources with well-measured
parallaxes (S/N > 8, parallax > 0) and valid BP/RP photometry, computes
absolute G magnitude, and accumulates a 2D log-density histogram of
(BP − RP) vs M_G.

Outputs:
    src/bhumi/static/cmd_density.json

Usage:
    uv run python scripts/precompute_cmd_density.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from bhumi.data import GAIA_SOURCE_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "src" / "bhumi" / "static"

# Axis ranges (must match app.js CMD plot)
BP_RP_RANGE = (-1.0, 6.0)  # mag
MG_RANGE = (-6.0, 16.0)  # mag (note: plotted with inverted y-axis)
BP_RP_BIN = 0.02  # mag per bin
MG_BIN = 0.05  # mag per bin

# Parallax quality cut
MIN_PARALLAX_SNR = 8.0


def main() -> None:
    t0 = time.time()

    files = sorted(GAIA_SOURCE_DIR.glob("GaiaSource_*.hdf5"))
    n_files = len(files)
    print(f"Found {n_files} GaiaSource files in {GAIA_SOURCE_DIR}")

    # Histogram accumulator
    xedges = np.arange(BP_RP_RANGE[0], BP_RP_RANGE[1] + BP_RP_BIN, BP_RP_BIN)
    yedges = np.arange(MG_RANGE[0], MG_RANGE[1] + MG_BIN, MG_BIN)
    hist = np.zeros((len(yedges) - 1, len(xedges) - 1), dtype=np.float64)

    n_total = 0
    n_selected = 0

    for i, filepath in enumerate(files):
        try:
            with h5py.File(filepath, "r") as f:
                n_rows = len(f["source_id"])

                parallax = f["parallax"][:]
                parallax_error = f["parallax_error"][:]
                g_mag = f["phot_g_mean_mag"][:]
                bp_mag = f["phot_bp_mean_mag"][:]
                rp_mag = f["phot_rp_mean_mag"][:]

            n_total += n_rows

            # Quality filter
            good = (
                np.isfinite(parallax)
                & (parallax > 0)
                & np.isfinite(parallax_error)
                & (parallax_error > 0)
                & ((parallax / parallax_error) > MIN_PARALLAX_SNR)
                & np.isfinite(g_mag)
                & np.isfinite(bp_mag)
                & np.isfinite(rp_mag)
            )
            n_good = int(np.sum(good))
            if n_good == 0:
                continue

            n_selected += n_good

            # Compute color and absolute magnitude
            bp_rp = bp_mag[good] - rp_mag[good]
            dist_pc = 1000.0 / parallax[good]  # parallax in mas -> distance in pc
            abs_g = g_mag[good] - 5.0 * np.log10(dist_pc) + 5.0

            # Accumulate histogram: y = M_G, x = BP-RP
            h, _, _ = np.histogram2d(abs_g, bp_rp, bins=[yedges, xedges])
            hist += h

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

    # Convert to log10 density (empty bins -> NaN -> null in JSON)
    log_hist = np.where(hist > 0, np.log10(hist), np.nan)

    # Save as JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    xcenters = ((xedges[:-1] + xedges[1:]) / 2).tolist()
    ycenters = ((yedges[:-1] + yedges[1:]) / 2).tolist()
    z_list = []
    for row in log_hist:
        z_list.append([None if np.isnan(v) else round(float(v), 3) for v in row])

    payload = {"x": xcenters, "y": ycenters, "z": z_list}
    outpath = OUTPUT_DIR / "cmd_density.json"
    with open(outpath, "w") as fout:
        json.dump(payload, fout, separators=(",", ":"))
    size_mb = outpath.stat().st_size / 1024 / 1024
    print(f"Wrote {outpath} ({size_mb:.1f} MB)")
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
