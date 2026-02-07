"""Derived astrophysical quantities and orbital analysis.

Computes Galactic coordinates, distances, velocities, and orbital
parameters from Gaia DR3 source data using astropy and gala.
"""

from __future__ import annotations

import logging
from typing import Any

import astropy.coordinates as coord
import astropy.units as u
import gala.dynamics as gd
import gala.potential as gp
import numpy as np

logger = logging.getLogger(__name__)

# Galactocentric frame defaults
coord.galactocentric_frame_defaults.set("v4.0")
GALCEN_FRAME = coord.Galactocentric(
    galcen_distance=8.275 * u.kpc,  # galcen_v_sun=[8.4, 251.0, 8.4] * u.km / u.s
)

# Milky Way potential for orbit integration
MW_POTENTIAL = gp.MilkyWayPotential(version="v2")

# Orbit integration parameters: 8 Gyr backward
ORBIT_DT = -0.5 * u.Myr
ORBIT_N_STEPS = 16_000


def compute_derived_quantities(source: dict[str, Any]) -> dict[str, Any]:
    """Compute derived astrometric and photometric quantities.

    Adds distance, distance modulus, Galactic coordinates, tangential
    velocity, and formatted display strings to the source data.

    Args:
        source: Raw source data dictionary from data.get_source().

    Returns:
        New dictionary with original + derived fields.
    """
    result = dict(source)

    parallax = source.get("parallax")
    parallax_error = source.get("parallax_error")

    # Distance and distance modulus
    if parallax is not None and parallax > 0:
        distance_kpc = 1.0 / parallax  # parallax in mas → d = 1/plx kpc
        distance_pc = distance_kpc * 1000.0
        dist_mod = 5.0 * np.log10(distance_pc / 10.0)
        result["distance_kpc"] = round(distance_kpc, 4)
        result["distance_modulus"] = round(dist_mod, 3)
    else:
        result["distance_kpc"] = None
        result["distance_modulus"] = None

    # Tangential velocity: v_t = 4.74047 * mu_tot / parallax [km/s]
    # where mu_tot in mas/yr and parallax in mas
    pmra = source.get("pmra")
    pmdec = source.get("pmdec")
    if parallax is not None and parallax > 0 and pmra is not None and pmdec is not None:
        mu_tot = np.sqrt(pmra**2 + pmdec**2)
        v_tan = 4.74047 * mu_tot / parallax
        result["pm_total"] = round(float(mu_tot), 4)
        result["v_tan_km_s"] = round(float(v_tan), 2)
    else:
        result["pm_total"] = None
        result["v_tan_km_s"] = None

    # S/N values for proper motions
    if pmra is not None and source.get("pmra_error") is not None:
        pmra_err = source["pmra_error"]
        result["pmra_snr"] = round(abs(pmra) / pmra_err, 1) if pmra_err > 0 else None
    else:
        result["pmra_snr"] = None

    if pmdec is not None and source.get("pmdec_error") is not None:
        pmdec_err = source["pmdec_error"]
        result["pmdec_snr"] = (
            round(abs(pmdec) / pmdec_err, 1) if pmdec_err > 0 else None
        )
    else:
        result["pmdec_snr"] = None

    # Radial velocity S/N
    rv = source.get("radial_velocity")
    rv_err = source.get("radial_velocity_error")
    if rv is not None and rv_err is not None and rv_err > 0:
        result["rv_snr"] = round(abs(rv) / rv_err, 1)
    else:
        result["rv_snr"] = None

    # Whether the star has full 6D phase-space information
    result["has_radial_velocity"] = rv is not None
    result["has_6d"] = (
        rv is not None
        and parallax is not None
        and parallax > 0
        and pmra is not None
        and pmdec is not None
    )

    return result


def _build_skycoord(source: dict[str, Any]) -> coord.SkyCoord:
    """Build an astropy SkyCoord from source data.

    Args:
        source: Source dictionary with ra, dec, parallax, pmra, pmdec,
                and optionally radial_velocity.

    Returns:
        SkyCoord in ICRS with distance and velocity info.
    """
    kwargs: dict[str, Any] = {
        "ra": source["ra"] * u.deg,
        "dec": source["dec"] * u.deg,
        "distance": (1.0 / source["parallax"]) * u.kpc,
        "pm_ra_cosdec": source["pmra"] * u.mas / u.yr,
        "pm_dec": source["pmdec"] * u.mas / u.yr,
    }
    if source.get("radial_velocity") is not None:
        kwargs["radial_velocity"] = source["radial_velocity"] * u.km / u.s

    return coord.SkyCoord(**kwargs)


def compute_galactocentric(source: dict[str, Any]) -> dict[str, Any] | None:
    """Compute Galactocentric Cartesian and cylindrical velocities.

    Args:
        source: Source dictionary (must have full 6D phase-space info).

    Returns:
        Dictionary with Galactocentric position and velocity components,
        or None if the source lacks 6D information.
    """
    if not source.get("has_6d", False):
        return None

    sc = _build_skycoord(source)
    galcen = sc.transform_to(GALCEN_FRAME)
    w = gd.PhaseSpacePosition(galcen.data)

    # Cartesian
    x = galcen.x.to(u.kpc).value
    y = galcen.y.to(u.kpc).value
    z = galcen.z.to(u.kpc).value
    v_x = galcen.v_x.to(u.km / u.s).value
    v_y = galcen.v_y.to(u.km / u.s).value
    v_z = galcen.v_z.to(u.km / u.s).value

    # Cylindrical
    cyl = w.cylindrical
    R = cyl.rho.to(u.kpc).value
    phi = cyl.phi.to(u.deg).value
    z_cyl = cyl.z.to(u.kpc).value
    v_R = cyl.v_rho.to(u.km / u.s).value
    v_phi = (
        (cyl.rho * cyl.pm_phi)
        .to(u.km / u.s, equivalencies=u.dimensionless_angles())
        .value
    )
    v_z_cyl = cyl.v_z.to(u.km / u.s).value

    return {
        "cartesian": {
            "x_kpc": round(float(x), 4),
            "y_kpc": round(float(y), 4),
            "z_kpc": round(float(z), 4),
            "v_x_km_s": round(float(v_x), 2),
            "v_y_km_s": round(float(v_y), 2),
            "v_z_km_s": round(float(v_z), 2),
        },
        "cylindrical": {
            "R_kpc": round(float(R), 4),
            "phi_deg": round(float(phi), 2),
            "z_kpc": round(float(z_cyl), 4),
            "v_R_km_s": round(float(v_R), 2),
            "v_phi_km_s": round(float(v_phi), 2),
            "v_z_km_s": round(float(v_z_cyl), 2),
        },
    }


def compute_orbit(source: dict[str, Any]) -> dict[str, Any] | None:
    """Integrate the Galactic orbit and compute orbital parameters.

    Uses MilkyWayPotential(version="v2") and integrates 8 Gyr backward.

    Args:
        source: Source dictionary (must have full 6D phase-space info).

    Returns:
        Dictionary with orbit projections (arrays for plotting) and
        orbital parameter summary, or None if no 6D info.
    """
    if not source.get("has_6d", False):
        return None

    sc = _build_skycoord(source)
    galcen = sc.transform_to(GALCEN_FRAME)
    w0 = gd.PhaseSpacePosition(galcen.data)

    orbit = MW_POTENTIAL.integrate_orbit(w0, dt=ORBIT_DT, n_steps=ORBIT_N_STEPS)

    # Extract orbit projections for plotting
    x = orbit.x.to(u.kpc).value
    y = orbit.y.to(u.kpc).value
    z = orbit.z.to(u.kpc).value
    cyl = orbit.cylindrical
    R = cyl.rho.to(u.kpc).value

    # Downsample for the frontend (every 4th point → ~4000 points)
    step = 4
    projections = {
        "x": x[::step].tolist(),
        "y": y[::step].tolist(),
        "z": z[::step].tolist(),
        "R": R[::step].tolist(),
    }

    # Orbital parameters
    pericenter = orbit.pericenter().to(u.kpc).value
    apocenter = orbit.apocenter().to(u.kpc).value
    eccentricity = orbit.eccentricity()
    zmax = orbit.zmax().to(u.kpc).value

    params = {
        "pericenter_kpc": round(float(pericenter), 3),
        "apocenter_kpc": round(float(apocenter), 3),
        "eccentricity": round(float(eccentricity), 4),
        "zmax_kpc": round(float(zmax), 3),
    }

    return {
        "projections": projections,
        "params": params,
    }
