"""FastAPI application for the Bhumi Gaia DR3 source viewer."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from bhumi import data, science

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bhumi — Gaia DR3 Viewer")

# Paths to templates and static files (relative to this module)
_PACKAGE_DIR = Path(__file__).parent
_TEMPLATES_DIR = _PACKAGE_DIR / "templates"
_STATIC_DIR = _PACKAGE_DIR / "static"

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    source_id: str | None = Query(default=None),
) -> HTMLResponse:
    """Landing page, or source detail page if source_id is provided."""
    if source_id is None:
        return templates.TemplateResponse(request, "index.html")

    # Validate and sanitize input: must be a valid int64
    try:
        source_id_int = int(source_id)
        if source_id_int < 0 or source_id_int > 2**63 - 1:
            raise ValueError("Out of range for int64")
    except (ValueError, OverflowError):
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "error": f"Invalid source ID: '{source_id}'. Please enter a valid Gaia DR3 source_id (19-digit integer)."
            },
        )

    # Verify the source exists before rendering the page
    source = data.get_source(source_id_int)
    if source is None:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": f"Source ID {source_id_int} not found in the Gaia DR3 data."},
        )

    enriched = science.compute_derived_quantities(source)

    # Fetch value-added catalog data and variability/multiplicity info
    try:
        andrae = data.get_andrae_params(source_id_int)
    except Exception:
        logger.exception("Andrae lookup failed for %d", source_id_int)
        andrae = None
    try:
        zhang = data.get_zhang_params(source_id_int)
    except Exception:
        logger.exception("Zhang lookup failed for %d", source_id_int)
        zhang = None
    try:
        variability = data.get_variability(source_id_int)
    except Exception:
        logger.exception("Variability lookup failed for %d", source_id_int)
        variability = None
    try:
        nss_orbit = data.get_nss_orbit(source_id_int)
    except Exception:
        logger.exception("NSS orbit lookup failed for %d", source_id_int)
        nss_orbit = None

    # Compute orbit server-side for the narrative description
    galcen = None
    orbit_params = None
    if enriched.get("has_6d"):
        try:
            galcen = science.compute_galactocentric(enriched)
            orbit_result = science.compute_orbit(enriched)
            if orbit_result is not None:
                orbit_params = orbit_result["params"]
        except Exception:
            logger.exception("Server-side orbit failed for %d", source_id_int)

    narrative = science.generate_narrative(zhang, galcen, orbit_params)

    return templates.TemplateResponse(
        request,
        "source.html",
        {
            "source": enriched,
            "source_id": source_id_int,
            "andrae": andrae,
            "zhang": zhang,
            "variability": variability,
            "nss_orbit": nss_orbit,
            "narrative": narrative,
        },
    )


# ---------------------------------------------------------------------------
# JSON API endpoints (called by frontend JS for plots)
# ---------------------------------------------------------------------------


@app.get("/api/random")
async def api_random() -> dict:
    """Return a random source_id from the Gaia DR3 data."""
    source_id = data.get_random_source_id()
    if source_id is None:
        raise HTTPException(status_code=500, detail="Could not pick a random source")
    return {"source_id": str(source_id)}


@app.get("/api/random/spectra")
async def api_random_spectra() -> dict:
    """Return a random source_id that has both RVS and XP spectra."""
    source_id = data.get_random_source_id_with_spectra()
    if source_id is None:
        raise HTTPException(
            status_code=500, detail="Could not find a random source with spectra"
        )
    return {"source_id": str(source_id)}


@app.get("/api/random/orbit")
async def api_random_orbit() -> dict:
    """Return a random source_id that has full 6D phase-space info."""
    source_id = data.get_random_source_id_with_orbit()
    if source_id is None:
        raise HTTPException(
            status_code=500, detail="Could not find a random source with orbit"
        )
    return {"source_id": str(source_id)}


@app.get("/api/source/{source_id}")
async def api_source(source_id: int) -> dict:
    """Return source astrometry and derived quantities as JSON."""
    source = data.get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return science.compute_derived_quantities(source)


@app.get("/api/cmd/{source_id}")
async def api_cmd(source_id: int) -> dict:
    """Return CMD neighbor data for plotting."""
    logger.info("CMD request for source_id=%d", source_id)
    result = data.get_cmd_neighbors(source_id)
    if result is None:
        logger.warning("CMD data not available for source_id=%d", source_id)
        raise HTTPException(status_code=404, detail="CMD data not available")
    return result


@app.get("/api/rvs/{source_id}")
async def api_rvs(source_id: int) -> dict:
    """Return the RVS mean spectrum, if available."""
    # First get the source to check has_rvs flag
    source = data.get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")

    has_rvs = source.get("has_rvs")
    result = data.get_rvs_spectrum(source_id, has_rvs=has_rvs)
    if result is None:
        raise HTTPException(status_code=404, detail="No RVS spectrum available")
    return result


@app.get("/api/xp/{source_id}")
async def api_xp(source_id: int) -> dict:
    """Return the XP sampled mean spectrum, if available."""
    source = data.get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")

    has_xp = source.get("has_xp_continuous")
    result = data.get_xp_spectrum(source_id, has_xp_continuous=has_xp)
    if result is None:
        raise HTTPException(status_code=404, detail="No XP spectrum available")
    return result


@app.get("/api/orbit/{source_id}")
async def api_orbit(source_id: int) -> dict:
    """Return orbit projections and orbital parameters."""
    logger.info("Orbit request for source_id=%d", source_id)
    source = data.get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")

    enriched = science.compute_derived_quantities(source)
    logger.info(
        "Orbit: source_id=%d has_6d=%s parallax=%s pmra=%s pmdec=%s rv=%s",
        source_id,
        enriched.get("has_6d"),
        enriched.get("parallax"),
        enriched.get("pmra"),
        enriched.get("pmdec"),
        enriched.get("radial_velocity"),
    )

    try:
        galcen = science.compute_galactocentric(enriched)
        logger.info("Galactocentric result: %s", "computed" if galcen else "None")
        orbit = science.compute_orbit(enriched)
        logger.info("Orbit result: %s", "computed" if orbit else "None")
    except Exception as e:
        logger.error(
            "Orbit computation failed for source_id=%d: %s", source_id, e, exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Orbit computation failed: {str(e)}"
        )

    if orbit is None or galcen is None:
        detail = (
            f"Missing 6D phase-space info: has_6d={enriched.get('has_6d')}, "
            f"parallax={enriched.get('parallax')}, rv={enriched.get('radial_velocity')}"
        )
        logger.warning("Orbit not available for source_id=%d: %s", source_id, detail)
        raise HTTPException(status_code=404, detail=detail)

    return {
        "galactocentric": galcen,
        "orbit": orbit,
    }
