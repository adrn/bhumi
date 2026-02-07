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
            {"error": f"Invalid source ID: '{source_id}'. Please enter a valid Gaia DR3 source_id (19-digit integer)."},
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
    return templates.TemplateResponse(
        request,
        "source.html",
        {"source": enriched, "source_id": source_id_int},
    )


# ---------------------------------------------------------------------------
# JSON API endpoints (called by frontend JS for plots)
# ---------------------------------------------------------------------------


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
    result = data.get_cmd_neighbors(source_id)
    if result is None:
        raise HTTPException(status_code=404, detail="CMD data not available")
    return result


@app.get("/api/rvs/{source_id}")
async def api_rvs(source_id: int) -> dict:
    """Return the RVS mean spectrum, if available."""
    result = data.get_rvs_spectrum(source_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No RVS spectrum available")
    return result


@app.get("/api/orbit/{source_id}")
async def api_orbit(source_id: int) -> dict:
    """Return orbit projections and orbital parameters."""
    source = data.get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")

    enriched = science.compute_derived_quantities(source)

    galcen = science.compute_galactocentric(enriched)
    orbit = science.compute_orbit(enriched)

    if orbit is None or galcen is None:
        raise HTTPException(
            status_code=404,
            detail="Full 6D phase-space information not available for this source",
        )

    return {
        "galactocentric": galcen,
        "orbit": orbit,
    }
