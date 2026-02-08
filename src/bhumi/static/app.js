/* Bhumi frontend — fetches JSON from API endpoints and renders Plotly charts. */

const PLOTLY_BG = "#1a1d27";
const PLOTLY_GRID = "#2a2d3a";
const PLOTLY_TEXT = "#e0e0e8";
const PLOTLY_DIM = "#8888a0";
const PLOTLY_ACCENT = "#6ea8fe";
const PLOTLY_HIGHLIGHT = "#ffd866";

/** Shared Plotly layout defaults for dark theme. */
function baseLayout(title, xLabel, yLabel, extra = {}) {
  return {
    title: { text: title, font: { color: PLOTLY_TEXT, size: 14 } },
    paper_bgcolor: PLOTLY_BG,
    plot_bgcolor: PLOTLY_BG,
    font: { color: PLOTLY_TEXT, size: 11 },
    margin: { t: 40, r: 20, b: 50, l: 60 },
    xaxis: {
      title: xLabel,
      gridcolor: PLOTLY_GRID,
      zerolinecolor: PLOTLY_GRID,
      color: PLOTLY_DIM,
    },
    yaxis: {
      title: yLabel,
      gridcolor: PLOTLY_GRID,
      zerolinecolor: PLOTLY_GRID,
      color: PLOTLY_DIM,
    },
    ...extra,
  };
}

/** Fetch JSON from an API endpoint, return null on error. */
async function apiFetch(path) {
  try {
    const resp = await fetch(path);
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try {
        const body = await resp.json();
        if (body.detail) detail += `: ${body.detail}`;
      } catch (_) {}
      console.error("API error:", path, detail);
      return null;
    }
    return await resp.json();
  } catch (e) {
    console.error("API error:", path, e);
    return null;
  }
}

/** Build an HTML table body from key-value pairs. */
function buildTableHTML(rows) {
  return (
    "<tbody>" +
    rows
      .map(
        ([label, value]) =>
          `<tr><td class="label">${label}</td><td>${value}</td></tr>`,
      )
      .join("") +
    "</tbody>"
  );
}

// ---------------------------------------------------------------------------
// CMD Plot
// ---------------------------------------------------------------------------

async function loadCMD() {
  const container = document.getElementById("cmd-plot");
  const data = await apiFetch(`/api/cmd/${SOURCE_ID}`);
  if (!data) {
    container.innerHTML = '<p class="loading">CMD data not available.</p>';
    return;
  }

  const density = {
    x: data.bp_rp,
    y: data.abs_g,
    type: "histogram2d",
    colorscale: [
      [0, "rgba(0,0,0,0)"],
      [0.01, "#1a1a4e"],
      [0.1, "#2a4858"],
      [0.3, "#3a7ca5"],
      [0.5, "#6ea8fe"],
      [0.7, "#a0c4ff"],
      [1.0, "#e0e0e8"],
    ],
    colorbar: {
      title: { text: "log\u2081\u2080(Count)", font: { color: PLOTLY_TEXT } },
      tickfont: { color: PLOTLY_DIM },
    },
    xbins: { start: -1, end: 5, size: 0.05 },
    ybins: { start: -6, end: 16, size: 0.1 },
    zsmooth: false,
    zauto: false,
    zmin: 0.5,
    name: `Neighbors (${data.n_stars})`,
    hoverinfo: "z",
  };

  const target = {
    x: [data.target_bp_rp],
    y: [data.target_abs_g],
    mode: "markers",
    type: "scatter",
    marker: {
      color: PLOTLY_HIGHLIGHT,
      size: 10,
      symbol: "star",
      line: { color: "#fff", width: 1 },
    },
    name: "This source",
  };

  container.innerHTML = "";

  const layout = baseLayout(
    "Color\u2013Magnitude Diagram",
    "BP \u2212 RP (mag)",
    "M<sub>G</sub> (mag)",
    {
      xaxis: {
        title: "BP \u2212 RP (mag)",
        range: [-1, 5],
        gridcolor: PLOTLY_GRID,
        zerolinecolor: PLOTLY_GRID,
        color: PLOTLY_DIM,
      },
      yaxis: {
        title: "M<sub>G</sub> (mag)",
        range: [16, -6],
        gridcolor: PLOTLY_GRID,
        zerolinecolor: PLOTLY_GRID,
        color: PLOTLY_DIM,
      },
      width: 500,
      height: 500,
      showlegend: true,
      legend: { x: 0.02, y: 0.02, bgcolor: "rgba(0,0,0,0)" },
    },
  );

  // Render, then apply log scale to density counts
  Plotly.newPlot(container, [density, target], layout, {
    responsive: true,
  }).then(() => {
    const z = container.data[0].z;
    if (z) {
      const logZ = z.map((row) =>
        row.map((v) => (v > 0 ? Math.log10(v) : null)),
      );
      Plotly.restyle(container, { z: [logZ], zauto: true, zmin: null }, [0]);
    }
  });
}

// ---------------------------------------------------------------------------
// RVS Spectrum
// ---------------------------------------------------------------------------

async function loadRVS() {
  const section = document.getElementById("rvs-section");
  const container = document.getElementById("rvs-plot");

  const data = await apiFetch(`/api/rvs/${SOURCE_ID}`);
  if (!data) {
    section.style.display = "none";
    return;
  }

  container.innerHTML = "";

  const spectrum = {
    x: data.wavelength,
    y: data.flux,
    mode: "lines",
    type: "scatter",
    line: { color: PLOTLY_ACCENT, width: 1.2 },
    name: "Flux",
  };

  // Error band (flux ± error)
  const upper = data.flux.map((f, i) => f + data.flux_error[i]);
  const lower = data.flux.map((f, i) => f - data.flux_error[i]);

  const errorBand = {
    x: data.wavelength.concat([...data.wavelength].reverse()),
    y: upper.concat([...lower].reverse()),
    fill: "toself",
    fillcolor: "rgba(110, 168, 254, 0.15)",
    line: { color: "transparent" },
    type: "scatter",
    name: "±1σ",
    hoverinfo: "skip",
  };

  const layout = baseLayout(
    "RVS Mean Spectrum",
    "Wavelength (nm)",
    "Normalized flux",
    {
      showlegend: true,
      legend: { x: 0.02, y: 0.98, bgcolor: "rgba(0,0,0,0)" },
    },
  );

  Plotly.newPlot(container, [errorBand, spectrum], layout, {
    responsive: true,
  });
}

// ---------------------------------------------------------------------------
// Orbit
// ---------------------------------------------------------------------------

async function loadOrbit() {
  if (!HAS_6D) return;

  const loadingEl = document.getElementById("orbit-loading");
  const plotsEl = document.getElementById("orbit-plots");
  const galcenEl = document.getElementById("galcen-section");

  let data;
  try {
    const resp = await fetch(`/api/orbit/${SOURCE_ID}`);
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try {
        const body = await resp.json();
        if (body.detail) detail = body.detail;
      } catch (_) {}
      console.error("Orbit API error:", detail);
      loadingEl.textContent = `Orbit failed: ${detail}`;
      return;
    }
    data = await resp.json();
  } catch (e) {
    console.error("Orbit fetch error:", e);
    loadingEl.textContent = `Orbit request failed: ${e.message}`;
    return;
  }

  loadingEl.style.display = "none";
  plotsEl.style.display = "grid";
  galcenEl.style.display = "block";

  const proj = data.orbit.projections;
  const orbitTrace = (x, y) => ({
    x,
    y,
    mode: "lines",
    type: "scattergl",
    line: { color: PLOTLY_ACCENT, width: 0.8 },
    hoverinfo: "skip",
    showlegend: false,
  });

  // Starting point marker
  const startMarker = (x, y) => ({
    x: [x],
    y: [y],
    mode: "markers",
    type: "scatter",
    marker: {
      color: PLOTLY_HIGHLIGHT,
      size: 8,
      symbol: "star",
      line: { color: "#fff", width: 1 },
    },
    name: "Current position",
    showlegend: false,
  });

  // x-y projection
  Plotly.newPlot(
    "orbit-xy",
    [orbitTrace(proj.x, proj.y), startMarker(proj.x[0], proj.y[0])],
    baseLayout("x–y", "x (kpc)", "y (kpc)", {
      yaxis: {
        scaleanchor: "x",
        gridcolor: PLOTLY_GRID,
        zerolinecolor: PLOTLY_GRID,
        color: PLOTLY_DIM,
        title: "y (kpc)",
      },
    }),
    { responsive: true },
  );

  // x-z projection
  Plotly.newPlot(
    "orbit-xz",
    [orbitTrace(proj.x, proj.z), startMarker(proj.x[0], proj.z[0])],
    baseLayout("x–z", "x (kpc)", "z (kpc)", {
      yaxis: {
        scaleanchor: "x",
        gridcolor: PLOTLY_GRID,
        zerolinecolor: PLOTLY_GRID,
        color: PLOTLY_DIM,
        title: "z (kpc)",
      },
    }),
    { responsive: true },
  );

  // R-z projection
  Plotly.newPlot(
    "orbit-Rz",
    [orbitTrace(proj.R, proj.z), startMarker(proj.R[0], proj.z[0])],
    baseLayout("R–z", "R (kpc)", "z (kpc)", {
      yaxis: {
        scaleanchor: "x",
        gridcolor: PLOTLY_GRID,
        zerolinecolor: PLOTLY_GRID,
        color: PLOTLY_DIM,
        title: "z (kpc)",
      },
    }),
    { responsive: true },
  );

  // Galactocentric velocities
  const gc = data.galactocentric;
  document.getElementById("vel-cartesian").innerHTML = buildTableHTML([
    ["x", `${gc.cartesian.x_kpc} kpc`],
    ["y", `${gc.cartesian.y_kpc} kpc`],
    ["z", `${gc.cartesian.z_kpc} kpc`],
    ["v<sub>x</sub>", `${gc.cartesian.v_x_km_s} km/s`],
    ["v<sub>y</sub>", `${gc.cartesian.v_y_km_s} km/s`],
    ["v<sub>z</sub>", `${gc.cartesian.v_z_km_s} km/s`],
  ]);

  document.getElementById("vel-cylindrical").innerHTML = buildTableHTML([
    ["R", `${gc.cylindrical.R_kpc} kpc`],
    ["φ", `${gc.cylindrical.phi_deg}°`],
    ["z", `${gc.cylindrical.z_kpc} kpc`],
    ["v<sub>R</sub>", `${gc.cylindrical.v_R_km_s} km/s`],
    ["v<sub>φ</sub>", `${gc.cylindrical.v_phi_km_s} km/s`],
    ["v<sub>z</sub>", `${gc.cylindrical.v_z_km_s} km/s`],
  ]);

  // Orbital parameters
  const op = data.orbit.params;
  document.getElementById("orbit-params").innerHTML = buildTableHTML([
    ["Pericenter", `${op.pericenter_kpc} kpc`],
    ["Apocenter", `${op.apocenter_kpc} kpc`],
    ["Eccentricity", `${op.eccentricity}`],
    ["z<sub>max</sub>", `${op.zmax_kpc} kpc`],
  ]);
}

// ---------------------------------------------------------------------------
// Initialize on page load
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  loadCMD();
  loadRVS();
  loadOrbit();
});
