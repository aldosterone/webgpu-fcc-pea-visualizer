# WebGPU FCC PEA Visualizer

## Overview

WebGPU FCC PEA Visualizer is a browser-based geospatial visualization tool for exploring population and population density across FCC Partial Economic Areas (PEAs). The application aggregates tract-level population data into PEA-level summaries, renders land-based PEAs with WebGPU, and supports interactive inspection through hover tooltips, click-to-drill-down detail views, boundary overlays, and display toggles for dithering and color inversion.

PEA 416, which corresponds to the Gulf of Mexico water area, is intentionally excluded because the visualization is designed for land-based county aggregation and population analysis only.

## Features

- Browser-native WebGPU rendering.
- National PEA view with drill-down to tract-level detail.
- Population and population-density visualization modes.
- Dithering controls with three modes:
  - Off.
  - Bayer ordered dithering.
  - Blue-noise dithering.
- Optional boundary overlays.
- Hover highlighting and feature tooltips.
- Click-to-select PEA detail views.
- GPU-based picking for interactive feature identification.
- Light and dark presentation modes.

## Data

The visualization uses three primary data sources:

- `data/peaFile.geojson`: FCC PEA boundary geometries.
- `data/tracts.json`: Census tract topology.
- `data/tract_to_pea.json`: Mapping from tract GEOIDs to PEA numbers.

Population and density values are computed by joining tract geometries to their corresponding PEAs and aggregating tract-level population and land area. Density is derived as population per square mile.

## Data Preparation

The repository includes precomputed data files used by the visualization. Separate offline preprocessing scripts were used to download FCC PEA data, obtain census geography and population data, compute tract-to-PEA overlap and aggregation, and export the GeoJSON and mapping files consumed by the application.

These preprocessing steps are not required to run the visualization itself.

## How It Works

1. The app loads the PEA geometry, tract topology, and tract-to-PEA mapping.
2. The tract topology is converted into GeoJSON features.
3. Each tract is matched to a PEA using its GEOID.
4. Tract population and land area are aggregated to the PEA level.
5. PEA polygons are triangulated with Earcut.
6. The geometry and per-feature attributes are uploaded to the GPU.
7. WebGPU shaders render the map, apply dithering, and support interaction.
8. Hover and click picking identify the selected feature and update the view.

The rendering pipeline uses WebGPU for drawing, boundary overlays, highlighting, and feature picking. The app uses a square-root transform when converting normalized values into display intensity so the visualization remains readable across a wide population range.

## Controls

- **Dither**: cycles through Off, Bayer, and Blue Noise rendering.
- **Boundaries**: toggles boundary outlines.
- **Color**: switches between light and dark mode.
- **Visualization mode**: switches between total population and population density.
- **Mouse wheel**: zooms in and out.
- **Mouse drag**: pans the map.
- **Hover**: shows tooltips for the feature under the cursor.
- **Click on a PEA**: opens tract-level detail view for that PEA.
- **Back**: returns from detail view to the national view.

## Browser Requirements

This application requires a browser with WebGPU support. For best results, use a current Chromium-based browser or another browser build with WebGPU enabled.

The app must be served from a local or remote web server; opening the HTML file directly from the filesystem may not work correctly in all browsers.

## Project Structure

- `index.html` — Application shell and UI layout.
- `style.css` — Interface styling.
- `main.js` — WebGPU initialization, data loading, aggregation, rendering, interaction, and UI logic.
- `data/peaFile.geojson` — FCC PEA boundary geometries.
- `data/tracts.json` — Census tract topology.
- `data/tract_to_pea.json` — Tract-to-PEA mapping table.
- `assets/HDR_LA_0(256x256).png` — Blue-noise texture used for dithering.

## Design Notes

The map supports two visual encodings: total population and population density. Dithering is presented as a display technique rather than a data transformation, and it is intended to make tonal differences easier to perceive on the screen.

The national view and detail view are both cached and recomputed through GPU buffer updates so that the user can move between levels of detail without reloading the entire application.

## Limitations

- PEA 416 is excluded by design because it is a water area and does not belong in a county-based land aggregation workflow.
- Rendering performance depends on browser and GPU support.
- The tool assumes the included data files are present and correctly formatted.
- Feature matching relies on the tract-to-PEA mapping data being complete and consistent.

## Getting Started

1. Clone the repository.
2. Serve the project over HTTP.
3. Open the application in a WebGPU-capable browser.
4. Interact with the map using the on-screen controls.

If you are using a local development server, any simple static server should be sufficient.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
