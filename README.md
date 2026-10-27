# fire-prediction-model-csci4502

Group project for CSCI4502

## A Closer Look at Our Data:

## gridMET

gridMET blends spatial attributes of weather data (provided by PRISM) with desirable temporal attributes. This produces a high-spatial resolution meterological data (spanning 1972-yesterday)
Each "gridMET grid" corresponds to a centroid of 4-km x 4-km pixel.

### Primary Climate Variables:

- Maximum temperature
- Minimum temperature
- Precipitation Accumulation
- Downward surface shortwave radiation
- Wind-velocity
- Humidity

### Derived Variables

- ASCE
- Energy Release Component
- Burning Index

### What gridMET is and is not

gridMET is not satellite imagery or a typical CSV dataset. It is a gridded model stored as a massive stack of 3D digital maps. Each layer represents a single dat, and each square represents a 4-km cell.

## LANDFIRE

LANDFIRE is a shared interagency wildfire data program across the US. It provides many geo-spatial products of biological and ecological data including vegetation types and fire behavior fuel. The dataset is mainly GIS raster data and also some tabular database data. :w

### Raster Data

- 30x30 meter raster grids delivered in GeoTIFF.
- Each pixel score corresponds to a specific piece of environmental data.

### Vector Data

- Mapping of perimeter boundary of a 2024 wildfire, burn, or logging project.

### Attribute Tables

- Every raster and vectored layer is backed by a database table
- Provides the mapping between pixel values and real environmental data.

## MTBS

MTBS is an interagency program that maps burn severity and extent of large fires across all lands in the continental US, Hawaii, Alaska, and PR.

### Fire Perimeter Polygons

- ESRI shapefiles and feature layers containing fire centroid points and boundary acres
- Tabluar attributes as well like fire type, ignition/containment details, reporting agency, e.t.c.

### Burn Severity Raster

- Similar to LANDFIRE the resolution is 30x30 meter squares.
- Measurements collected using near-infrared and short-wave infrared bands to highlight areas where a burn occurred
- Comparing pre-fire and post-fire imagery to measure how much change occurred in a given 30-meter resolution.
- Classifying burn severity.
