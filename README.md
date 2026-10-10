# Fossil Fuel CO₂ Emissions Dashboard

FastAPI backend + Leaflet/Chart.js frontend. One dashboard, switched per fuel
(solid, liquid, gas, cement, gas flaring, plus all sources). World map by continent;
click a country to zoom in and see its history, rank and fuel mix.

## Run locally
    pip install -r requirements.txt
    python build_data.py          # rebuilds data/processed.csv (already included)
    uvicorn app.main:app --reload
    # open http://127.0.0.1:8000   (API docs at /docs)

## Host it
    uvicorn app.main:app --host 0.0.0.0 --port $PORT
Works on Render, Railway, Fly.io or a VPS. Runtime needs only app/, static/,
data/processed.csv and data/world.geojson (packages: fastapi, uvicorn, pandas).

## GeoPy step (build_data.py)
The supplied Latitude/Longitude columns contain errors (e.g. China in Helsinki, Cameroon in
New York) and 992 blanks. build_data.py validates every coordinate with GeoPy's geodesic
distance against the country's boundary polygon and repairs bad ones from (1) a GeoPy/Nominatim
cache, (2) the polygon's representative point, (3) manual overrides for tiny territories.
`python build_data.py --online` also queries Nominatim through GeoPy for anything still
unresolved and caches it in data/geocache.json.

## API
- GET /api/meta
- GET /api/map?fuel=solid|liquid|gas|cement|flaring|total&year=2014&mode=year|cumulative
- GET /api/country/{ISO3}?fuel=...&year=...&mode=...
- GET /api/world.geojson, GET /healthz

## Notes on the data
- Units: the source (CDIAC) is in thousand metric tons of carbon; the API serves million
  tons of carbon (Mt C). Multiply by 3.664 for CO₂ mass.
- A few negative values (revisions) are clamped to 0.
- FRG/GDR -> Germany, Sabah/Sarawak/Peninsular Malaysia -> Malaysia, etc. are merged; defunct
  states (USSR, Czechoslovakia, Yugoslavia...) are kept as "historical" (rankings and details,
  not on the map).
