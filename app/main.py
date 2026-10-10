"""FastAPI backend for the fossil-fuel CO2 emissions dashboard."""
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STATIC = ROOT / "static"

# Global Carbon Budget 2025 (v15) reports million tonnes of CO2 (Mt CO2) - served as-is.
UNIT = "Mt CO₂"
KT_TO_MT = 1

FUELS = {
    "solid":   {"label": "Coal",        "hint": "Solid fuel",        "column": "solid"},
    "liquid":  {"label": "Oil",         "hint": "Liquid fuel",       "column": "liquid"},
    "gas":     {"label": "Gas",         "hint": "Natural gas",       "column": "gas"},
    "cement":  {"label": "Cement",      "hint": "Cement production", "column": "cement"},
    "flaring": {"label": "Flaring",     "hint": "Gas flaring",       "column": "flaring"},
    "other":   {"label": "Other",       "hint": "Other industry",    "column": "other"},
    "total":   {"label": "All sources", "hint": "Total emissions",   "column": "total"},
}
MIX_KEYS = ("solid", "liquid", "gas", "cement", "flaring", "other")
FuelKey = Literal["solid", "liquid", "gas", "cement", "flaring", "other", "total"]
CONTINENTS = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]

if not (DATA / "processed.csv").exists():
    raise RuntimeError("data/processed.csv missing - run `python build_data.py` first.")

DF = pd.read_csv(DATA / "processed.csv")
NUM_COLS = ["total", "solid", "liquid", "gas", "cement", "flaring", "other"]
DF[NUM_COLS] = DF[NUM_COLS].clip(lower=0)  # blanks (NaN) mean "no record" and stay NaN
_has_total = DF[DF["total"].notna()]
ENTITIES = (DF.groupby("key")
              .agg(iso3=("iso3", "first"), country=("country", "first"),
                   continent=("continent", "first"), historical=("historical", "first"),
                   lat=("lat", "first"), lon=("lon", "first"))
              .join(_has_total.groupby("key")["year"].agg(first_year="min", last_year="max"))
              .reset_index())
YEAR_MIN, YEAR_MAX = int(DF["year"].min()), int(DF["year"].max())

app = FastAPI(title="Fossil-fuel CO2 Emissions Dashboard", version="1.0.0")
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


def clean(v):
    return None if pd.isna(v) else v


def period_values(fuel: str, year: int, mode: str) -> pd.DataFrame:
    """One row per entity with its value (Mt CO2) for the chosen fuel and period."""
    col = FUELS[fuel]["column"]
    if mode == "year":
        sub = DF[DF["year"] == year][["key", col]]
    else:  # cumulative from the first record up to and including `year`
        sub = DF[DF["year"] <= year].groupby("key", as_index=False)[col].sum(min_count=1)
    sub = sub.rename(columns={col: "value"})
    sub["value"] = sub["value"] * KT_TO_MT
    return sub


@app.get("/api/meta")
def meta():
    return {
        "fuels": [{"key": k, "label": v["label"], "hint": v["hint"]} for k, v in FUELS.items()],
        "continents": CONTINENTS,
        "year_min": YEAR_MIN,
        "year_max": YEAR_MAX,
        "unit": UNIT,
        "unit_note": "Million tonnes of CO2 from fossil fuels and industry (Global Carbon Budget 2025). "
                     "Excludes international aviation and shipping.",
    }


@app.get("/api/map")
def map_data(fuel: FuelKey = "solid",
             year: int = Query(YEAR_MAX, ge=YEAR_MIN, le=YEAR_MAX),
             mode: Literal["year", "cumulative"] = "year"):
    """Emissions per country for the map, ranking panel and continent summary."""
    vals = period_values(fuel, year, mode)
    df = ENTITIES.merge(vals, on="key", how="left")
    # blank (NaN) values = "no record for this period"

    world_total = float(df["value"].sum())
    df = df.sort_values("value", ascending=False, na_position="last")
    df["rank"] = df["value"].rank(ascending=False, method="min")
    df["share"] = df["value"] / world_total if world_total else None
    df["continent_rank"] = df.groupby("continent")["value"].rank(ascending=False, method="min")

    cont = (df.groupby("continent")["value"].agg(["sum", "count"])
              .reindex(CONTINENTS).fillna(0).reset_index())
    cont["share"] = cont["sum"] / world_total if world_total else 0

    rows = [{
        "key": r.key, "iso3": clean(r.iso3), "country": r.country, "continent": r.continent,
        "historical": bool(r.historical), "lat": clean(r.lat), "lon": clean(r.lon),
        "value": None if pd.isna(r.value) else round(float(r.value), 4),
        "rank": None if pd.isna(r.rank) else int(r.rank),
        "continent_rank": None if pd.isna(r.continent_rank) else int(r.continent_rank),
        "share": None if pd.isna(r.share) else round(float(r.share), 5),
    } for r in df.itertuples()]
    return {
        "fuel": fuel, "year": year, "mode": mode, "unit": UNIT,
        "world_total": round(world_total, 3),
        "max_value": round(float(df["value"].max()), 3) if df["value"].notna().any() else 0,
        "countries_with_data": int(df["value"].notna().sum()),
        "continents": [{"continent": r.continent, "total": round(float(r.sum), 3),
                        "countries": int(r.count), "share": round(float(r.share), 5)}
                       for r in cont.itertuples()],
        "countries": rows,
    }


@app.get("/api/country/{key}")
def country(key: str, fuel: FuelKey = "solid",
            year: int = Query(YEAR_MAX, ge=YEAR_MIN, le=YEAR_MAX),
            mode: Literal["year", "cumulative"] = "year"):
    """Time series and facts for one country (zoomed-in view)."""
    key = key.upper()
    ent = ENTITIES[ENTITIES["key"] == key]
    if ent.empty:
        raise HTTPException(404, f"Unknown country: {key}")
    e = ent.iloc[0]
    sub = DF[DF["key"] == key].sort_values("year")
    series = {"years": sub["year"].tolist()}
    for k, f in FUELS.items():
        series[k] = [None if pd.isna(v) else round(float(v) * KT_TO_MT, 4) for v in sub[f["column"]]]

    col = FUELS[fuel]["column"]
    peak_row = sub.loc[sub[col].idxmax()] if sub[col].notna().any() and sub[col].max() > 0 else None
    # mix for the chosen period
    if mode == "year":
        window = sub[sub["year"] == year]
    else:
        window = sub[sub["year"] <= year]
    mix = {k: round(float(window[FUELS[k]["column"]].sum() * KT_TO_MT), 3) for k in MIX_KEYS}
    pc = sub[sub["year"] == year]["per_capita"]
    return {
        "key": key, "iso3": clean(e.iso3), "country": e.country, "continent": e.continent,
        "historical": bool(e.historical), "lat": clean(e.lat), "lon": clean(e.lon),
        "first_year": int(e.first_year), "last_year": int(e.last_year),
        "unit": UNIT, "fuel": fuel, "year": year, "mode": mode,
        "peak": None if peak_row is None else {
            "year": int(peak_row["year"]), "value": round(float(peak_row[col] * KT_TO_MT), 3)},
        "mix": mix,
        "per_capita_t_carbon": None if pc.empty or pd.isna(pc.iloc[0]) else round(float(pc.iloc[0]), 3),
        "series": series,
    }


@app.get("/api/world.geojson")
def world_geojson():
    return FileResponse(DATA / "world.geojson", media_type="application/geo+json",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/healthz")
def healthz():
    return JSONResponse({"status": "ok", "rows": int(len(DF)), "entities": int(len(ENTITIES))})


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
