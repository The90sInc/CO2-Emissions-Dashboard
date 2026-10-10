"""
Preprocess the Global Carbon Budget (GCB 2025 v15) national fossil-CO2 file for the dashboard.

What it does
------------
1. Reads data/GCB2025v15_MtCO2_flat.csv (Mt CO2; coal / oil / gas / cement / flaring / other).
2. Drops non-country rows (Global, International Aviation / Shipping, Antarctica) and merges
   three ISO-less series into their modern country (Kuwaiti Oil Fires -> Kuwait,
   Pacific Islands (Palau) -> Palau, Ryukyu Islands -> Japan).
3. Assigns continents and tidy display names.
4. Geocodes every country. The GCB file has no coordinates, so each country gets one from, in order:
     a) a cached GeoPy/Nominatim result      (data/geocache.json)
     b) data/geoseed.json                    (coordinates validated in the previous build)
        - checked with GeoPy's geodesic distance against the country's boundary polygon
     c) the boundary polygon's representative point (Natural Earth, data/world.geojson)
5. Writes data/processed.csv, which the FastAPI app loads.

Usage
-----
    python build_data.py                 # offline: cache + seed + polygons
    python build_data.py --online        # also asks Nominatim (via GeoPy) for anything unresolved
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pycountry
import pycountry_convert as pcc
from geopy.distance import geodesic
from shapely.geometry import Point, shape
from shapely.ops import nearest_points

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SRC_CSV = DATA / "GCB2025v15_MtCO2_flat.csv"
GEOJSON = DATA / "world.geojson"
CACHE = DATA / "geocache.json"
SEED = DATA / "geoseed.json"
OUT = DATA / "processed.csv"

PARTS = ["Coal", "Oil", "Gas", "Cement", "Flaring", "Other"]
RENAME = {"Total": "total", "Coal": "solid", "Oil": "liquid", "Gas": "gas", "Cement": "cement",
          "Flaring": "flaring", "Other": "other", "Per Capita": "per_capita"}
VALUE_COLS = ["total", "solid", "liquid", "gas", "cement", "flaring", "other"]

DROP_ISO = {"WLD", "XIA", "XIS", "ATA"}          # world, aviation, shipping, Antarctica
MERGE_INTO = {"Kuwaiti Oil Fires": "KWT", "Pacific Islands (Palau)": "PLW", "Ryukyu Islands": "JPN"}

DISPLAY = {
    "USA": "United States", "Viet Nam": "Vietnam", "Brunei Darussalam": "Brunei",
    "Micronesia (Federated States of)": "Micronesia", "State of Palestine": "Palestine",
    "Faeroe Islands": "Faroe Islands", "Macao": "Macau", "Democratic Republic of the Congo": "DR Congo",
    "Wallis and Futuna Islands": "Wallis and Futuna", "Saint Helena": "Saint Helena",
    "Bonaire, Saint Eustatius and Saba": "Bonaire, St Eustatius & Saba",
}

CONTINENT_NAMES = {"AF": "Africa", "AS": "Asia", "EU": "Europe", "NA": "North America",
                   "SA": "South America", "OC": "Oceania"}
CONTINENT_OVERRIDE = {"RUS": "Europe", "TUR": "Asia", "TLS": "Asia", "CYP": "Asia", "GEO": "Asia",
                      "ARM": "Asia", "AZE": "Asia", "KAZ": "Asia", "KSV": "Europe",
                      "SXM": "North America", "BES": "North America", "CUW": "North America",
                      "REU": "Africa", "SHN": "Africa", "FSM": "Oceania", "PLW": "Oceania",
                      "MHL": "Oceania", "NRU": "Oceania", "KIR": "Oceania", "TUV": "Oceania"}
POLY_ALIAS = {"KSV": "KOS"}                        # GCB uses KSV for Kosovo, Natural Earth uses KOS
POLY_TOLERANCE_KM = 250


def continent_of(iso3):
    if iso3 in CONTINENT_OVERRIDE:
        return CONTINENT_OVERRIDE[iso3]
    try:
        c = pycountry.countries.get(alpha_3=iso3)
        return CONTINENT_NAMES[pcc.country_alpha2_to_continent_code(c.alpha_2)]
    except Exception:
        return None


def load_polygons():
    gj = json.loads(GEOJSON.read_text())
    polys = {}
    for f in gj["features"]:
        p = f["properties"]
        code = p.get("ISO_A3_EH") if p.get("ISO_A3_EH") not in (None, "-99") else p.get("ADM0_A3")
        polys[code] = shape(f["geometry"])
        if p.get("ADM0_A3"):
            polys.setdefault(p["ADM0_A3"], polys[code])
    return polys


def km_to_polygon(geom, lat, lon):
    """Geodesic distance (GeoPy) from a point to the nearest point of a country polygon."""
    pt = Point(lon, lat)
    if geom.contains(pt):
        return 0.0
    near = nearest_points(geom, pt)[0]
    return geodesic((lat, lon), (near.y, near.x)).km


def rep_point(geom):
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    pt = geom.representative_point()
    return pt.y, pt.x


def geocode_online(cache, items):
    from geopy.extra.rate_limiter import RateLimiter
    from geopy.geocoders import Nominatim
    geo = Nominatim(user_agent="co2-emissions-dashboard", timeout=10)
    limited = RateLimiter(geo.geocode, min_delay_seconds=1.1)
    for iso3, q in items:
        try:
            r = limited(q)
        except Exception as e:
            print(f"  geocode failed for {q}: {e}", file=sys.stderr)
            continue
        if r:
            cache[iso3] = [r.latitude, r.longitude]
            print(f"  geocoded {q}: {r.latitude:.3f}, {r.longitude:.3f}")
    CACHE.write_text(json.dumps(cache, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--src", default=str(SRC_CSV))
    args = ap.parse_args()

    # keep_default_na=False: otherwise pandas would read some codes/names as NaN
    raw = pd.read_csv(args.src, keep_default_na=False, na_values=[""])
    raw = raw.rename(columns={"ISO 3166-1 alpha-3": "iso3", "Year": "year", **RENAME})
    raw["iso3"] = raw["iso3"].fillna(raw["Country"].map(MERGE_INTO))
    raw = raw[raw["iso3"].notna() & ~raw["iso3"].isin(DROP_ISO)].copy()

    # --- tidy / merge ----------------------------------------------------------------
    # fill a missing Total from its parts where parts exist
    parts = raw[["solid", "liquid", "gas", "cement", "flaring", "other"]].sum(axis=1, min_count=1)
    fill = raw["total"].isna() & parts.notna() & (parts > 0)
    raw.loc[fill, "total"] = parts[fill]
    print(f"Filled Total from its parts in {int(fill.sum())} rows")

    names = (raw[raw["Country"].isin(MERGE_INTO) == False]
             .groupby("iso3")["Country"].first())          # noqa: E712
    main_rows = raw[~raw["Country"].isin(MERGE_INTO)][["iso3", "year", "per_capita"]]
    agg = raw.groupby(["iso3", "year"], as_index=False)[VALUE_COLS].sum(min_count=1)
    out = agg.merge(main_rows, on=["iso3", "year"], how="left")
    out["key"] = out["iso3"]
    out["country"] = out["iso3"].map(names).map(lambda n: DISPLAY.get(n, n))
    out["continent"] = out["iso3"].map(continent_of)
    out["historical"] = False
    bad = out[out["continent"].isna()]["iso3"].unique()
    if len(bad):
        print("  ! no continent, dropped:", list(bad), file=sys.stderr)
    out = out[out["continent"].notna()]
    # drop entities with no emissions at all
    has_any = out.groupby("iso3")["total"].transform(lambda s: s.notna().any())
    out = out[has_any]

    # --- geocodes ----------------------------------------------------------------------
    polys = load_polygons()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    seed = json.loads(SEED.read_text()) if SEED.exists() else {}
    ents = out.groupby("iso3").agg(country=("country", "first")).reset_index()
    coords, how, pending = {}, {}, []
    for iso3, name in zip(ents["iso3"], ents["country"]):
        geom = polys.get(POLY_ALIAS.get(iso3, iso3))
        s = (seed[iso3]["lat"], seed[iso3]["lon"]) if iso3 in seed else None
        if iso3 in cache:
            coords[iso3], how[iso3] = tuple(cache[iso3]), "GeoPy/Nominatim cache"
        elif s and geom is not None and km_to_polygon(geom, *s) <= POLY_TOLERANCE_KM:
            coords[iso3], how[iso3] = s, "seed (validated with GeoPy against polygon)"
        elif geom is not None:
            coords[iso3], how[iso3] = rep_point(geom), "polygon representative point"
        elif s:
            coords[iso3], how[iso3] = s, "seed (no polygon to validate)"
        else:
            pending.append((iso3, name))
    if args.online and pending:
        print(f"Geocoding {len(pending)} places via GeoPy/Nominatim ...")
        geocode_online(cache, pending)
        for iso3, _ in pending:
            if iso3 in cache:
                coords[iso3], how[iso3] = tuple(cache[iso3]), "GeoPy/Nominatim"
    out["lat"] = out["iso3"].map(lambda k: coords.get(k, (None, None))[0])
    out["lon"] = out["iso3"].map(lambda k: coords.get(k, (None, None))[1])

    cols = ["key", "iso3", "country", "continent", "historical", "year", "total", "solid", "liquid",
            "gas", "cement", "flaring", "other", "per_capita", "lat", "lon"]
    out = out[cols].sort_values(["key", "year"])
    out.to_csv(OUT, index=False)

    print(f"Wrote {OUT.relative_to(ROOT)}: {len(out):,} rows, {out['key'].nunique()} countries, "
          f"years {out['year'].min()}-{out['year'].max()}")
    from collections import Counter
    for k, v in Counter(how.values()).items():
        print(f"   coordinates from {k}: {v}")
    if pending:
        left = [n for i, n in pending if i not in coords]
        if left:
            print(f"   no coordinates (listed on charts only): {', '.join(left)}")
    print("Countries per continent:", dict(out.drop_duplicates('iso3')['continent'].value_counts()))


if __name__ == "__main__":
    main()
