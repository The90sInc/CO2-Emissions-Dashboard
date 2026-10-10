"""
Preprocess the CDIAC fossil-fuel CO2 dataset for the dashboard.

What it does
------------
1. Cleans country names -> ISO3 codes, display names and continents.
   Predecessor series that continue a modern country (e.g. FRG / GDR -> Germany,
   Sabah / Sarawak -> Malaysia) are merged so a country's history is continuous.
   Defunct entities with no modern equivalent (USSR, Czechoslovakia, ...) are kept
   as "historical" entries: they appear in rankings / details but not on the map.
2. Repairs geocodes. The supplied Latitude/Longitude columns contain errors
   (e.g. China in Helsinki, Cameroon in New York) and 992 blanks. Each coordinate is
   validated with GeoPy (geodesic distance) against the country's boundary polygon;
   bad ones are replaced by, in order:
     a) a cached GeoPy/Nominatim geocode        (data/geocache.json)
     b) the polygon's representative point
     c) a manual override for tiny territories with no polygon
3. Writes data/processed.csv, which the FastAPI app loads.

Usage
-----
    python build_data.py                 # offline, uses cache + polygons
    python build_data.py --online        # also queries Nominatim via GeoPy for
                                         # anything unresolved (1 req/sec, cached)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pycountry
import pycountry_convert as pcc
from geopy.distance import geodesic
from shapely.geometry import Point, shape

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SRC_CSV = DATA / "fossil_fuel_emission_with_geocodes.csv"
GEOJSON = DATA / "world.geojson"
CACHE = DATA / "geocache.json"
OUT = DATA / "processed.csv"

FUELS = ["Solid Fuel", "Liquid Fuel", "Gas Fuel", "Cement", "Gas Flaring"]

# ---- name -> ISO3 (only where pycountry's lookup would fail or be wrong) -------
NAME_TO_ISO3 = {
    "ANTIGUA & BARBUDA": "ATG", "BOSNIA & HERZEGOVINA": "BIH", "BRUNEI (DARUSSALAM)": "BRN",
    "CHINA (MAINLAND)": "CHN", "COTE D IVOIRE": "CIV", "CONGO": "COG",
    "DEMOCRATIC PEOPLE S REPUBLIC OF KOREA": "PRK",
    "DEMOCRATIC REPUBLIC OF THE CONGO (FORMERLY ZAIRE)": "COD",
    "FALKLAND ISLANDS (MALVINAS)": "FLK", "FRANCE (INCLUDING MONACO)": "FRA",
    "HONG KONG SPECIAL ADMINSTRATIVE REGION OF CHINA": "HKG",
    "MACAU SPECIAL ADMINSTRATIVE REGION OF CHINA": "MAC",
    "ISLAMIC REPUBLIC OF IRAN": "IRN", "ITALY (INCLUDING SAN MARINO)": "ITA",
    "LAO PEOPLE S DEMOCRATIC REPUBLIC": "LAO", "LIBYAN ARAB JAMAHIRIYAH": "LBY",
    "MACEDONIA": "MKD", "MYANMAR (FORMERLY BURMA)": "MMR",
    "OCCUPIED PALESTINIAN TERRITORY": "PSE", "PLURINATIONAL STATE OF BOLIVIA": "BOL",
    "REPUBLIC OF CAMEROON": "CMR", "REPUBLIC OF KOREA": "KOR", "REPUBLIC OF MOLDOVA": "MDA",
    "REPUBLIC OF SOUTH SUDAN": "SSD", "REPUBLIC OF SUDAN": "SDN", "SUDAN": "SDN",
    "RUSSIAN FEDERATION": "RUS", "SAINT MARTIN (DUTCH PORTION)": "SXM",
    "SAO TOME & PRINCIPE": "STP", "ST. KITTS-NEVIS": "KNA", "ST. PIERRE & MIQUELON": "SPM",
    "ST. VINCENT & THE GRENADINES": "VCT", "SWAZILAND": "SWZ", "SYRIAN ARAB REPUBLIC": "SYR",
    "TIMOR-LESTE (FORMERLY EAST TIMOR)": "TLS", "UNITED REPUBLIC OF TANZANIA": "TZA",
    "UNITED STATES OF AMERICA": "USA", "VIET NAM": "VNM", "VENEZUELA": "VEN",
    "WALLIS AND FUTUNA ISLANDS": "WLF", "YEMEN": "YEM", "BONAIRE, SAINT EUSTATIUS, AND SABA": "BES",
    "FAEROE ISLANDS": "FRO", "FEDERATED STATES OF MICRONESIA": "FSM", "REUNION": "REU",
    "TURKEY": "TUR", "CAPE VERDE": "CPV", "CURACAO": "CUW", "SAINT HELENA": "SHN", "GUINEA BISSAU": "GNB", "GEORGIA": "GEO", "NIGER": "NER", "NIGERIA": "NGA",
    "YUGOSLAVIA (MONTENEGRO & SERBIA)": "SRB",
    "PACIFIC ISLANDS (PALAU)": "PLW", "KUWAITI OIL FIRES": "KWT",
    "FORMER PANAMA CANAL ZONE": "PAN", "ST. KITTS-NEVIS-ANGUILLA": "KNA",
    # predecessor series merged into the modern country
    "FEDERAL REPUBLIC OF GERMANY": "DEU", "FORMER GERMAN DEMOCRATIC REPUBLIC": "DEU",
    "FORMER YEMEN": "YEM", "FORMER DEMOCRATIC YEMEN": "YEM",
    "PENINSULAR MALAYSIA": "MYS", "SABAH": "MYS", "SARAWAK": "MYS",
    "EAST & WEST PAKISTAN": "PAK", "TANGANYIKA": "TZA", "ZANZIBAR": "TZA",
    "DEMOCRATIC REPUBLIC OF VIETNAM": "VNM", "REPUBLIC OF SOUTH VIETNAM": "VNM",
    "JAPAN (EXCLUDING THE RUYUKU ISLANDS)": "JPN", "RYUKYU ISLANDS": "JPN",
}

# entities with no modern equivalent: kept, shown as "historical"
HISTORICAL = {
    "USSR": ("Soviet Union (USSR)", "Europe"),
    "CZECHOSLOVAKIA": ("Czechoslovakia", "Europe"),
    "YUGOSLAVIA (FORMER SOCIALIST FEDERAL REPUBLIC)": ("Yugoslavia (SFR)", "Europe"),
    "UNITED KOREA": ("United Korea (pre-1945)", "Asia"),
    "RWANDA-URUNDI": ("Rwanda-Urundi", "Africa"),
    "NETHERLAND ANTILLES": ("Netherlands Antilles", "North America"),
    "NETHERLAND ANTILLES AND ARUBA": ("Netherlands Antilles & Aruba", "North America"),
    "FRENCH EQUATORIAL AFRICA": ("French Equatorial Africa", "Africa"),
    "FRENCH WEST AFRICA": ("French West Africa", "Africa"),
    "FRENCH INDO-CHINA": ("French Indo-China", "Asia"),
    "RHODESIA-NYASALAND": ("Rhodesia-Nyasaland", "Africa"),
    "LEEWARD ISLANDS": ("Leeward Islands", "North America"),
    "FEDERATION OF MALAYA-SINGAPORE": ("Federation of Malaya-Singapore", "Asia"),
}
DROP = {"ANTARCTIC FISHERIES"}

DISPLAY = {  # nicer display names than pycountry's formal ones
    "USA": "United States", "GBR": "United Kingdom", "RUS": "Russia", "CHN": "China",
    "IRN": "Iran", "KOR": "South Korea", "PRK": "North Korea", "VNM": "Vietnam",
    "TZA": "Tanzania", "BOL": "Bolivia", "VEN": "Venezuela", "SYR": "Syria",
    "LAO": "Laos", "MDA": "Moldova", "MKD": "North Macedonia", "COD": "DR Congo",
    "COG": "Congo", "CIV": "Côte d'Ivoire", "BRN": "Brunei", "FSM": "Micronesia",
    "HKG": "Hong Kong", "MAC": "Macau", "PSE": "Palestine", "TLS": "Timor-Leste",
    "FLK": "Falkland Islands", "SWZ": "Eswatini", "CPV": "Cape Verde", "MMR": "Myanmar",
    "TUR": "Türkiye", "SHN": "Saint Helena", "VGB": "British Virgin Islands", "CUW": "Curaçao", "CZE": "Czechia", "FRA": "France", "ITA": "Italy", "LBY": "Libya",
}

CONTINENT_NAMES = {"AF": "Africa", "AS": "Asia", "EU": "Europe", "NA": "North America",
                   "SA": "South America", "OC": "Oceania", "AN": "Antarctica"}
CONTINENT_OVERRIDE = {"RUS": "Europe", "TUR": "Asia", "TLS": "Asia", "CYP": "Asia",
                      "GEO": "Asia", "ARM": "Asia", "AZE": "Asia", "KAZ": "Asia",
                      "SXM": "North America", "BES": "North America", "CUW": "North America",
                      "REU": "Africa", "SHN": "Africa", "FSM": "Oceania", "PLW": "Oceania",
                      "MHL": "Oceania", "NRU": "Oceania", "KIR": "Oceania", "TUV": "Oceania"}

# tiny territories that have no polygon in the 110m boundary file and whose supplied
# coordinates are known to be wrong / missing (lat, lon)
COORD_OVERRIDE = {
    "HKG": (22.32, 114.17), "MAC": (22.20, 113.54), "DMA": (15.42, -61.35),
    "STP": (0.19, 6.61), "KNA": (17.34, -62.75), "SXM": (18.04, -63.07),
    "ANT": (12.20, -68.94), "FRA": (46.6, 2.4), "ITA": (42.8, 12.5), "LBY": (27.0, 17.0),
    "COD": (-2.9, 23.7), "PSE": (31.9, 35.2), "BES": (12.17, -68.28), "SRB": (44.0, 20.9),
    "VGB": (18.42, -64.64), "CUW": (12.17, -68.99), "AIA": (18.22, -63.05), "ABW": (12.52, -69.97),
}

POLY_TOLERANCE_KM = 250  # supplied coord must lie within this distance of the country


def resolve(name: str):
    """Return (key, iso3|None, display, continent|None, historical)."""
    n = name.strip().upper()
    if n in DROP:
        return None
    if n in HISTORICAL:
        disp, cont = HISTORICAL[n]
        return ("H_" + n[:12].replace(" ", "_"), None, disp, cont, True)
    iso3 = NAME_TO_ISO3.get(n)
    if iso3 is None:
        try:
            iso3 = pycountry.countries.lookup(n.title()).alpha_3
        except LookupError:
            try:
                iso3 = pycountry.countries.search_fuzzy(n.title())[0].alpha_3
            except LookupError:
                print(f"  ! unresolved name: {name}", file=sys.stderr)
                return None
    c = pycountry.countries.get(alpha_3=iso3)
    disp = DISPLAY.get(iso3) or (getattr(c, "common_name", None) or (c.name if c else name.title()))
    cont = CONTINENT_OVERRIDE.get(iso3)
    if cont is None:
        try:
            cont = CONTINENT_NAMES[pcc.country_alpha2_to_continent_code(c.alpha_2)]
        except Exception:
            cont = None
    return (iso3, iso3, disp, cont, False)


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
    """Approximate geodesic distance (GeoPy) from a point to the polygon's nearest vertex/edge."""
    pt = Point(lon, lat)
    if geom.contains(pt):
        return 0.0
    from shapely.ops import nearest_points
    near = nearest_points(geom, pt)[0]
    return geodesic((lat, lon), (near.y, near.x)).km


def rep_point(geom):
    """Representative point inside the largest part of the country (for markers)."""
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    pt = geom.representative_point()
    return pt.y, pt.x


def geocode_online(cache, queries):
    from geopy.extra.rate_limiter import RateLimiter
    from geopy.geocoders import Nominatim
    geo = Nominatim(user_agent="co2-emissions-dashboard", timeout=10)
    limited = RateLimiter(geo.geocode, min_delay_seconds=1.1)
    for iso3, q in queries:
        if iso3 in cache:
            continue
        try:
            r = limited(q)
        except Exception as e:  # network / service errors: skip, try again next run
            print(f"  geocode failed for {q}: {e}", file=sys.stderr)
            continue
        if r:
            cache[iso3] = [r.latitude, r.longitude]
            print(f"  geocoded {q}: {r.latitude:.3f}, {r.longitude:.3f}")
    CACHE.write_text(json.dumps(cache, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true", help="query Nominatim via GeoPy for unresolved coordinates")
    ap.add_argument("--src", default=str(SRC_CSV))
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    polys = load_polygons()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    # --- 1. names -> codes ----------------------------------------------------
    names = {n: resolve(n) for n in df["Country"].unique()}
    df["_r"] = df["Country"].map(names)
    df = df[df["_r"].notna()].copy()
    df["key"] = df["_r"].map(lambda r: r[0])
    df["iso3"] = df["_r"].map(lambda r: r[1])
    df["country"] = df["_r"].map(lambda r: r[2])
    df["continent"] = df["_r"].map(lambda r: r[3])
    df["historical"] = df["_r"].map(lambda r: r[4])
    df = df[df["continent"].notna()]

    # --- 2. merge predecessor series, keep clean columns -------------------------
    df["Year"] = df["Year"].astype(int)
    cols = ["Total"] + FUELS + ["Bunker fuels (Not in Total)"]
    first = df.sort_values("Total", ascending=False).groupby(["key", "Year"], as_index=False).first()
    agg = df.groupby(["key", "Year"], as_index=False)[cols].sum()
    pc = df.sort_values("Total", ascending=False).groupby(["key", "Year"])["Per Capita"].first().reset_index()
    out = agg.merge(pc, on=["key", "Year"]).merge(
        first[["key", "Year", "iso3", "country", "continent", "historical"]], on=["key", "Year"])
    # display name / continent per key: take the modern one
    meta = (df.sort_values("historical").groupby("key")
              .agg(iso3=("iso3", "first"), country=("country", "first"),
                   continent=("continent", "first"), historical=("historical", "first")))
    # prefer the modern name for merged keys (e.g. "Germany", not "Federal Republic of Germany")
    for key in meta.index:
        if not meta.loc[key, "historical"]:
            c = pycountry.countries.get(alpha_3=key)
            meta.loc[key, "country"] = DISPLAY.get(key) or (getattr(c, "common_name", None) or c.name)
    out = out.drop(columns=["iso3", "country", "continent", "historical"]).merge(meta, left_on="key", right_index=True)

    # --- 3. geocodes: validate with GeoPy, repair --------------------------------
    supplied = (df.dropna(subset=["Latitude", "Longitude"])
                  .groupby("key")[["Latitude", "Longitude"]].agg(lambda s: s.mode().iloc[0]))
    report, coords, pending = [], {}, []
    for key, row in meta.iterrows():
        iso3 = row["iso3"]
        geom = polys.get(iso3) if isinstance(iso3, str) else None
        cand = tuple(supplied.loc[key]) if key in supplied.index else None
        src = None
        if iso3 in COORD_OVERRIDE:
            coords[key], src = COORD_OVERRIDE[iso3], "manual override"
        elif geom is not None:
            if cand and km_to_polygon(geom, *cand) <= POLY_TOLERANCE_KM:
                coords[key], src = cand, "supplied (validated)"
            elif key in cache:
                coords[key], src = tuple(cache[key]), "geopy/nominatim cache"
            else:
                coords[key], src = rep_point(geom), "polygon representative point"
                report.append((row["country"], "supplied coord invalid" if cand else "missing", cand))
        elif isinstance(iso3, str):
            if key in cache:
                coords[key], src = tuple(cache[key]), "geopy/nominatim cache"
            elif cand:
                coords[key], src = cand, "supplied (no polygon to validate)"
            else:
                pending.append((key, row["country"]))
    if args.online and pending:
        print(f"Geocoding {len(pending)} places via GeoPy/Nominatim ...")
        geocode_online(cache, [(k, c) for k, c in pending])
        for k, _ in pending:
            if k in cache:
                coords[k] = tuple(cache[k])
    out["lat"] = out["key"].map(lambda k: coords.get(k, (None, None))[0])
    out["lon"] = out["key"].map(lambda k: coords.get(k, (None, None))[1])

    # --- 4. write -------------------------------------------------------------------
    out = out.rename(columns={"Year": "year", "Total": "total", "Solid Fuel": "solid",
                              "Liquid Fuel": "liquid", "Gas Fuel": "gas", "Cement": "cement",
                              "Gas Flaring": "flaring", "Per Capita": "per_capita",
                              "Bunker fuels (Not in Total)": "bunker"})
    out = out[["key", "iso3", "country", "continent", "historical", "year", "total", "solid", "liquid",
               "gas", "cement", "flaring", "per_capita", "bunker", "lat", "lon"]].sort_values(["key", "year"])
    out.to_csv(OUT, index=False)

    print(f"\nWrote {OUT.relative_to(ROOT)}: {len(out):,} rows, {out['key'].nunique()} entities "
          f"({int(meta['historical'].sum())} historical), years {out['year'].min()}-{out['year'].max()}")
    print(f"Coordinates repaired for {len(report)} countries with polygons:")
    for name, why, cand in report:
        print(f"   - {name}: {why} {cand if cand else ''}")
    unresolved = [c for k, c in pending if k not in coords]
    if unresolved:
        print(f"No coordinates for {len(unresolved)} places (shown in lists only; run with --online): "
              + ", ".join(unresolved))
    dup = out.groupby(["key", "year"]).size()
    print("Duplicate (entity, year) rows:", int((dup > 1).sum()))


if __name__ == "__main__":
    main()
