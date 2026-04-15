"""
ingest.py — Load and clean BTS DB1B Market + T-100 Segment data
================================================================
Reads the raw CSVs you downloaded from transtats.bts.gov, filters them
to our Upper Midwest scope, cleans the messy bits, and writes parquet
files that the rest of the pipeline reads from.

Designed to be forgiving about column name variations — BTS uses
different cases and occasional renames across years and tables. If a
required field really isn't there, this script tells you exactly what
it found so you can debug or paste it to Claude.

Run this ONCE after downloading data. Subsequent scripts read from
the parquet outputs, not the raw CSVs (much faster).

Usage:
    python src/ingest.py
"""

import pandas as pd
from pathlib import Path

from config import (
    DB1B_DIR, T100_DIR, DATA_DIR,
    FOCUS_ORIGINS, LEISURE_DESTINATIONS, HUB_AIRPORTS,
)

RELEVANT_AIRPORTS = set(FOCUS_ORIGINS) | set(LEISURE_DESTINATIONS) | set(HUB_AIRPORTS)


# ---------- Column name mapping ----------
# Map from our logical column name → list of possible source column names.
# BTS uses different cases/underscores across tables and years. Matching is
# case-insensitive, so list variants here if you see new ones.
DB1B_COLUMN_MAP = {
    "itin_id":         ["ItinID", "ITIN_ID"],
    "mkt_id":          ["MktID", "MKT_ID"],
    "year":            ["Year", "YEAR"],
    "quarter":         ["Quarter", "QUARTER"],
    "origin":          ["Origin", "ORIGIN"],
    "dest":            ["Dest", "DEST"],
    "origin_city_mkt": ["OriginCityMarketID", "ORIGIN_CITY_MARKET_ID"],
    "dest_city_mkt":   ["DestCityMarketID", "DEST_CITY_MARKET_ID"],
    "passengers":      ["Passengers", "PASSENGERS"],
    "mkt_fare":        ["MktFare", "MARKET_FARE", "MKT_FARE"],
    "mkt_distance":    ["MktDistance", "MARKET_DISTANCE", "MKT_DISTANCE"],
    "nonstop_miles":   ["NonStopMiles", "NONSTOP_MILES"],
    "tk_carrier":      ["TkCarrier", "TICKET_CARRIER", "TK_CARRIER"],
}

T100_COLUMN_MAP = {
    "year":       ["YEAR", "Year"],
    "month":      ["MONTH", "Month"],
    "carrier":    ["UNIQUE_CARRIER", "UniqueCarrier", "CARRIER", "Carrier"],
    "origin":     ["ORIGIN", "Origin"],
    "dest":       ["DEST", "Dest"],
    "passengers": ["PASSENGERS", "Passengers"],
    "seats":      ["SEATS", "Seats"],
    "distance":   ["DISTANCE", "Distance"],
    "departures": ["DEPARTURES_PERFORMED", "Departures_Performed"],
}

# Columns we absolutely require to make the pipeline work.
DB1B_REQUIRED = {"origin", "dest", "passengers"}
T100_REQUIRED = {"origin", "dest", "passengers", "seats", "departures"}


def resolve_columns(actual_columns, column_map):
    """
    Match actual CSV columns against our candidate lists, case-insensitively.
    Returns dict: actual_col_name -> logical_name.
    """
    actual_lower = {c.lower().strip(): c for c in actual_columns}
    resolved = {}
    for logical, candidates in column_map.items():
        for cand in candidates:
            if cand.lower() in actual_lower:
                resolved[actual_lower[cand.lower()]] = logical
                break
    return resolved


def load_csv_with_mapping(path: Path, column_map: dict, required: set) -> pd.DataFrame:
    """Read a CSV, keep only columns we care about, rename to logical names."""
    header = pd.read_csv(path, nrows=0)
    resolved = resolve_columns(header.columns, column_map)

    if not resolved:
        raise ValueError(
            f"{path.name}: no recognized columns. "
            f"Actual columns were: {list(header.columns)}"
        )

    found_logical = set(resolved.values())
    missing = required - found_logical
    if missing:
        raise ValueError(
            f"{path.name}: missing REQUIRED columns {missing}.\n"
            f"  Found logical columns: {sorted(found_logical)}\n"
            f"  Actual columns in file: {list(header.columns)}\n"
            f"  → Add the correct variant to the COLUMN_MAP in src/ingest.py, "
            f"or paste this error to Claude."
        )

    df = pd.read_csv(
        path,
        usecols=list(resolved.keys()),
        low_memory=False,
    )
    df = df.rename(columns=resolved)
    return df


def load_db1b() -> pd.DataFrame:
    csv_files = sorted(DB1B_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSVs in {DB1B_DIR}. Download DB1B Market from "
            "transtats.bts.gov and unzip CSVs here."
        )

    print(f"[db1b] Found {len(csv_files)} files")
    frames = []
    for f in csv_files:
        print(f"[db1b]   {f.name} ...", end=" ", flush=True)
        df = load_csv_with_mapping(f, DB1B_COLUMN_MAP, DB1B_REQUIRED)
        df["origin"] = df["origin"].astype("string").str.strip()
        df["dest"]   = df["dest"].astype("string").str.strip()
        df = df[df["origin"].isin(RELEVANT_AIRPORTS) | df["dest"].isin(RELEVANT_AIRPORTS)]
        print(f"{len(df):,} relevant rows")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    print(f"[db1b] Total relevant rows: {len(out):,}")
    return out


def load_t100() -> pd.DataFrame:
    csv_files = sorted(T100_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSVs in {T100_DIR}. Download T-100 Domestic Segment "
            "(All Carriers) from transtats.bts.gov."
        )

    print(f"[t100] Found {len(csv_files)} files")
    frames = []
    for f in csv_files:
        print(f"[t100]   {f.name} ...", end=" ", flush=True)
        df = load_csv_with_mapping(f, T100_COLUMN_MAP, T100_REQUIRED)
        df["origin"] = df["origin"].astype("string").str.strip()
        df["dest"]   = df["dest"].astype("string").str.strip()
        df = df[df["origin"].isin(RELEVANT_AIRPORTS) | df["dest"].isin(RELEVANT_AIRPORTS)]
        print(f"{len(df):,} relevant rows")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    print(f"[t100] Total relevant rows: {len(out):,}")
    return out


def build_od_demand(db1b: pd.DataFrame) -> pd.DataFrame:
    """Aggregate DB1B from itinerary-level to OD-pair totals."""
    agg_spec = {"passengers": "sum"}
    if "mkt_fare" in db1b.columns:
        agg_spec["mkt_fare"] = "mean"
    if "mkt_distance" in db1b.columns:
        agg_spec["mkt_distance"] = "mean"
    if "itin_id" in db1b.columns:
        agg_spec["itin_id"] = "count"

    od = db1b.groupby(["origin", "dest"], as_index=False).agg(agg_spec)
    od = od.rename(columns={
        "passengers": "true_passengers",
        "mkt_fare": "avg_fare",
        "mkt_distance": "avg_distance",
        "itin_id": "n_itineraries",
    })
    # DB1B is a 10% sample — gross up to full-population estimate
    od["true_passengers"] = od["true_passengers"] * 10
    return od


def build_od_supply(t100: pd.DataFrame) -> pd.DataFrame:
    """Aggregate T-100 to OD-pair direct-service totals."""
    agg_spec = {
        "passengers": "sum",
        "seats": "sum",
        "departures": "sum",
    }
    if "distance" in t100.columns:
        agg_spec["distance"] = "mean"

    supply = t100.groupby(["origin", "dest"], as_index=False).agg(agg_spec)
    supply = supply.rename(columns={
        "passengers": "direct_passengers",
        "seats": "direct_seats",
        "departures": "direct_departures",
        "distance": "avg_segment_distance",
    })
    supply["load_factor"] = (
        supply["direct_passengers"] / supply["direct_seats"]
    ).where(supply["direct_seats"] > 0)
    return supply


def main():
    print("=" * 60)
    print("Bridging the Gap — Data Ingestion")
    print("=" * 60)

    db1b = load_db1b()
    t100 = load_t100()

    print("\n[aggregate] Building OD-level demand from DB1B...")
    od_demand = build_od_demand(db1b)
    print(f"[aggregate] {len(od_demand):,} unique OD pairs with demand")

    print("\n[aggregate] Building OD-level supply from T-100...")
    od_supply = build_od_supply(t100)
    print(f"[aggregate] {len(od_supply):,} unique OD pairs with direct supply")

    demand_out = DATA_DIR / "od_demand.parquet"
    supply_out = DATA_DIR / "od_supply.parquet"
    od_demand.to_parquet(demand_out, index=False)
    od_supply.to_parquet(supply_out, index=False)
    print(f"\n[write] {demand_out}")
    print(f"[write] {supply_out}")

    print("\n[sanity] Top 10 OD pairs by true demand:")
    print(od_demand.nlargest(10, "true_passengers").to_string(index=False))

    print("\nDone. Next step: python src/features.py")


if __name__ == "__main__":
    main()
