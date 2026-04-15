"""
features.py — Build the ML-ready feature matrix
=================================================
Takes the OD demand and supply parquet files from ingest.py and:
  1. Joins them into a single OD-level table
  2. Adds structural features (the ex-ante variables from Mini #3 —
     population, distance, income, hub proximity, etc.)
  3. Labels each pair as "has direct service" or "ghost candidate"
  4. Writes the feature matrix for the model to consume

Structural features only — NO fares, NO load factors, NO anything that
wouldn't exist for a route that doesn't yet exist. This is the anti-
target-leakage strategy.

Usage:
    python src/features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

from config import DATA_DIR, FOCUS_ORIGINS, LEISURE_DESTINATIONS, HUB_AIRPORTS

# ---------- Airport metadata ----------
# Hard-coded lookup for the airports in our scope. In a bigger project you'd
# pull this from an OpenFlights dump or the BTS Master Coordinate file, but
# for a focused regional analysis this is honest and fast.
#
# Columns: lat, lon, metro_population (rough MSA), median_income (rough MSA),
#          is_hub, is_leisure, is_focus_origin
AIRPORT_META = {
    # Focus origins (Upper Midwest rural)
    "DBQ": (42.402, -90.709,  97_000, 61_000),
    "MCW": (43.158, -93.331,  43_000, 54_000),
    "FOD": (42.551, -94.193,  36_000, 53_000),
    "BRL": (40.783, -91.125,  47_000, 52_000),
    "ALO": (42.557, -92.400, 170_000, 55_000),
    "DEC": (39.835, -88.866, 104_000, 51_000),
    "UIN": (39.943, -91.195,  75_000, 50_000),
    "MWA": (37.755, -89.011,  57_000, 48_000),
    "CGI": (37.225, -89.571,  97_000, 49_000),
    "IRK": (40.093, -92.545,  25_000, 46_000),
    "IWD": (46.527, -90.131,  15_000, 44_000),
    "CMX": (47.168, -88.489,  37_000, 47_000),
    "RHI": (45.631, -89.467,  37_000, 51_000),
    "EAU": (44.865, -91.484, 170_000, 58_000),
    "LSE": (43.879, -91.257, 140_000, 57_000),
    # Leisure destinations
    "MCO": (28.429, -81.309, 2_700_000, 63_000),
    "LAS": (36.080, -115.152, 2_300_000, 62_000),
    "PHX": (33.434, -112.012, 4_900_000, 66_000),
    "TPA": (27.975, -82.533, 3_200_000, 61_000),
    "FLL": (26.072, -80.152, 1_950_000, 62_000),
    "MIA": (25.793, -80.291, 2_700_000, 58_000),
    "SAN": (32.733, -117.189, 3_300_000, 83_000),
    "LAX": (33.942, -118.408, 10_000_000, 77_000),
    "DEN": (39.861, -104.673, 3_000_000, 85_000),
    "SLC": (40.788, -111.978, 1_260_000, 82_000),
    "RSW": (26.536, -81.755, 770_000, 61_000),
    "PBI": (26.683, -80.095, 1_500_000, 70_000),
    "CUN": (21.037, -86.877, 900_000, 30_000),  # intl
    "PUJ": (18.567, -68.363, 300_000, 20_000),  # intl
    # Hubs
    "ORD": (41.979, -87.904, 9_500_000, 78_000),
    "MSP": (44.882, -93.222, 3_700_000, 85_000),
    "DTW": (42.212, -83.353, 4_400_000, 67_000),
    "MDW": (41.785, -87.752, 9_500_000, 78_000),
    "STL": (38.748, -90.370, 2_800_000, 70_000),
    "MKE": (42.947, -87.897, 1_570_000, 68_000),
    "MCI": (39.299, -94.714, 2_200_000, 73_000),
}

META_DF = pd.DataFrame.from_dict(
    AIRPORT_META,
    orient="index",
    columns=["lat", "lon", "metro_pop", "median_income"],
).reset_index().rename(columns={"index": "airport"})


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points, in statute miles."""
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_feature_matrix() -> pd.DataFrame:
    """
    Build the full OD-level feature matrix for modeling.

    Each row is a (origin, dest) pair. Features are all STRUCTURAL —
    they'd exist for a route whether or not it's currently flown.
    """
    demand = pd.read_parquet(DATA_DIR / "od_demand.parquet")
    supply = pd.read_parquet(DATA_DIR / "od_supply.parquet")

    # Outer join so we keep pairs that only exist in demand (ghost candidates)
    od = demand.merge(supply, on=["origin", "dest"], how="outer")
    od["true_passengers"]   = od["true_passengers"].fillna(0)
    od["direct_passengers"] = od["direct_passengers"].fillna(0)
    od["direct_seats"]      = od["direct_seats"].fillna(0)
    od["direct_departures"] = od["direct_departures"].fillna(0)

    # has_direct_service = T-100 recorded at least meaningful direct flights
    # (we use a threshold rather than >0 to filter out charter noise)
    od["has_direct_service"] = od["direct_departures"] >= 50

    # Join origin metadata
    od = od.merge(
        META_DF.add_prefix("o_"),
        left_on="origin", right_on="o_airport", how="left",
    ).drop(columns=["o_airport"])

    # Join dest metadata
    od = od.merge(
        META_DF.add_prefix("d_"),
        left_on="dest", right_on="d_airport", how="left",
    ).drop(columns=["d_airport"])

    # Drop pairs where we don't have metadata for both ends
    # (these are outside our scope — e.g. flights to random third airports)
    before = len(od)
    od = od.dropna(subset=["o_lat", "d_lat"])
    print(f"[features] Kept {len(od):,} of {before:,} pairs after metadata join")

    # ---------- Structural features ----------
    od["distance_mi"] = haversine_miles(
        od["o_lat"], od["o_lon"], od["d_lat"], od["d_lon"]
    )
    od["pop_product"]   = od["o_metro_pop"] * od["d_metro_pop"]
    od["pop_sum"]       = od["o_metro_pop"] + od["d_metro_pop"]
    od["income_mean"]   = (od["o_median_income"] + od["d_median_income"]) / 2
    od["income_ratio"]  = od["d_median_income"] / od["o_median_income"]

    # Is the destination a major leisure market? (1/0 flag)
    od["dest_is_leisure"] = od["dest"].isin(LEISURE_DESTINATIONS).astype(int)
    od["orig_is_focus"]   = od["origin"].isin(FOCUS_ORIGINS).astype(int)
    od["dest_is_hub"]     = od["dest"].isin(HUB_AIRPORTS).astype(int)

    # Gravity-model style feature: pop_product / distance^2.
    # This is a classic transportation-demand heuristic and it's completely
    # exogenous to whether a route is currently flown.
    od["gravity"] = od["pop_product"] / (od["distance_mi"] ** 2 + 1)

    return od


def main():
    print("=" * 60)
    print("Bridging the Gap — Feature Engineering")
    print("=" * 60)

    od = build_feature_matrix()

    out = DATA_DIR / "features.parquet"
    od.to_parquet(out, index=False)
    print(f"\n[write] {out}")
    print(f"[write] {len(od):,} rows, {len(od.columns)} columns")

    print("\n[sanity] Breakdown of direct-service status:")
    print(od["has_direct_service"].value_counts().to_string())

    print("\n[sanity] Top 10 ghost candidates (high demand, no direct service):")
    ghosts = od[~od["has_direct_service"]].nlargest(10, "true_passengers")
    print(ghosts[["origin", "dest", "true_passengers", "distance_mi"]].to_string(index=False))

    print("\nDone. Next step: python src/model.py")


if __name__ == "__main__":
    main()
