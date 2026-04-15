"""
config.py — Bridging the Gap: Ghost Route Detection
====================================================
Single source of truth for project scope, paths, and constants.
Change things here, not scattered across the codebase.
"""

from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
DB1B_DIR     = DATA_DIR / "db1b"
T100_DIR     = DATA_DIR / "t100"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
FIG_DIR      = OUTPUTS_DIR / "figures"
TABLES_DIR   = OUTPUTS_DIR / "tables"

for d in (FIG_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Scope: Upper Midwest EAS focus ----------
# These are the rural/small-market airports we're analyzing. Dubuque is our
# hero case, the rest are peer markets with similar structural profiles.
# NOTE: not all of these are currently EAS-subsidized in 2026, but all have
# historically received subsidy or are the kind of small-community airport
# the program exists to serve. We'll filter by what's actually in the data.
FOCUS_ORIGINS = [
    "DBQ",  # Dubuque, IA — hero case
    "MCW",  # Mason City, IA
    "FOD",  # Fort Dodge, IA
    "BRL",  # Burlington, IA
    "ALO",  # Waterloo, IA
    "DEC",  # Decatur, IL
    "UIN",  # Quincy, IL
    "MWA",  # Marion/Carbondale, IL
    "CGI",  # Cape Girardeau, MO
    "IRK",  # Kirksville, MO
    "IWD",  # Ironwood, MI
    "CMX",  # Hancock/Houghton, MI
    "RHI",  # Rhinelander, WI
    "EAU",  # Eau Claire, WI
    "LSE",  # La Crosse, WI
]

# Leisure destinations we expect to see "ghost" demand toward.
# These are the big warm-weather / entertainment markets people from the
# Upper Midwest actually fly to for vacation.
LEISURE_DESTINATIONS = [
    "MCO",  # Orlando
    "LAS",  # Las Vegas
    "PHX",  # Phoenix
    "TPA",  # Tampa
    "FLL",  # Fort Lauderdale
    "MIA",  # Miami
    "SAN",  # San Diego
    "LAX",  # Los Angeles
    "DEN",  # Denver (ski gateway)
    "SLC",  # Salt Lake City (ski gateway)
    "RSW",  # Fort Myers
    "PBI",  # West Palm Beach
    "CUN",  # Cancun (intl leisure)
    "PUJ",  # Punta Cana
]

# Major hub airports we expect to be the connecting points for the
# hub-and-spoke leakage pattern.
HUB_AIRPORTS = [
    "ORD",  # Chicago O'Hare
    "MSP",  # Minneapolis
    "DTW",  # Detroit
    "MDW",  # Chicago Midway
    "STL",  # St. Louis
    "MKE",  # Milwaukee
    "MCI",  # Kansas City
]

# ---------- Model hyperparameters ----------
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_leaf": 3,
    "random_state": 42,
    "n_jobs": -1,
}

# Temporal split: train on earlier data, test on most recent quarter.
# This is the anti-leakage strategy from Mini-deliverable #3.
HOLDOUT_QUARTER = ("2025", "Q3")  # adjust to whatever's actually latest

# ---------- Economic constants for Subsidy Efficiency Ratio ----------
# Rough EAS subsidy averages — we'll refine from the DOT EAS report.
# Placeholder values; update with real figures from DOT's annual EAS report.
EAS_AVG_SUBSIDY_PER_PASSENGER = 300.0  # dollars (legacy hub-and-spoke)
ASSUMED_DIRECT_LOAD_FACTOR    = 0.80   # what we project for direct leisure
ASSUMED_AIRCRAFT_SEATS        = 76     # typical regional jet (E175)
