"""
figures.py — Generate the four report figures
===============================================
Produces the figures described in Mini-deliverable #5:
  Fig 1: Hub-and-spoke leakage flow (top leakage pairs)
  Fig 2: Model validation scatter (predicted vs actual on holdout)
  Fig 3: Subsidy Efficiency Ratio comparison (bar chart)
  Fig 4: Top 10 Ghost Routes (horizontal bar chart)

All figures use a consistent style. Saved as PNG (for the poster) and
PDF (for the report).

Usage:
    python src/figures.py
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

from config import (
    DATA_DIR, FIG_DIR, TABLES_DIR,
    EAS_AVG_SUBSIDY_PER_PASSENGER, ASSUMED_DIRECT_LOAD_FACTOR,
    ASSUMED_AIRCRAFT_SEATS,
)

# ---------- Style ----------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Consistent color palette — legacy = muted red, proposed = teal
COLOR_LEGACY   = "#c44536"
COLOR_PROPOSED = "#2a9d8f"
COLOR_NEUTRAL  = "#264653"
COLOR_ACCENT   = "#e9c46a"


def save_both(fig, name: str):
    """Save a figure as both PNG and PDF."""
    fig.savefig(FIG_DIR / f"{name}.png")
    fig.savefig(FIG_DIR / f"{name}.pdf")
    print(f"[write] {FIG_DIR / name}.png + .pdf")


# ---------- Figure 1: Leakage flow ----------
def figure_1_leakage():
    """
    Top leakage pairs — rural origins with high DB1B demand to leisure
    destinations but low direct service. Shown as a horizontal bar chart
    of leakage intensity (true demand minus direct passengers carried).

    (A full Sankey would be nicer but adds a dependency and is fragile —
    the bar chart version makes the same point cleanly.)
    """
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    df = df[df["orig_is_focus"] == 1]
    df = df[df["dest_is_leisure"] == 1]
    df["leaked_demand"] = df["true_passengers"] - df["direct_passengers"]
    top = df.nlargest(12, "leaked_demand").iloc[::-1]  # reversed for bar order

    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = top["origin"] + " → " + top["dest"]
    ax.barh(labels, top["leaked_demand"], color=COLOR_LEGACY, alpha=0.85)
    ax.set_xlabel("Annual passengers routed through hubs instead of direct")
    ax.set_title("Figure 1. Passenger Leakage: Demand the Hub-and-Spoke Model Is Hiding")
    ax.grid(axis="x", alpha=0.3)
    fig.text(0.01, -0.02,
             "Source: BTS DB1B Market + T-100 Segment. Leakage = DB1B true demand − T-100 direct carriage.",
             fontsize=8, style="italic", color="#555")
    save_both(fig, "fig1_leakage")
    plt.close(fig)


# ---------- Figure 2: Model validation ----------
def figure_2_validation():
    """
    Predicted vs actual passenger volume on the holdout set.
    The y=x reference line shows perfect prediction; tight clustering
    around the line = model works.
    """
    pred = pd.read_parquet(DATA_DIR / "holdout_predictions.parquet")

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(pred["y_true"], pred["y_pred"],
               s=22, alpha=0.55, color=COLOR_NEUTRAL, edgecolors="none")

    lim = max(pred["y_true"].max(), pred["y_pred"].max()) * 1.05
    ax.plot([1, lim], [1, lim], ls="--", color=COLOR_LEGACY,
            lw=1.5, label="Perfect prediction (y=x)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, lim)
    ax.set_ylim(1, lim)
    ax.set_xlabel("Actual annual passengers (DB1B, log scale)")
    ax.set_ylabel("Random Forest prediction (log scale)")
    ax.set_title("Figure 2. Ghost Route Model Validation (Holdout Set)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, which="both", alpha=0.25)

    # Annotate with overall fit
    from sklearn.metrics import r2_score
    r2 = r2_score(pred["y_true_log"], pred["y_pred_log"])
    ax.text(0.98, 0.05, f"log-scale R² = {r2:.2f}",
            transform=ax.transAxes, ha="right", fontsize=11,
            bbox=dict(facecolor="white", edgecolor="#ccc"))

    fig.text(0.01, -0.02,
             "Predictions from Random Forest trained on structural features only (Mini #3 anti-leakage strategy).",
             fontsize=8, style="italic", color="#555")
    save_both(fig, "fig2_validation")
    plt.close(fig)


# ---------- Figure 3: Subsidy Efficiency Ratio ----------
def figure_3_ser():
    """
    Compare the Subsidy Efficiency Ratio of legacy hub-and-spoke EAS
    flights vs the projected SER for the top ghost routes if they were
    subsidized at the same level but operated direct with our predicted
    load factors.

    CRUCIAL: this is where we connect Figure 2 (model) to Figure 3 (economics)
    — the Figure 2 predictions ARE the load-factor inputs for Figure 3. This
    resolves the Mini #5 "how do these figures relate?" feedback.
    """
    ghosts = pd.read_csv(TABLES_DIR / "top_ghost_routes.csv")
    top10 = ghosts.head(10).copy()

    # Projected load factor: predicted passengers / (seats × assumed frequency)
    # Assume daily service with a 76-seat regional jet, 365 days/yr
    annual_seats = ASSUMED_AIRCRAFT_SEATS * 365
    top10["projected_lf"] = np.clip(
        top10["predicted_passengers"] / annual_seats, 0.3, 0.95
    )
    # Subsidy cost per passenger at that load factor, assuming the same
    # total subsidy dollars currently spent per flight.
    top10["projected_subsidy_per_pax"] = (
        EAS_AVG_SUBSIDY_PER_PASSENGER * 0.4 / top10["projected_lf"]
    )

    categories = ["Legacy EAS\n(hub-and-spoke)", "Proposed Direct\n(Ghost Routes)"]
    values = [
        EAS_AVG_SUBSIDY_PER_PASSENGER,
        top10["projected_subsidy_per_pax"].mean(),
    ]
    colors = [COLOR_LEGACY, COLOR_PROPOSED]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar(categories, values, color=colors, alpha=0.9, width=0.55)
    ax.set_ylabel("Average subsidy cost per passenger (USD)")
    ax.set_title("Figure 3. Subsidy Efficiency: Legacy vs Proposed Direct Service")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"${val:,.0f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", alpha=0.3)

    fig.text(0.01, -0.02,
             "Proposed SER derived from Figure 2 model predictions × assumed 76-seat regional jet, daily service.",
             fontsize=8, style="italic", color="#555")
    save_both(fig, "fig3_ser")
    plt.close(fig)


# ---------- Figure 4: Top 10 Ghost Routes ----------
def figure_4_top_ghosts():
    """Ranked horizontal bar of top 10 ghost routes by predicted demand."""
    ghosts = pd.read_csv(TABLES_DIR / "top_ghost_routes.csv").head(10).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = ghosts["origin"] + " → " + ghosts["dest"]
    ax.barh(labels, ghosts["predicted_passengers"],
            color=COLOR_PROPOSED, alpha=0.9)

    for i, (_, row) in enumerate(ghosts.iterrows()):
        ax.text(row["predicted_passengers"] * 1.01, i,
                f"{int(row['predicted_passengers']):,}",
                va="center", fontsize=9)

    ax.set_xlabel("Model-predicted annual passenger demand")
    ax.set_title("Figure 4. Top 10 Ghost Routes: Highest Unmet Direct Demand")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, ghosts["predicted_passengers"].max() * 1.18)

    fig.text(0.01, -0.02,
             "Routes with meaningful DB1B demand and no current direct service (<50 departures/yr). Model: Random Forest.",
             fontsize=8, style="italic", color="#555")
    save_both(fig, "fig4_top_ghosts")
    plt.close(fig)


def main():
    print("=" * 60)
    print("Bridging the Gap — Figure Generation")
    print("=" * 60)
    figure_1_leakage()
    figure_2_validation()
    figure_3_ser()
    figure_4_top_ghosts()
    print("\nAll figures generated. Report-writing time.")


if __name__ == "__main__":
    main()
