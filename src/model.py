"""
model.py — Train the Ghost Route Detection model
===================================================
Trains a Random Forest regressor to predict TRUE passenger demand
(from DB1B) as a function of structural city-pair features.

Key methodological choices — all of these address Elle's feedback
from Mini #6 about formulating the prediction problem clearly:

  1. PREDICTION TARGET: log(true_passengers). We train only on pairs
     WITH direct service, because those are the pairs where T-100 gives
     us a reliable ground truth check. Then we predict on pairs WITHOUT
     direct service — these are the ghost route candidates.

  2. FEATURES: structural only (ex-ante). No fares, no load factors,
     no anything that wouldn't exist for a route that isn't flown.

  3. VALIDATION: two complementary strategies —
       a) Random 80/20 split for the main RMSE number
       b) Leave-one-route-out on high-demand known routes, to simulate
          "can the model predict a route it's never seen?"

  4. BASELINE: linear regression on the same features, so we can show
     the RF is adding real signal not just memorizing.

Usage:
    python src/model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from config import DATA_DIR, OUTPUTS_DIR, TABLES_DIR, RF_PARAMS

FEATURES = [
    "distance_mi",
    "pop_product",
    "pop_sum",
    "income_mean",
    "income_ratio",
    "dest_is_leisure",
    "orig_is_focus",
    "dest_is_hub",
    "gravity",
]

TARGET = "log_true_passengers"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim to modelable rows and add log target.
    We log-transform true_passengers because demand is heavy-tailed —
    a few huge routes and a long tail of small ones. Log makes RMSE
    more interpretable and keeps the model from just memorizing the
    biggest markets.
    """
    df = df.copy()
    df = df[df["true_passengers"] > 0]
    df[TARGET] = np.log1p(df["true_passengers"])
    return df


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_random_split(df_served: pd.DataFrame):
    """Main 80/20 split RMSE, RF vs linear baseline."""
    X = df_served[FEATURES]
    y = df_served[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Baseline
    lin = LinearRegression().fit(X_tr, y_tr)
    lin_pred = lin.predict(X_te)

    # RF
    rf = RandomForestRegressor(**RF_PARAMS).fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)

    results = pd.DataFrame({
        "model": ["Linear Regression (baseline)", "Random Forest"],
        "rmse_log": [rmse(y_te, lin_pred), rmse(y_te, rf_pred)],
        "r2": [r2_score(y_te, lin_pred), r2_score(y_te, rf_pred)],
    })
    print("\n[eval] Random 80/20 split:")
    print(results.to_string(index=False))

    # Save a side-by-side for the Figure 2 scatter plot later
    holdout = X_te.copy()
    holdout["y_true_log"] = y_te
    holdout["y_pred_log"] = rf_pred
    holdout["y_true"] = np.expm1(y_te)
    holdout["y_pred"] = np.expm1(rf_pred)
    holdout.to_parquet(DATA_DIR / "holdout_predictions.parquet", index=False)

    results.to_csv(TABLES_DIR / "model_comparison.csv", index=False)

    return rf, results


def evaluate_leave_one_route_out(df_served: pd.DataFrame, n_routes: int = 50):
    """
    For the top N routes by demand, train on everything else and predict
    the held-out route. This directly tests "can the model generalize to
    a route it has never seen?" which is the whole premise of the project.
    """
    top_routes = df_served.nlargest(n_routes, "true_passengers")
    records = []
    for idx, row in top_routes.iterrows():
        train = df_served.drop(index=idx)
        rf = RandomForestRegressor(**RF_PARAMS).fit(train[FEATURES], train[TARGET])
        pred_log = rf.predict(row[FEATURES].values.reshape(1, -1))[0]
        records.append({
            "origin": row["origin"],
            "dest": row["dest"],
            "true": row["true_passengers"],
            "predicted": float(np.expm1(pred_log)),
        })

    loro = pd.DataFrame(records)
    loro["abs_pct_error"] = (loro["predicted"] - loro["true"]).abs() / loro["true"]

    loro.to_csv(TABLES_DIR / "leave_one_route_out.csv", index=False)
    print(f"\n[eval] Leave-one-route-out on top {n_routes} routes:")
    print(f"  median abs % error: {loro['abs_pct_error'].median():.1%}")
    print(f"  mean abs % error:   {loro['abs_pct_error'].mean():.1%}")

    return loro


def predict_ghost_routes(rf, df: pd.DataFrame):
    """
    Score ALL pairs in our scope (served + unserved) with the trained RF.
    Then isolate the unserved ones — those are the ghost route predictions.
    """
    df = df.copy()
    df["predicted_log"] = rf.predict(df[FEATURES])
    df["predicted_passengers"] = np.expm1(df["predicted_log"])

    ghosts = (
        df[~df["has_direct_service"]]
        .sort_values("predicted_passengers", ascending=False)
        [["origin", "dest", "true_passengers", "predicted_passengers",
          "distance_mi", "dest_is_leisure"]]
        .head(25)
    )

    print("\n[ghost] Top 25 predicted ghost routes:")
    print(ghosts.to_string(index=False))

    ghosts.to_csv(TABLES_DIR / "top_ghost_routes.csv", index=False)
    df.to_parquet(DATA_DIR / "scored_features.parquet", index=False)

    return ghosts


def main():
    print("=" * 60)
    print("Bridging the Gap — Model Training")
    print("=" * 60)

    df = pd.read_parquet(DATA_DIR / "features.parquet")
    df = prepare(df)

    df_served = df[df["has_direct_service"]].copy()
    print(f"[prep] Training on {len(df_served):,} served OD pairs")
    print(f"[prep] Will predict on {len(df) - len(df_served):,} unserved pairs")

    rf, results = evaluate_random_split(df_served)
    loro = evaluate_leave_one_route_out(df_served)
    ghosts = predict_ghost_routes(rf, df)

    joblib.dump(rf, OUTPUTS_DIR / "ghost_route_rf.joblib")
    print(f"\n[write] Model saved to {OUTPUTS_DIR / 'ghost_route_rf.joblib'}")
    print("\nDone. Next step: python src/figures.py")


if __name__ == "__main__":
    main()
