# Bridging the Gap: Optimizing EAS Routes for Underserved Markets

## Project Overview
This project uses a "Moneyball" approach to identify systemic inefficiencies in the US Essential Air Service (EAS) program. By merging ticket-level demand data with actual flight supply, we identify "Ghost Routes"—high-demand direct paths currently ignored by the legacy hub-and-spoke model.

## Data Sources
* **DB1B Coupon Data:** Used for "True Demand" analysis.
* **T-100 Segment Data:** Used for "Actual Supply" analysis.

## Technical Stack
* **Database:** SQL for large-scale BTS file ingestion.
* **Modeling:** Random Forest Regressor via Scikit-Learn.
* **Validation:** Root Mean Square Error (RMSE) vs. Linear Regression baseline.