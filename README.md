# Bridging the Gap: Optimizing EAS Routes for Underserved Markets

## Project Overview
[cite_start]This project uses a "Moneyball" approach to identify systemic inefficiencies in the US Essential Air Service (EAS) program. [cite_start]By merging ticket-level demand data with actual flight supply, we identify "Ghost Routes"—high-demand direct paths currently ignored by the legacy hub-and-spoke model[cite: 84].

## Data Sources
* [cite_start]**DB1B Coupon Data:** Used for "True Demand" analysis[cite: 42, 80].
* [cite_start]**T-100 Segment Data:** Used for "Actual Supply" analysis[cite: 43, 82].

## Technical Stack
* [cite_start]**Database:** SQL for large-scale BTS file ingestion[cite: 44, 102].
* [cite_start]**Modeling:** Random Forest Regressor via Scikit-Learn[cite: 45, 92].
* [cite_start]**Validation:** Root Mean Square Error ($RMSE$) vs. Linear Regression baseline[cite: 52, 93].