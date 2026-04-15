# Bridging the Gap

**Measuring Invisible Demand in the U.S. Essential Air Service Program**

SI 699 Capstone · Big Data Analytics · Winter 2026
Affaan Waheed · University of Michigan School of Information

---

## Project summary

The U.S. Essential Air Service (EAS) program subsidizes commercial flights from small communities to major airline hubs based on observed direct-route ridership. The dataset the program uses to make funding decisions is, by construction, a sample of survivors: it counts only the rural travelers who actually used the local airport, not those who long ago gave up and now drive hours to a hub for a direct flight.

This project applies Abraham Wald's 1943 critique of bullet-hole analysis on returning bombers to a modern federal program. Using one quarter of the BTS Origin and Destination Survey (DB1B) public release, I trace the full coupon-by-coupon flight paths of 1.88 million tickets touching fifteen small-community airports across the Upper Midwest and measure direct-versus-connecting flow patterns directly.

**Central finding (structural):** Across every focus origin → leisure destination pair examined, the direct flight count is zero, and all observed demand routes through hub airports. Per-route absolute volumes are modest at single-quarter, single-region scale and should not be over-interpreted as a sufficient case for launching new direct routes. The contribution is the demonstration that the EAS dataset is systematically blind to a class of demand that is recoverable from the same public source the program already has access to.

A supporting Random Forest model trained only on structural features (population, distance, income, gravity) independently corroborates the same routes by predicting substantial demand based on geography and demographics alone. The model outperforms a linear regression baseline by a meaningful margin (R² 0.48 vs 0.18) and is reliable for ranking and order-of-magnitude estimation.

## Methodology in one paragraph

The prediction target is log-transformed total passenger demand between origin–destination city pairs, sourced from the BTS DB1B public release (a 10% sample of all U.S. itineraries, grossed up by 10×). Each itinerary is classified as direct (one coupon) or connecting (two or more coupons) based on the BTS record layout, which lets us distinguish travelers who got where they wanted to go without a stop from travelers who routed through a hub. Features are structural only — metro population, geographic distance, median income, leisure-destination flag, and a gravity-model term — and deliberately exclude any transactional variable (current fares, current load factors) that would not exist for a route the system does not currently fly. The model trains on city pairs with meaningful direct service (where DB1B provides reliable ground truth) and predicts on pairs with no direct service. Validation uses both an 80/20 random holdout and a leave-one-route-out scheme on the top fifty known routes, which directly simulates the deployment scenario of predicting a route the model has never seen.

## How to reproduce (recommended path)

1. Clone the repo:
   ```bash
   git clone https://github.com/affaan02/si699-final-project.git
   cd si699-final-project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download at least one quarterly DB1B Public Release zip from the BTS Origin and Destination Survey page:
   https://www.bts.gov/topics/airlines-and-airports/origin-and-destination-survey-data

   Drop the zip into `data/raw_zips/`. (The current analysis uses Q4 2024.)

4. Open `bridging_the_gap.ipynb` in Jupyter or VSCode and **Run All**.

The notebook handles extraction, parsing, filtering, aggregation, model training, validation, and figure generation end-to-end. Total runtime ~5–10 minutes per quarter on a standard laptop. Output figures land in `outputs/figures/` and tables in `outputs/tables/`.

## Repo layout

```
si699-final-project/
├── bridging_the_gap.ipynb   ← the main notebook, run this end-to-end
├── README.md
├── requirements.txt
├── .gitignore
├── src/                      ← modular Python implementation (same logic, importable)
│   ├── config.py             ← single source of truth for scope and paths
│   ├── ingest.py             ← BTS .asc parser + filtering
│   ├── features.py           ← structural feature engineering
│   ├── model.py              ← Random Forest training + LORO validation
│   ├── figures.py            ← matplotlib figure generation
│   ├── inspect_data.py       ← schema diagnostic (run if ingest fails)
│   └── run_all.py            ← chains ingest → features → model → figures
├── data/
│   ├── raw_zips/             ← put downloaded BTS .zip files here (gitignored)
│   └── db1b_asc/             ← extracted .asc files (gitignored)
└── outputs/
    ├── figures/              ← fig1..fig4 PNG + PDF
    └── tables/               ← model_comparison.csv, top_ghost_routes.csv, etc.
```

## A note on the data strategy pivot

The original project plan called for reconciling DB1B (passenger demand) with the BTS T-100 Domestic Segment table (carrier-reported flight supply). During execution I realized DB1B alone carries both signals: the record layout includes the number of coupons per itinerary, which directly distinguishes direct from connecting tickets. Using DB1B alone simplified the pipeline, eliminated reconciliation errors that would have come from merging two datasets with different sampling units, and ensured both sides of the comparison are measured on the same population of travelers.

## Limitations

- **Single-quarter, single-region scope.** All results in the current run are derived from Q4 2024 DB1B and 15 Upper Midwest focus airports. The structural pattern (zero direct service, positive connecting demand) is consistent across every pair examined; the absolute magnitudes will be more meaningful when the analysis is extended to additional quarters and a national focus airport list. The pipeline supports this directly — drop more zips into `data/raw_zips/` and re-run.
- **Quarterly seasonality.** The 4× annualization assumes seasonal consistency. Q4 includes Thanksgiving and December but not the summer leisure peak.
- **Hand-coded airport metadata.** The model uses lat/lon/population/income for the 36 airports in scope, hard-coded in `src/features.py`. Scaling nationally requires pulling from the BTS Master Coordinate file and Census MSA tables — mechanical work, but adds a data dependency.
- **Self-referential survivorship bias.** The training set is drawn from city pairs that currently have direct service, which is itself a survivor population. The leave-one-route-out validation partially addresses this but cannot fully eliminate the concern.
- **Methodology applies in markets with a drivable hub alternative.** Does NOT apply in geographically isolated markets like Alaska or small island communities, where the leakage signal would read near zero not because demand is absent but because alternative travel modes do not exist.

## Deliverables

- **Final report** — narrative writeup with full methodology, results, limitations, and policy implications.
- **This repository** — fully reproduces every figure and table in the report.
- **Expo poster** — visual summary presented at the UMSI Student Project Expo.

## Tools and acknowledgments

This project was developed with assistance from generative AI tools for code debugging (including the parser for the BTS pipe-delimited file format) and for editorial feedback during writing. All analytical decisions, methodological choices, the survivorship-bias framing, the data interpretation, and the policy recommendations are my own.
