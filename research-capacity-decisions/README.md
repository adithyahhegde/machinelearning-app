# From Demand Forecasting to Capacity Decisions

**Project:** From Demand Forecasting to Capacity Decisions: Evaluating the Operational Value of Machine Learning Forecasts

This research workspace implements a reproducible empirical study of whether better demand forecasts necessarily produce better operational capacity decisions.

## Research question
Does higher demand-forecast accuracy necessarily translate into better operational capacity decisions?

## Study design
Historical NYC Yellow Taxi pickup demand is transformed into hourly demand by service zone. Forecasts are generated using simple, interpretable models and then passed through a transparent capacity rule. Forecast quality and downstream operational performance are evaluated separately.

The study deliberately avoids claiming that the number of forecasted trips equals physical taxi fleet size. Capacity is defined as **hourly service-capacity units**: the number of demand requests that the modeled operation is assumed able to serve during an hour. Under-capacity and excess-capacity are evaluated with scenario-based asymmetric costs rather than invented real-world dollar values.

## Models
- Seasonal Naive baseline
- Ridge Regression
- Random Forest
- XGBoost

## Operational evaluation
For a forecast d-hat and buffer beta:

`C_t = ceil(d-hat_t * (1 + beta))`

The analysis reports:
- MAE, RMSE and sMAPE
- capacity utilization
- under-capacity frequency
- excess capacity
- service-level attainment
- normalized asymmetric operational cost under multiple under-capacity/over-capacity penalty ratios

## Reproducibility principles
- Time-ordered train/validation/test splits
- No future-data leakage
- Model selection uses validation data only
- Final results use an untouched test period
- Public data are downloaded by the pipeline rather than committed to Git
- All assumptions are explicit and sensitivity-tested

## Repository layout
```text
research-capacity-decisions/
├── README.md
├── requirements.txt
├── data/README.md
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── forecasting_models.py
│   ├── capacity_simulation.py
│   └── evaluation.py
├── analysis/
│   └── run_analysis.py
├── results/
│   ├── tables/
│   └── figures/
└── .github/workflows/
    └── research.yml
```

## Important methodological note
The NYC taxi application is an operational test bed, not a claim that this simulation reproduces the complete economics of taxi fleet management. The contribution is the empirical evaluation of forecast-to-decision alignment using a transparent capacity proxy and public high-frequency demand data.
