# From Demand Forecasting to Capacity Decisions

**Project:** From Demand Forecasting to Capacity Decisions: Evaluating the Operational Value of Machine Learning Forecasts

## Research question
**Does higher demand-forecast accuracy necessarily translate into better operational capacity decisions?**

## Empirical setting
Public NYC Yellow Taxi trip records are aggregated into hourly pickup demand by taxi zone. NYC TLC publishes the trip records monthly in Parquet format and provides pickup timestamps and locations in the yellow-taxi data.

The operational unit is deliberately defined as **hourly service-capacity units**, not physical taxi fleet size. The data do not identify available vehicles or vehicle schedules, so the experiment is a transparent decision proxy rather than a literal fleet-sizing model.

## Models
- Seasonal Naive
- Ridge Regression
- Random Forest
- XGBoost

## Decision rule
For forecast \(\hat d_t\) and capacity buffer \(\beta\):

`C_t = ceil(max(0, d_hat_t) * (1 + beta))`

Under-capacity and excess-capacity are evaluated separately and combined using normalized asymmetric penalty scenarios.

## Evaluation
Forecast metrics:
- MAE
- RMSE
- sMAPE

Operational metrics:
- under-capacity amount and frequency;
- excess capacity;
- service-level attainment;
- utilization;
- normalized asymmetric operational cost.

## Reproducibility rules
- chronological train/validation/test split;
- no future-demand features;
- zone selection based only on information available before the test period;
- model selection using validation data only;
- all candidate models evaluated on the untouched test period;
- no raw Parquet files committed to Git;
- exact source months and configuration recorded in the run manifest.

## Research status
The literature review and methodology audit are now in the repository. Before any paper claims are written, the experiment must pass the audit in `paper/methodology_audit.md` and produce actual out-of-sample results.

## Repository layout
```text
research-capacity-decisions/
├── README.md
├── requirements.txt
├── data/README.md
├── analysis/run_analysis.py
├── paper/
│   ├── research_blueprint.md
│   ├── literature_review.md
│   └── methodology_audit.md
├── results/
│   ├── tables/
│   └── figures/
└── .github/workflows/research.yml
```
