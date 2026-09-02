# Research Blueprint

## Working title
**From Demand Forecasting to Capacity Decisions: Evaluating the Operational Value of Machine Learning Forecasts**

## Core research question
Does higher demand-forecast accuracy necessarily translate into better operational capacity decisions?

## Motivation
Forecasting studies commonly report statistical accuracy, while operations decisions are affected by asymmetric consequences of insufficient and excess capacity. This study therefore evaluates forecast quality and downstream capacity performance as separate outcomes.

## Empirical setting
NYC Yellow Taxi trip records provide public, high-frequency demand observations. The unit of analysis is hourly pickup demand by taxi zone. The study does not equate forecasted trips with physical vehicle count. Instead, it evaluates a stylized hourly service-capacity requirement in demand units.

NYC TLC states that yellow taxi records contain pickup/drop-off timestamps and locations and that trip data are published monthly in Parquet format. The official TLC page is the source of record for the dataset and download links.

## Experimental pipeline
1. Aggregate trip records into hourly pickup demand by selected high-volume zones.
2. Construct calendar and lag features using only information available before the forecast hour.
3. Split observations chronologically into train, validation, and test periods.
4. Fit Seasonal Naive, Ridge, Random Forest, and XGBoost models.
5. Select the model using validation RMSE only.
6. Refit on train + validation and evaluate all candidate models on the untouched test period.
7. Convert the selected forecast into capacity using a simple buffer rule.
8. Evaluate operational performance under several asymmetric under-capacity/over-capacity penalty ratios.
9. Conduct sensitivity analysis over capacity buffers.

## Hypotheses
These are empirical hypotheses, not assumed findings.

**H1:** Forecast accuracy and operational cost are positively aligned when under-capacity and excess-capacity penalties are symmetric.

**H2:** Under asymmetric penalties, the model with the lowest statistical forecast error need not minimize operational cost.

**H3:** Greater demand volatility increases the divergence between forecast-accuracy rankings and operational-performance rankings.

## Operational metrics
- Under-capacity amount
- Excess-capacity amount
- Under-capacity frequency
- Service-level attainment
- Capacity utilization
- Normalized asymmetric operational cost

## Important limitations
- The capacity rule is a transparent decision proxy, not a full taxi-fleet optimization model.
- Penalty values are scenario weights, not observed taxi-company financial costs.
- The data do not identify individual vehicle schedules or true available fleet capacity.
- Results should therefore be interpreted as evidence about forecast-to-capacity alignment, not as a literal fleet-sizing prescription.

## Literature standard
The final paper must use verified peer-reviewed journal sources. Any citation supplied by an external research model must be independently verified before inclusion. No DOI, journal article, dataset claim, novelty claim, or numerical result should be included merely because a secondary model produced it.
