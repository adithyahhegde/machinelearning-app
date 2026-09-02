# Methodology Audit — Before Results Are Trusted

## Current strengths
- Public primary dataset.
- Chronological train/validation/test design.
- Lag features use prior observations only.
- Model selection is separated from final test evaluation.
- Capacity is explicitly a stylized service-capacity proxy rather than a claim about physical taxi fleet size.
- Operational penalties are normalized scenario weights, not invented dollar estimates.

## Issues that must be resolved before the final paper

### 1. Zone-selection leakage
The first implementation selected the highest-volume zones using the entire study period. That allows future test-period demand to influence which zones enter the sample. This must be changed so zone selection is based only on the training portion (or a fixed, externally defined zone list).

### 2. Operational comparison must include every forecasting model
The original implementation evaluated operational cost only for the validation-selected model. That is insufficient for H2. The final analysis must generate test forecasts for all candidate models and compute operational metrics for every model under every cost/buffer scenario.

### 3. Do not call the proxy “fleet capacity”
The TLC trip records do not identify vehicle schedules or available fleet capacity. The paper must use terms such as “hourly service-capacity units” or “demand-service capacity.”

### 4. No fabricated economic calibration
Penalty ratios (1:1, 3:1, 5:1, 10:1) are sensitivity scenarios. They are not estimates of taxi-company costs. The paper must say this explicitly.

### 5. Aggregation and zero-demand handling
After hourly aggregation, every selected zone must have a complete hourly index. Missing combinations should be treated as zero observed pickups only when the source file is present for that period; invalid or missing source data must not silently become zero.

### 6. Robustness analysis
The final results should include at least:
- multiple capacity buffers;
- multiple under/over-capacity penalty ratios;
- a low-volatility versus high-volatility comparison;
- model ranking under forecast metrics versus operational cost.

## Acceptance criteria
The experiment is ready for paper writing only when:
1. the pipeline completes without errors;
2. the run manifest records exact study dates, months, selected zones, model settings and source URLs;
3. all four models have out-of-sample forecasts;
4. operational tables cover all models and scenarios;
5. no test information is used for model/zone selection;
6. figures and tables are generated from the same run;
7. results are reproducible from a clean GitHub Actions run.
