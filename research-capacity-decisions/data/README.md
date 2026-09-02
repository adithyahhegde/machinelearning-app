# Data

The study uses public NYC Taxi & Limousine Commission trip-record data.

Raw trip files are intentionally **not committed** to the repository because they are large. The analysis downloads only the monthly files required by the configured experiment and aggregates them immediately to hourly pickup demand by zone.

## Required fields
- pickup timestamp
- pickup location zone identifier

Optional fields may be used only for descriptive checks. The core forecasting experiment does not require fares, tips, or passenger income information.

## Data integrity rules
1. Keep the exact source URL/month in the run manifest.
2. Record row counts after filtering and aggregation.
3. Drop invalid timestamps and impossible records before aggregation.
4. Keep the final test period completely untouched during model selection.
5. Do not commit raw trip records.
