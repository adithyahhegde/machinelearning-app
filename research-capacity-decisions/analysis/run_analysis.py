"""Reproducible demand-to-capacity experiment.

The default CI mode is a small smoke test. Set FULL_RUN=1 for the configured
full study period. Raw TLC Parquet files are read remotely with DuckDB and
immediately aggregated to hourly pickup demand.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
for p in (TABLES, FIGURES):
    p.mkdir(parents=True, exist_ok=True)

FULL_RUN = os.getenv("FULL_RUN", "0") == "1"
START = pd.Timestamp(os.getenv("START", "2023-01-01"))
END = pd.Timestamp(os.getenv("END", "2024-12-31 23:00:00"))
if not FULL_RUN:
    START = pd.Timestamp("2023-01-01")
    END = pd.Timestamp("2023-02-28 23:00:00")

# Keep the experiment focused on the largest service zones. This also makes
# the operational unit explicit: hourly service demand within a zone.
MAX_ZONES = int(os.getenv("MAX_ZONES", "20"))
ZONE_SELECTION = os.getenv("ZONE_SELECTION", "top")


def tlc_url(ts: pd.Timestamp) -> str:
    return f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{ts.year}-{ts.month:02d}.parquet"


def month_range(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1)
    last = pd.Timestamp(end.year, end.month, 1)
    while cur <= last:
        yield cur
        cur = cur + pd.offsets.MonthBegin(1)


def load_hourly_demand(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    con = duckdb.connect()
    pieces = []
    for month in month_range(start, end):
        url = tlc_url(month)
        q = f"""
        SELECT
          date_trunc('hour', tpep_pickup_datetime) AS hour,
          CAST(PULocationID AS INTEGER) AS zone,
          COUNT(*)::DOUBLE AS demand
        FROM read_parquet('{url}')
        WHERE tpep_pickup_datetime >= TIMESTAMP '{start}'
          AND tpep_pickup_datetime <= TIMESTAMP '{end}'
          AND PULocationID IS NOT NULL
        GROUP BY 1, 2
        """
        try:
            pieces.append(con.execute(q).df())
        except Exception as exc:
            raise RuntimeError(f"Could not read {url}: {exc}") from exc
    if not pieces:
        raise RuntimeError("No TLC data were loaded.")
    raw = pd.concat(pieces, ignore_index=True)
    raw["hour"] = pd.to_datetime(raw["hour"])
    if ZONE_SELECTION == "top":
        top = raw.groupby("zone")["demand"].sum().nlargest(MAX_ZONES).index
        raw = raw[raw["zone"].isin(top)].copy()
    # Complete the hourly index for every selected zone; missing hours mean 0.
    hours = pd.date_range(start.floor("h"), end.floor("h"), freq="h")
    zones = np.sort(raw["zone"].unique())
    idx = pd.MultiIndex.from_product([hours, zones], names=["hour", "zone"])
    out = raw.set_index(["hour", "zone"]).reindex(idx, fill_value=0).reset_index()
    out["demand"] = out["demand"].astype(float)
    return out.sort_values(["zone", "hour"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["dow"] = x["hour"].dt.dayofweek
    x["hour_of_day"] = x["hour"].dt.hour
    x["month"] = x["hour"].dt.month
    x["is_weekend"] = (x["dow"] >= 5).astype(int)
    # Lag features are computed within each zone only, preventing cross-zone leakage.
    g = x.groupby("zone")["demand"]
    for lag in (1, 2, 24, 48, 168):
        x[f"lag_{lag}"] = g.shift(lag)
    x["roll_24"] = g.transform(lambda s: s.shift(1).rolling(24, min_periods=12).mean())
    return x.dropna().reset_index(drop=True)


def smape(y, pred):
    denom = np.abs(y) + np.abs(pred)
    return float(np.mean(np.where(denom == 0, 0, 2 * np.abs(y - pred) / denom)))


def forecast_by_model(train: pd.DataFrame, test: pd.DataFrame):
    feature_cols = [
        "dow", "hour_of_day", "month", "is_weekend",
        "lag_1", "lag_2", "lag_24", "lag_48", "lag_168", "roll_24",
    ]
    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=150, max_depth=18, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=250, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
            random_state=42, n_jobs=2,
        ),
    }
    rows = []
    preds = {}
    Xtr, ytr = train[feature_cols], train["demand"]
    Xte = test[feature_cols]
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = np.maximum(model.predict(Xte), 0)
        preds[name] = pred
        rows.append({
            "model": name,
            "MAE": mean_absolute_error(test["demand"], pred),
            "RMSE": math.sqrt(mean_squared_error(test["demand"], pred)),
            "sMAPE": smape(test["demand"].to_numpy(), pred),
        })
    # Seasonal naive uses the 168-hour lag.
    pred = np.maximum(test["lag_168"].to_numpy(), 0)
    preds["SeasonalNaive"] = pred
    rows.append({
        "model": "SeasonalNaive",
        "MAE": mean_absolute_error(test["demand"], pred),
        "RMSE": math.sqrt(mean_squared_error(test["demand"], pred)),
        "sMAPE": smape(test["demand"].to_numpy(), pred),
    })
    return pd.DataFrame(rows).sort_values("RMSE"), preds


def operational_metrics(actual, forecast, beta, under_cost, over_cost):
    capacity = np.ceil(np.maximum(forecast, 0) * (1 + beta))
    under = np.maximum(actual - capacity, 0)
    over = np.maximum(capacity - actual, 0)
    served = np.minimum(actual, capacity)
    return {
        "beta": beta,
        "under_cost_ratio": under_cost,
        "over_cost_ratio": over_cost,
        "under_capacity": float(under.mean()),
        "excess_capacity": float(over.mean()),
        "service_level": float((served / np.maximum(actual, 1)).mean()),
        "under_capacity_frequency": float((under > 0).mean()),
        "total_cost": float((under_cost * under + over_cost * over).mean()),
        "utilization": float((served / np.maximum(capacity, 1)).mean()),
    }


def main():
    demand = load_hourly_demand(START, END)
    data = add_features(demand)
    # Time split: early period train, middle period validation, final period test.
    t0, t1 = data["hour"].min(), data["hour"].max()
    span = t1 - t0
    train_end = t0 + span * 0.60
    val_end = t0 + span * 0.80
    train = data[data["hour"] <= train_end].copy()
    val = data[(data["hour"] > train_end) & (data["hour"] <= val_end)].copy()
    test = data[data["hour"] > val_end].copy()

    # Select the model using validation RMSE only. Test data remain untouched.
    val_scores, _ = forecast_by_model(train, val)
    chosen = val_scores.iloc[0]["model"]

    # Retrain chosen model on train+validation, then forecast test.
    trainval = pd.concat([train, val], ignore_index=True)
    test_scores, test_preds = forecast_by_model(trainval, test)
    test_pred = test_preds[chosen]

    forecast_table = pd.DataFrame({"model": test_scores["model"], "MAE": test_scores["MAE"], "RMSE": test_scores["RMSE"], "sMAPE": test_scores["sMAPE"]})
    forecast_table.to_csv(TABLES / "forecast_accuracy.csv", index=False)
    val_scores.to_csv(TABLES / "validation_accuracy.csv", index=False)

    rows = []
    for beta in (0.0, 0.10, 0.20, 0.30):
        for under, over in ((1, 1), (3, 1), (5, 1), (10, 1)):
            m = operational_metrics(test["demand"].to_numpy(), test_pred, beta, under, over)
            m["model"] = chosen
            rows.append(m)
    op = pd.DataFrame(rows)
    op.to_csv(TABLES / "operational_sensitivity.csv", index=False)

    manifest = {
        "project": "From Demand Forecasting to Capacity Decisions",
        "start": str(START), "end": str(END),
        "full_run": FULL_RUN, "zones": int(data["zone"].nunique()),
        "observations": int(len(data)), "selected_model_validation_rmse": chosen,
        "data_source": "NYC TLC Yellow Taxi Trip Record Data",
    }
    (RESULTS / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print("Validation-selected model:", chosen)
    print(test_scores.to_string(index=False))
    print(op.to_string(index=False))


if __name__ == "__main__":
    main()
