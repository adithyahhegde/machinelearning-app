"""Reproducible demand-to-capacity experiment.

FULL_RUN=1 runs the configured 2023-2024 study. The default CI run is a
small smoke test. Raw TLC Parquet files are queried remotely with DuckDB.
"""
from __future__ import annotations
import json, math, os
from pathlib import Path
import duckdb, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[1]; RESULTS=ROOT/"results"; TABLES=RESULTS/"tables"; FIGURES=RESULTS/"figures"
for p in (TABLES,FIGURES): p.mkdir(parents=True,exist_ok=True)
FULL_RUN=os.getenv("FULL_RUN","0")=="1"
START=pd.Timestamp(os.getenv("START","2023-01-01")); END=pd.Timestamp(os.getenv("END","2024-12-31 23:00:00"))
if not FULL_RUN: START=pd.Timestamp("2023-01-01"); END=pd.Timestamp("2023-02-28 23:00:00")
MAX_ZONES=int(os.getenv("MAX_ZONES","20"))

FEATURES=["dow","hour_of_day","month","is_weekend","lag_1","lag_2","lag_24","lag_48","lag_168","roll_24"]

def months(start,end):
    cur=pd.Timestamp(start.year,start.month,1); last=pd.Timestamp(end.year,end.month,1)
    while cur<=last: yield cur; cur=cur+pd.offsets.MonthBegin(1)

def tlc_url(ts): return f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{ts.year}-{ts.month:02d}.parquet"

def load_raw_hourly(start,end):
    con=duckdb.connect(); pieces=[]; urls=[]
    for m in months(start,end):
        url=tlc_url(m); urls.append(url)
        q=f"""SELECT date_trunc('hour',tpep_pickup_datetime) hour, CAST(PULocationID AS INTEGER) zone, COUNT(*)::DOUBLE demand
              FROM read_parquet('{url}')
              WHERE tpep_pickup_datetime>=TIMESTAMP '{start}' AND tpep_pickup_datetime<=TIMESTAMP '{end}' AND PULocationID IS NOT NULL
              GROUP BY 1,2"""
        pieces.append(con.execute(q).df())
    if not pieces: raise RuntimeError("No TLC data loaded")
    raw=pd.concat(pieces,ignore_index=True); raw.hour=pd.to_datetime(raw.hour)
    return raw,urls

def complete_hours(raw,start,end,zones):
    hours=pd.date_range(start.floor("h"),end.floor("h"),freq="h")
    idx=pd.MultiIndex.from_product([hours,np.sort(zones)],names=["hour","zone"])
    out=raw.set_index(["hour","zone"]).reindex(idx,fill_value=0).reset_index()
    out.demand=out.demand.astype(float)
    return out.sort_values(["zone","hour"]).reset_index(drop=True)

def features(df):
    x=df.copy(); x["dow"]=x.hour.dt.dayofweek; x["hour_of_day"]=x.hour.dt.hour; x["month"]=x.hour.dt.month; x["is_weekend"]=(x.dow>=5).astype(int)
    g=x.groupby("zone").demand
    for lag in (1,2,24,48,168): x[f"lag_{lag}"]=g.shift(lag)
    x["roll_24"]=g.transform(lambda s:s.shift(1).rolling(24,min_periods=12).mean())
    return x.dropna().reset_index(drop=True)

def smape(y,p):
    d=np.abs(y)+np.abs(p); return float(np.mean(np.where(d==0,0,2*np.abs(y-p)/d)))

def models():
    return {
      "Ridge":make_pipeline(StandardScaler(),Ridge(alpha=10.0)),
      "RandomForest":RandomForestRegressor(n_estimators=150,max_depth=18,min_samples_leaf=3,random_state=42,n_jobs=-1),
      "XGBoost":XGBRegressor(n_estimators=250,max_depth=6,learning_rate=0.05,subsample=.8,colsample_bytree=.8,objective="reg:squarederror",random_state=42,n_jobs=2),
    }

def fit_predict(train,test):
    preds={}
    for name,model in models().items():
        model.fit(train[FEATURES],train.demand); preds[name]=np.maximum(model.predict(test[FEATURES]),0)
    preds["SeasonalNaive"]=np.maximum(test.lag_168.to_numpy(),0)
    return preds

def accuracy(actual,pred):
    return {"MAE":float(mean_absolute_error(actual,pred)),"RMSE":float(math.sqrt(mean_squared_error(actual,pred))),"sMAPE":smape(np.asarray(actual),np.asarray(pred))}

def op_metrics(actual,pred,beta,under_cost,over_cost):
    cap=np.ceil(np.maximum(pred,0)*(1+beta)); under=np.maximum(actual-cap,0); over=np.maximum(cap-actual,0); served=np.minimum(actual,cap)
    return {"beta":beta,"under_cost_ratio":under_cost,"over_cost_ratio":over_cost,"under_capacity":float(under.mean()),"excess_capacity":float(over.mean()),"service_level":float((served/np.maximum(actual,1)).mean()),"under_capacity_frequency":float((under>0).mean()),"total_cost":float((under_cost*under+over_cost*over).mean()),"utilization":float((served/np.maximum(cap,1)).mean())}

def main():
    raw,urls=load_raw_hourly(START,END)
    # IMPORTANT: select zones using training-period demand only, never the test period.
    t_cut=START+(END-START)*0.60
    top=raw.loc[raw.hour<=t_cut].groupby("zone").demand.sum().nlargest(MAX_ZONES).index
    data=complete_hours(raw,START,END,top); data=features(data)
    t0,t1=data.hour.min(),data.hour.max(); span=t1-t0; train_end=t0+span*.60; val_end=t0+span*.80
    train=data[data.hour<=train_end].copy(); val=data[(data.hour>train_end)&(data.hour<=val_end)].copy(); test=data[data.hour>val_end].copy()
    val_preds=fit_predict(train,val); val_rows=[{"model":m,**accuracy(val.demand,p)} for m,p in val_preds.items()]; val_table=pd.DataFrame(val_rows).sort_values("RMSE")
    chosen=val_table.iloc[0].model
    trainval=pd.concat([train,val],ignore_index=True); test_preds=fit_predict(trainval,test)
    acc=pd.DataFrame([{"model":m,**accuracy(test.demand,p)} for m,p in test_preds.items()]).sort_values("RMSE")
    acc.to_csv(TABLES/"forecast_accuracy.csv",index=False); val_table.to_csv(TABLES/"validation_accuracy.csv",index=False)
    rows=[]
    for m,p in test_preds.items():
        for beta in (0,.10,.20,.30):
            for uc,oc in ((1,1),(3,1),(5,1),(10,1)):
                row=op_metrics(test.demand.to_numpy(),p,beta,uc,oc); row["model"]=m; row["validation_selected"]=(m==chosen); rows.append(row)
    op=pd.DataFrame(rows); op.to_csv(TABLES/"operational_sensitivity_all_models.csv",index=False)
    # Best operational model in each scenario and alignment with RMSE ranking.
    best=op.loc[op.groupby(["beta","under_cost_ratio","over_cost_ratio"]).total_cost.idxmin()].copy(); best.to_csv(TABLES/"best_operational_model_by_scenario.csv",index=False)
    manifest={"project":"From Demand Forecasting to Capacity Decisions","start":str(START),"end":str(END),"full_run":FULL_RUN,"zones":int(data.zone.nunique()),"zone_ids":[int(z) for z in sorted(top)],"observations":int(len(data)),"validation_selected_model":chosen,"source_month_urls":urls,"data_source":"NYC TLC Yellow Taxi Trip Record Data"}
    (RESULTS/"run_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2)); print("Validation-selected model:",chosen); print(acc.to_string(index=False)); print(best.to_string(index=False))

if __name__=="__main__": main()
