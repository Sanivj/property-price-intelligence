import joblib
import json
import numpy as np
import shap
import os
import requests

MODEL_URL="https://huggingface.co/Sanivj1881/property-price-model/blob/main/price_model.pkl"
COLS_URL="https://huggingface.co/Sanivj1881/property-price-model/blob/main/columns.json"

os.makedirs("models",exist_ok=True)
os.makedirs("artifacts",exist_ok=True)

def download(url,path):
    if not os.path.exists(path):
        r=requests.get(url)
        open(path,"wb").write(r.content)

download(MODEL_URL,"models/price_model.pkl")
download(COLS_URL,"artifacts/columns.json")

model=joblib.load("models/price_model.pkl")

with open("artifacts/columns.json") as f:
  columns=json.load(f)

explainer=shap.TreeExplainer(model)

def prepare_input(data_dict):
  x=np.zeros(len(columns))
  for k,v in data_dict.items():
    if k in columns:
      x[columns.index(k)]=v
  return x.reshape(1,-1)

def predict_price(data_dict):
  sample=prepare_input(data_dict)

  pred_price=model.predict(sample)[0]

  tree_preds=np.array([t.predict(sample)[0] for t in model.estimators_])
  low=np.percentile(tree_preds,10)
  high=np.percentile(tree_preds,90)

  shap_values=explainer.shap_values(sample)[0]
  top_idx=np.argsort(np.abs(shap_values))[-5:][::-1]

  factors=[]
  for i in top_idx:
    effect="increased" if shap_values[i]>0 else "decreased"
    factors.append(f"{columns[i]} {effect} price")

  return{
      "predicted_price":float(round(pred_price,2)),
      "fair_range":[float(round(low,2)),float(round(high,2))],
      "top_factors":factors
  }
