from fastapi import FastAPI
from pydantic import BaseModel
from property_ai.src.predict import predict_price

app=FastAPI(title="Property Price Intelligence API")

class HouseFeatures(BaseModel):
  Gr_Liv_Area: float=1500
  Overall_Qual: float=5
  Garage_Cars: float=2
  Year_Built: float=2000

@app.get("/")
def home():
  return {"message":"Property Price Intelligence API is running"}

@app.post("/predict")
def predict(data: HouseFeatures):
  input_dict={
      "Gr Liv Area":data.Gr_Liv_Area,
      "Overall Qual":data.Overall_Qual,
      "Garage Cars":data.Garage_Cars,
      "Year Built":data.Year_Built
  }
  result=predict_price(input_dict)
  return result
