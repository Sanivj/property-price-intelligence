# Property Price Intelligence System

An ML-powered API that estimates fair property value and explains pricing factors.

## Features
- Predicts property price using Random Forest
- Provides fair market range
- Explains top factors affecting valuation
- Detects overpriced / underpriced listings
- FastAPI deployment

## Example Response
{
  "predicted_price": 143532.77,
  "fair_range": [103500.0, 181316.0],
  "top_factors": [
    "Overall Qual decreased price",
    "Gr Liv Area increased price"
  ]
}

## Run locally
pip install -r requirements.txt
uvicorn app.api:app --reload
