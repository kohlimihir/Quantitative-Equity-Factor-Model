"""
FastAPI service for Azure App Service
Reads data from Azure Blob Storage
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
from azure.storage.blob import BlobServiceClient
from io import StringIO, BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Equity Factor Model API",
    description="REST API for equity factor model predictions and analytics",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure Blob Storage configuration
STORAGE_ACCOUNT = os.getenv('AZURE_STORAGE_ACCOUNT', '')
STORAGE_KEY = os.getenv('AZURE_STORAGE_KEY', '')

def get_blob_client():
    """Get Azure Blob Service Client"""
    if not STORAGE_ACCOUNT or not STORAGE_KEY:
        logger.error("Azure storage credentials not configured")
        return None
    
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT};AccountKey={STORAGE_KEY};EndpointSuffix=core.windows.net"
    return BlobServiceClient.from_connection_string(connection_string)

def load_csv_from_blob(container_name: str, blob_name: str):
    """Load CSV from Azure Blob Storage"""
    try:
        blob_client = get_blob_client()
        if not blob_client:
            return None
        
        blob = blob_client.get_blob_client(container=container_name, blob=blob_name)
        blob_data = blob.download_blob().readall()
        return pd.read_csv(BytesIO(blob_data))
    except Exception as e:
        logger.error(f"Error loading {container_name}/{blob_name}: {e}")
        return None

class PredictionResponse(BaseModel):
    ticker: str
    date: str
    prediction: float
    actual: Optional[float] = None
    sector: str
    rank: Optional[int] = None

class PerformanceMetrics(BaseModel):
    mean_ic: float
    ic_ir: float
    sharpe: float
    win_rate: float

@app.get("/")
def root():
    return {
        "message": "Equity Factor Model API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predictions": "/predictions",
            "top_stocks": "/top-stocks",
            "performance": "/performance",
            "features": "/features",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    storage_ok = STORAGE_ACCOUNT and STORAGE_KEY
    return {
        "status": "healthy" if storage_ok else "degraded",
        "storage_configured": storage_ok,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/predictions", response_model=List[PredictionResponse])
def get_predictions(
    date: Optional[str] = Query(None, description="Date in YYYY-MM format"),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get model predictions"""
    df = load_csv_from_blob('predictions', 'ensemble_optimize_predictions.csv')
    
    if df is None:
        raise HTTPException(status_code=404, detail="Predictions not found. Run model pipeline first.")
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Rename column if needed
    if 'ensemble_pred' in df.columns:
        df['prediction'] = df['ensemble_pred']
    
    # Filter by date if provided
    if date:
        target_date = pd.to_datetime(date)
        df = df[df['date'] == target_date]
    else:
        latest_date = df['date'].max()
        df = df[df['date'] == latest_date]
    
    # Get top predictions
    df = df.nlargest(limit, 'prediction')
    
    results = []
    for idx, row in df.iterrows():
        results.append(PredictionResponse(
            ticker=row['ticker'],
            date=row['date'].strftime('%Y-%m'),
            prediction=float(row['prediction']),
            actual=float(row['actual']) if pd.notna(row['actual']) else None,
            sector=row['sector']
        ))
    
    return results

@app.get("/predictions/{ticker}")
def get_ticker_predictions(ticker: str):
    """Get predictions for a specific ticker"""
    df = load_csv_from_blob('predictions', 'ensemble_optimize_predictions.csv')
    
    if df is None:
        raise HTTPException(status_code=404, detail="Predictions not found")
    
    df['date'] = pd.to_datetime(df['date'])
    
    if 'ensemble_pred' in df.columns:
        df['prediction'] = df['ensemble_pred']
    
    ticker_data = df[df['ticker'] == ticker.upper()].sort_values('date', ascending=False)
    
    if len(ticker_data) == 0:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")
    
    results = []
    for _, row in ticker_data.iterrows():
        results.append({
            "ticker": row['ticker'],
            "date": row['date'].strftime('%Y-%m'),
            "prediction": float(row['prediction']),
            "actual": float(row['actual']) if pd.notna(row['actual']) else None,
            "sector": row['sector']
        })
    
    return results

@app.get("/top-stocks")
def get_top_stocks(
    n: int = Query(20, ge=1, le=100),
    date: Optional[str] = None
):
    """Get top N stocks by prediction"""
    df = load_csv_from_blob('predictions', 'ensemble_optimize_predictions.csv')
    
    if df is None:
        raise HTTPException(status_code=404, detail="Predictions not found")
    
    df['date'] = pd.to_datetime(df['date'])
    
    if 'ensemble_pred' in df.columns:
        df['prediction'] = df['ensemble_pred']
    
    if date:
        target_date = pd.to_datetime(date)
        df = df[df['date'] == target_date]
    else:
        latest_date = df['date'].max()
        df = df[df['date'] == latest_date]
    
    top_stocks = df.nlargest(n, 'prediction')
    
    return {
        "date": top_stocks['date'].iloc[0].strftime('%Y-%m'),
        "count": len(top_stocks),
        "stocks": top_stocks[['ticker', 'prediction', 'sector']].to_dict('records')
    }

@app.get("/performance", response_model=PerformanceMetrics)
def get_performance():
    """Get model performance metrics"""
    oot = load_csv_from_blob('reports', 'oot_validation_report.csv')
    
    if oot is None:
        raise HTTPException(status_code=404, detail="Performance metrics not found")
    
    return PerformanceMetrics(
        mean_ic=float(oot['OOT_Mean_IC'].iloc[0]),
        ic_ir=float(oot['OOT_IC_IR'].iloc[0]),
        sharpe=float(oot['OOT_Sharpe'].iloc[0]),
        win_rate=float(oot['OOT_WinRate'].iloc[0])
    )

@app.get("/features")
def get_feature_importance():
    """Get feature importance rankings"""
    df = load_csv_from_blob('reports', 'feature_ic_summary.csv')
    
    if df is None:
        raise HTTPException(status_code=404, detail="Feature importance not found")
    
    df = df.sort_values('mean_ic', ascending=False)
    
    return {
        "count": len(df),
        "features": df[['feature', 'mean_ic', 'ic_ir']].to_dict('records')
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
