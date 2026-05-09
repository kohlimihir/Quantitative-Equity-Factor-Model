"""Test script to verify dashboard data loading"""
import pandas as pd
import sys

def test_predictions():
    """Test predictions loading"""
    try:
        df = pd.read_csv('data/ensemble_optimize_predictions.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        # Rename ensemble_pred to prediction
        if 'ensemble_pred' in df.columns:
            df['prediction'] = df['ensemble_pred']
        
        print("✅ Predictions loaded successfully")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Tickers: {df['ticker'].nunique()}")
        return True
    except Exception as e:
        print(f"❌ Error loading predictions: {e}")
        return False

def test_shap():
    """Test SHAP values loading"""
    try:
        df = pd.read_csv('data/shap_values.csv')
        print("✅ SHAP values loaded successfully")
        print(f"   Rows: {len(df)}")
        return True
    except Exception as e:
        print(f"⚠️  SHAP values not found (optional): {e}")
        return False

def test_performance():
    """Test performance metrics loading"""
    try:
        oot = pd.read_csv('reports/oot_validation_report.csv')
        print("✅ OOT validation report loaded")
        print(f"   Rows: {len(oot)}")
        print(f"   Columns: {oot.columns.tolist()}")
        
        # Check for expected columns
        expected_cols = ['OOT_Mean_IC', 'OOT_IC_IR', 'OOT_Sharpe', 'OOT_WinRate']
        missing = [col for col in expected_cols if col not in oot.columns]
        if missing:
            print(f"   ⚠️  Missing columns: {missing}")
    except Exception as e:
        print(f"⚠️  OOT report not found (optional): {e}")
    
    try:
        turnover = pd.read_csv('reports/turnover_optimized.csv')
        print("✅ Turnover report loaded")
        print(f"   Rows: {len(turnover)}")
        print(f"   Columns: {turnover.columns.tolist()}")
        
        # Check for expected columns
        if 'turnover' in turnover.columns:
            print(f"   Mean turnover: {turnover['turnover'].mean()*100:.1f}%")
        else:
            print(f"   ⚠️  'turnover' column not found")
    except Exception as e:
        print(f"⚠️  Turnover report not found (optional): {e}")

def test_features():
    """Test feature importance loading"""
    try:
        df = pd.read_csv('reports/feature_ic_summary.csv')
        print("✅ Feature importance loaded")
        print(f"   Features: {len(df)}")
        return True
    except Exception as e:
        print(f"⚠️  Feature importance not found (optional): {e}")
        return False

if __name__ == "__main__":
    print("Testing Dashboard Data Loading...")
    print("=" * 50)
    
    predictions_ok = test_predictions()
    print()
    test_shap()
    print()
    test_performance()
    print()
    test_features()
    
    print()
    print("=" * 50)
    if predictions_ok:
        print("✅ Dashboard should work! Run: streamlit run app.py")
    else:
        print("❌ Dashboard will fail. Run model first: python run_all.py --config optimized_low_turnover")
