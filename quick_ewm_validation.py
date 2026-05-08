from leakage_detector import LeakageDetector
import pandas as pd
import numpy as np

# Create simple test data
dates = pd.date_range('2024-01-01', periods=3, freq='MS')
data = []
for i, date in enumerate(dates):
    for ticker in ['AAPL', 'MSFT']:
        data.append({
            'date': date,
            'ticker': ticker,
            'raw_rank': 0.3 + i * 0.1,
            'smoothed_rank': 0.3 + i * 0.05,
            'sector': 'Tech'
        })

df = pd.DataFrame(data)
detector = LeakageDetector(verbose=False)
result = detector.run_ewm_leakage_validation(df)
print(f'EWM Validation: {result["ewm_overall_assessment"]}')
print(f'Tests: {result["ewm_tests_passed"]}/{result["total_ewm_tests"]} passed')