"""Quick validation of generated reports."""
import pandas as pd
import os

print("Validating Generated Reports")
print("="*60)

# Check CSV report
csv_path = "reports/leakage_audit_report.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"\n✓ CSV Report: {csv_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Categories: {df['category'].unique().tolist()}")
    print(f"  Severity Levels: {df['severity'].unique().tolist()}")
    
    # Check for key features
    assert 'automated_flags' in df.columns, "Missing automated_flags column"
    assert 'recommendations' in df.columns, "Missing recommendations column"
    assert 'severity' in df.columns, "Missing severity column"
    print("  ✓ All required columns present")
    
    # Check for performance anomaly
    perf_rows = df[df['test_name'].str.contains('PERFORMANCE', na=False)]
    if not perf_rows.empty:
        print(f"  ✓ Performance anomaly section present")
else:
    print(f"✗ CSV report not found: {csv_path}")

# Check text report
txt_path = "reports/leakage_audit_report_detailed.txt"
if os.path.exists(txt_path):
    with open(txt_path, 'r') as f:
        content = f.read()
    print(f"\n✓ Text Report: {txt_path}")
    print(f"  Size: {len(content)} characters")
    
    # Check for key sections
    sections = [
        "COMPREHENSIVE LEAKAGE DETECTION AUDIT REPORT",
        "PERFORMANCE ANOMALY ANALYSIS",
        "TEST RESULTS BY CATEGORY",
        "CRITICAL VIOLATIONS",
        "AUTOMATED FLAGS SUMMARY",
        "RECOMMENDATIONS"
    ]
    
    for section in sections:
        if section in content:
            print(f"  ✓ Section present: {section}")
        else:
            print(f"  ✗ Section missing: {section}")
else:
    print(f"✗ Text report not found: {txt_path}")

print("\n" + "="*60)
print("Validation Complete!")
