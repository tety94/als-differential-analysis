import pandas as pd
import logging

def generate_missing_report(df, output_folder):
    logging.info("🔍 Avvio missing report")

    missing = df.isna().sum()
    missing_percent = df.isna().mean() * 100
    report = pd.DataFrame({'missing_count': missing, 'missing_percent': missing_percent})
    report = report.sort_values('missing_percent', ascending=False)
    report_path = f"{output_folder}/missing_report.csv"
    report.to_csv(report_path)
    print(f"✅ Report valori null salvato in: {report_path}")
    logging.info(f"✅ Report valori null salvato in: {report_path}")
    return report
