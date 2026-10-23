"""
Main Pipeline Orchestrator
Executes the full end-to-end data ingestion, cleaning, feature engineering,
validation, and export pipeline. Suitable for automated batch processing and CLI usage.
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_ingestion import DataIngestion
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.data_validation import DataValidator

def setup_logging(log_file_path: str = "outputs/preprocessing_log.txt"):
    """Configures simultaneous file and console logging."""
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def generate_before_after_comparison(df_raw: pd.DataFrame, df_clean: pd.DataFrame, df_quarantined: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rigorous before-and-after data health metrics.
    """
    raw_rows = len(df_raw)
    clean_rows = len(df_clean)
    quarantine_rows = len(df_quarantined)

    # Missing counts
    raw_missing_cells = int(df_raw.isna().sum().sum())
    clean_missing_cells = int(df_clean.isna().sum().sum())
    missing_reduction_pct = round(((raw_missing_cells - clean_missing_cells) / max(raw_missing_cells, 1)) * 100, 2)

    # Duplicate counts
    raw_exact_dups = int(df_raw.duplicated().sum())
    clean_exact_dups = int(df_clean.duplicated().sum())

    raw_id_dups = int(df_raw["Order_ID"].duplicated().sum()) if "Order_ID" in df_raw.columns else 0
    clean_id_dups = int(df_clean["Order_ID"].duplicated().sum()) if "Order_ID" in df_clean.columns else 0

    # Data types check
    raw_text_cols = len(df_raw.select_dtypes(include=["object", "string"]).columns)
    clean_text_cols = len(df_clean.select_dtypes(include=["object", "string"]).columns)

    metrics = [
        {"Metric": "Total Records (Rows)", "Raw_Dataset": raw_rows, "Cleaned_Dataset": clean_rows, "Delta": clean_rows - raw_rows},
        {"Metric": "Total Features (Columns)", "Raw_Dataset": len(df_raw.columns), "Cleaned_Dataset": len(df_clean.columns), "Delta": len(df_clean.columns) - len(df_raw.columns)},
        {"Metric": "Missing Value Cells", "Raw_Dataset": raw_missing_cells, "Cleaned_Dataset": clean_missing_cells, "Delta": clean_missing_cells - raw_missing_cells},
        {"Metric": "Missing Value Reduction (%)", "Raw_Dataset": "0.0%", "Cleaned_Dataset": f"{missing_reduction_pct}%", "Delta": f"+{missing_reduction_pct}%"},
        {"Metric": "Exact Duplicate Rows", "Raw_Dataset": raw_exact_dups, "Cleaned_Dataset": clean_exact_dups, "Delta": -raw_exact_dups},
        {"Metric": "Duplicate Business Keys (Order_ID)", "Raw_Dataset": raw_id_dups, "Cleaned_Dataset": clean_id_dups, "Delta": -raw_id_dups},
        {"Metric": "Quarantined Corrupted Records", "Raw_Dataset": 0, "Cleaned_Dataset": quarantine_rows, "Delta": quarantine_rows},
        {"Metric": "Text / String Columns", "Raw_Dataset": raw_text_cols, "Cleaned_Dataset": clean_text_cols, "Delta": clean_text_cols - raw_text_cols}
    ]

    return pd.DataFrame(metrics)

def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    setup_logging("outputs/preprocessing_log.txt")
    logger = logging.getLogger("PipelineRunner")

    logger.info("=" * 70)
    logger.info("DATA INGESTION, CLEANING & PREPROCESSING PIPELINE INITIATED")
    logger.info("=" * 70)

    # 1. Ingestion
    raw_csv_path = "data/raw/retail_sales_raw.csv"
    if not os.path.exists(raw_csv_path):
        logger.error(f"Raw dataset not found at {raw_csv_path}. Generating benchmark dataset...")
        from scripts.generate_dataset import generate_messy_retail_dataset
        df_gen = generate_messy_retail_dataset(num_records=12200)
        os.makedirs("data/raw", exist_ok=True)
        df_gen.to_csv(raw_csv_path, index=False)

    ingestor = DataIngestion()
    df_raw = ingestor.load_dataset(raw_csv_path)
    raw_profile = ingestor.profile_raw_data(df_raw)

    logger.info(f"Raw dataset successfully ingested: {len(df_raw):,} records, {len(df_raw.columns)} features.")

    # 2. Data Cleaning
    cleaner = DataCleaner()
    df_cleaned, df_quarantined, audit_log = cleaner.clean_pipeline(df_raw)

    # 3. Feature Engineering
    engineer = FeatureEngineer()
    df_featured, disc_report = engineer.run_feature_engineering_pipeline(df_cleaned)

    # 4. Data Quality Validation
    validator = DataValidator()
    quality_report = validator.validate_dataset(df_featured)

    # 5. Before and After Comparison
    comparison_summary = generate_before_after_comparison(df_raw, df_featured, df_quarantined)

    # 6. Export Deliverables
    logger.info("Exporting all pipeline outputs...")

    # A. Cleaned dataset
    out_clean_path = "outputs/cleaned_dataset.csv"
    processed_path = "data/processed/cleaned_retail_sales.csv"
    df_featured.to_csv(out_clean_path, index=False)
    df_featured.to_csv(processed_path, index=False)
    logger.info(f"Cleaned dataset saved to: {out_clean_path} ({len(df_featured):,} records)")

    # B. Quarantined records
    out_quarantine_path = "outputs/quarantined_records.csv"
    df_quarantined.to_csv(out_quarantine_path, index=False)
    logger.info(f"Quarantined records saved to: {out_quarantine_path} ({len(df_quarantined):,} records)")

    # C. Data Quality Report
    out_quality_path = "outputs/data_quality_report.csv"
    quality_report.to_csv(out_quality_path, index=False)
    logger.info(f"Data quality report saved to: {out_quality_path}")

    # D. Cleaning Summary
    out_summary_path = "outputs/cleaning_summary.csv"
    comparison_summary.to_csv(out_summary_path, index=False)
    logger.info(f"Cleaning summary saved to: {out_summary_path}")

    logger.info("=" * 70)
    logger.info("PIPELINE EXECUTION SUCCESSFULLY COMPLETED")
    logger.info("=" * 70)

    # Print Summary Table to stdout
    print("\n" + "=" * 50)
    print("      DATA CLEANING & QUALITY EXECUTIVE SUMMARY      ")
    print("=" * 50)
    print(comparison_summary.to_string(index=False))
    print("\n" + "=" * 50)
    print("            DATA QUALITY CONTRACT CHECKS             ")
    print("=" * 50)
    print(quality_report[["Rule_ID", "Column_Name", "Pass_Fail", "Failures"]].to_string(index=False))
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
