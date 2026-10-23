"""
Feature Engineering Module
Constructs standardized date attributes, financial calculations (Gross Sales, Discount Amount,
Net Sales, Profit Margin Percentage), customer lifetime aggregations, and transaction value bands.
Provides mathematical discrepancy audits between raw reported sales and calculated financials.
"""

import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Constructs high-value analytical and temporal features for retail business intelligence.
    """

    VALUE_BANDS = [
        (0, 50, "Low (< $50)"),
        (50, 250, "Medium ($50 - $250)"),
        (250, 1000, "High ($250 - $1,000)"),
        (1000, float("inf"), "Enterprise (> $1,000)")
    ]

    def __init__(self):
        self.discrepancy_report: Dict[str, Any] = {}

    def extract_temporal_features(self, df: pd.DataFrame, date_col: str = "Order_Date") -> pd.DataFrame:
        """
        Derives granular calendar and seasonality features from Order_Date.
        """
        df_feat = df.copy()
        if date_col not in df_feat.columns:
            logger.warning(f"Date column '{date_col}' not found. Skipping date feature extraction.")
            return df_feat

        dt_series = pd.to_datetime(df_feat[date_col], errors="coerce")

        df_feat["Order_Year"] = dt_series.dt.year.astype("Int64")
        df_feat["Order_Month"] = dt_series.dt.month.astype("Int64")
        df_feat["Order_Month_Name"] = dt_series.dt.month_name()
        df_feat["Order_Quarter"] = dt_series.dt.quarter.astype("Int64")
        df_feat["Order_Day"] = dt_series.dt.day.astype("Int64")
        df_feat["Order_Day_Name"] = dt_series.dt.day_name()
        df_feat["Is_Weekend"] = dt_series.dt.day_of_week.isin([5, 6]).astype(int)

        logger.info(f"Derived 7 temporal features from '{date_col}'.")
        return df_feat

    def compute_sales_financials(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Derives core unit economics:
        - Gross_Sales = |Quantity| * Unit_Price
        - Discount_Amount = Gross_Sales * (Discount / 100)
        - Net_Sales = Gross_Sales - Discount_Amount
        - Unit_Cost = Cost / |Quantity|
        - Profit_Margin_Percentage = (Profit / Net_Sales) * 100 (where Net_Sales > 0)
        Audits discrepancy against any pre-existing 'Sales' column.
        """
        df_feat = df.copy()

        # Check required columns
        req_cols = ["Quantity", "Unit_Price", "Discount"]
        for col in req_cols:
            if col not in df_feat.columns:
                raise KeyError(f"Required column '{col}' missing for sales feature engineering.")

        # Ensure numeric
        qty = df_feat["Quantity"].abs().astype(float)
        unit_price = df_feat["Unit_Price"].astype(float)
        discount = df_feat["Discount"].fillna(0.0).astype(float)

        # 1. Gross Sales
        df_feat["Gross_Sales"] = (qty * unit_price).round(2)

        # 2. Discount Amount
        df_feat["Discount_Amount"] = (df_feat["Gross_Sales"] * (discount / 100.0)).round(2)

        # 3. Net Sales
        df_feat["Net_Sales"] = (df_feat["Gross_Sales"] - df_feat["Discount_Amount"]).round(2)

        # 4. Profit & Profit Margin
        if "Profit" in df_feat.columns and "Cost" in df_feat.columns:
            # Reconcile Profit = Net_Sales - Cost
            # Where Profit was missing or inconsistent, use Net_Sales - Cost
            calculated_profit = (df_feat["Net_Sales"] - df_feat["Cost"].astype(float)).round(2)
            df_feat["Calculated_Profit"] = calculated_profit
            # If Transaction_Type is Return, profit is negative return processing cost
            if "Transaction_Type" in df_feat.columns:
                return_mask = df_feat["Transaction_Type"] == "Return"
                df_feat.loc[return_mask, "Net_Sales"] = -df_feat.loc[return_mask, "Net_Sales"]
                df_feat.loc[return_mask, "Calculated_Profit"] = -round(df_feat.loc[return_mask, "Gross_Sales"] * 0.15, 2)

        # Calculate Profit Margin Percentage (safely avoiding 0 division)
        profit_col = "Calculated_Profit" if "Calculated_Profit" in df_feat.columns else "Profit"
        if profit_col in df_feat.columns:
            valid_margin_mask = (df_feat["Net_Sales"] > 0) & (df_feat[profit_col].notna())
            df_feat["Profit_Margin_Percentage"] = np.nan
            df_feat.loc[valid_margin_mask, "Profit_Margin_Percentage"] = (
                (df_feat.loc[valid_margin_mask, profit_col] / df_feat.loc[valid_margin_mask, "Net_Sales"]) * 100.0
            ).round(2)

        # 5. Financial Discrepancy Audit
        discrepancy_stats = {}
        if "Sales" in df_feat.columns:
            raw_sales = pd.to_numeric(df_feat["Sales"], errors="coerce")
            sales_diff = (raw_sales - df_feat["Net_Sales"]).abs()
            has_diff = sales_diff > 0.05  # tolerance threshold 5 cents
            discrepancy_count = int(has_diff.sum())
            max_diff = float(sales_diff.max()) if not sales_diff.empty else 0.0
            mean_diff = float(sales_diff[has_diff].mean()) if discrepancy_count > 0 else 0.0

            discrepancy_stats = {
                "discrepancies_detected": discrepancy_count,
                "discrepancy_percentage": round((discrepancy_count / len(df_feat)) * 100, 2),
                "max_dollar_discrepancy": round(max_diff, 2),
                "mean_discrepancy": round(mean_diff, 2)
            }
            logger.info(
                f"Sales Discrepancy Audit: {discrepancy_count} rows ({discrepancy_stats['discrepancy_percentage']}%) "
                f"differed from newly calculated Net_Sales. Aligned to mathematically rigorous Net_Sales."
            )

        self.discrepancy_report = discrepancy_stats
        return df_feat, discrepancy_stats

    def compute_customer_and_transaction_features(
        self, df: pd.DataFrame, cust_col: str = "Customer_ID"
    ) -> pd.DataFrame:
        """
        Computes customer purchasing behavior aggregations and assigns transaction value tiers.
        """
        df_feat = df.copy()

        # 1. Transaction Value Tier
        if "Net_Sales" in df_feat.columns:
            def assign_band(val):
                if pd.isna(val):
                    return "Unknown"
                for low, high, label in FeatureEngineer.VALUE_BANDS:
                    if low <= val < high:
                        return label
                return "Unknown"

            df_feat["Transaction_Value_Band"] = df_feat["Net_Sales"].apply(assign_band)

        # 2. Customer Behavioral Metrics
        if cust_col in df_feat.columns and "Net_Sales" in df_feat.columns:
            # Frequency and Monetary Aggregations
            cust_summary = df_feat.groupby(cust_col).agg(
                Customer_Purchase_Count=("Order_ID", "count") if "Order_ID" in df_feat.columns else (cust_col, "count"),
                Customer_Total_Spend=("Net_Sales", "sum")
            ).reset_index()

            # Customer Segment Tiering
            def segment_customer(row):
                cnt = row["Customer_Purchase_Count"]
                spend = row["Customer_Total_Spend"]
                if cnt >= 5 or spend >= 2500:
                    return "VIP Customer"
                elif cnt >= 2:
                    return "Repeat Customer"
                else:
                    return "Single-Order Customer"

            cust_summary["Customer_Segment"] = cust_summary.apply(segment_customer, axis=1)

            # Merge back into dataset
            df_feat = df_feat.merge(cust_summary, on=cust_col, how="left")
            logger.info("Derived Customer_Purchase_Count, Customer_Total_Spend, and Customer_Segment.")

        return df_feat

    def run_feature_engineering_pipeline(self, df_clean: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Orchestrates full feature engineering pipeline.
        """
        logger.info(f"Initiating feature engineering on {len(df_clean):,} clean records...")
        step1 = self.extract_temporal_features(df_clean)
        step2, audit_meta = self.compute_sales_financials(step1)
        step3 = self.compute_customer_and_transaction_features(step2)

        logger.info(f"Feature engineering completed. Output columns count: {step3.shape[1]}")
        return step3, audit_meta
