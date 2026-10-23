"""
Unit Tests for Feature Engineering Module
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer

@pytest.fixture
def clean_test_df():
    return pd.DataFrame({
        "Order_ID": ["ORD-1", "ORD-2", "ORD-3", "ORD-4"],
        "Order_Date": pd.to_datetime(["2024-01-15", "2024-06-22", "2024-09-08", "2024-11-28"]),
        "Customer_ID": ["CUST-1", "CUST-1", "CUST-2", "CUST-3"],
        "Quantity": [2, 1, 4, 10],
        "Unit_Price": [100.0, 50.0, 25.0, 150.0],
        "Discount": [10.0, 0.0, 20.0, 15.0],
        "Cost": [120.0, 30.0, 60.0, 900.0],
        "Profit": [60.0, 20.0, 20.0, 375.0]
    })

def test_temporal_feature_extraction(clean_test_df):
    fe = FeatureEngineer()
    res = fe.extract_temporal_features(clean_test_df, date_col="Order_Date")

    assert "Order_Year" in res.columns
    assert "Order_Month" in res.columns
    assert "Order_Quarter" in res.columns
    assert "Order_Day_Name" in res.columns
    assert "Is_Weekend" in res.columns

    # 2024-01-15 is a Monday (Is_Weekend == 0)
    assert res.iloc[0]["Order_Year"] == 2024
    assert res.iloc[0]["Order_Month"] == 1
    assert res.iloc[0]["Order_Month_Name"] == "January"
    assert res.iloc[0]["Is_Weekend"] == 0

    # 2024-06-22 is a Saturday (Is_Weekend == 1)
    assert res.iloc[1]["Is_Weekend"] == 1

def test_sales_financial_formulas(clean_test_df):
    fe = FeatureEngineer()
    res, audit = fe.compute_sales_financials(clean_test_df)

    # Row 0: Quantity=2, Unit_Price=100.0, Discount=10.0%
    # Gross_Sales = 2 * 100 = 200.0
    # Discount_Amount = 200 * 0.10 = 20.0
    # Net_Sales = 200 - 20 = 180.0
    # Profit = 180 - Cost(120) = 60.0
    # Profit_Margin_Percentage = 60 / 180 * 100 = 33.33%
    row0 = res.iloc[0]
    assert row0["Gross_Sales"] == 200.0
    assert row0["Discount_Amount"] == 20.0
    assert row0["Net_Sales"] == 180.0
    assert row0["Calculated_Profit"] == 60.0
    assert abs(row0["Profit_Margin_Percentage"] - 33.33) < 0.02

def test_transaction_value_bands(clean_test_df):
    fe = FeatureEngineer()
    res, _ = fe.compute_sales_financials(clean_test_df)
    res = fe.compute_customer_and_transaction_features(res)

    assert "Transaction_Value_Band" in res.columns
    # Row 0 Net Sales is 180.0 -> Medium ($50 - $250)
    assert res.iloc[0]["Transaction_Value_Band"] == "Medium ($50 - $250)"
    # Row 3 Net Sales is 10 * 150 - 15% = 1500 - 225 = 1275 -> Enterprise (> $1,000)
    assert res.iloc[3]["Transaction_Value_Band"] == "Enterprise (> $1,000)"

def test_customer_aggregations(clean_test_df):
    fe = FeatureEngineer()
    res, _ = fe.compute_sales_financials(clean_test_df)
    res = fe.compute_customer_and_transaction_features(res)

    assert "Customer_Purchase_Count" in res.columns
    assert "Customer_Segment" in res.columns

    # CUST-1 ordered twice
    cust1_rows = res[res["Customer_ID"] == "CUST-1"]
    assert cust1_rows.iloc[0]["Customer_Purchase_Count"] == 2
    assert cust1_rows.iloc[0]["Customer_Segment"] == "Repeat Customer"
