"""
Unit and Integration Tests for Data Cleaning Module
"""

import pytest
import pandas as pd
import numpy as np
from src.data_cleaning import DataCleaner

@pytest.fixture
def sample_raw_df():
    """Provides a controlled micro-dataset with representative anomalies."""
    return pd.DataFrame({
        "Order_ID": ["ORD-1", "ORD-1", "ORD-2", "ORD-3", "ORD-4", "ORD-4"],
        "Order_Date": ["2024-05-10", "2024-05-10", "15/06/2024", "invalid_date", "2024-08-20", "2024-08-20"],
        "Customer_ID": ["CUST-10", "CUST-10", "CUST-11", "CUST-12", "CUST-13", "CUST-13"],
        "Customer_Name": ["Alice Smith", "Alice Smith", "  Bob Jones  ", "N/A", "Carol Danvers", "Carol Danvers"],
        "Product": ["Dell XPS 15 Laptop", "Dell XPS 15 Laptop", "Ergonomic Mesh Office Chair", "Gel Ink Pen Pack (24ct)", "Apple iPhone 15 Pro", "Apple iPhone 15 Pro"],
        "Category": ["technology", "technology", np.nan, "office supplies", "Technology", "Technology"],
        "Quantity": [" 2 ", " 2 ", "-1", "5", "300", "300"],
        "Unit_Price": ["$1,299.99", "$1,299.99", "289.00", "18.50", "1099.00", "1099.00"],
        "Sales": ["$2,339.98", "$2,339.98", "260.10", "92.50", "329700.00", "329700.00"],
        "Discount": ["10%", "10%", "0.10", "null", "0%", "0%"],
        "Cost": ["1800.00", "1800.00", np.nan, "37.50", "2250.00", "2250.00"],
        "Profit": [539.98, 539.98, -20.0, 55.0, 3000.0, 3000.0],
        "City": ["NYC", "NYC", "los angeles", "Chicago", "San Francisco", "San Francisco"],
        "Region": ["East", "East", "West", "Central", "West", "West"],
        "Payment_Method": ["credit card", "credit card", "PayPal", "MISSING", "Debit Card", "Debit Card"],
        "Order_Status": ["delivered", "delivered", "shipped", "pending_review", "Delivered", "Delivered"]
    })

def test_empty_rows_removal():
    cleaner = DataCleaner()
    df_with_empty = pd.DataFrame({
        "ColA": ["Value", "", np.nan],
        "ColB": ["Value", "   ", None]
    })
    res = cleaner.remove_empty_rows(df_with_empty)
    assert len(res) == 1
    assert res.iloc[0]["ColA"] == "Value"

def test_placeholder_normalization():
    cleaner = DataCleaner()
    df_test = pd.DataFrame({
        "Status": ["Active", "N/A", "none", "?", "MISSING", ""]
    })
    clean_df = cleaner.normalize_placeholders(df_test)
    assert pd.isna(clean_df.loc[1, "Status"])
    assert pd.isna(clean_df.loc[2, "Status"])
    assert pd.isna(clean_df.loc[3, "Status"])
    assert pd.isna(clean_df.loc[4, "Status"])
    assert pd.isna(clean_df.loc[5, "Status"])
    assert clean_df.loc[0, "Status"] == "Active"

def test_exact_duplicates_removal(sample_raw_df):
    cleaner = DataCleaner()
    clean_df, quaran_df = cleaner.handle_duplicates(sample_raw_df, id_col="Order_ID")
    # Rows 0 and 1 are exact duplicates, rows 4 and 5 are exact duplicates
    assert len(clean_df) == 4
    assert clean_df["Order_ID"].tolist() == ["ORD-1", "ORD-2", "ORD-3", "ORD-4"]

def test_conflicting_id_quarantine():
    cleaner = DataCleaner()
    df_conflicts = pd.DataFrame({
        "Order_ID": ["ORD-100", "ORD-100"],
        "Customer_ID": ["CUST-1", "CUST-2"],
        "Sales": [100.0, 500.0]
    })
    clean_df, quaran_df = cleaner.handle_duplicates(df_conflicts, id_col="Order_ID")
    assert len(clean_df) == 0
    assert len(quaran_df) == 2
    assert "Conflicting duplicate" in quaran_df.iloc[0]["Quarantine_Reason"]

def test_numeric_string_parsing():
    cleaner = DataCleaner()
    s = pd.Series(["$1,299.99", "  89.50 ", "15%", "N/A", "invalid"])
    parsed = cleaner.parse_numeric_strings(s)
    assert parsed.iloc[0] == 1299.99
    assert parsed.iloc[1] == 89.50
    assert parsed.iloc[2] == 15.0
    assert pd.isna(parsed.iloc[3])
    assert pd.isna(parsed.iloc[4])

def test_discount_fractional_scale_normalization():
    cleaner = DataCleaner()
    df = pd.DataFrame({"Discount": ["0.15", "0.20", "10%", "0%"]})
    clean_df, _ = cleaner.clean_and_coerce_types(df)
    assert clean_df["Discount"].iloc[0] == 15.0
    assert clean_df["Discount"].iloc[1] == 20.0
    assert clean_df["Discount"].iloc[2] == 10.0
    assert clean_df["Discount"].iloc[3] == 0.0

def test_category_and_city_normalization():
    cleaner = DataCleaner()
    df = pd.DataFrame({
        "Category": ["  technology ", "OFFICE SUPPLIES"],
        "City": ["nyc", "los angeles"],
        "Order_Status": ["pending_review", "delivered"]
    })
    cleaned = cleaner.standardize_text_fields(df)
    assert cleaned["Category"].iloc[0] == "Technology"
    assert cleaned["Category"].iloc[1] == "Office Supplies"
    assert cleaned["City"].iloc[0] == "New York"
    assert cleaned["City"].iloc[1] == "Los Angeles"
    assert cleaned["Order_Status"].iloc[0] == "Processing"
    assert cleaned["Order_Status"].iloc[1] == "Delivered"

def test_missing_category_imputation():
    cleaner = DataCleaner()
    df = pd.DataFrame({
        "Product": ["Dell XPS 15 Laptop", "Heavy-Duty File Folders (Box 100)"],
        "Category": [np.nan, None]
    })
    cleaned = cleaner.handle_missing_values(df)
    assert cleaned["Category"].iloc[0] == "Technology"
    assert cleaned["Category"].iloc[1] == "Office Supplies"

def test_sales_returns_classification():
    cleaner = DataCleaner()
    df = pd.DataFrame({
        "Quantity": [-2.0, 5.0],
        "Unit_Price": [50.0, 20.0],
        "Discount": [0.0, 0.0]
    })
    cleaned, quaran = cleaner.enforce_business_rules(df)
    assert cleaned["Transaction_Type"].iloc[0] == "Return"
    assert cleaned["Transaction_Type"].iloc[1] == "Sale"
