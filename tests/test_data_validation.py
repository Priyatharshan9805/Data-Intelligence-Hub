"""
Automated Data Quality Validation Tests
"""

import pytest
import pandas as pd
import numpy as np
from src.data_validation import DataValidator

@pytest.fixture
def clean_valid_dataset():
    """Provides a perfectly conforming dataset that should pass all validation rules."""
    return pd.DataFrame({
        "Order_ID": ["ORD-1", "ORD-2", "ORD-3"],
        "Order_Date": pd.to_datetime(["2024-05-10", "2024-06-12", "2024-08-20"]),
        "Customer_ID": ["CUST-1", "CUST-2", "CUST-3"],
        "Customer_Name": ["Alice", "Bob", "Carol"],
        "Product": ["Dell XPS 15 Laptop", "Ergonomic Mesh Office Chair", "All-Weather Waterproof Jacket"],
        "Category": ["Technology", "Furniture", "Clothing"],
        "Quantity": [1, 2, 1],
        "Unit_Price": [1299.99, 289.00, 120.00],
        "Gross_Sales": [1299.99, 578.00, 120.00],
        "Discount": [10.0, 0.0, 15.0],
        "Discount_Amount": [130.00, 0.00, 18.00],
        "Net_Sales": [1169.99, 578.00, 102.00],
        "Cost": [900.00, 320.00, 58.00],
        "Calculated_Profit": [269.99, 258.00, 44.00],
        "Profit_Margin_Percentage": [23.08, 44.64, 43.14],
        "Order_Status": ["Delivered", "Shipped", "Processing"],
        "Transaction_Type": ["Sale", "Sale", "Sale"]
    })

def test_validator_passes_on_valid_data(clean_valid_dataset):
    validator = DataValidator()
    report = validator.validate_dataset(clean_valid_dataset)
    assert not report.empty
    assert (report["Pass_Fail"] == "PASS").all()
    assert report["Failures"].sum() == 0

def test_validator_detects_duplicate_order_ids(clean_valid_dataset):
    corrupted = clean_valid_dataset.copy()
    corrupted.loc[1, "Order_ID"] = "ORD-1"  # Duplicate Order_ID
    validator = DataValidator()
    report = validator.validate_dataset(corrupted)
    v02 = report[report["Rule_ID"] == "V02"].iloc[0]
    assert v02["Pass_Fail"] == "FAIL"
    assert v02["Failures"] == 1

def test_validator_detects_invalid_categories(clean_valid_dataset):
    corrupted = clean_valid_dataset.copy()
    corrupted.loc[0, "Category"] = "Unapproved Category"
    validator = DataValidator()
    report = validator.validate_dataset(corrupted)
    v10 = report[report["Rule_ID"] == "V10"].iloc[0]
    assert v10["Pass_Fail"] == "FAIL"
    assert v10["Failures"] == 1

def test_validator_detects_sales_math_mismatch(clean_valid_dataset):
    corrupted = clean_valid_dataset.copy()
    corrupted.loc[0, "Net_Sales"] = 9999.99  # Math discrepancy
    validator = DataValidator()
    report = validator.validate_dataset(corrupted)
    v09 = report[report["Rule_ID"] == "V09"].iloc[0]
    assert v09["Pass_Fail"] == "FAIL"
    assert v09["Failures"] == 1

def test_validator_handles_empty_dataset():
    validator = DataValidator()
    report = validator.validate_dataset(pd.DataFrame())
    v01 = report[report["Rule_ID"] == "V01"].iloc[0]
    assert v01["Pass_Fail"] == "FAIL"
    assert v01["Failures"] == 1
