"""
Data Quality Validation Module
Provides automated enterprise data contract validation, structural verification,
and generates structured compliance reports with Rule IDs and corrective actions.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Automated data quality gatekeeper and contract compliance verifier.
    Generates structured auditing reports for executive governance.
    """

    def __init__(self):
        self.validation_results: List[Dict[str, Any]] = []

    def add_result(
        self,
        rule_id: str,
        column_name: str,
        description: str,
        records_checked: int,
        failures: int,
        recommended_action: str
    ):
        """Appends validation check outcome to audit report."""
        status = "PASS" if failures == 0 else "FAIL"
        entry = {
            "Rule_ID": rule_id,
            "Column_Name": column_name,
            "Validation_Description": description,
            "Records_Checked": records_checked,
            "Failures": failures,
            "Pass_Fail": status,
            "Recommended_Action": recommended_action if failures > 0 else "None required (Meets quality threshold)"
        }
        self.validation_results.append(entry)
        level = logging.INFO if status == "PASS" else logging.WARNING
        logger.log(level, f"[{status}] {rule_id} ({column_name}): {description} - Failures: {failures:,}")

    def validate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes complete quality validation ruleset against a processed dataset.
        Returns a DataFrame representing the quality contract report.
        """
        self.validation_results.clear()
        n_rows = len(df)

        # Rule V01: Dataset Non-Emptiness
        self.add_result(
            rule_id="V01",
            column_name="DATASET",
            description="Verify dataset is non-empty and contains records",
            records_checked=n_rows,
            failures=0 if n_rows > 0 else 1,
            recommended_action="Ensure raw ingestion stage received a valid non-empty data file"
        )

        if n_rows == 0:
            return pd.DataFrame(self.validation_results)

        # Rule V02: Primary Identifier Uniqueness
        if "Order_ID" in df.columns:
            id_dups = int(df["Order_ID"].duplicated().sum())
            self.add_result(
                rule_id="V02",
                column_name="Order_ID",
                description="Zero duplicate business primary keys (Order_ID uniqueness)",
                records_checked=n_rows,
                failures=id_dups,
                recommended_action="Quarantine duplicate Order_ID instances for business identity resolution"
            )

        # Rule V03: Order_Date Datetime Validity
        if "Order_Date" in df.columns:
            invalid_dates = int(pd.to_datetime(df["Order_Date"], errors="coerce").isna().sum())
            self.add_result(
                rule_id="V03",
                column_name="Order_Date",
                description="All order dates must be valid ISO datetime objects",
                records_checked=n_rows,
                failures=invalid_dates,
                recommended_action="Check raw date format parsing parameters in data cleaning module"
            )

        # Rule V04: Temporal Boundary Constraints
        if "Order_Date" in df.columns:
            dt = pd.to_datetime(df["Order_Date"], errors="coerce")
            future_dates = int((dt > pd.Timestamp.now()).sum())
            self.add_result(
                rule_id="V04",
                column_name="Order_Date",
                description="Order date must not occur in the future",
                records_checked=n_rows,
                failures=future_dates,
                recommended_action="Filter out prospective test orders or rectify clock drift in upstream ingestion"
            )

        # Rule V05: Missing Critical Fields (Customer_ID)
        if "Customer_ID" in df.columns:
            missing_cust = int(df["Customer_ID"].isna().sum())
            self.add_result(
                rule_id="V05",
                column_name="Customer_ID",
                description="Zero missing Customer_ID identifiers in processed records",
                records_checked=n_rows,
                failures=missing_cust,
                recommended_action="Assign synthetic guest identifiers or reject anonymous orders"
            )

        # Rule V06: Unit Price Non-Negativity
        if "Unit_Price" in df.columns:
            neg_price = int((df["Unit_Price"] <= 0).sum())
            self.add_result(
                rule_id="V06",
                column_name="Unit_Price",
                description="Unit_Price must be strictly positive (> $0.00)",
                records_checked=n_rows,
                failures=neg_price,
                recommended_action="Inspect product master pricing catalog; quarantine zero/negative priced rows"
            )

        # Rule V07: Discount Scale Integrity
        if "Discount" in df.columns:
            invalid_disc = int(((df["Discount"] < 0) | (df["Discount"] > 100)).sum())
            self.add_result(
                rule_id="V07",
                column_name="Discount",
                description="Discount rate must be bounded between 0.0% and 100.0%",
                records_checked=n_rows,
                failures=invalid_disc,
                recommended_action="Normalize fractional discounts to percentages and clamp invalid promotions"
            )

        # Rule V08: Financial Math Consistency - Gross Sales
        if all(col in df.columns for col in ["Gross_Sales", "Quantity", "Unit_Price"]):
            expected_gross = (df["Quantity"].abs() * df["Unit_Price"]).round(2)
            gross_mismatch = int(((df["Gross_Sales"] - expected_gross).abs() > 0.05).sum())
            self.add_result(
                rule_id="V08",
                column_name="Gross_Sales",
                description="Gross_Sales must equal |Quantity| * Unit_Price within $0.05 tolerance",
                records_checked=n_rows,
                failures=gross_mismatch,
                recommended_action="Recalculate Gross_Sales using vectorized feature engineering formulas"
            )

        # Rule V09: Financial Math Consistency - Net Sales
        if all(col in df.columns for col in ["Net_Sales", "Gross_Sales", "Discount_Amount"]):
            # For ordinary sales, Net_Sales = Gross_Sales - Discount_Amount
            # For returns, Net_Sales = -(Gross_Sales - Discount_Amount)
            expected_net = (df["Gross_Sales"] - df["Discount_Amount"]).round(2)
            if "Transaction_Type" in df.columns:
                return_mask = df["Transaction_Type"] == "Return"
                expected_net[return_mask] = -expected_net[return_mask]

            net_mismatch = int(((df["Net_Sales"] - expected_net).abs() > 0.05).sum())
            self.add_result(
                rule_id="V09",
                column_name="Net_Sales",
                description="Net_Sales must equal Gross_Sales - Discount_Amount (signed for returns)",
                records_checked=n_rows,
                failures=net_mismatch,
                recommended_action="Check discount deduction and return sign convention in feature engineering"
            )

        # Rule V10: Categorical Domain Consistency (Category)
        valid_categories = {"Technology", "Furniture", "Office Supplies", "Clothing", "Home & Kitchen"}
        if "Category" in df.columns:
            invalid_cat = int((~df["Category"].isin(valid_categories)).sum())
            self.add_result(
                rule_id="V10",
                column_name="Category",
                description=f"Category must belong to approved master taxonomy: {sorted(list(valid_categories))}",
                records_checked=n_rows,
                failures=invalid_cat,
                recommended_action="Map unapproved category strings to approved taxonomy via product mapping"
            )

        # Rule V11: Categorical Domain Consistency (Order_Status)
        valid_statuses = {"Delivered", "Shipped", "Processing", "Cancelled", "Refunded"}
        if "Order_Status" in df.columns:
            invalid_status = int((~df["Order_Status"].isin(valid_statuses)).sum())
            self.add_result(
                rule_id="V11",
                column_name="Order_Status",
                description=f"Order_Status must belong to approved lifecycle states: {sorted(list(valid_statuses))}",
                records_checked=n_rows,
                failures=invalid_status,
                recommended_action="Standardize casing and map legacy fulfillment statuses"
            )

        # Rule V12: Safe Profit Margin Computation (Zero / Negative Denominator Handling)
        if "Profit_Margin_Percentage" in df.columns:
            # Check for Inf, -Inf, or NaN in ordinary positive sales
            ordinary_sales = (df.get("Transaction_Type", "Sale") == "Sale") & (df.get("Net_Sales", 1) > 0)
            inf_margins = int(np.isinf(df.loc[ordinary_sales, "Profit_Margin_Percentage"]).sum())
            self.add_result(
                rule_id="V12",
                column_name="Profit_Margin_Percentage",
                description="Profit_Margin_Percentage must not contain infinite values (zero denominator protection)",
                records_checked=int(ordinary_sales.sum()),
                failures=inf_margins,
                recommended_action="Apply conditional zero-check guard before computing margin division"
            )

        # Rule V13: Outlier Bounds Verification
        if "Unit_Price" in df.columns and "Is_Outlier_Unit_Price" in df.columns:
            # Verify no price exceeds 5000 (after capping)
            extreme_prices = int((df["Unit_Price"] > 5000).sum())
            self.add_result(
                rule_id="V13",
                column_name="Unit_Price",
                description="Extreme prices must be bounded after Winsorizing outlier treatment",
                records_checked=n_rows,
                failures=extreme_prices,
                recommended_action="Verify IQR capping bounds in outlier treatment step"
            )

        report_df = pd.DataFrame(self.validation_results)
        return report_df
