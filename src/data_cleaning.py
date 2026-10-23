"""
Data Cleaning and Preprocessing Module
Implements industry-grade data scrubbing: placeholder normalization, type coercion,
exact & conflict duplicate handling, context-aware imputation, categorical standardization,
IQR outlier bounding, and business rule enforcement.
"""

import logging
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Comprehensive data cleaning engine for retail and e-commerce transactional data.
    Maintains full audit trails and separates quarantined records for governance.
    """

    PLACEHOLDERS = [
        "nan", "null", "none", "n/a", "na", "?", "missing", "unknown",
        "-", "--", "null value", "undefined"
    ]

    CITY_STANDARDIZATION_MAP = {
        "nyc": "New York",
        "new york": "New York",
        "new york city": "New York",
        "la": "Los Angeles",
        "los angeles": "Los Angeles",
        "sf": "San Francisco",
        "san francisco": "San Francisco",
        "san fran": "San Francisco",
        "chicago": "Chicago",
        "houston": "Houston",
        "dallas": "Dallas",
        "boston": "Boston",
        "seattle": "Seattle",
        "atlanta": "Atlanta",
        "miami": "Miami",
        "philadelphia": "Philadelphia",
        "charlotte": "Charlotte"
    }

    PRODUCT_CATEGORY_MAP = {
        "Dell XPS 15 Laptop": "Technology",
        "Apple iPhone 15 Pro": "Technology",
        "Sony WH-1000XM5 Headphones": "Technology",
        "Samsung 34' Curved Monitor": "Technology",
        "Logitech MX Master 3 Mouse": "Technology",
        "Mechanical Gaming Keyboard": "Technology",
        "Anker USB-C Fast Charger": "Technology",
        "Ergonomic Mesh Office Chair": "Furniture",
        "Motorized Standing Desk": "Furniture",
        "Solid Oak Coffee Table": "Furniture",
        "Bookshelf 5-Tier Wooden": "Furniture",
        "Modern Fabric Recliner": "Furniture",
        "Heavy-Duty File Folders (Box 100)": "Office Supplies",
        "Gel Ink Pen Pack (24ct)": "Office Supplies",
        "Stapler with 5000 Staples": "Office Supplies",
        "Desktop Document Organizer": "Office Supplies",
        "Whiteboard Magnetic 36x24": "Office Supplies",
        "Men's Tailored Wool Blazer": "Clothing",
        "Women's Cashmere Sweater": "Clothing",
        "All-Weather Waterproof Jacket": "Clothing",
        "Athletic Running Shoes": "Clothing",
        "Breathable Cotton Crew Socks (6pk)": "Clothing",
        "Cold Brew Coffee Maker 1L": "Home & Kitchen",
        "Cast Iron Dutch Oven 6-Quart": "Home & Kitchen",
        "Stainless Steel Chef Knife Set": "Home & Kitchen",
        "Smart Electric Kettle 1.7L": "Home & Kitchen",
        "HEPA Air Purifier Compact": "Home & Kitchen"
    }

    CATEGORY_MARGIN_MAP = {
        "Technology": 0.35,
        "Furniture": 0.42,
        "Office Supplies": 0.55,
        "Clothing": 0.50,
        "Home & Kitchen": 0.48
    }

    def __init__(self):
        self.cleaning_audit_log: List[Dict[str, Any]] = []
        self.quarantined_records: List[pd.DataFrame] = []

    def log_action(self, step: str, column: str, description: str, records_impacted: int, details: Optional[Dict] = None):
        """Records cleaning transformation for auditability."""
        log_entry = {
            "step": step,
            "column": column,
            "description": description,
            "records_impacted": records_impacted,
            "details": details or {}
        }
        self.cleaning_audit_log.append(log_entry)
        logger.info(f"[{step}] {column}: {description} (Impacted: {records_impacted:,})")

    def remove_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes completely blank or all-NaN rows."""
        initial_len = len(df)
        # Check if all columns are NaN or empty whitespace
        is_empty = df.isna().all(axis=1) | df.map(lambda x: str(x).strip() in ("", "nan", "None") if pd.notna(x) else True).all(axis=1)
        empty_count = int(is_empty.sum())
        df_cleaned = df[~is_empty].copy()

        if empty_count > 0:
            self.log_action("Empty Rows Removal", "ALL", "Removed completely unpopulated records", empty_count)

        return df_cleaned

    def normalize_placeholders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes textual null placeholders ('N/A', '?', 'none', etc.) to np.nan."""
        df_clean = df.copy()
        total_placeholders_replaced = 0

        for col in df_clean.columns:
            if df_clean[col].dtype == object or isinstance(df_clean[col].dtype, pd.StringDtype):
                # Clean whitespace
                s_str = df_clean[col].astype(str).str.strip()
                s_lower = s_str.str.lower()
                mask = s_lower.isin(self.PLACEHOLDERS) | (s_str == "")
                count = int(mask.sum())
                if count > 0:
                    df_clean.loc[mask, col] = np.nan
                    total_placeholders_replaced += count

        self.log_action(
            "Placeholder Normalization", "ALL_TEXT",
            "Converted invalid string placeholders ('N/A', '?', 'null', blanks) to standard NaN",
            total_placeholders_replaced
        )
        return df_clean

    def handle_duplicates(self, df: pd.DataFrame, id_col: str = "Order_ID") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Differentiates between:
        1. Exact duplicate rows (identical across every field) -> safely dropped.
        2. Conflicting business ID duplicates (same Order_ID, differing attributes) -> quarantined.
        3. Legitimate repeat purchases (same Customer_ID) -> preserved.
        """
        initial_len = len(df)

        # 1. Exact duplicates across all columns
        exact_dups_mask = df.duplicated(keep="first")
        exact_dups_count = int(exact_dups_mask.sum())
        df_no_exact = df[~exact_dups_mask].copy()

        self.log_action(
            "Duplicate Removal", "ALL_COLUMNS",
            "Identified and removed exact full-row duplicate records",
            exact_dups_count
        )

        # 2. Conflicting business key duplicates on id_col
        if id_col in df_no_exact.columns:
            # Drop NaN IDs first for ID uniqueness check
            valid_id_mask = df_no_exact[id_col].notna() & (df_no_exact[id_col].astype(str).str.strip() != "")
            id_dups_mask = df_no_exact[valid_id_mask].duplicated(subset=[id_col], keep=False)

            conflict_indices = df_no_exact[valid_id_mask][id_dups_mask].index
            conflict_records = df_no_exact.loc[conflict_indices].copy()
            conflict_records["Quarantine_Reason"] = f"Conflicting duplicate primary key: {id_col}"

            if not conflict_records.empty:
                self.quarantined_records.append(conflict_records)
                self.log_action(
                    "Duplicate Business Key Quarantine", id_col,
                    "Quarantined records with duplicate Order_ID containing conflicting attributes",
                    len(conflict_records)
                )
                df_no_exact = df_no_exact.drop(index=conflict_indices)

        return df_no_exact, pd.concat(self.quarantined_records, ignore_index=True) if self.quarantined_records else pd.DataFrame()

    def parse_numeric_strings(self, series: pd.Series) -> pd.Series:
        """Strips currency symbols, commas, percent signs, and whitespace."""
        s = series.astype(str).str.strip()
        # Remove '$', ',', '%', and whitespace
        s = s.str.replace("$", "", regex=False)
        s = s.str.replace(",", "", regex=False)
        s = s.str.replace("%", "", regex=False)
        s = s.str.strip()
        s = s.replace(self.PLACEHOLDERS + ["", "nan"], np.nan)
        return pd.to_numeric(s, errors="coerce")

    def clean_and_coerce_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Enforces proper datatypes without silently masking invalid values.
        Unparseable dates or corrupted vital keys are isolated into quarantine.
        """
        df_clean = df.copy()
        quarantined = []

        # 1. Date Parsing
        if "Order_Date" in df_clean.columns:
            # Parse mixed date formats safely
            parsed_dates = pd.to_datetime(df_clean["Order_Date"], errors="coerce", format="mixed")
            missing_or_unparseable = parsed_dates.isna()
            current_time = pd.Timestamp.now()
            future_dates = (parsed_dates > current_time)

            invalid_date_mask = missing_or_unparseable | future_dates
            unparseable_count = int(invalid_date_mask.sum())

            if unparseable_count > 0:
                quarantine_dates = df_clean[invalid_date_mask].copy()
                quarantine_dates["Quarantine_Reason"] = np.where(
                    future_dates[invalid_date_mask],
                    "Future transaction date exceeding system timestamp",
                    "Unparseable or missing Order_Date"
                )
                quarantined.append(quarantine_dates)
                df_clean = df_clean[~invalid_date_mask].copy()
                parsed_dates = parsed_dates[~invalid_date_mask]
                self.log_action("Date Parsing & Quarantine", "Order_Date",
                                "Isolated records with missing, invalid, or future dates", unparseable_count)

            df_clean["Order_Date"] = parsed_dates

        # 2. Numeric Field Cleaning
        numeric_cols = ["Quantity", "Unit_Price", "Sales", "Discount", "Cost", "Profit"]
        for col in numeric_cols:
            if col in df_clean.columns:
                raw_notna = df_clean[col].notna().sum()
                cleaned_num = self.parse_numeric_strings(df_clean[col])
                coerced_nan_count = int((df_clean[col].notna() & cleaned_num.isna()).sum())

                if coerced_nan_count > 0:
                    self.log_action("Type Coercion", col,
                                    f"Parsed string numbers; {coerced_nan_count} non-numeric values converted to NaN",
                                    coerced_nan_count)

                df_clean[col] = cleaned_num

        # 3. Discount Scale Normalization
        # Detect if discount values are given as fraction (<= 1.0) or percentage (1 to 100)
        if "Discount" in df_clean.columns:
            disc_valid = df_clean["Discount"].dropna()
            # If values exist and max is <= 1.0 (e.g. 0.05, 0.15, 0.20), scale to standard 0-100%
            fractional_mask = df_clean["Discount"].notna() & (df_clean["Discount"] > 0) & (df_clean["Discount"] <= 1.0)
            frac_count = int(fractional_mask.sum())
            if frac_count > 0:
                df_clean.loc[fractional_mask, "Discount"] = df_clean.loc[fractional_mask, "Discount"] * 100.0
                self.log_action("Discount Scale Alignment", "Discount",
                                "Normalized decimal fractions (e.g. 0.15) to percentage scale (15.0%)", frac_count)

        quarantine_df = pd.concat(quarantined, ignore_index=True) if quarantined else pd.DataFrame()
        return df_clean, quarantine_df

    def standardize_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trims whitespace, standardizes casing, and resolves regional/categorical synonyms.
        """
        df_clean = df.copy()

        # Whitespace stripping across all string columns
        for col in df_clean.select_dtypes(include=["object", "string"]).columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
            df_clean[col] = df_clean[col].replace(["nan", ""], np.nan)

        # Standardize Category: Title Case
        if "Category" in df_clean.columns:
            mask = df_clean["Category"].notna()
            df_clean.loc[mask, "Category"] = df_clean.loc[mask, "Category"].astype(str).str.strip().str.title()
            self.log_action("Text Normalization", "Category", "Enforced Title Case formatting", int(mask.sum()))

        # Standardize Order_Status: Title Case & canonical status taxonomy mapping
        status_synonym_map = {
            "pending_review": "Processing",
            "pending review": "Processing",
            "pending": "Processing",
            "in transit": "Shipped",
            "returned": "Refunded"
        }
        if "Order_Status" in df_clean.columns:
            mask = df_clean["Order_Status"].notna()
            raw_status = df_clean.loc[mask, "Order_Status"].astype(str).str.strip().str.lower()
            mapped = raw_status.map(status_synonym_map)
            df_clean.loc[mask, "Order_Status"] = mapped.combine_first(
                df_clean.loc[mask, "Order_Status"].astype(str).str.strip().str.title()
            )
            self.log_action("Text Normalization", "Order_Status", "Enforced Title Case and lifecycle status taxonomy", int(mask.sum()))

        # Standardize Payment_Method: Title Case
        if "Payment_Method" in df_clean.columns:
            mask = df_clean["Payment_Method"].notna()
            df_clean.loc[mask, "Payment_Method"] = df_clean.loc[mask, "Payment_Method"].astype(str).str.strip().str.title()
            self.log_action("Text Normalization", "Payment_Method", "Enforced Title Case formatting", int(mask.sum()))

        # Standardize City using canonical lookup map
        if "City" in df_clean.columns:
            city_mask = df_clean["City"].notna()
            raw_cities = df_clean.loc[city_mask, "City"].astype(str).str.strip().str.lower()
            mapped_cities = raw_cities.map(self.CITY_STANDARDIZATION_MAP)
            replaced_mask = mapped_cities.notna() & (mapped_cities != df_clean.loc[city_mask, "City"])
            replaced_count = int(replaced_mask.sum())
            df_clean.loc[city_mask, "City"] = mapped_cities.combine_first(
                df_clean.loc[city_mask, "City"].astype(str).str.strip().str.title()
            )
            self.log_action("City Standardization", "City",
                            "Mapped city abbreviations and colloquialisms to official names", replaced_count)

        # Standardize IDs: Uppercase
        for id_col in ["Order_ID", "Customer_ID"]:
            if id_col in df_clean.columns:
                mask = df_clean[id_col].notna()
                df_clean.loc[mask, id_col] = df_clean.loc[mask, id_col].astype(str).str.strip().str.upper()

        return df_clean

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Statistically and domain-grounded missing value treatment:
        - Category: Imputed via deterministic Product -> Category lookup.
        - Discount: Missing discounts represent standard 0% (no promotion applied).
        - Payment_Method: Mode imputation ('Credit Card') or 'Unknown'.
        - City: Mode by Region when available, else 'Unknown'.
        - Cost: Deterministically reconstructed via Unit_Price and Category Margin.
        - Customer_Name: Retain missingness indicator or set to 'Guest Customer'.
        - Order_Status: Imputed with mode ('Delivered').
        """
        df_clean = df.copy()

        # 1. Category Imputation via Product mapping
        if "Category" in df_clean.columns and "Product" in df_clean.columns:
            df_clean["Category"] = df_clean["Category"].astype(object)
            cat_missing_mask = df_clean["Category"].isna() & df_clean["Product"].notna()
            imputed_cats = df_clean.loc[cat_missing_mask, "Product"].map(self.PRODUCT_CATEGORY_MAP)
            valid_imputes = imputed_cats.notna()
            count = int(valid_imputes.sum())
            if count > 0:
                df_clean.loc[cat_missing_mask & valid_imputes, "Category"] = imputed_cats[valid_imputes]
                self.log_action("Domain Imputation", "Category",
                                "Imputed missing categories via deterministic product catalog lookup", count)

        # 2. Discount Imputation: Missing = 0.0%
        if "Discount" in df_clean.columns:
            disc_missing_mask = df_clean["Discount"].isna()
            count = int(disc_missing_mask.sum())
            if count > 0:
                df_clean["Discount"] = df_clean["Discount"].fillna(0.0)
                self.log_action("Domain Imputation", "Discount",
                                "Imputed missing discount rates with 0.0% (standard non-discounted pricing)", count)

        # 3. Cost Imputation: Reconstruct via Unit_Price and typical Category Margin
        if "Cost" in df_clean.columns and "Unit_Price" in df_clean.columns:
            cost_missing_mask = df_clean["Cost"].isna() & df_clean["Unit_Price"].notna()
            count = int(cost_missing_mask.sum())
            if count > 0:
                # Default margin is 0.40 if category not mapped
                margins = df_clean["Category"].map(self.CATEGORY_MARGIN_MAP).fillna(0.40)
                est_cost_per_unit = df_clean["Unit_Price"] * (1.0 - margins)
                qty = df_clean["Quantity"].abs().fillna(1.0)
                reconstructed_cost = round(est_cost_per_unit * qty, 2)
                df_clean.loc[cost_missing_mask, "Cost"] = reconstructed_cost[cost_missing_mask]
                df_clean["Cost_Imputed"] = cost_missing_mask
                self.log_action("Statistical Cost Reconstruction", "Cost",
                                "Reconstructed missing order costs using unit prices and category margin baselines", count)

        # 4. Payment_Method: Mode imputation
        if "Payment_Method" in df_clean.columns:
            pay_missing_mask = df_clean["Payment_Method"].isna()
            count = int(pay_missing_mask.sum())
            if count > 0:
                mode_val = df_clean["Payment_Method"].mode()[0] if not df_clean["Payment_Method"].dropna().empty else "Credit Card"
                df_clean["Payment_Method"] = df_clean["Payment_Method"].fillna(mode_val)
                self.log_action("Mode Imputation", "Payment_Method",
                                f"Imputed missing payment channels with mode ('{mode_val}')", count)

        # 5. City: Imputed with modal city of the respective Region
        if "City" in df_clean.columns and "Region" in df_clean.columns:
            city_missing_mask = df_clean["City"].isna()
            count = int(city_missing_mask.sum())
            if count > 0:
                region_city_mode = df_clean.groupby("Region")["City"].apply(
                    lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"
                ).to_dict()
                imputed_cities = df_clean.loc[city_missing_mask, "Region"].map(region_city_mode).fillna("Unknown")
                df_clean.loc[city_missing_mask, "City"] = imputed_cities
                self.log_action("Hierarchical Imputation", "City",
                                "Imputed missing city records using regional modal metropolitan hubs", count)

        # 6. Customer_Name: Fill with 'Unknown Customer' without fabricating personal data
        if "Customer_Name" in df_clean.columns:
            cname_missing_mask = df_clean["Customer_Name"].isna()
            count = int(cname_missing_mask.sum())
            if count > 0:
                df_clean["Customer_Name"] = df_clean["Customer_Name"].fillna("Unknown Customer")
                self.log_action("Audit Labeling", "Customer_Name",
                                "Labeled missing customer names as 'Unknown Customer' to avoid fabricating PII", count)

        # 7. Order_Status: Imputed with mode ('Delivered')
        if "Order_Status" in df_clean.columns:
            status_missing_mask = df_clean["Order_Status"].isna()
            count = int(status_missing_mask.sum())
            if count > 0:
                mode_status = df_clean["Order_Status"].mode()[0] if not df_clean["Order_Status"].dropna().empty else "Delivered"
                df_clean["Order_Status"] = df_clean["Order_Status"].fillna(mode_status)
                self.log_action("Mode Imputation", "Order_Status",
                                f"Imputed unassigned order statuses with mode ('{mode_status}')", count)

        return df_clean

    def detect_and_treat_outliers(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
        method: str = "iqr",
        iqr_factor: float = 3.0,
        treatment: str = "cap"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detects and treats extreme outliers using IQR or Z-Score methodology.
        Outliers are capped (Winsorized) at boundary thresholds to preserve sample size
        while mitigating extreme skew in downstream analytics.
        """
        df_clean = df.copy()
        outlier_summary = {}
        if numeric_cols is None:
            numeric_cols = ["Quantity", "Unit_Price"]

        for col in numeric_cols:
            if col not in df_clean.columns:
                continue

            series = df_clean[col].dropna()
            # Focus on absolute values or positive ranges
            val_series = series[series > 0]
            if len(val_series) < 10:
                continue

            if method == "iqr":
                q1 = val_series.quantile(0.25)
                q3 = val_series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = max(0.0, q1 - iqr_factor * iqr)
                upper_bound = q3 + iqr_factor * iqr
            else:  # z-score
                mean = val_series.mean()
                std = val_series.std()
                lower_bound = max(0.0, mean - 3 * std)
                upper_bound = mean + 3 * std

            upper_outliers_mask = df_clean[col] > upper_bound
            outlier_count = int(upper_outliers_mask.sum())

            outlier_summary[col] = {
                "method": method,
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "outliers_detected": outlier_count
            }

            if outlier_count > 0 and treatment == "cap":
                df_clean[f"Is_Outlier_{col}"] = upper_outliers_mask
                df_clean.loc[upper_outliers_mask, col] = round(upper_bound, 2)
                self.log_action(
                    f"Outlier Treatment ({treatment.capitalize()})", col,
                    f"Capped {outlier_count} extreme outlier values at conservative threshold ({upper_bound:.2f})",
                    outlier_count
                )

        return df_clean, outlier_summary

    def enforce_business_rules(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Enforces domain invariants:
        - Identifies sales returns (Quantity < 0) and tags with Transaction_Type='Return' instead of deletion.
        - Quarantines unrecoverable records with Zero or Non-Positive Unit Price.
        - Validates discount constraints [0.0, 100.0].
        """
        df_clean = df.copy()
        quarantined = []

        # 1. Sales Returns Policy
        if "Quantity" in df_clean.columns:
            df_clean["Transaction_Type"] = "Sale"
            return_mask = df_clean["Quantity"] < 0
            return_count = int(return_mask.sum())
            if return_count > 0:
                df_clean.loc[return_mask, "Transaction_Type"] = "Return"
                # For reporting and downstream calculation, store absolute quantity but retain Return flag
                self.log_action("Business Rule Enforcement", "Quantity",
                                f"Identified {return_count} product return transactions; tagged Transaction_Type='Return'",
                                return_count)

            # Check for non-integer corrupted quantities
            corrupted_qty_mask = df_clean["Quantity"].isna() | (df_clean["Quantity"] == 0)
            corrupted_count = int(corrupted_qty_mask.sum())
            if corrupted_count > 0:
                q_records = df_clean[corrupted_qty_mask].copy()
                q_records["Quarantine_Reason"] = "Zero or unparseable Quantity"
                quarantined.append(q_records)
                df_clean = df_clean[~corrupted_qty_mask].copy()

        # 2. Non-negative Unit Price
        if "Unit_Price" in df_clean.columns:
            invalid_price_mask = df_clean["Unit_Price"].isna() | (df_clean["Unit_Price"] <= 0)
            invalid_count = int(invalid_price_mask.sum())
            if invalid_count > 0:
                q_prices = df_clean[invalid_price_mask].copy()
                q_prices["Quarantine_Reason"] = "Zero or negative Unit_Price"
                quarantined.append(q_prices)
                df_clean = df_clean[~invalid_price_mask].copy()
                self.log_action("Business Rule Quarantine", "Unit_Price",
                                "Quarantined records with invalid non-positive prices", invalid_count)

        # 3. Discount Boundary Constraints
        if "Discount" in df_clean.columns:
            invalid_disc_mask = (df_clean["Discount"] < 0) | (df_clean["Discount"] > 100)
            invalid_disc_count = int(invalid_disc_mask.sum())
            if invalid_disc_count > 0:
                df_clean.loc[df_clean["Discount"] < 0, "Discount"] = 0.0
                df_clean.loc[df_clean["Discount"] > 100, "Discount"] = 100.0
                self.log_action("Discount Constraint Clamping", "Discount",
                                "Clamped out-of-bound discounts to valid range [0, 100]", invalid_disc_count)

        quarantine_df = pd.concat(quarantined, ignore_index=True) if quarantined else pd.DataFrame()
        return df_clean, quarantine_df

    def clean_pipeline(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
        """
        Executes end-to-end cleaning pipeline and returns (clean_df, quarantined_df, audit_log).
        """
        logger.info(f"Starting data cleaning pipeline on {len(df_raw):,} raw records...")
        self.cleaning_audit_log.clear()
        self.quarantined_records.clear()

        # 1. Remove empty rows
        step1 = self.remove_empty_rows(df_raw)

        # 2. Normalize string placeholders
        step2 = self.normalize_placeholders(step1)

        # 3. Handle exact duplicates and primary key conflicts
        step3, quaran_dups = self.handle_duplicates(step2, id_col="Order_ID")

        # 4. Clean and coerce types
        step4, quaran_types = self.clean_and_coerce_types(step3)

        # 5. Standardize text and categorical fields
        step5 = self.standardize_text_fields(step4)

        # 6. Statistical and domain missing value imputation
        step6 = self.handle_missing_values(step5)

        # 7. Outlier detection and Winsorizing
        step7, outlier_meta = self.detect_and_treat_outliers(step6, numeric_cols=["Quantity", "Unit_Price"])

        # 8. Business rule enforcement
        clean_df, quaran_rules = self.enforce_business_rules(step7)

        # Consolidate all quarantined records
        all_quarantine = []
        if not quaran_dups.empty:
            all_quarantine.append(quaran_dups)
        if not quaran_types.empty:
            all_quarantine.append(quaran_types)
        if not quaran_rules.empty:
            all_quarantine.append(quaran_rules)

        final_quarantine = pd.concat(all_quarantine, ignore_index=True) if all_quarantine else pd.DataFrame()

        logger.info(
            f"Cleaning completed: {len(clean_df):,} clean records retained, "
            f"{len(final_quarantine):,} records quarantined, "
            f"{len(self.cleaning_audit_log)} audit events logged."
        )

        return clean_df, final_quarantine, self.cleaning_audit_log
