"""
Dataset Generator Script for Data Preprocessing Project
Generates an industry-standard, realistic messy retail sales dataset with 12,500 records.
Clearly labeled as a synthetic benchmark dataset for training and testing data cleaning pipelines.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_messy_retail_dataset(num_records: int = 12500, random_seed: int = 42):
    random.seed(random_seed)
    np.random.seed(random_seed)

    products_by_category = {
        "Technology": [
            ("Dell XPS 15 Laptop", 1299.99, 900.00),
            ("Apple iPhone 15 Pro", 1099.00, 750.00),
            ("Sony WH-1000XM5 Headphones", 349.99, 210.00),
            ("Samsung 34' Curved Monitor", 499.99, 320.00),
            ("Logitech MX Master 3 Mouse", 99.99, 55.00),
            ("Mechanical Gaming Keyboard", 129.50, 70.00),
            ("Anker USB-C Fast Charger", 39.99, 18.00)
        ],
        "Furniture": [
            ("Ergonomic Mesh Office Chair", 289.00, 160.00),
            ("Motorized Standing Desk", 450.00, 260.00),
            ("Solid Oak Coffee Table", 199.50, 110.00),
            ("Bookshelf 5-Tier Wooden", 149.00, 80.00),
            ("Modern Fabric Recliner", 550.00, 310.00)
        ],
        "Office Supplies": [
            ("Heavy-Duty File Folders (Box 100)", 24.99, 11.00),
            ("Gel Ink Pen Pack (24ct)", 18.50, 7.50),
            ("Stapler with 5000 Staples", 15.99, 6.00),
            ("Desktop Document Organizer", 29.99, 12.50),
            ("Whiteboard Magnetic 36x24", 45.00, 22.00)
        ],
        "Clothing": [
            ("Men's Tailored Wool Blazer", 189.00, 95.00),
            ("Women's Cashmere Sweater", 145.00, 70.00),
            ("All-Weather Waterproof Jacket", 120.00, 58.00),
            ("Athletic Running Shoes", 85.00, 42.00),
            ("Breathable Cotton Crew Socks (6pk)", 19.99, 8.00)
        ],
        "Home & Kitchen": [
            ("Cold Brew Coffee Maker 1L", 34.99, 16.00),
            ("Cast Iron Dutch Oven 6-Quart", 79.99, 38.00),
            ("Stainless Steel Chef Knife Set", 69.99, 32.00),
            ("Smart Electric Kettle 1.7L", 59.99, 28.00),
            ("HEPA Air Purifier Compact", 119.99, 65.00)
        ]
    }

    # Flatten product lookup for mapping
    product_category_map = {}
    for cat, prod_list in products_by_category.items():
        for p_name, _, _ in prod_list:
            product_category_map[p_name] = cat

    regions_cities = {
        "East": ["New York", "Boston", "Philadelphia"],
        "West": ["Los Angeles", "San Francisco", "Seattle"],
        "Central": ["Chicago", "Houston", "Dallas"],
        "South": ["Atlanta", "Miami", "Charlotte"]
    }

    city_region_map = {}
    all_cities = []
    for reg, cities in regions_cities.items():
        for c in cities:
            city_region_map[c] = reg
            all_cities.append(c)

    payment_methods = ["Credit Card", "Debit Card", "PayPal", "Bank Transfer", "Cash on Delivery"]
    order_statuses = ["Delivered", "Shipped", "Processing", "Cancelled", "Refunded"]

    first_names = ["James", "Emma", "Liam", "Olivia", "Noah", "Sophia", "Oliver", "Ava", "William", "Isabella",
                   "Lucas", "Mia", "Henry", "Evelyn", "Alexander", "Harper", "Sebastian", "Camila", "Michael", "Gianna",
                   "Daniel", "Abigail", "Matthew", "Emily", "Aiden", "Ella", "David", "Elizabeth", "Joseph", "Sofia",
                   "Priya", "Arun", "Rohan", "Ananya", "Kavya", "Vikram", "Rajesh", "Meera", "Aditya", "Sneha"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
                  "Patel", "Sharma", "Kumar", "Singh", "Reddy", "Wilson", "Anderson", "Taylor", "Thomas", "Moore"]

    # Generate pool of ~1800 unique customers for realistic repeat transactions
    num_customers = 1800
    customer_pool = []
    for i in range(1, num_customers + 1):
        cid = f"CUST-{i:05d}"
        cname = f"{random.choice(first_names)} {random.choice(last_names)}"
        customer_pool.append((cid, cname))

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 3, 31)
    date_range_days = (end_date - start_date).days

    rows = []
    order_counter = 10001

    for _ in range(num_records):
        order_id = f"ORD-{order_counter}"
        order_counter += 1

        # Customer selection (simulating frequent buyers)
        cust_id, cust_name = random.choice(customer_pool)

        # Date generation with diverse real-world formatting
        rand_days = random.randint(0, date_range_days)
        rand_dt = start_date + timedelta(days=rand_days, hours=random.randint(8, 22), minutes=random.randint(0, 59))

        date_style = random.random()
        if date_style < 0.45:
            order_date_str = rand_dt.strftime("%Y-%m-%d")
        elif date_style < 0.70:
            order_date_str = rand_dt.strftime("%d/%m/%Y")
        elif date_style < 0.85:
            order_date_str = rand_dt.strftime("%m-%d-%Y")
        elif date_style < 0.97:
            order_date_str = rand_dt.strftime("%Y/%m/%d %H:%M:%S")
        else:
            # Corrupted / invalid date strings (~3%)
            order_date_str = random.choice(["invalid_date", "2024-02-31", "2025-13-45", "UNKNOWN", ""])

        # Category and Product
        category = random.choice(list(products_by_category.keys()))
        prod_tuple = random.choice(products_by_category[category])
        product_name = prod_tuple[0]
        base_unit_price = prod_tuple[1]
        base_cost = prod_tuple[2]

        # Location
        city = random.choice(all_cities)
        region = city_region_map[city]

        # Quantity
        q_rand = random.random()
        if q_rand < 0.85:
            quantity = random.randint(1, 8)
        elif q_rand < 0.95:
            quantity = random.randint(9, 20)
        elif q_rand < 0.98:
            # Sales returns represented as negative quantity (~3%)
            quantity = -random.randint(1, 3)
        else:
            # Outlier quantity (~2%)
            quantity = random.randint(150, 600)

        # Unit Price with potential outlier or string formatting
        unit_price = base_unit_price
        if random.random() < 0.015:
            # High price outlier
            unit_price = round(base_unit_price * random.uniform(15, 40), 2)

        # Discount (mix of percentage [0-30] and decimal [0.0 - 0.30])
        disc_choice = random.random()
        if disc_choice < 0.30:
            discount_val = 0.0
            discount_str = "0%"
        elif disc_choice < 0.65:
            disc_pct = random.choice([5, 10, 15, 20, 25, 30])
            discount_val = float(disc_pct)
            discount_str = f"{disc_pct}%" if random.random() < 0.5 else str(disc_pct)
        elif disc_choice < 0.93:
            # Decimal fraction representation
            disc_pct = random.choice([0.05, 0.10, 0.15, 0.20, 0.25])
            discount_val = disc_pct * 100
            discount_str = str(disc_pct)
        else:
            # Missing discount
            discount_val = np.nan
            discount_str = random.choice(["", "N/A", "null", "none", "?"])

        # Financials
        effective_disc_pct = discount_val if pd.notna(discount_val) else 0.0
        gross_sales = round(abs(quantity) * unit_price, 2)
        discount_amt = round(gross_sales * (effective_disc_pct / 100.0), 2)
        net_sales = round(gross_sales - discount_amt, 2)
        cost_total = round(abs(quantity) * base_cost, 2)
        profit = round(net_sales - cost_total, 2) if quantity > 0 else -round(gross_sales * 0.15, 2)

        # Format messy string values for numeric columns
        sales_str = str(net_sales)
        if random.random() < 0.20:
            sales_str = f"${net_sales:,.2f}"
        elif random.random() < 0.10:
            sales_str = f"  {net_sales} "

        price_str = str(unit_price)
        if random.random() < 0.15:
            price_str = f"${unit_price:.2f}"
        elif random.random() < 0.08:
            price_str = f" {unit_price}  "

        cost_str = str(cost_total)
        if random.random() < 0.05:
            # Missing cost in raw data
            cost_str = random.choice(["", "MISSING", "N/A", "null"])

        # Formatting quantity
        qty_str = str(quantity)
        if random.random() < 0.08:
            qty_str = f"{float(quantity):.1f}"
        elif random.random() < 0.05:
            qty_str = f"  {quantity}  "
        elif random.random() < 0.005:
            qty_str = "corrupted_qty"

        # Categorical text variations & messy casing
        cat_str = category
        if random.random() < 0.15:
            cat_str = category.lower()
        elif random.random() < 0.10:
            cat_str = category.upper()
        elif random.random() < 0.08:
            cat_str = f"  {category}  "
        elif random.random() < 0.03:
            cat_str = random.choice(["", "N/A", "Unknown", "none"])

        city_str = city
        if city == "New York" and random.random() < 0.25:
            city_str = random.choice(["new york", "NYC", "New York City", "  New York  "])
        elif city == "Los Angeles" and random.random() < 0.25:
            city_str = random.choice(["los angeles", "LA", "Los Angeles  ", "LOS ANGELES"])
        elif city == "San Francisco" and random.random() < 0.25:
            city_str = random.choice(["san francisco", "SF", "San Fran"])
        elif random.random() < 0.03:
            city_str = random.choice(["", "MISSING", "null", "  "])

        status = random.choice(order_statuses)
        status_str = status
        if random.random() < 0.15:
            status_str = status.lower()
        elif random.random() < 0.08:
            status_str = status.upper()
        elif random.random() < 0.02:
            status_str = random.choice(["", "N/A", "Pending_Review"])

        pay_method = random.choice(payment_methods)
        pay_str = pay_method
        if random.random() < 0.04:
            pay_str = random.choice(["", "N/A", "MISSING", "null"])

        cname_str = cust_name
        if random.random() < 0.035:
            cname_str = random.choice(["", "   ", "N/A", "?", "UNKNOWN"])

        rows.append({
            "Order_ID": order_id,
            "Order_Date": order_date_str,
            "Customer_ID": cust_id,
            "Customer_Name": cname_str,
            "Product": product_name,
            "Category": cat_str,
            "Quantity": qty_str,
            "Unit_Price": price_str,
            "Sales": sales_str,
            "Discount": discount_str,
            "Cost": cost_str,
            "Profit": profit,
            "City": city_str,
            "Region": region,
            "Payment_Method": pay_str,
            "Order_Status": status_str
        })

    df = pd.DataFrame(rows)

    # 1. Inject exact duplicate rows (~1.5% -> 180 rows)
    num_exact_dups = 180
    dup_indices = np.random.choice(df.index, size=num_exact_dups, replace=False)
    dup_rows = df.loc[dup_indices].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 2. Inject duplicate business keys (conflicting Order_IDs, ~35 rows)
    num_conflicts = 35
    conflict_indices = np.random.choice(range(500, len(df) - 200), size=num_conflicts, replace=False)
    for idx in conflict_indices:
        target_oid = df.at[idx, "Order_ID"]
        # Create a conflicting duplicate record with same Order_ID but altered customer/sales
        conflict_row = df.loc[idx].copy()
        conflict_row["Customer_ID"] = f"CUST-{random.randint(9000, 9999)}"
        conflict_row["Sales"] = "$9,999.99"
        conflict_row["Quantity"] = "99"
        df.loc[len(df)] = conflict_row

    # 3. Inject empty rows (all nulls or whitespace, ~10 rows)
    for _ in range(10):
        empty_row = {col: random.choice(["", "  ", None, np.nan]) for col in df.columns}
        df.loc[len(df)] = empty_row

    # Shuffle the dataset so anomalies are distributed throughout
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    return df

def main():
    os.makedirs("data/raw", exist_ok=True)
    csv_path = "data/raw/retail_sales_raw.csv"
    xlsx_path = "data/raw/retail_sales_raw.xlsx"

    print("Generating synthetic benchmark retail dataset (>= 10,000 rows)...")
    df = generate_messy_retail_dataset(num_records=12200, random_seed=42)
    print(f"Dataset generated. Shape: {df.shape}")

    # Save to CSV
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Saved CSV to: {csv_path}")

    # Save sample / full to Excel for multi-format ingestion testing (first 2500 rows for Excel performance)
    df.iloc[:2500].to_excel(xlsx_path, index=False, engine="openpyxl")
    print(f"Saved Excel format sample to: {xlsx_path}")

if __name__ == "__main__":
    main()
