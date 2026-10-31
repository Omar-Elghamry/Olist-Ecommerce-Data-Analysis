
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==================================================
# 1. CUSTOMERS DATASET
# ==================================================

print("\n" + "=" * 50)
print("           CUSTOMERS DATA ANALYSIS")
print("=" * 50)

customer = pd.read_csv("olist_customers_dataset.csv")

rows, columns = customer.shape

print("\n--- Dataset Overview ---")
print("Number of Rows:", rows)
print("Number of Columns:", columns)
print("\nFirst 5 Rows:\n", customer.head(5))
print("\nColumn Names:\n", customer.columns)
print("\nData Types:\n", customer.dtypes)

print("\n--- Data Quality ---")
print("Missing Values:\n", customer.isnull().sum())
print("Duplicate Rows:", customer.duplicated().sum())

print("\n--- Customer Analysis ---")
print("Unique Customers:", customer["customer_unique_id"].nunique())

print("\nNumber of Customers by City:")
print(customer.groupby("customer_city")["customer_id"].count())

print("\nTop 5 Cities by Number of Customers:")
print(customer.groupby("customer_city")["customer_id"].count().nlargest(5))


# ==================================================
# 2. ORDERS DATASET
# ==================================================

print("\n" + "=" * 50)
print("             ORDERS DATA ANALYSIS")
print("=" * 50)

order = pd.read_csv("olist_orders_dataset.csv")

order["order_purchase_timestamp"] = pd.to_datetime(
    order["order_purchase_timestamp"]
)

rows, columns = order.shape

print("\n--- Dataset Overview ---")
print("Number of Rows:", rows)
print("Number of Columns:", columns)
print("\nFirst 5 Rows:\n", order.head(5))
print("\nColumn Names:\n", order.columns)
print("\nData Types:\n", order.dtypes)

print("\n--- Data Quality ---")
print("Missing Values:\n", order.isnull().sum())
print("Duplicate Rows:", order.duplicated().sum())

print("\n--- Orders Analysis ---")
print("Unique Orders:", order["order_id"].nunique())

print("\nNumber of Orders by Status:")
print(order.groupby("order_status")["order_id"].count())

print("\nTop 5 Order Statuses:")
print(order.groupby("order_status")["order_id"].count().nlargest(5))

print("\nFirst Order Date:", order["order_purchase_timestamp"].min())
print("Last Order Date:", order["order_purchase_timestamp"].max())

print("\nMonthly Number of Orders:")
print(
    order.groupby(
        order["order_purchase_timestamp"].dt.to_period("M")
    )["order_id"].count()
)


# ==================================================
# 3. ORDER ITEMS DATASET
# ==================================================

print("\n" + "=" * 50)
print("           ORDER ITEMS ANALYSIS")
print("=" * 50)

item = pd.read_csv("olist_order_items_dataset.csv")

print("\n--- Revenue Analysis ---")

total_revenue = item["price"].sum()
print("Total Product Revenue:", total_revenue)

print("\nTop 5 Best-Selling Products by Items Sold:")
print(
    item.groupby("product_id")["order_item_id"]
    .count()
    .nlargest(5)
)

print("\n--- Monthly Revenue Analysis ---")

merged_orders_items = pd.merge(
    item,
    order,
    on="order_id",
    how="inner"
)

monthly_revenue = merged_orders_items.groupby(
    merged_orders_items["order_purchase_timestamp"].dt.to_period("M")
)["price"].sum()

print("Monthly Product Revenue:")
print(monthly_revenue)

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_revenue.index.to_timestamp(),
    monthly_revenue.values,
    marker=".",
    markersize=10
)

plt.title("Monthly Product Revenue")
plt.xlabel("Month & Year")
plt.ylabel("Revenue (BRL)")
plt.grid(True)
plt.tight_layout()
plt.show()


# # ==================================================
# # 4. PAYMENTS DATASET
# # ==================================================

print("\n" + "=" * 50)
print("            PAYMENTS DATA ANALYSIS")
print("=" * 50)

payment = pd.read_csv("olist_order_payments_dataset.csv")

rows, columns = payment.shape

print("\n--- Dataset Overview ---")
print("Number of Rows:", rows)
print("Number of Columns:", columns)
print("\nFirst 5 Rows:\n", payment.head(5))
print("\nData Types:\n", payment.dtypes)

print("\n--- Data Quality ---")
print("Missing Values:\n", payment.isnull().sum())
print("Duplicate Rows:", payment.duplicated().sum())

print("\n--- Payment Analysis ---")

print("Number of Payment Records by Payment Type:")
print(payment.groupby("payment_type")["order_id"].count())

print("\nTotal Payment Value:", payment["payment_value"].sum())


# ==================================================
# 5. PRODUCTS DATASET
# ==================================================

print("\n" + "=" * 50)
print("            PRODUCTS DATA ANALYSIS")
print("=" * 50)

product = pd.read_csv("olist_products_dataset.csv")

rows, columns = product.shape

print("\n--- Dataset Overview ---")
print("Number of Rows:", rows)
print("Number of Columns:", columns)
print("\nFirst 5 Rows:\n", product.head(5))
print("\nData Types:\n", product.dtypes)

print("\n--- Data Quality ---")
print("Missing Values:\n", product.isnull().sum())
print("Duplicate Rows:", product.duplicated().sum())

print("\n--- Product Analysis ---")

print("Unique Products:", product["product_id"].nunique())

print("\nTop 5 Categories by Number of Products:")
print(
    product.groupby("product_category_name")["product_id"]
    .count()
    .nlargest(5)
)


# ==================================================
# 6. REVIEWS DATASET
# ==================================================

print("\n" + "=" * 50)
print("             REVIEWS DATA ANALYSIS")
print("=" * 50)

reviews = pd.read_csv("olist_order_reviews_dataset.csv")

rows, columns = reviews.shape

print("\n--- Dataset Overview ---")
print("Number of Rows:", rows)
print("Number of Columns:", columns)
print("\nFirst 5 Rows:\n", reviews.head(5))
print("\nData Types:\n", reviews.dtypes)

print("\n--- Data Quality ---")
print("Missing Values:\n", reviews.isnull().sum())
print("Duplicate Rows:", reviews.duplicated().sum())

print("\n--- Customer Satisfaction Analysis ---")

print("Average Review Score:", reviews["review_score"].mean())

print("\nNumber of Reviews by Score:")
print(reviews.groupby("review_score")["review_id"].count())


# ==================================================
# 7. SELLERS DATASET
# ==================================================

print("\n" + "=" * 50)
print("             SELLERS DATA ANALYSIS")
print("=" * 50)

seller = pd.read_csv("olist_sellers_dataset.csv")

rows, columns = seller.shape

print("\n--- Dataset Overview ---")
print("Number of Rows:", rows)
print("Number of Columns:", columns)
print("\nFirst 5 Rows:\n", seller.head(5))
print("\nData Types:\n", seller.dtypes)

print("\n--- Data Quality ---")
print("Missing Values:\n", seller.isnull().sum())
print("Duplicate Rows:", seller.duplicated().sum())

print("\n--- Seller Analysis ---")

print("Unique Sellers:", seller["seller_id"].nunique())

print("\nTop 5 States by Number of Sellers:")
print(
    seller.groupby("seller_state")["seller_id"]
    .count()
    .nlargest(5)
)


# ==================================================
# 8. GEOLOCATION DATASET
# ==================================================

print("\n" + "=" * 50)
print("           GEOLOCATION DATA ANALYSIS")
print("=" * 50)

geolocation = pd.read_csv("olist_geolocation_dataset.csv")

rows, columns = geolocation.shape

print("\n--- Dataset Overview ---")
print("Number of Rows:", rows)
print("Number of Columns:", columns)
print("\nFirst 5 Rows:\n", geolocation.head(5))
print("\nData Types:\n", geolocation.dtypes)

print("\n--- Data Quality ---")
print("Missing Values:\n", geolocation.isnull().sum())
print("Duplicate Rows:", geolocation.duplicated().sum())

print("\n--- Geolocation Analysis ---")

print(
    "Unique ZIP Code Prefixes:",
    geolocation["geolocation_zip_code_prefix"].nunique()
)


# ==================================================
# 9. CATEGORY TRANSLATION DATASET
# ==================================================

print("\n" + "=" * 50)
print("          CATEGORY TRANSLATION ANALYSIS")
print("=" * 50)

category = pd.read_csv("product_category_name_translation.csv")

rows, columns = category.shape

print("\n--- Dataset Overview ---")
print("Number of Rows:", rows)
print("Number of Columns:", columns)
print("\nFirst 5 Rows:\n", category.head(5))

print("\n--- Data Quality ---")
print("Missing Values:\n", category.isnull().sum())
print("Duplicate Rows:", category.duplicated().sum())

print("\n--- Category Analysis ---")

print(
    "Unique English Categories:",
    category["product_category_name_english"].nunique()
)


# ==================================================
# END OF PHASE 1
# ==================================================

print("\n" + "=" * 50)
print("       PHASE 1: DATA EXPLORATION COMPLETED")
print("=" * 50)



print("=" * 60)
print("DATA CLEANING")
print("=" * 60)


# ============================================================
# 1. ORDERS DATASET
# ============================================================

print("\n========== ORDERS DATASET ==========")

missing = order.isnull().sum()
print("Missing Values:\n", missing[missing > 0])

print("\nDuplicate Order IDs:", order["order_id"].duplicated().sum())

order["order_approved_at"] = pd.to_datetime(order["order_approved_at"])
order["order_delivered_carrier_date"] = pd.to_datetime(order["order_delivered_carrier_date"])
order["order_delivered_customer_date"] = pd.to_datetime(order["order_delivered_customer_date"])
order["order_estimated_delivery_date"] = pd.to_datetime(order["order_estimated_delivery_date"])

print(
    "Delivered Orders With Missing Delivery Date:",
    ((order["order_status"] == "delivered")
     & (order["order_delivered_customer_date"].isnull())).sum()
)

order["missing_delivery"] = order["order_delivered_customer_date"].isnull()
print(order.groupby("order_status")["missing_delivery"].sum())

invalid_date = (order["order_delivered_customer_date"] < order["order_purchase_timestamp"])
print("Invalid Delivery Dates:", invalid_date.sum())

orders_clean = order.copy()
orders_clean.loc[invalid_date, "order_delivered_customer_date"] = pd.NaT


# ============================================================
# 2. ORDER ITEMS DATASET
# ============================================================

print("\n========== ORDER ITEMS DATASET ==========")

missing = item.isnull().sum()
print("Missing Values:\n", missing[missing != 0])

duplicate = item.duplicated()
print("Duplicate Rows:", duplicate.sum())

print("Invalid Prices:", (item["price"] <= 0).sum())
print("Negative Freight Values:", (item["freight_value"] < 0).sum())

items_clean = item.copy()

items_clean.loc[item["price"] <= 0, "price"] = np.nan


# ============================================================
# 3. CUSTOMERS DATASET
# ============================================================

print("\n========== CUSTOMERS DATASET ==========")

missing = customer.isnull().sum()
print("Missing Values:\n", missing[missing > 0])

duplicate = customer.duplicated()
print("Duplicate Rows:", duplicate.sum())

print("Unique Customer IDs:", customer["customer_id"].nunique())

duplicate_id = customer["customer_id"].duplicated()
print("Duplicate Customer IDs:", duplicate_id.sum())

missing_cities = customer["customer_city"]
print("Missing Customer Cities:", missing_cities.isnull().sum())

customer_clean = customer.copy()

customer_clean.loc[missing_cities.isnull(), "customer_city"] = "unknown"


# ============================================================
# 4. PRODUCTS DATASET
# ============================================================

print("\n========== PRODUCTS DATASET ==========")

missing = product.isnull().sum()
print("Missing Values:\n", missing[missing > 0])

print("Duplicate Rows:", product.duplicated().sum())
print("Unique Product IDs:", product["product_id"].nunique())
print("Duplicate Product IDs:", product["product_id"].duplicated().sum())

print("Invalid Product Weights:", (product["product_weight_g"] <= 0).sum())

products_clean = product.copy()

products_clean.loc[products_clean["product_weight_g"] <= 0, "product_weight_g"] = np.nan

products_clean["product_category_name"] = (
    products_clean["product_category_name"].fillna("Unknown")
)


# ============================================================
# 5. PAYMENTS DATASET
# ============================================================

print("\n========== PAYMENTS DATASET ==========")

missing = payment.isnull().sum()
print("Missing Values:\n", missing[missing > 0])

print("Duplicate Rows:", payment.duplicated().sum())

print("Negative Payment Values:", (payment["payment_value"] < 0).sum())
print("Negative Payment Installments:", (payment["payment_installments"] < 0).sum())

payments_clean = payment.copy()

payments_clean.loc[payments_clean["payment_value"] < 0, "payment_value"] = np.nan

payments_clean.loc[payments_clean["payment_installments"] < 0, "payment_installments"] = np.nan


# ============================================================
# 6. SELLERS DATASET
# ============================================================

print("\n========== SELLERS DATASET ==========")

missing = seller.isnull().sum()
print("Missing Values:\n", missing[missing > 0])

print("Duplicate Rows:", seller.duplicated().sum())
print("Duplicate Seller IDs:", seller["seller_id"].duplicated().sum())

sellers_clean = seller.copy()

sellers_clean["seller_city"] = (
    sellers_clean["seller_city"].fillna("Unknown")
)


# ============================================================
# 7. REVIEWS DATASET
# ============================================================

print("\n========== REVIEWS DATASET ==========")

missing = reviews.isnull().sum()
print("Missing Values:\n", missing[missing > 0])

print("Duplicate Rows:", reviews.duplicated().sum())
print("Duplicate Review IDs:", reviews["review_id"].duplicated().sum())

print("Unique Review IDs:", reviews["review_id"].nunique())

reviews["review_answer_timestamp"] = pd.to_datetime(reviews["review_answer_timestamp"])
reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"])

incorrect_scores = (reviews["review_score"] < 1) | (reviews["review_score"] > 5)
print("Invalid Review Scores:", incorrect_scores.sum())

reviews_clean = reviews.copy()

reviews_clean.loc[incorrect_scores, "review_score"] = np.nan


# ============================================================
# 8. GEOLOCATION DATASET
# ============================================================

print("\n========== GEOLOCATION DATASET ==========")

missing = geolocation.isnull().sum()
print("Missing Values:\n", missing[missing > 0])

print("Duplicate Rows:", geolocation.duplicated().sum())
print("Duplicate ZIP Codes:", geolocation["geolocation_zip_code_prefix"].duplicated().sum())
print("Unique ZIP Codes:", geolocation["geolocation_zip_code_prefix"].nunique())

geolocation_clean = geolocation.copy()

geolocation_clean = geolocation_clean.drop_duplicates()

incorrect_latitude = (geolocation_clean["geolocation_lat"] < -90) | (geolocation_clean["geolocation_lat"] > 90)
print("Invalid Latitude Values:", incorrect_latitude.sum())

incorrect_longitude = (geolocation_clean["geolocation_lng"] < -180) | (geolocation_clean["geolocation_lng"] > 180)
print("Invalid Longitude Values:", incorrect_longitude.sum())

geolocation_clean.loc[incorrect_latitude, "geolocation_lat"] = np.nan
geolocation_clean.loc[incorrect_longitude, "geolocation_lng"] = np.nan

print("Remaining Duplicate Rows:", geolocation_clean.duplicated().sum())


# ============================================================
# 9. CATEGORY TRANSLATION DATASET
# ============================================================

print("\n========== CATEGORY TRANSLATION DATASET ==========")

missing=category.isnull().sum()
print("Missing Values:")
print(missing[missing>0])

print("Duplicate Rows:",category.duplicated().sum())

print("Duplicate Categories:",category["product_category_name"].duplicated().sum())

category_clean=category.drop_duplicates().copy()

print("\n" + "=" * 60)
print("DATA CLEANING SECTION FINISHED")
print("=" * 60)


# ============================================================
#  MERGE ORDERS WITH CUSTOMERS
# ============================================================

orders_customers = pd.merge(orders_clean, customer_clean, on="customer_id", how="left")

print("\n========== ORDERS + CUSTOMERS ==========")
print("First 5 Rows:")
print(orders_customers.head(5))

rows, columns = orders_customers.shape
print("Total Rows:", rows)
print("Total Columns:", columns)

missing = orders_customers["customer_unique_id"].isnull().sum()
print("Missing Customer Unique IDs:", missing)

print("Rows Before Merge:", orders_clean.shape[0])
print("Rows After Merge:", orders_customers.shape[0])
print("Are Rows Equal?", orders_clean.shape[0] == orders_customers.shape[0])


# ============================================================
#  MERGE ORDERS WITH ORDER ITEMS
# ============================================================

orders_items = pd.merge(orders_customers, items_clean, on="order_id", how="left")

print("\n========== ORDERS + ORDER ITEMS ==========")
print("First 5 Rows:")
print(orders_items.head(5))

rows, columns = orders_items.shape
print("Total Rows:", rows)
print("Total Columns:", columns)

print("Unique Orders:", orders_items["order_id"].nunique())

missing = orders_items["product_id"].isnull().sum()
print("Missing Product IDs:", missing)

print("Rows Before Merge:", orders_customers.shape[0])
print("Rows After Merge:", orders_items.shape[0])
print("Are Rows Equal?", orders_items.shape[0] == orders_customers.shape[0])

print("\nTotal Product Price by Order:")
print(orders_items.groupby("order_id")["price"].sum())


# ============================================================
# MERGE ORDER ITEMS WITH PRODUCTS
# ============================================================

orders_products = pd.merge(orders_items, products_clean, on="product_id", how="left")

print("\n========== ORDERS + PRODUCTS ==========")
print("First 5 Rows:")
print(orders_products.head(5))

rows, columns = orders_products.shape
print("Total Rows:", rows)
print("Total Columns:", columns)

print("Unique Products:", orders_products["product_id"].nunique())

missing = orders_products["product_category_name"].isnull().sum()
print("Missing Product Categories:", missing)

print("Rows Before Merge:", orders_items.shape[0])
print("Rows After Merge:", orders_products.shape[0])
print("Are Rows Equal?", orders_items.shape[0] == orders_products.shape[0])

print("\nTotal Product Price by Category:")
print(orders_products.groupby("product_category_name")["price"].sum())


# ============================================================
#  MERGE WITH SELLERS
# ============================================================

orders_sellers = pd.merge(orders_products, sellers_clean, on="seller_id", how="left")

print("\n========== ORDERS + SELLERS ==========")
print("First 5 Rows:")
print(orders_sellers.head(5))

rows, columns = orders_sellers.shape
print("Total Rows:", rows)
print("Total Columns:", columns)

print("Unique Sellers:", orders_sellers["seller_id"].nunique())

missing = orders_sellers["seller_city"].isnull().sum()
print("Missing Seller Cities:", missing)

print("Rows Before Merge:", orders_products.shape[0])
print("Rows After Merge:", orders_sellers.shape[0])
print("Are Rows Equal?", orders_products.shape[0] == orders_sellers.shape[0])

print("\nTotal Product Price by Seller City:")
print(orders_sellers.groupby("seller_city")["price"].sum())


# ============================================================
#  MERGE WITH CATEGORY TRANSLATION
# ============================================================

orders_translated = pd.merge(
    orders_sellers, category_clean, on="product_category_name", how="left"
)

print("\n========== ORDERS + CATEGORY TRANSLATION ==========")
print("First 5 Rows:")
print(orders_translated.head(5))

rows, columns = orders_translated.shape
print("Total Rows:", rows)
print("Total Columns:", columns)

print("Unique English Categories:", orders_translated["product_category_name_english"].nunique())

missing = orders_translated["product_category_name_english"].isnull().sum()
print("Missing English Category Names:", missing)

print("Rows Before Merge:", orders_sellers.shape[0])
print("Rows After Merge:", orders_translated.shape[0])
print("Are Rows Equal?", orders_sellers.shape[0] == orders_translated.shape[0])

print("\nTotal Product Price by English Category:")
print(orders_translated.groupby("product_category_name_english")["price"].sum())


# ============================================================
#  AGGREGATE PAYMENTS AND MERGE
# ============================================================

payments_grouped = payments_clean.groupby("order_id", as_index=False)["payment_value"].sum()

orders_payments = pd.merge(
    orders_translated, payments_grouped, on="order_id", how="left"
)

print("\n========== ORDERS + PAYMENTS ==========")
print("First 5 Rows:")
print(orders_payments.head(5))

rows, columns = orders_payments.shape
print("Total Rows:", rows)
print("Total Columns:", columns)

missing = orders_payments["payment_value"].isnull().sum()
print("Missing Payments:", missing)

print("Rows Before Merge:", orders_translated.shape[0])
print("Rows After Merge:", orders_payments.shape[0])
print("Are Rows Equal?", orders_translated.shape[0] == orders_payments.shape[0])


# ============================================================
#  AGGREGATE REVIEWS AND MERGE
# ============================================================

reviews_grouped = reviews_clean.groupby("order_id", as_index=False)["review_score"].mean()

orders_reviews = pd.merge(
    orders_payments, reviews_grouped, on="order_id", how="left"
)

print("\n========== ORDERS + REVIEWS ==========")
print("First 5 Rows:")
print(orders_reviews.head(5))

rows, columns = orders_reviews.shape
print("Total Rows:", rows)
print("Total Columns:", columns)

missing = orders_reviews["review_score"].isnull().sum()
print("Missing Reviews:", missing)

print("Rows Before Merge:", orders_payments.shape[0])
print("Rows After Merge:", orders_reviews.shape[0])
print("Are Rows Equal?", orders_payments.shape[0] == orders_reviews.shape[0])

print("\nAverage Review Score by English Category:")
print(orders_reviews.groupby("product_category_name_english")["review_score"].mean())



# ============================================================
# TASK 24: SALES ANALYSIS
# ============================================================

total_revenue = orders_reviews["price"].sum()
print("Total Revenue:", total_revenue)

total_orders = orders_reviews["order_id"].nunique()
print("Total Orders:", total_orders)

print("Total Product Items Sold:", orders_reviews["product_id"].count())

print("Average Order Value (AOV):", total_revenue / total_orders)


# ============================================================
# TASK 25: TOP 10 CATEGORIES BY REVENUE
# ============================================================

category_revenue = orders_reviews.groupby("product_category_name_english")["price"].sum()

sorted_categories = category_revenue.sort_values(ascending=False)

print("\n========== ALL CATEGORIES BY REVENUE ==========")
print(sorted_categories)

print("\n========== TOP 10 CATEGORIES BY REVENUE ==========")
print(sorted_categories.head(10))


plt.figure(figsize=(12,10))
plt.bar(sorted_categories.head(10).index,sorted_categories.head(10).values,width=0.6)
plt.title("TOP 10 CATEGORIES")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.xticks(rotation=45,ha="right")
plt.tight_layout()
plt.show()



