# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load dataset
df = pd.read_csv("Task_1/Sample - Superstore.csv", encoding="latin1")

print(df.head())

# 3. Check missing values and duplicate records
print("Missing values:")
print(df.isnull().sum())

print("Duplicate rows:", df.duplicated().sum())

# 4. Clean data
# Remove duplicate records
df = df.drop_duplicates()

# Fill missing values in numerical columns 
df["Sales"] = df["Sales"].fillna(0)
df["Quantity"] = df["Quantity"].fillna(0)
df["Discount"] = df["Discount"].fillna(0)

# Remove extra spaces from text columns
df["Customer Name"] = df["Customer Name"].str.strip()
df["Product Name"] = df["Product Name"].str.strip()
df["Category"] = df["Category"].str.strip()

print("Data cleaning completed successfully.")

# 5. Exploratory data analysis
print("Total sales:", df["Sales"].sum())
print("Total profit:", df["Profit"].sum())
print("Total quantity sold:", df["Quantity"].sum())
print("Total customers:", df["Customer ID"].nunique())

# 6. Category performance annlysis
category_sales = df.groupby("Category")["Sales"].sum()

print("Sales by category")
print(category_sales)

# 7. Regional performance analysis
region_sales = df.groupby("Region")["Sales"].sum()

print("Sales by region:")
print(region_sales)


# 8 . Product performance analysis
product_sales = df.groupby("Product Name")["Sales"].sum()

top_products = product_sales.sort_values(ascending=False).head(10)

print("Top 10 products by sales:")
print(top_products)

# 9. Customer performance analysis
customer_sales = df.groupby("Customer Name")["Sales"].sum()

top_customers = customer_sales.sort_values(ascending=False).head(10)

print("Top 10 customers by sales:")
print(top_customers)

# 10. Category-wise sales bar chart
plt.figure()
category_sales.plot(kind="bar")

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")

plt.gcf().canvas.manager.set_window_title("Sales by Category")
plt.savefig("category_sales.png", dpi=300, bbox_inches="tight")
plt.show()

# 11. Sales trend line chart
plt.figure()
df["Order Date"] = pd.to_datetime(df["Order Date"])

monthly_sales = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum()

monthly_sales.plot(kind="line")

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.gcf().canvas.manager.set_window_title("Monthly Sales Trend")
plt.savefig("monthly_sales_trend.png", dpi=300, bbox_inches="tight")
plt.show()

# 12. Sales and profit scatter plot
plt.figure()
plt.scatter(df["Sales"], df["Profit"])

plt.title("Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")

plt.gcf().canvas.manager.set_window_title("Sales vs Profit")
plt.savefig("sales_vs_profit.png", dpi=300, bbox_inches="tight")
plt.show()

# 13. Correlation heatmap
plt.figure()
correlation = df[["Sales", "Quantity", "Discount", "Profit"]].corr()
sns.heatmap(correlation, annot=True)

plt.title("Correlation Heatmap")

plt.gcf().canvas.manager.set_window_title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# 14. Business insights
print("Business Insights:")

print("Technology category has the highest sales.")
print("West region has the highest sales.")
print("South region has the lowest sales.")
print("Monthly sales show fluctuations over time.")
print("Sales reach their highest level toword the end of the dataset.")