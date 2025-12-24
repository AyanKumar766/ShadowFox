import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "Sample_Superstore.csv"

# LOAD DATA

df = pd.read_csv(DATA_PATH, encoding="latin1")

print("\n[INFO] Dataset Loaded Successfully")
print("[INFO] Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 Rows:")
print(df.head())

# BASIC ANALYSIS

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# SALES & PROFIT ANALYSIS

total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
avg_sales = df["Sales"].mean()
avg_profit = df["Profit"].mean()

print("\n====== KEY METRICS ======")
print("Total Sales:", total_sales)
print("Total Profit:", total_profit)
print("Average Sales:", avg_sales)
print("Average Profit:", avg_profit)

# CATEGORY ANALYSIS

category_sales = df.groupby("Category")["Sales"].sum()
category_profit = df.groupby("Category")["Profit"].sum()

print("\nSales by Category:")
print(category_sales)
print("\nProfit by Category:")
print(category_profit)

# Visualizations
plt.figure(figsize=(10, 5))
sns.barplot(x=category_sales.index, y=category_sales.values)
plt.title("Sales by Category")
plt.ylabel("Sales")
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x=category_profit.index, y=category_profit.values)
plt.title("Profit by Category")
plt.ylabel("Profit")
plt.show()

# SUB-CATEGORY ANALYSIS

sub_sales = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False)
sub_profit = df.groupby("Sub-Category")["Profit"].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=sub_sales.index, y=sub_sales.values)
plt.title("Sales by Sub-Category")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x=sub_profit.index, y=sub_profit.values)
plt.title("Profit by Sub-Category")
plt.xticks(rotation=45)
plt.show()

# REGION ANALYSIS

region_profit = df.groupby("Region")["Profit"].sum()

plt.figure(figsize=(8, 5))
sns.barplot(x=region_profit.index, y=region_profit.values)
plt.title("Profit by Region")
plt.ylabel("Profit")
plt.show()

# FINAL SUMMARY

print("\n====== INSIGHTS ======")
print("✓ Top-selling categories identified.")
print("✓ Most profitable categories identified.")
print("✓ Sub-categories ranked by sales and profit.")
print("✓ Regional profit comparison completed.")
print("✓ Visual insights generated.")

print("\n[INFO] Analysis Completed Successfully!")
