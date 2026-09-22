import matplotlib.pyplot as plt
import pandas as pd

# 1. Load the dataset
file_path = "task1_marketplace_delivery_sla_raw.csv"
df = pd.read_csv(file_path)

print("--- Initial Data Preview ---")
print(df.head(), "\n")

# 2. Standardize column text/types and parse dates
df["order_date_dt"] = pd.to_datetime(df["order_date"], format="%d-%m-%y")
df["shipped_at_dt"] = pd.to_datetime(df["shipped_at"], format="%d-%m-%y")
df["delivered_at_dt"] = pd.to_datetime(df["delivered_at"], format="%d-%m-%y")

# Clean text fields
df["is_test_clean"] = df["is_test_order"].str.strip().str.capitalize()
df["order_status_clean"] = df["order_status"].str.strip().str.capitalize()

# Standardize courier names to handle variations
courier_map = {
    "shipquick": "ShipQuick",
    "ShipQuick": "ShipQuick",
    "fastbee": "FastBee",
    "FastBee": "FastBee",
    "Fastbee": "FastBee",
    "bluedart": "Blue Dart",
    "Bluedart": "Blue Dart",
    "Blue Dart": "Blue Dart",
}
df["courier_clean"] = df["courier"].map(courier_map)

# Drop duplicate order IDs keeping the latest record
df_clean = df.drop_duplicates(subset=["order_id"], keep="last").copy()

# 3. Filter criteria:
# - Delivered in May 2026 (year == 2026, month == 5)
# - Exclude test orders (is_test_clean == 'No')
# - Only successfully delivered orders
mask = (
    (df_clean["delivered_at_dt"].dt.year == 2026)
    & (df_clean["delivered_at_dt"].dt.month == 5)
    & (df_clean["is_test_clean"] == "No")
    & (df_clean["order_status_clean"] == "Delivered")
)

may_orders = df_clean[mask].copy()

# 4. Calculate actual transit days and on-time status
may_orders["actual_transit_days"] = (
    may_orders["delivered_at_dt"] - may_orders["shipped_at_dt"]
).dt.days
may_orders["is_on_time"] = (
    may_orders["actual_transit_days"] <= may_orders["promised_days"]
)

print("--- Filtered May 2026 Orders & Delivery Status ---")
print(
    may_orders[
        [
            "order_id",
            "courier_clean",
            "shipped_at_dt",
            "delivered_at_dt",
            "actual_transit_days",
            "promised_days",
            "is_on_time",
        ]
    ].to_string(),
    "\n",
)

# 5. Aggregate by courier to generate the required summary dataframe
summary_df = (
    may_orders.groupby("courier_clean")
    .agg(
        eligible_orders=("order_id", "count"),
        on_time_orders=("is_on_time", "sum"),
    )
    .reset_index()
)

summary_df["on_time_rate"] = (
    summary_df["on_time_orders"] / summary_df["eligible_orders"]
).round(4)
summary_df = summary_df.rename(columns={"courier_clean": "courier"})

print("=== Courier-Level Delivery Performance Summary (May 2026) ===")
print(summary_df.to_string(index=False), "\n")

# 6. Visualization Requirement (Using Pure Matplotlib)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Chart 1: On-Time Rate by Courier
bars1 = axes[0].bar(
    summary_df["courier"],
    summary_df["on_time_rate"],
    color="#4b6584",
    width=0.5,
)
axes[0].set_title(
    "On-Time Delivery Rate by Courier (May 2026)", fontsize=13, fontweight="bold"
)
axes[0].set_xlabel("Courier Partner", fontsize=11)
axes[0].set_ylabel("On-Time Rate", fontsize=11)
axes[0].set_ylim(0, 1.2)

for bar in bars1:
  yval = bar.get_height()
  axes[0].text(
      bar.get_x() + bar.get_width() / 2.0,
      yval + 0.03,
      f"{yval:.2f}",
      ha="center",
      va="bottom",
      fontsize=10,
      fontweight="bold",
  )

# Chart 2: Eligible vs On-Time Orders Volume Comparison
x = range(len(summary_df["courier"]))
width = 0.35

bars2_eligible = axes[1].bar(
    [p - width / 2 for p in x],
    summary_df["eligible_orders"],
    width=width,
    label="eligible_orders",
    color="#20bf6b",
)
bars2_ontime = axes[1].bar(
    [p + width / 2 for p in x],
    summary_df["on_time_orders"],
    width=width,
    label="on_time_orders",
    color="#eb3b5a",
)

axes[1].set_title(
    "Eligible Orders vs. On-Time Orders by Courier",
    fontsize=13,
    fontweight="bold",
)
axes[1].set_xlabel("Courier Partner", fontsize=11)
axes[1].set_ylabel("Number of Orders", fontsize=11)
axes[1].set_xticks(x)
axes[1].set_xticklabels(summary_df["courier"])
axes[1].legend()
axes[1].set_ylim(0, max(summary_df["eligible_orders"]) + 1)

for bar in bars2_eligible:
  yval = bar.get_height()
  axes[1].text(
      bar.get_x() + bar.get_width() / 2.0,
      yval + 0.05,
      str(int(yval)),
      ha="center",
      va="bottom",
      fontsize=10,
  )

for bar in bars2_ontime:
  yval = bar.get_height()
  axes[1].text(
      bar.get_x() + bar.get_width() / 2.0,
      yval + 0.05,
      str(int(yval)),
      ha="center",
      va="bottom",
      fontsize=10,
  )

plt.tight_layout()
plt.savefig("delivery_performance_analysis.png", dpi=300)
print("Charts saved successfully as 'delivery_performance_analysis.png'.")
plt.show()