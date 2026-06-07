#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import statsmodels.formula.api as smf
from IPython.display import display

# Raw tables
df = pd.read_csv("../data/transaction_data.csv")
product_df = pd.read_csv("../data/product.csv")

print(df.shape)
print(product_df.shape)

display(df.head())
display(product_df.head())


# In[2]:


df_full = df.merge(product_df, on="PRODUCT_ID", how="left")

print(df_full.shape)
display(df_full.head())


# In[30]:


#Removing all non-retail items, cleaning dataset
# Remove coupons, fuel, and misc non-retail transactions
df_clean = df_full[
    (~df_full["COMMODITY_DESC"].isin(["COUPON/MISC ITEMS", "FUEL"])) &
    (~df_full["DEPARTMENT"].isin(["KIOSK-GAS", "MISC SALES TRAN"]))
].copy()

print(df_full.shape, "->", df_clean.shape)

# Optional check: top products after cleaning
product_totals = (
    df_clean.groupby("PRODUCT_ID")["QUANTITY"]
    .sum()
    .sort_values(ascending=False)
)

top_100 = product_totals.head(100)
top_100_df = top_100.reset_index().merge(product_df, on="PRODUCT_ID", how="left")

display(top_100_df[[
    "PRODUCT_ID", "QUANTITY", "DEPARTMENT", "COMMODITY_DESC", "SUB_COMMODITY_DESC"
]].head(30))


# In[4]:


#Create categories 

def map_category(row):
    if row['COMMODITY_DESC'] == 'HOT DOGS':
        return 'HOT_DOGS'
    elif 'HOT DOG BUNS' in str(row['SUB_COMMODITY_DESC']):
        return 'HOT_DOG_BUNS'
    elif 'SOFT DRINKS' in str(row['SUB_COMMODITY_DESC']):
        return 'SOFT_DRINKS'
    elif 'CHIPS' in str(row['COMMODITY_DESC']):
        return 'CHIPS'
    elif 'DRY NOODLES' in str(row['COMMODITY_DESC']):
        return 'PASTA'
    elif 'PASTA SAUCE' in str(row['COMMODITY_DESC']):
        return 'PASTA_SAUCE'
    elif 'SALSA' in str(row['SUB_COMMODITY_DESC']):
        return 'SALSA'
    elif 'TORTILLA/NACHO' in str(row['SUB_COMMODITY_DESC']):
        return 'TORTILLA_CHIPS'
    elif 'TORTILLA CHIPS' in str(row['SUB_COMMODITY_DESC']):
        return 'TORTILLA_CHIPS'
    elif 'HAMBURGER BUNS' in str(row['SUB_COMMODITY_DESC']):
        return 'HAMBURGER_BUNS'
    elif 'PATTY' in str(row['SUB_COMMODITY_DESC']):
        return 'HAMBURGER PATTY'
    elif 'CEREAL' in str(row['COMMODITY_DESC']):
        return 'CEREAL'
    elif 'FLUID MILK WHITE ONLY' in str(row['SUB_COMMODITY_DESC']):
        return 'MILK'
    else:
        return None

df_clean["CATEGORY"] = df_clean.apply(map_category, axis=1)

df_cat = df_clean[df_clean["CATEGORY"].notna()].copy()

print(df_cat.shape)
print(df_cat["CATEGORY"].value_counts())

display(df_cat[[
    "PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC", "SUB_COMMODITY_DESC", "CATEGORY"
]].head(30))


# In[5]:


#Extract package sizes
def mixed_to_float(token):
    token = str(token).strip().replace(",", ".")

    m = re.fullmatch(r"(\d+)\s+(\d+)/(\d+)", token)
    if m:
        a, b, c = m.groups()
        return float(a) + float(b) / float(c)

    m = re.fullmatch(r"(\d+)/(\d+)", token)
    if m:
        b, c = m.groups()
        return float(b) / float(c)

    m = re.fullmatch(r"\d*\.?\d+", token)
    if m:
        return float(token)

    return np.nan


def parse_size(size_text):
    if pd.isna(size_text) or str(size_text).strip() == "":
        return pd.Series([np.nan, np.nan, np.nan, np.nan])

    s = str(size_text).upper().strip()

    # normalize common variants
    s = s.replace("OUNCES", "OUNCE").replace("OUNCE", "OZ")
    s = s.replace("LBS", "LB").replace("POUNDS", "LB").replace("POUND", "LB")
    s = s.replace("LITERS", "LTR").replace("LITER", "LTR")
    s = s.replace("FL. OZ", "FL OZ").replace("FLOZ", "FL OZ")
    s = s.replace("COUNTS", "CT").replace("COUNT", "CT")
    s = s.replace("PACKS", "PK").replace("PACK", "PK")
    s = s.replace("PIECES", "PC").replace("PIECE", "PC")

    # strip junk before the number
    s = re.sub(r"^[^0-9\.]+", "", s)
    s = re.sub(r"\s+", " ", s)

    num = r"(\d+(?:\.\d+)?|\.\d+|\d+\s+\d+/\d+|\d+/\d+)"

    # multipacks like 8/2.25 OZ
    m = re.search(rf"(\d+)\s*(?:PK|CT)?\s*/\s*{num}\s*(FL OZ|OZ|LB|ML|LTR|L|GA|GAL|QT|PT)\b", s)
    if m:
        outer_n = float(m.group(1))
        inner_n = mixed_to_float(m.group(2))
        raw_value = outer_n * inner_n
        raw_unit = m.group(3)
    else:
        # normal forms like 16 OZ / 1.75LTR / 110CT
        m = re.search(rf"{num}\s*(FL OZ|OZ|LB|ML|LTR|L|GA|GAL|QT|PT|CT|PK|PC|CTN|STEM)\b", s)

        # attached patterns like 24OZ
        if not m:
            m = re.search(r"(\d+(?:\.\d+)?|\.\d+)(OZ|LB|ML|LTR|L|GA|GAL|QT|PT|CT|PK)\b", s)

        if not m:
            return pd.Series([np.nan, np.nan, np.nan, np.nan])

        raw_value = mixed_to_float(m.group(1))
        raw_unit = m.group(2)

    # standardize units
    if raw_unit == "LB":
        std_size, std_unit = raw_value * 16, "oz"
    elif raw_unit == "OZ":
        std_size, std_unit = raw_value, "oz"
    elif raw_unit == "FL OZ":
        std_size, std_unit = raw_value * 29.5735, "ml"
    elif raw_unit == "ML":
        std_size, std_unit = raw_value, "ml"
    elif raw_unit in ["LTR", "L"]:
        std_size, std_unit = raw_value * 1000, "ml"
    elif raw_unit in ["GA", "GAL"]:
        std_size, std_unit = raw_value * 3785.41, "ml"
    elif raw_unit == "QT":
        std_size, std_unit = raw_value * 946.353, "ml"
    elif raw_unit == "PT":
        std_size, std_unit = raw_value * 473.176, "ml"
    elif raw_unit in ["CT", "PK", "PC", "CTN", "STEM"]:
        std_size, std_unit = raw_value, "ct"
    else:
        std_size, std_unit = np.nan, np.nan

    return pd.Series([raw_value, raw_unit, std_size, std_unit])


product_df[["size_value_raw", "size_unit_raw", "std_size", "std_unit"]] = (
    product_df["CURR_SIZE_OF_PRODUCT"].apply(parse_size)
)

unparsed_sizes = (
    product_df.loc[
        product_df["CURR_SIZE_OF_PRODUCT"].fillna("").str.strip().ne("") &
        product_df["std_unit"].isna(),
        "CURR_SIZE_OF_PRODUCT"
    ]
    .value_counts()
    .head(30)
)

print("Top unparsed size strings:")
print(unparsed_sizes)


# In[6]:


#No very important units left out, so now merging sizes back into cleaned category data
# Keep only the parsed size fields from product_df
size_cols = product_df[[
    "PRODUCT_ID", "size_value_raw", "size_unit_raw", "std_size", "std_unit"
]].copy()

# Merge only the new parsed variables back
df = df_cat.merge(
    size_cols,
    on="PRODUCT_ID",
    how="left"
)

df = df[df["QUANTITY"] > 0].copy()

print(df.shape)
display(df.head())


# In[7]:


#Create the prices
# Price paid by customer per purchased pack
df["paid_price_per_pack"] = (
    (df["SALES_VALUE"] + df["COUPON_DISC"]) / df["QUANTITY"]
)

# Shelf price incl. loyalty logic from dataset guide
df["loyalty_price_per_pack"] = (
    (df["SALES_VALUE"] - (df["RETAIL_DISC"] + df["COUPON_MATCH_DISC"])) / df["QUANTITY"]
)

# Regular/non-loyalty shelf price
df["regular_price_per_pack"] = (
    (df["SALES_VALUE"] - df["COUPON_MATCH_DISC"]) / df["QUANTITY"]
)

price_cols = [
    "paid_price_per_pack",
    "loyalty_price_per_pack",
    "regular_price_per_pack"
]

df[price_cols] = df[price_cols].replace([np.inf, -np.inf], np.nan)

display(df[[
    "PRODUCT_ID", "CATEGORY", "QUANTITY", "SALES_VALUE",
    "RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC",
    "paid_price_per_pack", "loyalty_price_per_pack", "regular_price_per_pack"
]].head())


# In[8]:


#Convert the prices to price per standard unit
for col in price_cols:
    new_col = col.replace("_per_pack", "_per_std_unit")
    df[new_col] = np.where(
        df["std_size"] > 0,
        df[col] / df["std_size"],
        np.nan
    )

df["std_unit_label"] = df["std_unit"].map({
    "oz": "price per oz",
    "ml": "price per ml",
    "ct": "price per count"
})

display(df[[
    "PRODUCT_ID", "CATEGORY", "CURR_SIZE_OF_PRODUCT", "std_size", "std_unit",
    "paid_price_per_pack", "paid_price_per_std_unit",
    "regular_price_per_pack", "regular_price_per_std_unit"
]].head(20))


# In[9]:


# Broad line-level price promo
df["line_promo_broad"] = (
    (df["RETAIL_DISC"] < 0) |
    (df["COUPON_DISC"] < 0) |
    (df["COUPON_MATCH_DISC"] < 0)
).astype(int)

# Keep your original line-level flag name too
df["price_promo_flag"] = df["line_promo_broad"]

promo_check = (
    df.groupby(["PRODUCT_ID", "price_promo_flag", "std_unit"], as_index=False)
      .agg(
          median_paid_price_per_std_unit=("paid_price_per_std_unit", "median"),
          median_regular_price_per_std_unit=("regular_price_per_std_unit", "median"),
          n_obs=("PRODUCT_ID", "size")
      )
)

display(promo_check.head(20))


# In[10]:


#Weighted average here
def weighted_avg(values, weights):
    values = pd.Series(values)
    weights = pd.Series(weights)
    mask = values.notna() & weights.notna()

    if mask.sum() == 0:
        return np.nan
    if weights[mask].sum() == 0:
        return np.nan

    return np.average(values[mask], weights=weights[mask])


# In[11]:


#I have decided to keep these main units
# Check how many line-level observations each category has by standard unit
unit_obs_line = (
    df.groupby(["CATEGORY", "std_unit"], dropna=False)
      .agg(
          n_lines=("PRODUCT_ID", "size"),
          n_products=("PRODUCT_ID", "nunique"),
          total_qty=("QUANTITY", "sum")
      )
      .reset_index()
      .sort_values(["CATEGORY", "n_lines"], ascending=[True, False])
)

display(unit_obs_line)

main_units = {
    "CEREAL": "oz",
    "CHIPS": "oz",
    "HAMBURGER PATTY": "oz",
    "HAMBURGER_BUNS": "oz",
    "HOT_DOGS": "oz",
    "HOT_DOG_BUNS": "oz",
    "MILK": "ml",
    "PASTA": "oz",
    "PASTA_SAUCE": "oz",
    "SALSA": "oz",
    "SOFT_DRINKS": "oz",
    "TORTILLA_CHIPS": "oz"
}

df_model = df.copy()

# keep only rows whose parsed unit matches the chosen main unit for that category
df_model["main_unit"] = df_model["CATEGORY"].map(main_units)

df_model = df_model[
    df_model["main_unit"].notna() &
    (df_model["std_unit"] == df_model["main_unit"])
].copy()

print(df_model["CATEGORY"].value_counts())
print(df_model[["CATEGORY", "std_unit"]].drop_duplicates().sort_values(["CATEGORY", "std_unit"]))

cat_week = (
    df_model.groupby(["STORE_ID", "WEEK_NO", "CATEGORY"], as_index=False)
      .agg(
          demand_qty=("QUANTITY", "sum"),
          n_lines=("PRODUCT_ID", "size"),
          promo_lines=("line_promo_broad", "sum"),

          # pack-level prices
          avg_paid_price=("paid_price_per_pack", lambda x: weighted_avg(x, df_model.loc[x.index, "QUANTITY"])),
          avg_regular_price=("regular_price_per_pack", lambda x: weighted_avg(x, df_model.loc[x.index, "QUANTITY"])),

          # standard-unit prices, now safe because unit is fixed within category
          avg_paid_price_std=("paid_price_per_std_unit", lambda x: weighted_avg(x, df_model.loc[x.index, "QUANTITY"])),
          avg_regular_price_std=("regular_price_per_std_unit", lambda x: weighted_avg(x, df_model.loc[x.index, "QUANTITY"])),

          std_unit=("std_unit", "first")
      )
)

cat_week["price_promo_flag"] = (cat_week["promo_lines"] > 0).astype(int)

display(cat_week.head(20))


# In[12]:


# Create 4-week model periods from weekly data
cat_week_4w = cat_week.copy()

# Period 1 = weeks 1-4, period 2 = weeks 5-8, etc.
cat_week_4w["PERIOD_4W"] = ((cat_week_4w["WEEK_NO"] - 1) // 4) + 1

display(
    cat_week_4w[["STORE_ID", "WEEK_NO", "PERIOD_4W", "CATEGORY", "demand_qty", "price_promo_flag"]]
    .sort_values(["STORE_ID", "WEEK_NO", "CATEGORY"])
    .head(20)
)

print("Number of 4-week periods:", cat_week_4w["PERIOD_4W"].nunique())
print(sorted(cat_week_4w["PERIOD_4W"].unique()))


# In[13]:


#Basket co-occurence
basket_cat = df_cat[["BASKET_ID", "CATEGORY"]].drop_duplicates()

pairs = basket_cat.merge(basket_cat, on="BASKET_ID")
pairs = pairs[pairs["CATEGORY_x"] < pairs["CATEGORY_y"]]

pair_counts = (
    pairs.groupby(["CATEGORY_x", "CATEGORY_y"])
    .size()
    .reset_index(name="co_occurrence")
)

cat_counts = (
    basket_cat.groupby("CATEGORY")["BASKET_ID"]
    .nunique()
    .reset_index(name="count")
)

#The number of baskets that is used here, only contains the baskets that contian a product from one of the mapped categories. It does not make sense here to include all of the basket,
# also the ones that do not contain any product that we are considering
n_baskets = df_cat["BASKET_ID"].nunique()

#Usig all cleaned baskets is an option too, which brings the uplift to a much higher number, but it does not necessarliy improve the analysis
# Use all cleaned baskets as denominator, not only reduced category baskets
# n_baskets = df_clean["BASKET_ID"].nunique()

pair_counts = pair_counts.merge(
    cat_counts.rename(columns={"CATEGORY": "CATEGORY_x", "count": "count_x"}),
    on="CATEGORY_x",
    how="left"
)

pair_counts = pair_counts.merge(
    cat_counts.rename(columns={"CATEGORY": "CATEGORY_y", "count": "count_y"}),
    on="CATEGORY_y",
    how="left"
)

pair_counts["support_ab"] = pair_counts["co_occurrence"] / n_baskets
pair_counts["support_a"] = pair_counts["count_x"] / n_baskets
pair_counts["support_b"] = pair_counts["count_y"] / n_baskets
pair_counts["lift"] = pair_counts["support_ab"] / (pair_counts["support_a"] * pair_counts["support_b"])
pair_counts["confidence_x_to_y"] = pair_counts["co_occurrence"] / pair_counts["count_x"]
pair_counts["confidence_y_to_x"] = pair_counts["co_occurrence"] / pair_counts["count_y"]

results = pair_counts[
    (pair_counts["co_occurrence"] > 50) &
    (pair_counts["lift"] > 1.5)
].sort_values(by="lift", ascending=False)

display(results.head(50))


# In[14]:


#Checkign in how many baskets each product occurs
check_counts = (
    basket_cat.groupby("CATEGORY")["BASKET_ID"]
    .nunique()
    .reset_index(name="basket_count")
    .sort_values("basket_count", ascending=False)
)

display(check_counts)


# In[15]:


#Baseline demand, promo demand and uplift
# ============================================================
# Baseline demand in 4-week buckets + overall uplift per category
# ============================================================

# ---------- A. Overall category-level uplift (keep this for u_i_model) ----------
baseline_tbl_overall = (
    cat_week.loc[cat_week["price_promo_flag"] == 0]
    .groupby("CATEGORY", as_index=False)
    .agg(
        n_nonpromo=("demand_qty", "size"),
        baseline_mean=("demand_qty", "mean"),
        baseline_std=("demand_qty", "std"),
        total_demand_no_promo=("demand_qty", "sum")
    )
)

promo_tbl_overall = (
    cat_week.loc[cat_week["price_promo_flag"] == 1]
    .groupby("CATEGORY", as_index=False)
    .agg(
        n_promo=("demand_qty", "size"),
        demand_promo=("demand_qty", "mean"),
        total_demand_promo=("demand_qty", "sum")
    )
)

demand_summary = baseline_tbl_overall.merge(promo_tbl_overall, on="CATEGORY", how="left")

demand_summary["baseline_std"] = demand_summary["baseline_std"].fillna(0)

demand_summary["low_demand"] = (
    demand_summary["baseline_mean"] - demand_summary["baseline_std"]
).clip(lower=0)
demand_summary["regular_demand"] = demand_summary["baseline_mean"]
demand_summary["high_demand"] = (
    demand_summary["baseline_mean"] + demand_summary["baseline_std"]
)

# Raw uplift
demand_summary["uplift_units_raw"] = (
    demand_summary["demand_promo"] - demand_summary["baseline_mean"]
)
demand_summary["uplift_factor_raw"] = (
    demand_summary["demand_promo"] / demand_summary["baseline_mean"]
)
demand_summary["uplift_pct_raw"] = 100 * (
    demand_summary["uplift_units_raw"] / demand_summary["baseline_mean"]
)

# Model version: keep non-negative own-promotion uplift
demand_summary["u_i_model"] = demand_summary["uplift_units_raw"].clip(lower=0)

display(demand_summary.sort_values("CATEGORY"))


# ---------- B. 4-week period baseline demand ----------
# First aggregate to store-period-category, so the model periods are truly 4-week buckets
cat_period_4w = (
    cat_week_4w.groupby(["STORE_ID", "PERIOD_4W", "CATEGORY"], as_index=False)
    .agg(
        demand_qty_4w=("demand_qty", "sum"),
        n_weeks_in_bucket=("WEEK_NO", "nunique"),
        promo_weeks_in_bucket=("price_promo_flag", "sum"),
        std_unit=("std_unit", "first")
    )
)

# A 4-week bucket is considered promotional if at least one week in the bucket had a price promotion
cat_period_4w["price_promo_flag_4w"] = (
    cat_period_4w["promo_weeks_in_bucket"] > 0
).astype(int)

display(cat_period_4w.head(20))


# ---------- C. Overall non-promo fallback baseline ----------
overall_baseline_4w = (
    cat_period_4w.loc[cat_period_4w["price_promo_flag_4w"] == 0]
    .groupby("CATEGORY", as_index=False)
    .agg(
        baseline_mean_overall_4w=("demand_qty_4w", "mean"),
        baseline_std_overall_4w=("demand_qty_4w", "std"),
        n_nonpromo_buckets_overall=("demand_qty_4w", "size")
    )
)

overall_baseline_4w["baseline_std_overall_4w"] = (
    overall_baseline_4w["baseline_std_overall_4w"].fillna(0)
)


# ---------- D. Period-specific non-promo baseline ----------
baseline_period_4w = (
    cat_period_4w.loc[cat_period_4w["price_promo_flag_4w"] == 0]
    .groupby(["CATEGORY", "PERIOD_4W"], as_index=False)
    .agg(
        n_nonpromo_buckets=("demand_qty_4w", "size"),
        baseline_mean_4w=("demand_qty_4w", "mean"),
        baseline_std_4w=("demand_qty_4w", "std"),
        total_demand_no_promo_4w=("demand_qty_4w", "sum")
    )
)

baseline_period_4w["baseline_std_4w"] = baseline_period_4w["baseline_std_4w"].fillna(0)

# Merge overall fallback
baseline_period_4w = baseline_period_4w.merge(
    overall_baseline_4w,
    on="CATEGORY",
    how="left"
)

# If a category-period has too few non-promo observations, fall back to the overall category 4-week baseline
min_nonpromo_buckets = 3

baseline_period_4w["baseline_mean_final"] = np.where(
    baseline_period_4w["n_nonpromo_buckets"] >= min_nonpromo_buckets,
    baseline_period_4w["baseline_mean_4w"],
    baseline_period_4w["baseline_mean_overall_4w"]
)

baseline_period_4w["baseline_std_final"] = np.where(
    baseline_period_4w["n_nonpromo_buckets"] >= min_nonpromo_buckets,
    baseline_period_4w["baseline_std_4w"],
    baseline_period_4w["baseline_std_overall_4w"]
)

baseline_period_4w["baseline_std_final"] = baseline_period_4w["baseline_std_final"].fillna(0)

baseline_period_4w["low_demand_4w"] = (
    baseline_period_4w["baseline_mean_final"] - baseline_period_4w["baseline_std_final"]
).clip(lower=0)

baseline_period_4w["regular_demand_4w"] = baseline_period_4w["baseline_mean_final"]
baseline_period_4w["high_demand_4w"] = (
    baseline_period_4w["baseline_mean_final"] + baseline_period_4w["baseline_std_final"]
)

display(
    baseline_period_4w[
        [
            "CATEGORY",
            "PERIOD_4W",
            "n_nonpromo_buckets",
            "baseline_mean_final",
            "baseline_std_final",
            "low_demand_4w",
            "regular_demand_4w",
            "high_demand_4w"
        ]
    ].sort_values(["CATEGORY", "PERIOD_4W"])
)


# In[16]:


#Make this into a cleaner baselien demand scenarios table
baseline_scenarios = demand_summary[[
    "CATEGORY", "low_demand", "regular_demand", "high_demand"
]].copy()

baseline_scenarios = baseline_scenarios.rename(columns={
    "low_demand": "baseline_low",
    "regular_demand": "baseline_regular",
    "high_demand": "baseline_high"
})

display(baseline_scenarios.sort_values("CATEGORY"))

#Now also create a baslien 4-week table for the model
# Cleaner 4-week baseline demand scenarios table
baseline_scenarios_4w = baseline_period_4w[[
    "CATEGORY",
    "PERIOD_4W",
    "low_demand_4w",
    "regular_demand_4w",
    "high_demand_4w"
]].copy()

baseline_scenarios_4w = baseline_scenarios_4w.rename(columns={
    "PERIOD_4W": "PERIOD",
    "low_demand_4w": "baseline_low",
    "regular_demand_4w": "baseline_regular",
    "high_demand_4w": "baseline_high"
})

display(baseline_scenarios_4w.sort_values(["CATEGORY", "PERIOD"]))


# In[17]:


#Now a category price table
# Typical package size per category, based on the filtered df_model
# (so only the chosen main unit per category remains)
category_size_table = (
    df_model.groupby(["CATEGORY", "std_unit"], as_index=False)
    .agg(
        avg_pack_size=("std_size", lambda x: weighted_avg(x, df_model.loc[x.index, "QUANTITY"])),
        median_pack_size=("std_size", "median"),
        min_pack_size=("std_size", "min"),
        max_pack_size=("std_size", "max"),
        n_lines_size=("PRODUCT_ID", "size"),
        n_products=("PRODUCT_ID", "nunique"),
        total_qty=("QUANTITY", "sum")
    )
)

# Weekly prices at store-week-category level
weekly_price = (
    cat_week.groupby(["CATEGORY", "std_unit", "price_promo_flag"], as_index=False)
    .agg(
        avg_paid_price_pack=("avg_paid_price", "mean"),
        avg_regular_price_pack=("avg_regular_price", "mean"),
        avg_paid_price_std=("avg_paid_price_std", "mean"),
        avg_regular_price_std=("avg_regular_price_std", "mean"),
        n_obs=("CATEGORY", "size")
    )
)

price_table = weekly_price.pivot(
    index=["CATEGORY", "std_unit"],
    columns="price_promo_flag"
)

price_table.columns = [
    f"{metric}_{'promo' if promo == 1 else 'nonpromo'}"
    for metric, promo in price_table.columns
]
price_table = price_table.reset_index()

# Merge package size information
price_table = price_table.merge(
    category_size_table,
    on=["CATEGORY", "std_unit"],
    how="left"
)

# Model price should be package price, not std-unit price
price_table["P_i_model_pack"] = price_table["avg_regular_price_pack_nonpromo"]

# Discount depth based on package prices
price_table["delta_i_model_pack"] = 1 - (
    price_table["avg_paid_price_pack_promo"] /
    price_table["avg_paid_price_pack_nonpromo"]
)

price_table["delta_i_model_pack"] = price_table["delta_i_model_pack"].clip(lower=0, upper=1)

display(
    price_table[[
        "CATEGORY",
        "std_unit",
        "avg_pack_size",
        "median_pack_size",
        "P_i_model_pack",
        "delta_i_model_pack",
        "avg_paid_price_pack_nonpromo",
        "avg_paid_price_pack_promo",
        "avg_paid_price_std_nonpromo",
        "avg_paid_price_std_promo"
    ]].sort_values("CATEGORY")
)


# In[18]:


#Calculating promo frequencies and demand variability
promo_frequency = (
    cat_week.groupby("CATEGORY")
    .agg(
        n_store_weeks=("WEEK_NO", "size"),
        n_promo_weeks=("price_promo_flag", "sum")
    )
    .reset_index()
)

promo_frequency["promo_frequency"] = (
    promo_frequency["n_promo_weeks"] / promo_frequency["n_store_weeks"]
)

display(promo_frequency.sort_values("CATEGORY"))

8
demand_variability = (
    cat_week.groupby(["CATEGORY", "price_promo_flag"])["demand_qty"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

demand_variability["cv"] = demand_variability["std"] / demand_variability["mean"]

display(demand_variability.sort_values(["CATEGORY", "price_promo_flag"]))


# In[19]:


#Creating a wide table
def build_pair_promo_table(cat_week, cat_a, cat_b):
    sub = cat_week[cat_week["CATEGORY"].isin([cat_a, cat_b])].copy()

    wide = (
        sub.pivot_table(
            index=["STORE_ID", "WEEK_NO"],
            columns="CATEGORY",
            values=["demand_qty", "price_promo_flag"],
            aggfunc="first"
        )
        .reset_index()
    )

    # flatten columns
    wide.columns = [
        "_".join([str(c) for c in col if str(c) != ""]).strip("_")
        if isinstance(col, tuple) else col
        for col in wide.columns
    ]

    # fill missing demand/promo with zero
    for c in [f"demand_qty_{cat_a}", f"demand_qty_{cat_b}"]:
        if c not in wide.columns:
            wide[c] = 0
        wide[c] = wide[c].fillna(0)

    for c in [f"price_promo_flag_{cat_a}", f"price_promo_flag_{cat_b}"]:
        if c not in wide.columns:
            wide[c] = 0
        wide[c] = wide[c].fillna(0).astype(int)

    # pair states
    conds = [
        (wide[f"price_promo_flag_{cat_a}"] == 0) & (wide[f"price_promo_flag_{cat_b}"] == 0),
        (wide[f"price_promo_flag_{cat_a}"] == 1) & (wide[f"price_promo_flag_{cat_b}"] == 0),
        (wide[f"price_promo_flag_{cat_a}"] == 0) & (wide[f"price_promo_flag_{cat_b}"] == 1),
        (wide[f"price_promo_flag_{cat_a}"] == 1) & (wide[f"price_promo_flag_{cat_b}"] == 1),
    ]

    states = ["none", f"{cat_a}_only", f"{cat_b}_only", "both"]

    wide["promo_state"] = np.select(conds, states, default="unknown")

    state_summary = (
        wide.groupby("promo_state", as_index=False)
        .agg(
            **{
                f"mean_{cat_a}": (f"demand_qty_{cat_a}", "mean"),
                f"mean_{cat_b}": (f"demand_qty_{cat_b}", "mean"),
                "n_weeks": ("promo_state", "size")
            }
        )
    )

    effects = pd.DataFrame({
        "pair": [f"{cat_a} - {cat_b}"],
        f"{cat_a}_mean_when_{cat_b}_promo": [
            wide.loc[wide[f"price_promo_flag_{cat_b}"] == 1, f"demand_qty_{cat_a}"].mean()
        ],
        f"{cat_a}_mean_when_{cat_b}_notpromo": [
            wide.loc[wide[f"price_promo_flag_{cat_b}"] == 0, f"demand_qty_{cat_a}"].mean()
        ],
        f"{cat_b}_mean_when_{cat_a}_promo": [
            wide.loc[wide[f"price_promo_flag_{cat_a}"] == 1, f"demand_qty_{cat_b}"].mean()
        ],
        f"{cat_b}_mean_when_{cat_a}_notpromo": [
            wide.loc[wide[f"price_promo_flag_{cat_a}"] == 0, f"demand_qty_{cat_b}"].mean()
        ]
    })

    return wide, state_summary, effects


# In[20]:


hd_wide, hd_state_summary, hd_effects = build_pair_promo_table(
    cat_week=cat_week,
    cat_a="HOT_DOGS",
    cat_b="HOT_DOG_BUNS"
)

ps_wide, ps_state_summary, ps_effects = build_pair_promo_table(
    cat_week=cat_week,
    cat_a="PASTA",
    cat_b="PASTA_SAUCE"
)

display(hd_state_summary)
display(hd_effects)

display(ps_state_summary)
display(ps_effects)


# In[21]:


#Merge into pair tables
def add_prices_and_summarize_states(wide_df, weekly_price, cat_a, cat_b):
    price_a = (
        cat_week.loc[cat_week["CATEGORY"] == cat_a, ["STORE_ID", "WEEK_NO", "avg_paid_price"]]
        .rename(columns={"avg_paid_price": f"avg_paid_price_{cat_a}"})
    )

    price_b = (
        cat_week.loc[cat_week["CATEGORY"] == cat_b, ["STORE_ID", "WEEK_NO", "avg_paid_price"]]
        .rename(columns={"avg_paid_price": f"avg_paid_price_{cat_b}"})
    )

    wide_price = wide_df.merge(price_a, on=["STORE_ID", "WEEK_NO"], how="left")
    wide_price = wide_price.merge(price_b, on=["STORE_ID", "WEEK_NO"], how="left")

    state_price_summary = (
        wide_price.groupby("promo_state", as_index=False)
        .agg(
            **{
                f"mean_{cat_a}": (f"demand_qty_{cat_a}", "mean"),
                f"mean_{cat_b}": (f"demand_qty_{cat_b}", "mean"),
                f"price_{cat_a}": (f"avg_paid_price_{cat_a}", "mean"),
                f"price_{cat_b}": (f"avg_paid_price_{cat_b}", "mean"),
                "n_weeks": ("promo_state", "size")
            }
        )
    )

    return wide_price, state_price_summary


# In[22]:


#Tables now with prices
hd_wide_price, hd_state_price_summary = add_prices_and_summarize_states(
    wide_df=hd_wide,
    weekly_price=weekly_price,
    cat_a="HOT_DOGS",
    cat_b="HOT_DOG_BUNS"
)

ps_wide_price, ps_state_price_summary = add_prices_and_summarize_states(
    wide_df=ps_wide,
    weekly_price=weekly_price,
    cat_a="PASTA",
    cat_b="PASTA_SAUCE"
)

display(hd_state_price_summary)
display(ps_state_price_summary)


# In[23]:


#Now prepare a dataset to perform extra regression analysis
hd_reg = hd_wide_price.copy().rename(columns={
    "demand_qty_HOT_DOGS": "demand_hotdogs",
    "demand_qty_HOT_DOG_BUNS": "demand_buns",
    "price_promo_flag_HOT_DOGS": "promo_hotdogs",
    "price_promo_flag_HOT_DOG_BUNS": "promo_buns",
    "avg_paid_price_HOT_DOGS": "price_hotdogs",
    "avg_paid_price_HOT_DOG_BUNS": "price_buns"
})

ps_reg = ps_wide_price.copy().rename(columns={
    "demand_qty_PASTA": "demand_pasta",
    "demand_qty_PASTA_SAUCE": "demand_sauce",
    "price_promo_flag_PASTA": "promo_pasta",
    "price_promo_flag_PASTA_SAUCE": "promo_sauce",
    "avg_paid_price_PASTA": "price_pasta",
    "avg_paid_price_PASTA_SAUCE": "price_sauce"
})

display(hd_reg.head())
display(ps_reg.head())


# In[24]:


model_buns_2 = smf.ols(
    "demand_buns ~ promo_buns + promo_hotdogs + price_buns + price_hotdogs",
    data=hd_reg
).fit()

model_hotdogs_2 = smf.ols(
    "demand_hotdogs ~ promo_hotdogs + promo_buns + price_hotdogs + price_buns",
    data=hd_reg
).fit()

model_buns_fe = smf.ols(
    "demand_buns ~ promo_buns + promo_hotdogs + price_buns + price_hotdogs + C(STORE_ID) + C(WEEK_NO)",
    data=hd_reg
).fit()

model_hotdogs_fe = smf.ols(
    "demand_hotdogs ~ promo_hotdogs + promo_buns + price_hotdogs + price_buns + C(STORE_ID) + C(WEEK_NO)",
    data=hd_reg
).fit()

print(model_buns_2.summary())
print(model_hotdogs_2.summary())
print(model_buns_fe.summary())
print(model_hotdogs_fe.summary())


# In[25]:


model_sauce_2 = smf.ols(
    "demand_sauce ~ promo_sauce + promo_pasta + price_sauce + price_pasta",
    data=ps_reg
).fit()

model_pasta_2 = smf.ols(
    "demand_pasta ~ promo_pasta + promo_sauce + price_pasta + price_sauce",
    data=ps_reg
).fit()

model_sauce_fe = smf.ols(
    "demand_sauce ~ promo_sauce + promo_pasta + price_sauce + price_pasta + C(STORE_ID) + C(WEEK_NO)",
    data=ps_reg
).fit()

model_pasta_fe = smf.ols(
    "demand_pasta ~ promo_pasta + promo_sauce + price_pasta + price_sauce + C(STORE_ID) + C(WEEK_NO)",
    data=ps_reg
).fit()

print(model_sauce_2.summary())
print(model_pasta_2.summary())
print(model_sauce_fe.summary())
print(model_pasta_fe.summary())


# In[26]:


def run_pair_robustness(df_in, dep, own_promo, cross_promo, own_price, cross_price, label):
    df_ = df_in.copy()

    df_["both_promo"] = df_[own_promo] * df_[cross_promo]
    df_[f"log_{dep}"] = np.log1p(df_[dep])

    df_ = df_.replace([np.inf, -np.inf], np.nan).dropna(subset=[dep, own_promo, cross_promo, own_price, cross_price])

    m1 = smf.ols(
        f"{dep} ~ {own_promo} + {cross_promo}",
        data=df_
    ).fit()

    m2 = smf.ols(
        f"{dep} ~ {own_promo} + {cross_promo} + {own_price} + {cross_price}",
        data=df_
    ).fit()

    m3 = smf.ols(
        f"{dep} ~ {own_promo} + {cross_promo} + {own_price} + {cross_price} + both_promo + C(STORE_ID) + C(WEEK_NO)",
        data=df_
    ).fit()

    m4 = smf.ols(
        f"log_{dep} ~ {own_promo} + {cross_promo} + {own_price} + {cross_price} + both_promo + C(STORE_ID) + C(WEEK_NO)",
        data=df_
    ).fit()

    return {
        "label": label,
        "level_no_price": m1,
        "level_price": m2,
        "level_price_fe_interaction": m3,
        "log_price_fe_interaction": m4
    }

#Run robustness checks
hd_buns_robust = run_pair_robustness(
    df_in=hd_reg,
    dep="demand_buns",
    own_promo="promo_buns",
    cross_promo="promo_hotdogs",
    own_price="price_buns",
    cross_price="price_hotdogs",
    label="hotdogs_to_buns"
)

hd_hotdogs_robust = run_pair_robustness(
    df_in=hd_reg,
    dep="demand_hotdogs",
    own_promo="promo_hotdogs",
    cross_promo="promo_buns",
    own_price="price_hotdogs",
    cross_price="price_buns",
    label="buns_to_hotdogs"
)

ps_sauce_robust = run_pair_robustness(
    df_in=ps_reg,
    dep="demand_sauce",
    own_promo="promo_sauce",
    cross_promo="promo_pasta",
    own_price="price_sauce",
    cross_price="price_pasta",
    label="pasta_to_sauce"
)

ps_pasta_robust = run_pair_robustness(
    df_in=ps_reg,
    dep="demand_pasta",
    own_promo="promo_pasta",
    cross_promo="promo_sauce",
    own_price="price_pasta",
    cross_price="price_sauce",
    label="sauce_to_pasta"
)

print(hd_buns_robust["level_price_fe_interaction"].summary())
print(hd_hotdogs_robust["level_price_fe_interaction"].summary())
print(ps_sauce_robust["level_price_fe_interaction"].summary())
print(ps_pasta_robust["level_price_fe_interaction"].summary())


# In[27]:


#Now extracting the coefficients
def extract_key_coefs(model, model_name):
    coef_table = model.params.reset_index()
    coef_table.columns = ["variable", "coefficient"]
    coef_table["p_value"] = model.pvalues.values
    coef_table["model"] = model_name
    return coef_table

coef_results = pd.concat([
    extract_key_coefs(model_buns_2, "buns_price_controls"),
    extract_key_coefs(model_hotdogs_2, "hotdogs_price_controls"),
    extract_key_coefs(model_buns_fe, "buns_price_fe"),
    extract_key_coefs(model_hotdogs_fe, "hotdogs_price_fe"),
    extract_key_coefs(model_sauce_2, "sauce_price_controls"),
    extract_key_coefs(model_pasta_2, "pasta_price_controls"),
    extract_key_coefs(model_sauce_fe, "sauce_price_fe"),
    extract_key_coefs(model_pasta_fe, "pasta_price_fe")
], ignore_index=True)

display(coef_results)

key_vars = [
    "promo_buns", "promo_hotdogs", "price_buns", "price_hotdogs",
    "promo_pasta", "promo_sauce", "price_pasta", "price_sauce"
]

coef_results_key = coef_results[coef_results["variable"].isin(key_vars)].copy()

display(coef_results_key)


# In[28]:


#Input tables for model
# ============================================================
# Input tables for model
# ============================================================
import pandas as pd


# Demand scenarios per CATEGORY and 4-week PERIOD
# u_i_model is still category-level for now, so merge it in from demand_summary
model_demand = baseline_scenarios_4w.merge(
    demand_summary[["CATEGORY", "u_i_model", "uplift_pct_raw"]],
    on="CATEGORY",
    how="left"
)

display(model_demand.sort_values(["CATEGORY", "PERIOD"]))
selected_categories = ["HOT_DOGS", "HOT_DOG_BUNS", "PASTA", "PASTA_SAUCE"]

model_demand_selected = model_demand.loc[
    model_demand["CATEGORY"].isin(selected_categories)
].copy()

with pd.option_context("display.max_rows", None, "display.max_columns", None):
    display(model_demand_selected.sort_values(["CATEGORY", "PERIOD"]))

# Price inputs
model_price = price_table[[
    "CATEGORY",
    "std_unit",
    "avg_pack_size",
    "median_pack_size",
    "P_i_model_pack",
    "delta_i_model_pack",
    "avg_paid_price_pack_nonpromo",
    "avg_paid_price_pack_promo"
]].copy()

model_price = model_price.rename(columns={
    "P_i_model_pack": "P_i_model",
    "delta_i_model_pack": "delta_i_model",
    "avg_paid_price_pack_nonpromo": "paid_price_nonpromo",
    "avg_paid_price_pack_promo": "paid_price_promo"
})

display(model_price.sort_values("CATEGORY"))


# Spillover inputs from FE models
gamma_table = pd.DataFrame([
    {
        "promoted_category": "HOT_DOGS",
        "affected_category": "HOT_DOG_BUNS",
        "gamma_raw": model_buns_fe.params.get("promo_hotdogs", np.nan)
    },
    {
        "promoted_category": "HOT_DOG_BUNS",
        "affected_category": "HOT_DOGS",
        "gamma_raw": model_hotdogs_fe.params.get("promo_buns", np.nan)
    },
    {
        "promoted_category": "PASTA",
        "affected_category": "PASTA_SAUCE",
        "gamma_raw": model_sauce_fe.params.get("promo_pasta", np.nan)
    },
    {
        "promoted_category": "PASTA_SAUCE",
        "affected_category": "PASTA",
        "gamma_raw": model_pasta_fe.params.get("promo_sauce", np.nan)
    }
])

display(gamma_table)

