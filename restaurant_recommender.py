"""
=============================================================
Predictive Restaurant Recommender  |  Soulpage IT Solutions
=============================================================
Task     : Binary classification per (Customer, Location, Vendor)
Target   : 1 = customer likely to order from vendor, 0 = unlikely
Model    : Random Forest Classifier
Cold-start: New customers predicted via top vendor popularity

HOW TO RUN:
    python restaurant_recommender.py

REQUIREMENTS:
    pip install -r requirements.txt

FOLDER STRUCTURE:
    restaurant_project/
    ├── restaurant_recommender.py   <- this file
    ├── requirements.txt
    ├── data/
    │   ├── Train/
    │   │   ├── orders.csv
    │   │   ├── train_customers.csv
    │   │   ├── train_locations.csv
    │   │   └── vendors.csv
    │   └── Test/
    │       ├── test_customers.csv
    │       └── test_locations.csv
    └── output/
        └── submission.csv          <- generated after running
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────
# PATH SETUP  (works on Windows, Mac, Linux automatically)
# ─────────────────────────────────────────────────────────────
# Locate the project root as the folder containing this script
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR    = os.path.join(PROJECT_ROOT, "data", "Train")
TEST_DIR     = os.path.join(PROJECT_ROOT, "data", "Test")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def data_path(folder, filename):
    return os.path.join(folder, filename)


# ─────────────────────────────────────────────────────────────
# STEP 0 │ Load Data
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 0 │ Loading Data")
print("=" * 60)

train_customers = pd.read_csv(data_path(TRAIN_DIR, "train_customers.csv"))
test_customers  = pd.read_csv(data_path(TEST_DIR,  "test_customers.csv"))
train_locations = pd.read_csv(data_path(TRAIN_DIR, "train_locations.csv"))
test_locations  = pd.read_csv(data_path(TEST_DIR,  "test_locations.csv"))
vendors         = pd.read_csv(data_path(TRAIN_DIR, "vendors.csv"))
orders          = pd.read_csv(data_path(TRAIN_DIR, "orders.csv"), low_memory=False)

print(f"  train_customers : {train_customers.shape}")
print(f"  test_customers  : {test_customers.shape}")
print(f"  train_locations : {train_locations.shape}")
print(f"  test_locations  : {test_locations.shape}")
print(f"  vendors         : {vendors.shape}")
print(f"  orders          : {orders.shape}")


# ─────────────────────────────────────────────────────────────
# STEP 1 │ Data Exploration
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 │ Data Exploration")
print("=" * 60)

print("\n[1a] Top 10 vendors by order volume:")
print(orders["vendor_id"].value_counts().head(10))

print("\n[1b] Vendor category distribution:")
print(vendors["vendor_category_en"].value_counts())

print("\n[1c] Gender distribution (train customers):")
print(train_customers["gender"].value_counts())

print(f"\n[1d] Unique customers with orders : {orders['customer_id'].nunique():,}")
print(f"     Avg orders per customer       : {orders.groupby('customer_id').size().mean():.2f}")
print(f"     Unique positive combos        : {orders['CID X LOC_NUM X VENDOR'].nunique():,}")

print("\n[1e] Null counts in orders (key columns):")
print(orders[["vendor_id", "grand_total", "is_favorite",
              "deliverydistance", "vendor_rating"]].isnull().sum())


# ─────────────────────────────────────────────────────────────
# STEP 2 │ Pre-Processing
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 │ Pre-Processing")
print("=" * 60)

# 2a. Clean orders
orders = orders.dropna(subset=["vendor_id"])
orders["vendor_id"]   = orders["vendor_id"].astype(int)
orders["is_favorite"] = orders["is_favorite"].map({"Yes": 1, "No": 0})

# 2b. Positive (customer, location, vendor) pairs
positive_pairs = (
    orders[["customer_id", "LOCATION_NUMBER", "vendor_id"]]
    .drop_duplicates()
    .rename(columns={"LOCATION_NUMBER": "location_number"})
    .assign(target=1)
)
print(f"  Positive pairs (from orders) : {len(positive_pairs):,}")

# 2c. Customer age from birth year
for df in [train_customers, test_customers]:
    df["age"] = 2024 - df["dob"].fillna(0).astype(int)
    df.loc[df["dob"].isna(), "age"] = np.nan

# 2d. Encode gender (normalize inconsistent casing)
le_gender = LabelEncoder()
all_genders = pd.concat([train_customers["gender"], test_customers["gender"]]
                        ).fillna("Unknown").str.strip().str.title()
le_gender.fit(all_genders)
for df in [train_customers, test_customers]:
    df["gender_enc"] = le_gender.transform(
        df["gender"].fillna("Unknown").str.strip().str.title()
    )

# 2e. Remove GPS outliers from locations (3-sigma rule)
def remove_gps_outliers(df):
    df = df.copy()
    for col in ["latitude", "longitude"]:
        med, std = df[col].median(), df[col].std()
        bad = (df[col] > med + 3*std) | (df[col] < med - 3*std) | df[col].isna()
        df.loc[bad, col] = np.nan
    return df

train_locations = remove_gps_outliers(train_locations)
test_locations  = remove_gps_outliers(test_locations)

# 2f. Customer centroid (mean lat/lon across all their locations)
def customer_centroid(loc_df):
    return (
        loc_df.groupby("customer_id")[["latitude", "longitude"]]
        .mean().reset_index()
        .rename(columns={"latitude": "cust_lat", "longitude": "cust_lon"})
    )

cc_train = customer_centroid(train_locations)
cc_test  = customer_centroid(test_locations)
print(f"  Train centroids valid : {cc_train['cust_lat'].notna().sum():,}")
print(f"  Test  centroids valid : {cc_test['cust_lat'].notna().sum():,}")


# ─────────────────────────────────────────────────────────────
# STEP 3 │ Feature Engineering
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 │ Feature Engineering")
print("=" * 60)

# 3a. Vendor features
le_vcat = LabelEncoder()
vendors["vendor_category_enc"] = le_vcat.fit_transform(
    vendors["vendor_category_en"].fillna("Unknown")
)
vendor_feats = vendors[[
    "id", "latitude", "longitude", "vendor_rating", "delivery_charge",
    "serving_distance", "prepration_time", "discount_percentage",
    "vendor_category_enc", "rank"
]].rename(columns={"id": "vendor_id"}).copy()
vendor_feats["rank"] = pd.to_numeric(vendor_feats["rank"], errors="coerce")

# 3b. Customer x Vendor order history (strongest signal)
cv_hist = orders.groupby(["customer_id", "vendor_id"]).agg(
    cv_order_count = ("order_id",    "count"),
    cv_avg_total   = ("grand_total", "mean"),
    cv_is_fav      = ("is_favorite", "max"),
).reset_index()

# 3c. Customer overall order history
c_hist = orders.groupby("customer_id").agg(
    c_total_orders   = ("order_id",    "count"),
    c_avg_total      = ("grand_total", "mean"),
    c_unique_vendors = ("vendor_id",   "nunique"),
).reset_index()

# 3d. Vendor popularity + reach % (used for cold-start customers)
total_train_customers = orders["customer_id"].nunique()
v_pop = orders.groupby("vendor_id").agg(
    v_total_orders     = ("order_id",    "count"),
    v_unique_customers = ("customer_id", "nunique"),
).reset_index()
v_pop["v_reach_pct"] = v_pop["v_unique_customers"] / total_train_customers

print(f"  Vendor features     : {vendor_feats.shape}")
print(f"  Cust×Vendor history : {cv_hist.shape}")
print(f"  Customer history    : {c_hist.shape}")
print(f"  Vendor popularity   : {v_pop.shape}")


def build_features(pairs, cust_df, centroid_df, loc_df):
    """Attach all features to a (customer_id, location_number, vendor_id) dataframe."""
    d = pairs.copy()

    # Customer demographics
    d = d.merge(cust_df[["customer_id", "gender_enc", "age", "status", "verified"]],
                on="customer_id", how="left")

    # Customer centroid coordinates
    d = d.merge(centroid_df, on="customer_id", how="left")

    # Specific delivery-location coordinates
    lp = loc_df[["customer_id", "location_number", "latitude", "longitude"]].rename(
        columns={"latitude": "loc_lat", "longitude": "loc_lon"})
    d = d.merge(lp, on=["customer_id", "location_number"], how="left")

    # Vendor features (includes vendor lat/lon)
    d = d.merge(vendor_feats, on="vendor_id", how="left")

    # Distance: location → vendor (fall back to centroid if GPS missing)
    eff_lat = d["loc_lat"].fillna(d["cust_lat"])
    eff_lon = d["loc_lon"].fillna(d["cust_lon"])
    d["dist_loc_vendor"]  = np.sqrt((eff_lat       - d["latitude"]) ** 2 +
                                    (eff_lon       - d["longitude"]) ** 2)
    d["dist_cust_vendor"] = np.sqrt((d["cust_lat"] - d["latitude"]) ** 2 +
                                    (d["cust_lon"] - d["longitude"]) ** 2)

    # Order history features
    d = d.merge(c_hist,  on="customer_id",               how="left")
    d = d.merge(cv_hist, on=["customer_id", "vendor_id"], how="left")
    d = d.merge(v_pop[["vendor_id", "v_total_orders", "v_unique_customers", "v_reach_pct"]],
                on="vendor_id", how="left")

    # No prior interaction → fill with 0
    for col in ["cv_order_count", "cv_avg_total", "cv_is_fav",
                "c_total_orders", "c_avg_total", "c_unique_vendors"]:
        if col in d.columns:
            d[col] = d[col].fillna(0)

    return d


# ─────────────────────────────────────────────────────────────
# STEP 4 │ Build Training Dataset
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 │ Building Training Dataset")
print("=" * 60)

all_vendor_ids = vendors["id"].tolist()
vkdf = pd.DataFrame({"vendor_id": all_vendor_ids, "_key": 1})

# Only customers with at least one order (need label signal)
active_customers = orders["customer_id"].unique()
train_cl = (
    train_locations[train_locations["customer_id"].isin(active_customers)]
    [["customer_id", "location_number"]].drop_duplicates().assign(_key=1)
)
train_pairs = train_cl.merge(vkdf, on="_key").drop(columns="_key")
print(f"  Train candidate pairs : {len(train_pairs):,}")

# Assign labels: 1 if combo in orders, else 0
train_pairs = train_pairs.merge(
    positive_pairs, on=["customer_id", "location_number", "vendor_id"], how="left"
)
train_pairs["target"] = train_pairs["target"].fillna(0).astype(int)
n_pos = train_pairs["target"].sum()
print(f"  Positives     : {n_pos:,}  |  Negatives : {(train_pairs['target']==0).sum():,}")
print(f"  Positive rate : {n_pos / len(train_pairs):.5f}")

# Downsample negatives 5:1 to handle class imbalance
pos_df = train_pairs[train_pairs["target"] == 1]
neg_df = train_pairs[train_pairs["target"] == 0].sample(n=len(pos_df)*5, random_state=42)
train_bal = pd.concat([pos_df, neg_df]).reset_index(drop=True)
print(f"  Balanced dataset (5:1) : {len(train_bal):,} rows")

train_bal = build_features(train_bal, train_customers, cc_train, train_locations)
print(f"  Final train shape      : {train_bal.shape}")


# ─────────────────────────────────────────────────────────────
# STEP 5 │ Model Training & Validation
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 │ Model Training & Validation")
print("=" * 60)

FEATURE_COLS = [
    "gender_enc", "age", "status", "verified",           # demographics
    "c_total_orders", "c_avg_total", "c_unique_vendors",  # customer history
    "cv_order_count", "cv_avg_total", "cv_is_fav",        # cust×vendor history
    "vendor_rating", "delivery_charge", "serving_distance",
    "prepration_time", "discount_percentage",
    "vendor_category_enc", "rank",                        # vendor attributes
    "v_total_orders", "v_unique_customers", "v_reach_pct",# vendor popularity
    "dist_loc_vendor", "dist_cust_vendor",                # distance
]

X = train_bal[FEATURE_COLS].fillna(-1).values
y = train_bal["target"].values

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train : {X_tr.shape}  |  Val : {X_val.shape}")

print("\n  Training Random Forest (100 trees, max_depth=10) ...")
model = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    min_samples_leaf=50, n_jobs=-1, random_state=42
)
model.fit(X_tr, y_tr)

auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
print(f"\n  >>> Validation ROC-AUC : {auc:.4f}")

print("\n  Top 10 Feature Importances:")
feat_imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
for feat, imp in feat_imp.head(10).items():
    print(f"    {feat:<35s} {imp:.4f}")


# ─────────────────────────────────────────────────────────────
# STEP 6 │ Test Predictions & Submission
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 │ Test Predictions & Submission")
print("=" * 60)

# Build all test candidate pairs
test_cl = (
    test_locations[["customer_id", "location_number"]]
    .drop_duplicates().assign(_key=1)
)
test_pairs = test_cl.merge(vkdf, on="_key").drop(columns="_key")
print(f"  Test candidate pairs : {len(test_pairs):,}")

test_pairs = build_features(test_pairs, test_customers, cc_test, test_locations)
X_test = test_pairs[FEATURE_COLS].fillna(-1).values
test_pairs["prob"] = model.predict_proba(X_test)[:, 1]

# Cold-start strategy:
# All test customers are new (no prior order history).
# Recommend vendors in the top 20% by customer reach percentage.
top20_reach = v_pop["v_reach_pct"].quantile(0.80)
is_cold     = test_pairs["cv_order_count"] == 0
print(f"  Cold-start pairs        : {is_cold.sum():,} / {len(test_pairs):,}")
print(f"  Top-20% reach threshold : {top20_reach:.4f}")

test_pairs["target"] = (
    (test_pairs["prob"] >= 0.5) |
    (is_cold & (test_pairs["v_reach_pct"] >= top20_reach))
).astype(int)

print(f"  Predicted positives : {test_pairs['target'].sum():,} / {len(test_pairs):,}")
print(f"  Positive rate       : {test_pairs['target'].mean():.4f}")

# Build submission key: "CID X LOC_NUM X VENDOR"
test_pairs["CID X LOC_NUM X VENDOR"] = (
    test_pairs["customer_id"].astype(str)                   + " X " +
    test_pairs["location_number"].astype(int).astype(str)   + " X " +
    test_pairs["vendor_id"].astype(int).astype(str)
)

submission = (
    test_pairs[["CID X LOC_NUM X VENDOR", "target"]]
    .sort_values("CID X LOC_NUM X VENDOR")
    .reset_index(drop=True)
)

print(f"\n  Submission shape : {submission.shape}")
print("\n  Sample rows:")
print(submission.head(8).to_string(index=False))

out_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print(f"\n  Saved → {out_path}")

print("\n" + "=" * 60)
print(f"  DONE  |  ROC-AUC : {auc:.4f}")
print("=" * 60)
