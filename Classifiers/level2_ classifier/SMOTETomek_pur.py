import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score
)
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.combine import SMOTETomek  # <-- NOUVEL IMPORT ICI

# CONFIG
save_dir = "level2_classification/level_2_final"
os.makedirs(save_dir, exist_ok=True)

MODEL_TYPE = "xgb"   # "xgb" or "brf"
RARE_THRESHOLD = 0.002   # Garde les catégories qui apparaissent au moins à 0.2%
CORR_THRESHOLD = 0.50
USE_MANUAL_AMPLIFIERS = False  

# 1. LOAD DATA
train_df = pd.read_csv("UNSW_NB15_training-set.csv")
test_df  = pd.read_csv("UNSW_NB15_testing-set.csv")

print("=" * 60)
print("Original dataset shapes")
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

# 2. CLEAN DATA
def clean_df(df):
    df = df.copy()

    if "attack_cat" in df.columns:
        df["attack_cat"] = (
            df["attack_cat"]
            .astype(str)
            .str.strip()
            .replace("", "Normal")
            .replace("nan", "Normal")
            .fillna("Normal")
        )

    for col in ["sport", "dsport", "ct_ftp_cmd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

train_df = clean_df(train_df)
test_df  = clean_df(test_df)

# 3. KEEP ONLY ATTACK TRAFFIC (LEVEL 2)
train_l2 = train_df[train_df["label"] == 1].copy()
test_l2  = test_df[test_df["label"] == 1].copy()

print("\n" + "=" * 60)
print("Attack-only dataset shapes")
print("Train Level 2 shape:", train_l2.shape)
print("Test Level 2 shape :", test_l2.shape)

# 4. DEFINE TARGET AND FEATURES
target_col = "attack_cat"

y_train = train_l2[target_col].copy()
y_test  = test_l2[target_col].copy()

drop_cols = [target_col, "label"]
optional_drop = ["id", "srcip", "dstip", "stime", "ltime"]
drop_cols += [c for c in optional_drop if c in train_l2.columns]

X_train = train_l2.drop(columns=drop_cols).copy()
X_test  = test_l2.drop(columns=drop_cols).copy()

print("\nDropped columns:", drop_cols)
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

# 5. ENCODE TARGET
le_attack = LabelEncoder()
y_train_enc = le_attack.fit_transform(y_train)
y_test_enc  = le_attack.transform(y_test)

attack_cat_mapping = {
    cls_name: int(cls_id)
    for cls_name, cls_id in zip(
        le_attack.classes_,
        le_attack.transform(le_attack.classes_)
    )
}

print("\n" + "=" * 60)
print("Attack category mapping")
print(attack_cat_mapping)

# 6. FEATURE TYPING
candidate_categorical = ["proto", "service", "state"]
categorical_features = [c for c in candidate_categorical if c in X_train.columns]

binary_features = [
    col for col in X_train.columns
    if col not in categorical_features
    and set(X_train[col].dropna().unique()).issubset({0, 1})
]

numerical_features = [
    col for col in X_train.columns
    if col not in categorical_features + binary_features
]

print("\n" + "=" * 60)
print("Feature groups")
print("Categorical features:", categorical_features)
print("Binary features     :", binary_features)
print("Numerical features  :", len(numerical_features))

# 7. RARE CATEGORY GROUPING
common_categories = {}

for col in categorical_features:
    freq = X_train[col].astype(str).value_counts(normalize=True)
    common = set(freq[freq > RARE_THRESHOLD].index)
    common_categories[col] = sorted(list(common))

    X_train[col] = X_train[col].astype(str).apply(
        lambda x, c=common: x if x in c else "OTHER"
    )
    X_test[col] = X_test[col].astype(str).apply(
        lambda x, c=common: x if x in c else "OTHER"
    )

print("\n" + "=" * 60)
print("Common categories after rare grouping")
for col, vals in common_categories.items():
    print(f"{col}: {vals}")

# 8. DROP HIGHLY CORRELATED NUMERICAL FEATURES (SMART FILTERING)
protected_features = {
    "sbytes", "dbytes", "spkts", "dpkts", 
    "sload", "dload", "sttl", "dttl", 
    "smean", "dmean", "dur"
}

corr_matrix = X_train[numerical_features].corr().abs()
num_to_drop = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] > CORR_THRESHOLD:
            col_name_i = corr_matrix.columns[i]
            col_name_j = corr_matrix.columns[j]
            
            if col_name_i in num_to_drop or col_name_j in num_to_drop:
                continue
                
            if col_name_i in protected_features and col_name_j not in protected_features:
                num_to_drop.add(col_name_j)
            elif col_name_j in protected_features and col_name_i not in protected_features:
                num_to_drop.add(col_name_i)
            else:
                num_to_drop.add(col_name_i)

num_to_drop = list(num_to_drop)
numerical_features_filtered = [
    col for col in numerical_features
    if col not in num_to_drop
]

print("\n" + "=" * 60)
print("Highly correlated numerical features dropped (SMART FILTER):")
print(num_to_drop)
print("Count dropped:", len(num_to_drop))
print("Remaining numerical features:", len(numerical_features_filtered))

# 9. PREPROCESSOR
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features_filtered),
        ("cat", categorical_transformer, categorical_features),
        ("bin", "passthrough", binary_features)
    ],
    remainder="drop"
)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

print("\n" + "=" * 60)
print("Processed shapes")
print("Processed train shape:", X_train_proc.shape)
print("Processed test shape :", X_test_proc.shape)


# 11. TRAIN MODEL
if MODEL_TYPE == "xgb":
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_proc,
        y_train_enc,
        test_size=0.1,
        random_state=42,
        stratify=y_train_enc
    )

    print("\n" + "=" * 60)
    print("Train/Val split BEFORE SMOTETomek")
    print(f"Train: {X_tr.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test_proc.shape[0]}")

    # --- APPLICATION DE SMOTETomek ---
    print("\nApplying SMOTETomek to training data (this may take a few minutes)...")
    smt = SMOTETomek(random_state=42)
    X_tr_smote, y_tr_smote = smt.fit_resample(X_tr, y_tr)
    
    print(f"Train shape AFTER SMOTETomek: {X_tr_smote.shape[0]}")
    # ---------------------------------

    n_classes = len(le_attack.classes_)

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.2,
        eval_metric="mlogloss",
        early_stopping_rounds=40,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    print("\nTraining XGBoost with early stopping on SMOTETomek data...")
    clf.fit(
        X_tr_smote,
        y_tr_smote,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    print(f"\nBest iteration: {clf.best_iteration}")
    print(f"Best val mlogloss: {clf.best_score:.5f}")

elif MODEL_TYPE == "brf":
    n_classes = len(le_attack.classes_)
    clf = BalancedRandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        replacement=True,
        sampling_strategy="all"
    )

    print("\nTraining Balanced Random Forest...")
    clf.fit(X_train_proc, y_train_enc)

else:
    raise ValueError("MODEL_TYPE must be 'xgb' or 'brf'")

# 12. EVALUATION
y_pred = clf.predict(X_test_proc)

balanced_acc = balanced_accuracy_score(y_test_enc, y_pred)
macro_f1 = f1_score(y_test_enc, y_pred, average="macro")
weighted_f1 = f1_score(y_test_enc, y_pred, average="weighted")

print("\n" + "=" * 60)
print(f"LEVEL 2 FINAL RESULTS ({MODEL_TYPE.upper()})")
print(f"Balanced Accuracy : {balanced_acc:.4f}")
print(f"Macro F1-score    : {macro_f1:.4f}")
print(f"Weighted F1-score : {weighted_f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(
    y_test_enc,
    y_pred,
    target_names=le_attack.classes_,
    digits=4
))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test_enc, y_pred))

# 13. FEATURE IMPORTANCE (IF AVAILABLE)
if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop 20 most important features:")
    for i in range(min(20, len(feature_names))):
        print(f"{i+1:2d}. {feature_names[indices[i]]:<40} {importances[indices[i]]:.4f}")

# 14. SAVE EVERYTHING
joblib.dump(preprocessor, os.path.join(save_dir, "preprocessor_level2.pkl"))
joblib.dump(le_attack, os.path.join(save_dir, "attack_label_encoder.pkl"))
joblib.dump(clf, os.path.join(save_dir, f"{MODEL_TYPE}_level2_model.pkl"))

metadata = {
    "model_type": MODEL_TYPE,
    "target_column": target_col,
    "dropped_columns": drop_cols,
    "categorical_features": categorical_features,
    "binary_features": binary_features,
    "numerical_features_before_corr_filter": numerical_features,
    "numerical_features_after_corr_filter": numerical_features_filtered,
    "high_corr_features_dropped": num_to_drop,
    "rare_threshold": RARE_THRESHOLD,
    "correlation_threshold": CORR_THRESHOLD,
    "common_categories": common_categories,
    "attack_category_mapping": attack_cat_mapping,
    "processed_feature_names": feature_names.tolist()
}

if MODEL_TYPE == "xgb" and hasattr(clf, "best_iteration"):
    metadata["best_iteration"] = int(clf.best_iteration)

with open(os.path.join(save_dir, "level2_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("\n" + "=" * 60)
print(f"All artifacts saved in: {save_dir}")