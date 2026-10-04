import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    StandardScaler,
    OrdinalEncoder
)
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTENC


# CONFIG
SAVE_DIR = "level2_files" 
os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_PATH = "UNSW_NB15_training-set.csv"
TEST_PATH = "UNSW_NB15_testing-set.csv"

TARGET_COL = "attack_cat"

RARE_THRESHOLD = 0.002
CORR_THRESHOLD = 0.90

RANDOM_STATE = 42

PROTECTED_FEATURES = {
    "sbytes", "dbytes", "spkts", "dpkts",
    "sload", "dload", "sttl", "dttl",
    "smean", "dmean", "dur"
}
# UTILS

def print_section(title):
    print("\n" + "=" * 60)
    print(title)


def clean_df(df):
    df = df.copy()

    if TARGET_COL in df.columns:
        df[TARGET_COL] = (
            df[TARGET_COL]
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


def get_feature_groups(X):
    candidate_categorical = ["proto", "service", "state"]
    categorical_features = [
        col for col in candidate_categorical
        if col in X.columns
    ]

    binary_features = [
        col for col in X.columns
        if col not in categorical_features
        and set(X[col].dropna().unique()).issubset({0, 1})
    ]

    numerical_features = [
        col for col in X.columns
        if col not in categorical_features + binary_features
    ]

    return categorical_features, binary_features, numerical_features


def group_rare_categories(X_train, X_val, X_test, categorical_features):
    common_categories = {}

    for col in categorical_features:
        freq = X_train[col].astype(str).value_counts(normalize=True)
        common = set(freq[freq > RARE_THRESHOLD].index)

        common_categories[col] = sorted(list(common))

        X_train[col] = X_train[col].astype(str).apply(
            lambda x: x if x in common else "OTHER"
        )
        X_val[col] = X_val[col].astype(str).apply(
            lambda x: x if x in common else "OTHER"
        )
        X_test[col] = X_test[col].astype(str).apply(
            lambda x: x if x in common else "OTHER"
        )

    return X_train, X_val, X_test, common_categories


def drop_correlated_features(X_train, numerical_features):
    corr_matrix = X_train[numerical_features].corr().abs()
    features_to_drop = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] <= CORR_THRESHOLD:
                continue

            feature_i = corr_matrix.columns[i]
            feature_j = corr_matrix.columns[j]

            if feature_i in features_to_drop or feature_j in features_to_drop:
                continue

            if feature_i in PROTECTED_FEATURES and feature_j not in PROTECTED_FEATURES:
                features_to_drop.add(feature_j)
            elif feature_j in PROTECTED_FEATURES and feature_i not in PROTECTED_FEATURES:
                features_to_drop.add(feature_i)
            else:
                features_to_drop.add(feature_i)

    features_to_drop = sorted(list(features_to_drop))

    numerical_features_filtered = [
        col for col in numerical_features
        if col not in features_to_drop
    ]

    return numerical_features_filtered, features_to_drop


def build_preprocessor(numerical_features, categorical_features, binary_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
            ("bin", binary_transformer, binary_features)
        ],
        remainder="drop"
    )

    return preprocessor


def fill_missing_values(X_train, X_val, X_test, numerical_features, categorical_features, binary_features):
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    for col in numerical_features:
        median_value = X_train[col].median()

        X_train[col] = X_train[col].fillna(median_value)
        X_val[col] = X_val[col].fillna(median_value)
        X_test[col] = X_test[col].fillna(median_value)

    for col in categorical_features:
        X_train[col] = X_train[col].fillna("MISSING").astype(str)
        X_val[col] = X_val[col].fillna("MISSING").astype(str)
        X_test[col] = X_test[col].fillna("MISSING").astype(str)

    for col in binary_features:
        mode_value = X_train[col].mode()
        fill_value = mode_value.iloc[0] if not mode_value.empty else 0

        X_train[col] = X_train[col].fillna(fill_value)
        X_val[col] = X_val[col].fillna(fill_value)
        X_test[col] = X_test[col].fillna(fill_value)

    return X_train, X_val, X_test


def apply_smote_tomek(X_train, y_train, feature_order, categorical_features):
    categorical_indices = [
        feature_order.index(col)
        for col in categorical_features
    ]

    smote_nc = SMOTENC(
        categorical_features=categorical_indices,
        random_state=RANDOM_STATE
    )

    smote_tomek = SMOTETomek(
        smote=smote_nc,
        random_state=RANDOM_STATE
    )

    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

    if not isinstance(X_resampled, pd.DataFrame):
        X_resampled = pd.DataFrame(
            X_resampled,
            columns=feature_order
        )

    return X_resampled, y_resampled


def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)

    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print_section("LEVEL 2 FINAL RESULTS — XGBOOST")
    print(f"Balanced Accuracy : {balanced_acc:.4f}")
    print(f"Macro F1-score    : {macro_f1:.4f}")
    print(f"Weighted F1-score : {weighted_f1:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4
    ))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    return y_pred, balanced_acc, macro_f1, weighted_f1


def save_artifacts(
    preprocessor,
    ordinal_encoder,
    label_encoder,
    model,
    metadata
):
    joblib.dump(preprocessor, os.path.join(SAVE_DIR, "preprocessor_level2.pkl"))
    joblib.dump(ordinal_encoder, os.path.join(SAVE_DIR, "ordinal_encoder_level2.pkl")) # Conservé pour intégrité
    joblib.dump(label_encoder, os.path.join(SAVE_DIR, "attack_label_encoder.pkl"))
    joblib.dump(model, os.path.join(SAVE_DIR, "xgb_level2_model.pkl"))

    with open(os.path.join(SAVE_DIR, "level2_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print_section("Artifacts saved")
    print(f"All artifacts saved in: {SAVE_DIR}")

# 1. LOAD DATA
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print_section("Original dataset shapes")
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

# 2. CLEAN DATA
train_df = clean_df(train_df)
test_df = clean_df(test_df)

# 3. KEEP ONLY ATTACK TRAFFIC
train_l2 = train_df[train_df["label"] == 1].copy()
test_l2 = test_df[test_df["label"] == 1].copy()

print_section("Attack-only dataset shapes")
print("Train Level 2 shape:", train_l2.shape)
print("Test Level 2 shape :", test_l2.shape)

# 4. DEFINE FEATURES AND TARGET
y_train = train_l2[TARGET_COL].copy()
y_test = test_l2[TARGET_COL].copy()

drop_cols = [TARGET_COL, "label"]
optional_drop_cols = ["id", "srcip", "dstip", "stime", "ltime"]

drop_cols += [
    col for col in optional_drop_cols
    if col in train_l2.columns
]

X_train = train_l2.drop(columns=drop_cols, errors="ignore").copy()
X_test = test_l2.drop(columns=drop_cols, errors="ignore").copy()

print_section("Features and target")
print("Dropped columns:", drop_cols)
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

# 5. ENCODE TARGET
label_encoder = LabelEncoder()

y_train_enc = label_encoder.fit_transform(y_train)

unknown_classes = set(y_test) - set(label_encoder.classes_)
if unknown_classes:
    raise ValueError(f"Unknown attack categories in test set: {unknown_classes}")

y_test_enc = label_encoder.transform(y_test)

attack_cat_mapping = {
    class_name: int(class_id)
    for class_name, class_id in zip(
        label_encoder.classes_,
        label_encoder.transform(label_encoder.classes_)
    )
}

print_section("Attack category mapping")
print(attack_cat_mapping)


# 6. TRAIN / VALIDATION SPLIT
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
    X_train,
    y_train_enc,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=y_train_enc
)

print_section("Train / Validation split")
print("Raw train shape:", X_train_raw.shape)
print("Raw val shape  :", X_val_raw.shape)
print("Raw test shape :", X_test.shape)

# 7. FEATURE GROUPS
categorical_features, binary_features, numerical_features = get_feature_groups(X_train_raw)

print_section("Feature groups")
print("Categorical features:", categorical_features)
print("Binary features    :", binary_features)
print("Numerical features  :", len(numerical_features))

# 8. RARE CATEGORY GROUPING
X_train_raw, X_val_raw, X_test, common_categories = group_rare_categories(
    X_train_raw,
    X_val_raw,
    X_test,
    categorical_features
)

print_section("Common categories after rare grouping")
for col, values in common_categories.items():
    print(f"{col}: {values}")

# 9. DROP HIGHLY CORRELATED NUMERICAL FEATURES

numerical_features_filtered, high_corr_features_dropped = drop_correlated_features(
    X_train_raw,
    numerical_features
)

print_section("Correlation filtering")
print("Dropped numerical features:", high_corr_features_dropped)
print("Count dropped:", len(high_corr_features_dropped))
print("Remaining numerical features:", len(numerical_features_filtered))

# 10. PREPARE DATA FOR SMOTENC
feature_order = (
    numerical_features_filtered
    + categorical_features
    + binary_features
)

X_train_smote = X_train_raw[feature_order].copy()
X_val_model = X_val_raw[feature_order].copy()
X_test_model = X_test[feature_order].copy()

X_train_smote, X_val_model, X_test_model = fill_missing_values(
    X_train_smote,
    X_val_model,
    X_test_model,
    numerical_features_filtered,
    categorical_features,
    binary_features
)

# 11. ORDINAL ENCODING BEFORE SMOTENC
ordinal_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_train_smote[categorical_features] = ordinal_encoder.fit_transform(
    X_train_smote[categorical_features]
)

X_val_model[categorical_features] = ordinal_encoder.transform(
    X_val_model[categorical_features]
)

X_test_model[categorical_features] = ordinal_encoder.transform(
    X_test_model[categorical_features]
)

X_train_smote = X_train_smote.astype(float)
X_val_model = X_val_model.astype(float)
X_test_model = X_test_model.astype(float)

# 12. APPLY SMOTETOMEK WITH SMOTENC
print_section("Applying SMOTETomek with SMOTENC")
print("Train shape BEFORE SMOTETomek:", X_train_smote.shape)

X_train_resampled, y_train_resampled = apply_smote_tomek(
    X_train_smote,
    y_train_raw,
    feature_order,
    categorical_features
)

print("Train shape AFTER SMOTETomek :", X_train_resampled.shape)

# 13. FINAL PREPROCESSING
preprocessor = build_preprocessor(
    numerical_features_filtered,
    categorical_features,
    binary_features
)

print_section("Fitting final preprocessor")

X_tr = preprocessor.fit_transform(X_train_resampled)
X_val = preprocessor.transform(X_val_model)
X_test_proc = preprocessor.transform(X_test_model)

feature_names = preprocessor.get_feature_names_out()

print("Processed train shape:", X_tr.shape)
print("Processed val shape  :", X_val.shape)
print("Processed test shape :", X_test_proc.shape)

# 14. TRAIN XGBOOST
n_classes = len(label_encoder.classes_)

xgb_model = XGBClassifier(
    objective="multi:softprob",
    num_class=n_classes,
    n_estimators=2000,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.2,
    eval_metric="mlogloss",
    early_stopping_rounds=40,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist"
)

print_section("Training XGBoost")

xgb_model.fit(
    X_tr,
    y_train_resampled,
    eval_set=[(X_val, y_val_raw)],
    verbose=100
)

print(f"\nBest iteration: {xgb_model.best_iteration}")
print(f"Best val mlogloss: {xgb_model.best_score:.5f}")

# 15. EVALUATION
y_pred, balanced_acc, macro_f1, weighted_f1 = evaluate_model(
    xgb_model,
    X_test_proc,
    y_test_enc,
    label_encoder
)

# 16. FEATURE IMPORTANCE
if hasattr(xgb_model, "feature_importances_"):
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print_section("Top 20 most important features")

    for rank, idx in enumerate(indices[:20], start=1):
        print(
            f"{rank:2d}. "
            f"{feature_names[idx]:<40} "
            f"{importances[idx]:.4f}"
        )

# 17. SAVE ARTIFACTS
metadata = {
    "model": "XGBoost",
    "target_column": TARGET_COL,
    "dropped_columns": drop_cols,

    "categorical_features": categorical_features,
    "binary_features": binary_features,
    "numerical_features_before_corr_filter": numerical_features,
    "numerical_features_after_corr_filter": numerical_features_filtered,

    "high_corr_features_dropped": high_corr_features_dropped,
    "rare_threshold": RARE_THRESHOLD,
    "correlation_threshold": CORR_THRESHOLD,
    "common_categories": common_categories,

    "attack_category_mapping": attack_cat_mapping,
    "processed_feature_names": feature_names.tolist(),

    "smote_method": "SMOTETomek with SMOTENC before one-hot encoding",

    "best_iteration": int(xgb_model.best_iteration),
    "best_validation_mlogloss": float(xgb_model.best_score),

    "test_balanced_accuracy": float(balanced_acc),
    "test_macro_f1": float(macro_f1),
    "test_weighted_f1": float(weighted_f1)
}

save_artifacts(
    preprocessor=preprocessor,
    ordinal_encoder=ordinal_encoder,
    label_encoder=label_encoder,
    model=xgb_model,
    metadata=metadata
)