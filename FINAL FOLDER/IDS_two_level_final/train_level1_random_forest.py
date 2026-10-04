import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score


# Constantes utiles
N_ESTIMATORS = 200
MAX_DEPTH = 15

# Chargement des données

train_df = pd.read_csv("UNSW_NB15_training-set.csv")
test_df = pd.read_csv("UNSW_NB15_testing-set.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
#print(train_df.columns.tolist())

# Suppression des colonnes inutiles 

train_df = train_df.drop(columns=["id", "attack_cat"])
test_df = test_df.drop(columns=["id", "attack_cat"])

y_train = train_df["label"]
x_train = train_df.drop(columns=["label"])

y_test = test_df["label"]
x_test = test_df.drop(columns=["label"])


# Encodage des colonnes

label_encoders = {}
common_categories = {} # pour utiliser lors de la sauvgarde des résultats
for col in ["proto", "service", "state"]:
    # Top 10 catégories du train
    top_elements = train_df[col].value_counts().nlargest(10).index.tolist()
    allowed_categories = top_elements + ["others"]
    common_categories[col] = top_elements

    le = LabelEncoder()
    le.fit(allowed_categories)

    # Remplacer les valeurs hors-top par "others"
    x_train[col] = x_train[col].apply(lambda x: x if x in top_elements else "others")
    x_test[col]  = x_test[col].apply(lambda x: x if x in top_elements else "others")

    # Appliquer l'encodage
    x_train[col] = le.transform(x_train[col].astype(str))
    x_test[col] = le.transform(x_test[col].astype(str))

    label_encoders[col] = le

print("Encodage terminé. Nombre de colonnes :", x_train.shape[1])

# Suppression des features fortement corrélées

# Exclure "proto", "service" et "state" pour ne pas supprimer à tort
cat_cols = ["proto", "service", "state"]
num_cols = [c for c in x_train.select_dtypes(include=[np.number]).columns if c not in cat_cols]

corr_matrix = x_train[num_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

threshold = 0.85
CORR_THRESHOLD = threshold  # pour la sauvgarde des résultats 

to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

print(f"Features supprimées : {to_drop}")
x_train = x_train.drop(columns=to_drop)
x_test = x_test.drop(columns=to_drop)
final_features = x_train.columns.tolist() # pour la sauvgarde des résultats 
print("Nombre de colonnes finales :", x_train.shape[1])


# Scaling 

scaler = StandardScaler()
cols_to_scale = [c for c in x_train.columns if c not in ["proto", "service", "state"]]

x_train[cols_to_scale] = scaler.fit_transform(x_train[cols_to_scale])
x_test[cols_to_scale] = scaler.transform(x_test[cols_to_scale])

print("Scaling terminé sur les colonnes numériques.")
print("Colonnes scalées :", cols_to_scale)



# Modèle RandomForest
# Pour le raspberry réduire le nombre d'arbre à 50 n_jobs=-1 pour utiliser tous les coeurs CPU dispo. 
# max_depth= 15 pour éviter que les arbres grandissent indéfiniment 
rf = RandomForestClassifier(n_estimators = N_ESTIMATORS, max_depth = MAX_DEPTH, class_weight = "balanced", random_state = 42, n_jobs = -1)
rf.fit(x_train, y_train)

# Affichage des features les plus importantes 

importances = pd.Series(rf.feature_importances_, index=x_train.columns)
importances.nlargest(15).plot(kind="barh", title="Top 15 features importantes")
plt.tight_layout()
plt.show()


y_probs = rf.predict_proba(x_test)[:, 1]
# Calculer la courbe ROC pour choisir le seuil le plus adapté
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Trouver le seuil optimal (meilleur équilibre TPR/FPR) true positive rate et false positive rate 

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Seuil optimal selon ROC : {optimal_threshold:.3f}")

# Appliquer CE seuil
y_pred = (y_probs >= optimal_threshold).astype(int)

# Tracer la courbe
plt.figure(figsize=(8, 6))
auc_score = roc_auc_score(y_test, y_probs)  # ← ajouter avant
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], 
            color="red", label=f"Seuil optimal = {optimal_threshold:.3f}")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("FPR (Faux Positifs)")
plt.ylabel("TPR (Vrais Positifs / Recall)")
plt.title("ROC Curve - Level 1")
plt.legend()
plt.tight_layout()
plt.show()


print("Rapport de Classification (Level 1 - Binaire) :")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
disp.plot(cmap="Blues")
plt.title("Matrice de Confusion : Normal vs Attack")
plt.tight_layout()
plt.show()

SAVE_DIR = "level1_files"
os.makedirs(SAVE_DIR, exist_ok=True)

# Fichiers requis selon la spécification
joblib.dump(rf, os.path.join(SAVE_DIR, "ids_model.joblib"))
joblib.dump(label_encoders, os.path.join(SAVE_DIR, "ids_mapping.joblib"))
joblib.dump(final_features, os.path.join(SAVE_DIR, "ids_columns.joblib"))
joblib.dump(cat_cols, os.path.join(SAVE_DIR, "ids_cat_features.joblib"))

# Conservation du scaler (indispensable pour le pré-traitement futur même si non listé explicitement)
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler_level1.joblib"))
 
metadata = {
    "model_type":          "RandomForestClassifier",
    "n_estimators":        N_ESTIMATORS,
    "max_depth":           MAX_DEPTH,
    "class_weight":        "balanced",
    "target_column":       "label",
    "label_mapping":       {"0": "Normal", "1": "Attack"},
    "dropped_columns":     ["id", "attack_cat"],
    "categorical_features": cat_cols,
    "common_categories":   common_categories,
    "cols_to_scale":       cols_to_scale,
    "high_corr_dropped":   to_drop,
    "correlation_threshold": CORR_THRESHOLD,
    "final_features":      final_features,
    "optimal_threshold":   optimal_threshold,
    "auc_score":           auc_score,
}
 
with open(os.path.join(SAVE_DIR, "level1_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

