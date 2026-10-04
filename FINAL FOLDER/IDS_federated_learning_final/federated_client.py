import flwr as fl
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import sys


# BALANCING DATA
def balance_local_data(X, y):
    X = np.array(X)
    y = np.array(y)

    normal_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]

    # Safety check
    if len(normal_idx) == 0 or len(attack_idx) == 0:
        print("Warning: one class is missing, skipping balancing.")
        return X, y

    if len(normal_idx) < len(attack_idx):
        extra_normal_idx = np.random.choice(
            normal_idx,
            size=len(attack_idx) - len(normal_idx),
            replace=True
        )
        balanced_idx = np.concatenate([normal_idx, attack_idx, extra_normal_idx])
    else:
        extra_attack_idx = np.random.choice(
            attack_idx,
            size=len(normal_idx) - len(attack_idx),
            replace=True
        )
        balanced_idx = np.concatenate([normal_idx, attack_idx, extra_attack_idx])

    np.random.shuffle(balanced_idx)

    return X[balanced_idx], y[balanced_idx]


# PREPROCESSING
def load_local_data(partition):
    df = pd.read_csv("UNSW_NB15_training-set.csv").drop(
        ["id", "attack_cat"],
        axis=1,
        errors="ignore"
    )

    # Encode categorical columns
    for col in df.select_dtypes(include=["object", "str"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Shuffle before splitting
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df.drop("label", axis=1)
    y = df["label"]

    # Split into 3 clients
    X_splits = np.array_split(X, 3)
    y_splits = np.array_split(y, 3)

    X_local = X_splits[partition]
    y_local = y_splits[partition]

    # Local train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_local,
        y_local,
        test_size=0.2,
        random_state=42,
        stratify=y_local
    )

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Balance only training data
    X_train, y_train = balance_local_data(X_train, y_train)

    return X_train, y_train, X_val, y_val


# GET CLIENT PARTITION
partition_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

x_train, y_train, x_val, y_val = load_local_data(partition_id)

print(f"Client train size {partition_id}: {x_train.shape}")
print(f"Client validation size {partition_id}: {x_val.shape}")


# FLOWER CLIENT
class IDSClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(10, 10),
            max_iter=1,
            random_state=42
        )

        # Initialize weights
        self.model.partial_fit(x_train[:1], y_train[:1], classes=[0, 1])

    def get_parameters(self, config):
        return self.model.coefs_ + self.model.intercepts_

    def set_parameters(self, parameters):
        half = len(parameters) // 2
        self.model.coefs_ = parameters[:half]
        self.model.intercepts_ = parameters[half:]

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.model.partial_fit(x_train, y_train)

        print(f"Local training done - Client {partition_id}")

        return self.get_parameters(config), len(x_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        y_pred = self.model.predict(x_val)
        y_proba = self.model.predict_proba(x_val)

        accuracy = accuracy_score(y_val, y_pred)
        loss = log_loss(y_val, y_proba, labels=[0, 1])

        print(
            f"Client {partition_id} evaluation - "
            f"Accuracy: {accuracy:.4f}, Loss: {loss:.4f}"
        )

        return loss, len(x_val), {"accuracy": accuracy}


print(f"[CLIENT] IDS node started (Partition {partition_id})...")

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=IDSClient()
)
