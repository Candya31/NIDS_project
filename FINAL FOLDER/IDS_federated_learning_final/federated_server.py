import flwr as fl
import numpy as np
import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import time

warnings.filterwarnings('ignore',category=UserWarning)

# INIT CONFIG 
NUM_FEATURES = 42 

def get_initial_model():
    model = MLPClassifier(hidden_layer_sizes=(10, 10))
    # Initialisation
    model.partial_fit(np.zeros((1, NUM_FEATURES)), np.array([0]), classes=[0, 1])
    return model


def evaluate_final_model(parameters):
    print("\n" + "="*50)
    print(" EVALUATING FINAL GLOBAL MODEL")
    print("="*50)
    
    # Données de test pour l'évaluation finale
    test_df = pd.read_csv('UNSW_NB15_testing-set.csv').drop(['id', 'attack_cat'], axis=1, errors='ignore')
    
    # Prétraitement identique au client
    for col in test_df.select_dtypes(include=['object']).columns:
        test_df[col] = LabelEncoder().fit_transform(test_df[col].astype(str))
    
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']
    X_test = StandardScaler().fit_transform(X_test)

    # Créer le modèle et injecter les poids appris
    final_model = get_initial_model()
    half = len(parameters) // 2
    final_model.coefs_ = parameters[:half]
    final_model.intercepts_ = parameters[half:]

    # Prédictions avec seuil personnalisé
    THRESHOLD = 0.70
    y_proba = final_model.predict_proba(X_test)[:, 1] 
    y_pred = (y_proba >= THRESHOLD).astype(int)
    
    # Métriques

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("\nTableau de bord des performances :")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attaque"]))

    # Sauvegarde physique
    joblib.dump(final_model, 'federated_ids_model.joblib')
    print("\n Global model saved : 'federated_ids_model.joblib'")


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Appel à la méthode parente pour faire la moyenne (FedAvg)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Si c'est le dernier round (ici round 5)
            if server_round == 5:
                print(f"\n Round {server_round} done. Extracting final model...")
                # Convertir les paramètres Flower en tableaux NumPy
                params_list = fl.common.parameters_to_ndarrays(aggregated_parameters)
                evaluate_final_model(params_list)
        
        return aggregated_parameters, aggregated_metrics

# START
model_init = get_initial_model()
initial_parameters = fl.common.ndarrays_to_parameters(model_init.coefs_ + model_init.intercepts_)

strategy = SaveModelStrategy(
    min_fit_clients=3,
    min_available_clients=3,
    initial_parameters=initial_parameters,
)

print(" [SERVEUR] federated system ready. Waiting nodes...")
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)

time.sleep(3600)