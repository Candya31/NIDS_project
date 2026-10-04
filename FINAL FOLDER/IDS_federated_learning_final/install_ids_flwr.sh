#!/bin/bash

echo "Mise à jour système..."
sudo apt update && sudo apt upgrade -y

echo "Installation dépendances système..."
sudo apt install -y python3-venv python3-pip python3-dev build-essential

echo "Création environnement virtuel..."
python3 -m venv flwr_env

echo "Activation venv..."
source flwr_env/bin/activate

echo "Upgrade pip..."
pip install --upgrade pip

echo "Installation libs Python..."
pip install flwr pandas numpy scikit-learn joblib

echo "✅ Installation terminée"
echo "👉 Pour activer : source flwr_env/bin/activate"