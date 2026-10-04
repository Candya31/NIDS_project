#!/bin/bash

echo "-------------------------------------------------------"
echo "🛠️  Installation des dépendances pour l'IDS UNSW-NB15"
echo "-------------------------------------------------------"

# Mise à jour des dépôts
sudo apt-get update

# Installation des dépendances système pour Python et les calculs
echo "📦 Installation des paquets système..."
sudo apt-get install -y python3-pip python3-dev libatlas-base-dev gfortran

# Installation des bibliothèques Python
echo "🐍 Installation des bibliothèques Python via Pip..."
pip3 install --upgrade pip
pip3 install pandas numpy scikit-learn joblib matplotlib seaborn

echo "-------------------------------------------------------"
echo "✅ Installation terminée ! Vous pouvez lancer l'IDS."
echo "-------------------------------------------------------"