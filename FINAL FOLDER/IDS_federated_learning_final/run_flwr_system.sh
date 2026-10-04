#!/bin/bash

echo " Activation environnement..."
source flwr_env/bin/activate

echo " Lancement serveur Flower..."

# Lancer serveur en arrière-plan
python federated_server.py &
SERVER_PID=$!

sleep 5

echo " Lancement clients..."

# Client 0
python federated_client.py 0 &
CLIENT1=$!

# Client 1
python federated_client.py 1 &
CLIENT2=$!

# Client 2
python federated_client.py 2 &
CLIENT3=$!

echo "✅ Système fédéré lancé"
echo "Server PID: $SERVER_PID"
echo "Clients: $CLIENT1 $CLIENT2 $CLIENT3"

wait
