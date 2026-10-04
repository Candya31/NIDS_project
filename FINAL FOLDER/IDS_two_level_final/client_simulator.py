import socket
import json
import time
import pandas as pd

test_df = pd.read_csv('UNSW_NB15_testing-set.csv')

def envoyer_trafic(ip_serveur='127.0.0.1', port=5000, nb_paquets=20):
    print(f"Starting traffic simulation to IDS ({ip_serveur}:{port})...")
    
    samples = test_df.sample(nb_paquets)
    
    for i, (_, row) in enumerate(samples.iterrows()):
        paquet = row.to_dict()
        message = json.dumps(paquet).encode('utf-8')
        
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((ip_serveur, port))
            s.sendall(message)
            s.close()
            print(f"Packet {i+1} Sent successfully | Type : {row.get('attack_cat', 'Normal')}")
        except Exception as e:
            print(f"Packet {i+1} Failed to connect to IDS : {e}")
            
        time.sleep(1) # Delay between sends

if __name__ == "__main__":
    envoyer_trafic(ip_serveur='127.0.0.1', nb_paquets=20)