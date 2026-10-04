import socket
import json
import joblib
import pandas as pd
import numpy as np
import os

# PREPARATION

print("Loading Hierarchical IDS models and artifacts...")

# Level 1 (Random Forest)
l1_path = os.path.join("level1_files")

model_l1 = joblib.load(os.path.join(l1_path, 'ids_model.joblib'))
scaler_l1 = joblib.load(os.path.join(l1_path, 'scaler_level1.joblib'))
label_encoders_l1 = joblib.load(os.path.join(l1_path, 'ids_mapping.joblib'))

with open(os.path.join(l1_path, 'level1_metadata.json'), 'r') as f:
    meta_l1 = json.load(f)

cat_cols_l1 = meta_l1['categorical_features']
common_cats_l1 = meta_l1['common_categories']
cols_to_scale_l1 = meta_l1['cols_to_scale']
final_features_l1 = meta_l1['final_features']
optimal_threshold_l1 = meta_l1['optimal_threshold']


# Level 2 (XGBoost)
l2_path = os.path.join("level2_files")

model_l2 = joblib.load(os.path.join(l2_path, 'xgb_level2_model.pkl'))
preprocessor_l2 = joblib.load(os.path.join(l2_path, 'preprocessor_level2.pkl'))
ordinal_encoder_l2 = joblib.load(os.path.join(l2_path, 'ordinal_encoder_level2.pkl'))
le_l2 = joblib.load(os.path.join(l2_path, 'attack_label_encoder.pkl'))

with open(os.path.join(l2_path, 'level2_metadata.json'), 'r') as f:
    meta_l2 = json.load(f)

common_cats_l2 = meta_l2['common_categories']
cat_features_l2 = meta_l2['categorical_features']
num_features_l2 = meta_l2['numerical_features_after_corr_filter']
bin_features_l2 = meta_l2['binary_features']
feature_order_l2 = num_features_l2 + cat_features_l2 + bin_features_l2

print("Models and artifacts loaded successfully.")

# SERVER START

def start_hybrid_ids_server(host='0.0.0.0', port=5000):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"\nHYBRID IDS active on port {port}. Waiting for traffic...")

    while True:
        conn, addr = server_socket.accept()
        try:
            data = conn.recv(8192).decode('utf-8')
            if not data: continue
            
            packet_raw = json.loads(data)
            df_packet = pd.DataFrame([packet_raw])

            # STEP 1: LEVEL 1 FILTERING
            df_l1 = df_packet.copy()
            
            # Rare categories handling and encoding (Level 1)
            for col in cat_cols_l1:
                if col in df_l1.columns:
                    top_elements = common_cats_l1[col]
                    df_l1[col] = df_l1[col].apply(lambda x: x if x in top_elements else "others")
                    df_l1[col] = label_encoders_l1[col].transform(df_l1[col].astype(str))

            # Ensure presence of all required final columns
            for col in final_features_l1:
                if col not in df_l1.columns:
                    df_l1[col] = 0

            # Numerical scaling (Level 1)
            df_l1[cols_to_scale_l1] = scaler_l1.transform(df_l1[cols_to_scale_l1])

            # Final ordering
            X_l1 = df_l1[final_features_l1]
            
            # Prediction with the threshold optimized by the ROC curve
            y_probs_l1 = model_l1.predict_proba(X_l1)[0, 1]
            verdict_l1 = 1 if y_probs_l1 >= optimal_threshold_l1 else 0

            if verdict_l1 == 0:
                print(f"[{addr[0]}] RELIABLE TRAFFIC")
            else:
                # STEP 2: ATTACK DETERMINATION
                df_l2 = df_packet.copy()
                
                # 'clean_df' style cleaning
                for col in ["sport", "dsport", "ct_ftp_cmd"]:
                    if col in df_l2.columns:
                        df_l2[col] = pd.to_numeric(df_l2[col], errors="coerce").fillna(0)

                # Ensure columns are present in the correct order
                for col in feature_order_l2:
                    if col not in df_l2.columns:
                        df_l2[col] = 0
                        
                X_l2 = df_l2[feature_order_l2].copy()

                # Rare categories grouping (Level 2)
                for col in cat_features_l2:
                    common = set(common_cats_l2[col])
                    X_l2[col] = X_l2[col].astype(str).apply(lambda x: x if x in common else "OTHER")

                # Safety for OrdinalEncoder: handle potential NaNs
                X_l2[cat_features_l2] = X_l2[cat_features_l2].fillna("MISSING")

                # Ordinal Encoding
                X_l2[cat_features_l2] = ordinal_encoder_l2.transform(X_l2[cat_features_l2])

                # Transformation via Preprocessor (Imputation + Scaling + OneHot)
                X_l2_proc = preprocessor_l2.transform(X_l2)
                
                # Prediction
                y_probs_l2 = model_l2.predict_proba(X_l2_proc)[0]
                pred_l2_idx = np.argmax(y_probs_l2)
                attack_name = le_l2.inverse_transform([pred_l2_idx])[0]
                
                # Confidence score
                conf = y_probs_l2[pred_l2_idx] * 100
                
                print(f"[{addr[0]}] ALERT: {attack_name.upper()} detected ({conf:.2f}%)")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    start_hybrid_ids_server()