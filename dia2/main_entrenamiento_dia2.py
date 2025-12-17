import numpy as np
import sqlite3
import pandas as pd
from red_neuronal_completa import create_regression_network, create_classification_network
from metricas_evaluacion import RegressionMetrics, ClassificationMetrics, FeatureImportanceAnalyzer, plot_confusion_matrix
from validacion_cruzada import KFoldCV
from utils import StandardScaler

def ejecutar_entrenamiento_completo():
    # 1. Carga de datos unificada
    conn = sqlite3.connect('cleandataset.sqlite')
    df = pd.read_sql_query("SELECT * FROM player_attributes_dataset", conn)
    conn.close()

    # Red 1: 20 Atributos (Rúbrica 2.2.1)
    features_20 = ['crossing', 'finishing', 'short_passing', 'dribbling', 'volleys',
                   'acceleration', 'sprint_speed', 'agility', 'reactions', 'stamina',
                   'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
                   'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle']
    
    # Red 2: 15 Atributos (Rúbrica 2.2.1)
    features_15 = features_20[:15]

    X_r = df[features_20].values
    y_r = df['potential'].values
    
    # Perfil: 7 posiciones (Rúbrica 2.2.1)
    # Generamos perfil basado en scores técnicos existentes en tu tabla
    df['perfil'] = (df['technical_score'] // 14).astype(int).clip(0, 6)
    y_c = df['perfil'].values

    print("\n" + "="*60)
    print("SISTEMA INTEGRADO - ENTRENAMIENTO DÍA 2")
    print("="*60)

    # --- RED 1: REGRESIÓN ---
    print("\n>>> RED 1: REGRESIÓN (Arquitectura: 20-256-128-64-1)")
    kf = KFoldCV(n_splits=5)
    reg_metrics_list = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_r)):
        # Preparación de Folds con dimensiones correctas
        X_train, X_test = X_r[train_idx], X_r[test_idx]
        y_train = y_r[train_idx].reshape(-1, 1) # (N_train, 1)
        y_test = y_r[test_idx].reshape(-1, 1)   # (N_test, 1)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_r = create_regression_network(
        input_dim=20, 
        lr=0.001,      # Un LR más bajo ayuda a no saltarse el mínimo global
        l2_lambda=0.005, # Este es tu LAMBDA (Regularización L2)
        dropout_rate=0.2 # Este es tu DROPOUT (Apaga neuronas aleatoriamente)
    )
        
        # FIT CORREGIDO: X_val y y_val deben ser los conjuntos de TEST del fold
        model_r.fit(
        X_train_scaled, y_train, 
        X_val=X_test_scaled, y_val=y_test, 
        epochs=100,      # Aumentamos para dar tiempo a aprender
        batch_size=64,   # Tamaño ideal para estabilidad
        verbose=True     # Para que veas cómo baja el Loss en cada época
    )

        preds = model_r.predict(X_test_scaled)
        m = RegressionMetrics.compute_all(y_r[test_idx], preds, verbose=False)
        reg_metrics_list.append(m['R²'])
        print(f"Fold {fold+1} - R²: {m['R²']:.4f} | RMSE: {m['RMSE']:.4f}")

    model_r.save_weights('red1_potencial.pkl')

    # --- RED 2: CLASIFICACIÓN ---
    print("\n>>> RED 2: CLASIFICACIÓN (Arquitectura: 15-256-128-7)")
    n_classes = 7
    y_c_oh = np.eye(n_classes)[y_c] # One-hot
    
    scaler_c = StandardScaler()
    X_c_scaled = scaler_c.fit_transform(df[features_15].values)
    
    model_c = create_classification_network(input_dim=15, n_classes=n_classes)
    model_c.fit(X_c_scaled, y_c_oh, epochs=30, batch_size=128, verbose=True)
    model_c.save_weights('red2_perfil.pkl')

    # --- ENTREGABLES (Rúbrica 2.2.3) ---
    print("\n>>> GENERANDO REPORTES PARA RÚBRICA")
    analyzer = FeatureImportanceAnalyzer()
    importances = analyzer.permutation_importance(model_r, X_test_scaled, y_r[test_idx])
    analyzer.plot_importance(importances, features_20, save_path='shap_red1.png')
    
    c_preds = model_c.predict(X_c_scaled)
    plot_confusion_matrix(y_c, c_preds, save_path='matriz_red2.png')

if __name__ == "__main__":
    ejecutar_entrenamiento_completo()