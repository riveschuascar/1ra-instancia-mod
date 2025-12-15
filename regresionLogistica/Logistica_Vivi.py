import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from LogisticaMulticlase import RegresionLogisticaMulticlase, RegresionLogisticaBinaria 


import matplotlib.pyplot as plt
import seaborn as sns
# --- 1. Carga de Datos y Preprocesamiento Específico ---

DB_PATH_CLEAN = 'cleandataset.sqlite'

def load_clean_data(db_path):
    """Carga el dataset limpio generado por Huascar."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM player_attributes_dataset", conn)
    conn.close()
    return df

df_clean = load_clean_data(DB_PATH_CLEAN)
print(f"Dataset limpio cargado. Filas: {len(df_clean)}, Columnas: {len(df_clean.columns)}")

# --- 2. Creación de la Variable Objetivo de Posición (4 Clases) ---

# Simplificación de la posición (basado en el patrón de nombres) 
# Suponiendo que la posición original (antes de one-hot) estaba en df_raw o se puede deducir
# Dado que solo tenemos las columnas one-hot de work_rate y preferred_foot, 
# usaremos las columnas de atributos de juego para DEDUCIR la posición y crear la etiqueta 'y'.

def categorize_position(row):
    """Clasifica al jugador en 4 categorías principales basado en sus ratings."""
    # Usaremos ratings clave normalizados (0-100) para crear la variable objetivo 
    # Atributos: Goalkeeping (GK), Defence (DEF), Midfield (MID), Attack (ATK)

    # Nota: Los datos ya fueron normalizados por Huascar al rango 0-100.
    
    GK_score = row['gk_diving'] + row['gk_handling'] + row['gk_kicking']
    DEF_score = row['marking'] + row['standing_tackle'] + row['sliding_tackle']
    MID_score = row['ball_control'] + row['short_passing'] + row['long_passing'] + row['vision']
    ATK_score = row['finishing'] + row['shot_power'] + row['long_shots']

    # Asignación simple por el score más alto (solo para crear la etiqueta Y)
    scores = {
        0: GK_score,    # 0: Portero
        1: DEF_score,   # 1: Defensa
        2: MID_score,   # 2: Medio
        3: ATK_score    # 3: Atacante
    }
        
    max_score = max(scores.values())
    
    if max_score == GK_score: return 0
    elif max_score == ATK_score: return 3
    elif max_score == MID_score: return 2
    else: return 1 

df_clean['position_class'] = df_clean.apply(categorize_position, axis=1)

print("\nDistribución de la Variable Objetivo (Antes de Balanceo):")
print(df_clean['position_class'].value_counts(normalize=True))

# --- 3. Selección de Características y División de Datos ---

# 15 Atributos clave (ejemplo para Regresión Logística, incluye los scores de Huascar) 
# Asegurarse de que las columnas One-Hot también se incluyan si son predictoras
# Excluir overall_rating (variable objetivo de Mauri) y IDs.
FEATURES = [
    'estimated_age', 'potential', 'physical_score', 'technical_score', 'mental_score',
    'ball_control', 'dribbling', 'marking', 'standing_tackle', 'shot_power',
    'long_shots', 'reactions', 'vision', 'gk_diving', 'gk_handling', # 15 atributos + 3 scores
    'preferred_foot_Right', 'attacking_work_rate_high', 'defensive_work_rate_medium'
]

# Filtrar solo las columnas que existen después del preprocesamiento de Huascar
X_cols = [col for col in FEATURES if col in df_clean.columns]
X = df_clean[X_cols].values
y = df_clean['position_class'].values

# Dividir en entrenamiento y prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Balanceo del Dataset (Opcional, si la distribución es muy sesgada) ---
# Usaremos un balanceo simple de undersampling en el dataset de entrenamiento

def undersample_train(X, y):
    """Realiza undersampling para equilibrar las clases de entrenamiento."""
    X_balanced, y_balanced = [], []
    classes = np.unique(y)
    
    # Encontrar el tamaño de la clase minoritaria
    counts = pd.Series(y).value_counts()
    min_size = counts.min() # 

    for c in classes:
        # Seleccionar índices de la clase actual
        indices = np.where(y == c)[0]
        
        # Muestrear aleatoriamente hasta el tamaño de la clase minoritaria
        if len(indices) > min_size:
            np.random.shuffle(indices)
            indices = indices[:min_size]
        
        X_balanced.append(X[indices])
        y_balanced.append(y[indices])

    X_balanced = np.concatenate(X_balanced)
    y_balanced = np.concatenate(y_balanced)
    
    # Mezclar los datos finales
    perm = np.random.permutation(len(y_balanced))
    
    print(f"Dataset balanceado a {min_size} muestras por clase.")
    return X_balanced[perm], y_balanced[perm]

# Aplicar balanceo
X_train_bal, y_train_bal = undersample_train(X_train, y_train)

# --- 5. Implementación y Entrenamiento (Ajuste de Hiperparámetros) ---



ALPHA = 0.001       # Tasa de aprendizaje
LAMBDA = 0.0001      # Parámetro de regularización L2 para evitar overfitting 
ITERACIONES = 1000  # Número máximo de iteraciones para DG

K_CLASSES = 4 

model_ovr = RegresionLogisticaMulticlase(
    num_clases=K_CLASSES, 
    clasificador_binario=RegresionLogisticaBinaria 
)

model_ovr.fit(
    X_train_bal, y_train_bal, 
    alpha=ALPHA, 
    lambda_param=LAMBDA, 
    num_iteraciones=ITERACIONES
)

# --- 6. Evaluación del Modelo (Gráfica y Texto) ---

# Predicción en el conjunto de prueba
y_pred = model_ovr.predecir(X_test)

# Calcular Matriz de Confusión 
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1_score, support = precision_recall_fscore_support(
    y_test, y_pred, average=None, labels=[0, 1, 2, 3]
)

metrics_df = pd.DataFrame({
    'Clase': ['Portero (0)', 'Defensa (1)', 'Medio (2)', 'Atacante (3)'],
    'Precisión': precision.round(4),
    'Recall': recall.round(4),
    'F1-Score': f1_score.round(4)
})

# --- FUNCIÓN DE GRÁFICOS ---
def plot_results(cm, metrics_df, accuracy):
    """Genera la Matriz de Confusión como heatmap y tabla de métricas."""
    
    class_names = ['GK (0)', 'DEF (1)', 'MID (2)', 'ATK (3)']
    
    # 1. Matriz de Confusión (Heatmap)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=.5, linecolor='gray', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matriz de Confusión (Accuracy General: {accuracy:.4f})')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción del Modelo')
    plt.show() # Muestra la Matriz de Confusión
    
    # 2. Métricas por Clase (Tabla Gráfica)
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off') # Ocultar ejes
    
    # Prepara los datos para la tabla
    data = metrics_df[['Precisión', 'Recall', 'F1-Score']].values
    cols = metrics_df.columns[1:]
    rows = metrics_df['Clase'].values
    
    table = ax.table(cellText=data,
                     colLabels=cols,
                     rowLabels=rows,
                     loc='center',
                     cellLoc='center',
                     rowLoc='left')
    
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    ax.set_title('Métricas de Clasificación por Clase', fontsize=16)
    plt.show() # Muestra la Tabla de Métricas

# Ejecutar las funciones de gráfico
plot_results(cm, metrics_df, accuracy)

# --- 7. Análisis de Probabilidades y Jugadores Híbridos (Texto y Tabla) ---


def get_probabilities(model_ovr, X_test):
    probabilities = []
    for model in model_ovr.clasificadores:
        prob_k = model.predecir(X_test).flatten() 
        probabilities.append(prob_k)

    return np.column_stack(probabilities) 

probabilities = get_probabilities(model_ovr, X_test)
df_prob = pd.DataFrame(probabilities, columns=['Prob_GK', 'Prob_DEF', 'Prob_MID', 'Prob_ATK'])
df_prob['Predicción'] = y_pred
df_prob['Real'] = y_test

df_prob['Max_Prob'] = df_prob[['Prob_GK', 'Prob_DEF', 'Prob_MID', 'Prob_ATK']].max(axis=1)
df_prob['Second_Max_Prob'] = df_prob[['Prob_GK', 'Prob_DEF', 'Prob_MID', 'Prob_ATK']].apply(
    lambda row: row.nlargest(2).iloc[1], axis=1
)
df_prob['Prob_Diff'] = df_prob['Max_Prob'] - df_prob['Second_Max_Prob']

HYBRID_THRESHOLD = 0.15 
hybrid_players_indices = df_prob[df_prob['Prob_Diff'] < HYBRID_THRESHOLD].index

print(f"\n--- Análisis de Jugadores Híbridos (Prob_Diff < {HYBRID_THRESHOLD}) ---")
print(f"Número de jugadores híbridos identificados: {len(hybrid_players_indices)}")
print("\nEjemplo de Jugadores Híbridos:")
print(df_prob.loc[hybrid_players_indices, ['Prob_GK', 'Prob_DEF', 'Prob_MID', 'Prob_ATK', 'Predicción', 'Real', 'Prob_Diff']].head(5).to_markdown())


for i, model in enumerate(model_ovr.clasificadores):
    print(f"\nParámetros (Theta) del modelo OvR para Clase {i}:")
