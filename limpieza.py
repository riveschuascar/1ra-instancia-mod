import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# 1. Cargar datos
# -------------------------
print("Cargando datos desde la base de datos SQLite")

conn = sqlite3.connect("database.sqlite")

player_attr = pd.read_sql("""
SELECT *
FROM Player_Attributes
""", conn)

player = pd.read_sql("""
SELECT player_api_id, birthday
FROM Player
""", conn)

conn.close()

# -------------------------
# 2. Merge de tablas
# -------------------------
print("Realizando join de tablas")

df = player_attr.merge(player, on="player_api_id", how="left")

df["date"] = pd.to_datetime(df["date"])
df["birthday"] = pd.to_datetime(df["birthday"])

# -------------------------
# 3. Edad estimada
# -------------------------
print("Calculando edad estimada de los jugadores")

df["estimated_age"] = (
    (df["date"] - df["birthday"]).dt.days / 365.25
).round(1)

# -------------------------
# 4. Variables derivadas
# -------------------------
print("Creando variables derivadas")

df["physical_score"] = df[
    ["acceleration", "sprint_speed", "stamina", "strength"]
].mean(axis=1)

df["technical_score"] = df[
    ["ball_control", "dribbling", "short_passing"]
].mean(axis=1)

df["mental_score"] = df[
    ["positioning", "vision", "reactions"]
].mean(axis=1)

# -------------------------
# 5. Manejo de valores nulos
# Mediana por columna
# -------------------------
print("Manejando valores nulos")

numeric_cols = df.select_dtypes(include=np.number).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

df = df.fillna("Unknown")

# -------------------------
# 6. One-Hot Encoding
# -------------------------
print("Aplicando One-Hot Encoding a variables categóricas")

categorical_cols = [
    "preferred_foot",
    "attacking_work_rate",
    "defensive_work_rate"
]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# -------------------------
# 7. Tratamiento de outliers (IQR)
# -------------------------
print("Tratando outliers usando el método IQR")

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)

# -------------------------
# 8. Normalización (0–100)
# -------------------------
print("Normalizando variables numéricas al rango 0-100")

id_cols = ["id", "player_api_id", "player_fifa_api_id", "overall_rating"]

numeric_cols = df.select_dtypes(include=np.number).columns
scale_cols = [c for c in numeric_cols if c not in id_cols] # Evitar columnas de IDs

scaler = MinMaxScaler(feature_range=(0, 100))
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# -------------------------
# 9. Guardar dataset limpio en SQLite
# -------------------------
print("Guardando dataset limpio en la base de datos SQLite")

conn = sqlite3.connect("cleandataset.sqlite")
df.to_sql("player_attributes_dataset", conn, if_exists="replace", index=False)
conn.close()
