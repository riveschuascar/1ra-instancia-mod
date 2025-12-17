import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# 1. Cargar datos
# -------------------------
print("Cargando datos desde SQLite")

conn = sqlite3.connect("database.sqlite")

player_attr = pd.read_sql("""
SELECT id,
       player_api_id,
       date,
       acceleration,
       sprint_speed,
       stamina,
       strength,
       ball_control,
       dribbling,
       short_passing,
       positioning,
       vision,
       reactions
FROM Player_Attributes
""", conn)

player = pd.read_sql("""
SELECT player_api_id, birthday
FROM Player
""", conn)

conn.close()

# -------------------------
# 2. Merge
# -------------------------
print("Realizando merge")

df = player_attr.merge(player, on="player_api_id", how="left")

df["date"] = pd.to_datetime(df["date"])
df["birthday"] = pd.to_datetime(df["birthday"])

# -------------------------
# 3. Edad estimada
# -------------------------
print("Calculando edad estimada")

df["estimated_age"] = (
    (df["date"] - df["birthday"]).dt.days / 365.25
).round(1)

# -------------------------
# 4. Scores derivados
# -------------------------
print("Calculando scores derivados")

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
# 5. Selección final de columnas
# -------------------------
final_cols = [
    "id",
    "estimated_age",
    "physical_score",
    "technical_score",
    "mental_score"
]

df_final = df[final_cols].copy()

# -------------------------
# 6. Manejo simple de nulos
# -------------------------
print("Tratando valores nulos")

df_final[["physical_score", "technical_score", "mental_score"]] = (
    df_final[["physical_score", "technical_score", "mental_score"]]
    .fillna(df_final[["physical_score", "technical_score", "mental_score"]].median())
)

df_final["estimated_age"] = df_final["estimated_age"].fillna(
    df_final["estimated_age"].median()
)

# -------------------------
# 7. Normalización SOLO de scores a [0, 1]
# -------------------------
print("Escalando scores a rango [0, 1]")

scaler = MinMaxScaler(feature_range=(0, 1))
score_cols = ["physical_score", "technical_score", "mental_score"]
df_final[score_cols] = scaler.fit_transform(df_final[score_cols])

# -------------------------
# 8. Guardar dataset limpio
# -------------------------
print("Guardando dataset de integracion")

conn = sqlite3.connect("integrationdataset.sqlite")
df_final.to_sql(
    "player_attributes_dataset",
    conn,
    if_exists="replace",
    index=False
)
conn.close()

print("Proceso finalizado correctamente")
