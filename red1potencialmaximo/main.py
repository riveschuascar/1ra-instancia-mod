from dia2.red import *
from dia2.utils import SQLiteDataset, ReLU, Linear, mse_loss

class PotentialRegressionNetwork(NeuralNetwork):
    def __init__(self, lr=0.001, l2_lambda=0.001):
        super().__init__(lr)

        self.add(DenseLayer(20, 256, l2_lambda), ReLU())
        self.add(DenseLayer(256, 128, l2_lambda), ReLU())
        self.add(DenseLayer(128, 64, l2_lambda), ReLU())
        self.add(DenseLayer(64, 1), Linear())

# nombre de la base de datos SQLite
DB_PATH = 'cleandataset.sqlite'

# nombre de la tabla para sacar los datos con SQLiteDataset
TABLE_NAME = 'player_attributes_dataset'

# columnas de características y columna objetivo
FEATURE_COLS = [
    "overall_rating",
    "estimated_age",
    "physical_score",
    "technical_score",
    "mental_score",
    "acceleration",
    "sprint_speed",
    "agility",
    "reactions",
    "balance",
    "stamina",
    "strength",
    "shot_power",
    "long_shots",
    "positioning",
    "vision",
    "ball_control",
    "dribbling",
    "short_passing",
    "finishing"]

TARGET_COL = "potential"