from dia2.red import *
from dia2.utils import SQLiteDataset, ReLU, Softmax, mse_loss

class PositionClassificationNetwork(NeuralNetwork):
    def __init__(self, lr=0.001, l2_lambda=0.001):
        super().__init__(lr)

        self.add(DenseLayer(15, 256, l2_lambda), ReLU())
        self.add(DenseLayer(256, 128, l2_lambda), ReLU())
        self.add(DenseLayer(128, 7), Softmax())

# nombre de la base de datos SQLite
DB_PATH = 'cleandataset.sqlite'

# nombre de la tabla para sacar los datos con SQLiteDataset
TABLE_NAME = 'player_attributes_dataset'

# columnas de características
FEATURE_COLS = [
    "physical_score",
    "technical_score",
    "mental_score",
    "finishing",
    "heading_accuracy",
    "short_passing",
    "long_passing",
    "ball_control",
    "positioning",
    "interceptions",
    "marking",
    "standing_tackle",
    "sliding_tackle",
    "strength",
    "stamina"
]

# NOTE: definir la matriz o vector de 7 etiquetas para clasificación