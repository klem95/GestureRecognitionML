import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from Parameters import labels
from tensorflow.keras.utils import to_categorical

class BaseLSTMModel:

    batch_size = 5
    epochs = 1

    first_layer_rollouts = 50
    label_encoder = LabelEncoder()
    oneHot_encoder = OneHotEncoder(sparse=False)

    def __init__(self):
        self.data = []
        print("1")

    def encode_labels(self, t_size):

        pass

