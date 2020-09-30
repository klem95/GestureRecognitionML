import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

class BaseLSTMModel:

    def __init__(self):
        self.data = []