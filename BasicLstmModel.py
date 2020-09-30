import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from Parameters import Labeler
from tensorflow.keras.utils import to_categorical
import pandas as pd
import glob2


class BasicLstmModel:
    first_layer_rollouts = 50

    def __init__(self,batch_size, epochs):
        self.batch_size = batch_size
        self.epochs = epochs

        path = r'Data'  # use your path
        all_files = glob2.glob(path + "/*.csv")

        self.dfs = []
        for filename in all_files:
            self.dfs.append(pd.read_csv(filename))
        self.label_encoder = LabelEncoder()
        self.oneHot_encoder = OneHotEncoder(sparse=False)
        self.onehotLabels = self.encode_labels(all_files)

    def encode_labels(self,file_names):
        mappedFileNames = []
        for filename in file_names:
            mappedFileNames.append(filename[5:-7])
        integer_encoded = self.label_encoder.fit_transform(mappedFileNames)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = self.oneHot_encoder.fit_transform(integer_encoded)
        return onehot_encoded


    def train_model(self):
        xTrain = shape(t)
        yTrain = self.onehotLabels

        xTest = shape(t)
        yTest = self.onehotLabels


        print("Train")

    def make_prediction(self):
        print("Make Prediction")
