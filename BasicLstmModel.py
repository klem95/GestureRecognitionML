import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from Parameters import Labeler, RecordSettings
from tensorflow.keras.utils import to_categorical
import pandas as pd
import glob2
import tensorflow as tf
import csv


class BasicLstmModel:
    first_layer_rollouts = 50

    def __init__(self, batch_size, epochs):
        self.batch_size = batch_size
        self.epochs = epochs

        path = r'Data'  # use your path
        all_files = glob2.glob(path + "/*.csv")

        self.dfs = []
        labels = []
        for filename in all_files:
            #            self.dfs.append(np.array(pd.read_csv(filename)))
            with open(filename, newline='') as csvfile:
                firstLine = True
                # spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
                sample = []
                for row in spamreader:
                    if firstLine:
                        firstLine = False
                        continue
                    labels.append(filename[5:-7])
                    float_list = [float(s.replace(',', '')) for s in row]
                    self.dfs.append(np.asarray(float_list).astype(float))
                    #sample.append(np.asarray(float_list).astype(float))
                # print('------')
                # self.dfs.append(np.asarray(sample))

        self.label_encoder = LabelEncoder()
        self.oneHot_encoder = OneHotEncoder(sparse=False)
        self.onehotLabels = self.one_hot(labels);

        #self.onehotLabels = self.encode_labels(all_files)

    def encode_labels(self, file_names):
        mappedFileNames = []
        for filename in file_names:
            mappedFileNames.append(filename[5:-7])
        integer_encoded = self.label_encoder.fit_transform(mappedFileNames)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = self.oneHot_encoder.fit_transform(integer_encoded)
        return onehot_encoded

    def one_hot(self, data):
        integer_encoded = self.label_encoder.fit_transform(data)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = self.oneHot_encoder.fit_transform(integer_encoded)
        return onehot_encoded


    def train_model(self):
        xTrain = np.asarray(self.dfs)
        xTrain = xTrain.reshape(xTrain.shape[0],1,xTrain.shape[1])
        yTrain = np.asarray(self.onehotLabels).astype(float)

        xValidate = np.asarray(self.dfs)
        xValidate = xValidate.reshape(xValidate.shape[0],1,xValidate.shape[1])

        yValidate = np.asarray(self.onehotLabels).astype(float)

        print(type(xTrain))
       # print(type(xTrain[50]))
      #  print(xTrain[0])

       # print(type(xTrain[0][0]))
        #print(type(xTrain[0][0][0]))

      #  print(yTrain.shape)
     #   print(xTrain.shape)
      #  print(type(xTrain[50][0][0]))
        #print(len(xTrain[0]))
        #print(len(xTrain[0][0]))

        # print(xTrain)

        model = Sequential()
        model.add(LSTM(32, return_sequences=True, recurrent_dropout=0.1, input_shape=(30, 289)))
        model.add(Flatten())
        model.add(Dense(3, activation='softmax'))  # Classification
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()

        history = model.fit(xTrain, yTrain, epochs=self.epochs, validation_data=(xValidate, yValidate),
                            batch_size=self.batch_size, validation_split=0.1)

        print("Train")

    def make_prediction(self):
        print("Make Prediction")
