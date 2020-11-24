import numpy as np
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Flatten, Dropout, Conv1D, Reshape, Permute
from keras.optimizers import schedules
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import csv
import matplotlib.pyplot as plt
from numpy import load, save, genfromtxt
import glob2
# from livelossplot import PlotLossesKeras

label_encoder = LabelEncoder()
oneHot_encoder = OneHotEncoder(sparse=False)

from GestureRecognitionML import Tools


class LSTM_s():

    def __init__(self, lr, bs, e, split, f, loadModel=False, path=''):
        self.path = path
        self.batch_size = 20 if bs is None else bs
        self.learning_rate = 0.01 if lr is None else lr
        self.epochs = 400 if e is None else e
        self.validationDataEvery = 5 if split is None else split
        self.label_size = 0
        self.dataPath = 'Data' if f is None else f
        self.trained_model_path = 'Trained_models'  # use your path
        self.time_steps = 0
        self.feature_size = 0
        self.labels = []
        self.modelType = 'cnnlstm'

        self.train_dataset = []
        self.validation_dataset = []
        self.trainFiles = []
        self.validationFiles = []

        if (loadModel):
            self.model = Tools.loadModel(self.path, self.modelType)
        else:
            self.model = None

    def preprocess(self):
        all_files = glob2.glob(self.dataPath + "/*.csv")
        largestFrameCount = Tools.biggestDocLength(self.dataPath)

        i = 0
        for filename in sorted(all_files):
            with open(filename, newline='') as csvfile:
                print('loading: ' + filename)
                data = genfromtxt(csvfile, delimiter=';')
                result = Tools.format(data, largestFrameCount)

                if i % self.validationDataEvery == 0:
                    self.validation_dataset.append(result)
                    self.validationFiles.append(filename)
                else:
                    self.train_dataset.append(result)
                    self.trainFiles.append(filename)
            i += 1

        self.onehotTrainLabels = Tools.encode_labels(self.trainFiles)
        self.onehotValidationLabels = Tools.encode_labels(self.validationFiles)
        print('train_dataset shape')
        print(np.asarray(self.train_dataset).shape)
        print('traning onehot shape:')
        print(np.asarray(self.onehotTrainLabels).shape)

    def train_model(self):
        print('CNN Model')

        x_train = None
        y_train = None
        x_validation = None
        y_validation = None

        bufferedNumpy = Tools.loadFromBuffer(self.path, self.dataPath)

        if (bufferedNumpy == False):
            self.preprocess()
            x_train = np.asarray(self.train_dataset)
            y_train = np.asarray(self.onehotTrainLabels)
            x_validation = np.asarray(self.validation_dataset)
            y_validation = np.asarray(self.onehotValidationLabels)
            Tools.bufferFile(self.path, self.dataPath, np.asarray([x_train, y_train, x_validation, y_validation]))
        else:
            x_train = bufferedNumpy[0]
            y_train = bufferedNumpy[1]
            x_validation = bufferedNumpy[2]
            y_validation = bufferedNumpy[3]

        [x_validation, y_validation] = Tools.shuffleData(x_validation, y_validation)
        [x_train, y_train] = Tools.shuffleData(x_train, y_train)

        sequence = x_train.shape[0]
        joints = x_train.shape[1]
        frames = x_train.shape[2]
        coords = x_train.shape[3]
        # channels = x_train.shape[4]




        print(y_train.shape)

        x_train = x_train.reshape((x_train.shape[0], -1, x_train.shape[2]))
        x_validation = x_validation.reshape((x_validation.shape[0], -1, x_validation.shape[2]))
        print()
        #=(frames, joints),


        self.label_size = y_train.shape[1]

        lr_schedule = schedules.ExponentialDecay(
            initial_learning_rate=0.05,
            decay_steps=10000,
            decay_rate=0.9)

        model = Sequential()
        model.add(LSTM(50,  # (None, 30, 118, 3, 20)
                         recurrent_activation='tanh',
                         recurrent_dropout=0.2,
                         kernel_initializer='he_uniform',
                         return_sequences=True,
                         input_shape=(x_train.shape[1], x_train.shape[2]),
                         )
                  )
        model.add(LSTM(100,  # (None, 30, 118, 3, 20
                         recurrent_dropout=0.3,
                         recurrent_activation='tanh',
                         return_sequences=True,
                         kernel_initializer='he_uniform',
                         input_shape=(x_train.shape[1], x_train.shape[2]),
                         )
                  )
        model.add(Dropout(0.2))  # (None, 29, 117, 3, 20)
        model.add(Dense(200,  activation='tanh', kernel_regularizer=regularizers.l2(0.1)))
        model.add(Dropout(0.2))
        model.add(Dense(100,  activation='tanh', kernel_regularizer=regularizers.l2(0.1)))
        model.add(Flatten())

        model.add(Dense(self.label_size, activation='softmax'))  # Classification
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr_schedule),
                      metrics=['accuracy'])
        # print((joints, frames, coords, channels))
        model.summary()

        mcp_save = ModelCheckpoint(self.path + 'saved-models/' + self.modelType + '-bestWeights.h5',
                                   save_best_only=True,
                                   monitor='val_loss',
                                   mode='min')
        history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=(x_validation, y_validation), callbacks=[mcp_save])
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()

        Tools.saveModel(self.path, model, self.modelType)

    def predict(self, data, columnSize, zeroPad):
        print('data')
        print(data)

        formattedData = Tools.format(data, columnSize, zeroPad, removeFirstLine=False)
        shape = np.asarray([formattedData])
        print(shape.shape)
        score = self.model.predict(shape, verbose=0)
        return score