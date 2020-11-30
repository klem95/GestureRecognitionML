import numpy as np
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Flatten, Dropout, Conv2D, Reshape, Permute, MaxPooling2D
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


class cnnlstm():

    def __init__(self, lr, bs, e, split, f, loadModel=False, path=''):
        self.path = path
        self.batch_size = 20 if bs is None else bs
        self.learning_rate = 0.01 if lr is None else lr
        self.epochs = 400 if e is None else e
        self.validationDataEvery = 3 if split is None else split
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

        self.label_size = y_train.shape[1]

        lr_schedule = schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=10000,
            decay_rate=0.9)

        model = Sequential()


        model.add(Conv2D(20,  # input shape: (None, 32, 120, 3)
                         activation='tanh',
                         kernel_initializer='he_uniform',
                         data_format='channels_last',
                         input_shape=(joints, frames, coords), # 30 joints, 60 frames, 3 channels representing axis position coordinates (xyz)
                         kernel_size=(3, 3))) # Kernel size is set to 3, 3
        model.add(MaxPooling2D(pool_size=(2, 2), padding="same")) # Maxpool

        model.add(Conv2D(50, kernel_size=(2, 2), activation='tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2),  padding="same"))

        model.add(Conv2D(100, kernel_size=(3, 3), activation='tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        convJointSize = model.output_shape[1]
        convFrameSize = model.output_shape[2]
        model.add(Reshape((convJointSize, convFrameSize, -1)))
        model.add(Permute((2, 1, 3))) # Permuting the conv output shape such that frames are given as the sequential input for the LSTM layers
        model.add(Reshape((convFrameSize, -1))) # Reshaping the permuted output to match the (timesteps, features) LSTM input - while withholding the
        model.add(LSTM(units=20, input_shape=(model.output_shape), return_sequences=True, recurrent_dropout=0.2))
        model.add(LSTM(units=100, input_shape=(model.output_shape), recurrent_dropout=0.1))
        model.add(Dropout(0.2))
        model.add(Dense(300, activation='tanh', kernel_regularizer=regularizers.l2(0.1)))
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='tanh', kernel_regularizer=regularizers.l2(0.1)))
        model.add(Flatten())

        model.add(Dense(self.label_size, activation='softmax'))  # Classification
        model.compile(loss='categorical_crossentropy', optimizer=Adam(),
                      metrics=['accuracy'])
        print((joints, frames, coords))
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
        formattedData = Tools.format(data, columnSize, zeroPad, removeFirstLine=False)
        shape = np.asarray([formattedData])
        score = self.model.predict(shape, verbose=0)
        return score