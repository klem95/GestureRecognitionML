import numpy as np
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Flatten, Embedding, Dropout, Conv2D, MaxPooling2D, TimeDistributed, Conv3D, MaxPooling3D
from keras.optimizers import schedules
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import glob2
import csv
import matplotlib.pyplot as plt
from numpy import load, save, genfromtxt


class CNN_n_LSTM:

    def __init__(self, lr, bs, e, split, f, loadModel=False, path = ''):
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


        self.train_dataset = []
        self.validation_dataset = []
        self.trainFiles = []
        self.validationFiles = []

        if (loadModel):
            self.model = self.loadModel()
        else:
            self.model = None


    def biggestDocLength (self):
        all_files = glob2.glob(self.dataPath + "/*.csv")
        biggestRowCount = 0
        for filename in all_files:
            with open(filename, newline='') as csvfile:
                length = 0
                data = genfromtxt(csvfile, delimiter=';')

                for row in data:
                    length += 1
                if(length > biggestRowCount):
                    biggestRowCount = length
        return biggestRowCount


    def loadFromBuffer(self):
        try:
            npObject = load(self.path + 'numpy-buffers/' + self.dataPath + '-npBuffer.npy', allow_pickle=True)
            print('buffer loaded')
            return npObject
        except:
            print('no buffer')
            return False


    def bufferFile(self, npObject):
        print('saving data to buffer')
        save(self.path + 'numpy-buffers/' + self.dataPath + '-npBuffer.npy', npObject)

    def format(self, chunk, zeroPad=True):

        largestFrameCount = self.biggestDocLength()

        frames = []
        frame_count = 0
        firstLine = True
        for frame in chunk:
            if firstLine:
                firstLine = False
                continue
            coords = []  # x, y, z
            for col in range(0, len(frame[:-1])):
                if (col % 9 == 0 or col % 9 == 1 or col % 9 == 2):
                    coords.append(np.asarray([frame[col]]))

            joints = np.asarray(coords).reshape(-1, 3)  # produces 32 * 3
            frames.append(np.asarray(joints).astype(float))
            frame_count += 1

        transposed = np.transpose(np.asarray(frames), (1, 0, 2))
        transposed = transposed.reshape((transposed.shape[0], transposed.shape[1], transposed.shape[2], 1))

        if(zeroPad):
            result = np.zeros((transposed.shape[0], largestFrameCount, transposed.shape[2], transposed.shape[3]))
            result[:transposed.shape[0], :transposed.shape[1], : transposed.shape[2], :transposed.shape[3]] = transposed
            print(np.asarray(result).shape)
        else:
            result = transposed
        return result


    def preprocess(self):
        all_files = glob2.glob(self.dataPath + "/*.csv")
        i = 0

        for filename in sorted(all_files):
            with open(filename, newline='') as csvfile:
                print('loading: ' + filename)
                # dataScanner = csv.reader(csvfile, delimiter=';', quotechar='|')
                data = genfromtxt(csvfile, delimiter=';')

                result = self.format(data)

                if i % self.validationDataEvery == 0:
                    self.validation_dataset.append(result)
                    self.validationFiles.append(filename)
                else:
                    self.train_dataset.append(result)
                    self.trainFiles.append(filename)

            i += 1
        #print(np.asarray(self.trainFiles).shape)
        self.label_encoder = LabelEncoder()
        self.oneHot_encoder = OneHotEncoder(sparse=False)
        self.onehotTrainLabels = self.encode_labels(self.trainFiles)
        self.onehotValidationLabels = self.encode_labels(self.validationFiles)
        print('train_dataset shape')
        print(np.asarray(self.train_dataset).shape)
        print('traning onehot shape:')
        print(np.asarray(self.onehotTrainLabels).shape)

        for encodes in range(0, 3):
            inverted = self.label_encoder.inverse_transform(encodes)
            print('label 1')
            print(inverted)



    def encode_labels(self, file_names):
        mappedFileNames = []
        for filename in file_names:
            mappedFileNames.append(filename.split("_")[0])
            # print(file_names)
        integer_encoded = self.label_encoder.fit_transform(mappedFileNames)
        print(integer_encoded)
        print(mappedFileNames)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = self.oneHot_encoder.fit_transform(integer_encoded)


        return onehot_encoded



    def train_model(self):
        print('CNN Model')

        x_train = None
        y_train = None
        x_validation = None
        y_validation = None

        bufferedNumpy = self.loadFromBuffer()

        if(bufferedNumpy != False):
            x_train = bufferedNumpy[0]
            y_train = bufferedNumpy[1]
            x_validation = bufferedNumpy[2]
            y_validation = bufferedNumpy[3]
        else:
            self.preprocess()
            print('Preprocessing files')
            x_train = np.asarray(self.train_dataset)
            y_train = np.asarray(self.onehotTrainLabels)
            x_validation = np.asarray(self.validation_dataset)
            y_validation = np.asarray(self.onehotValidationLabels)
            self.bufferFile(np.asarray([x_train, y_train, x_validation, y_validation]))



        sequence = x_train.shape[0]
        joints = x_train.shape[1]
        frames = x_train.shape[2]
        coords = x_train.shape[3]
        channels = x_train.shape[4]


        self.label_size = y_train.shape[1]



        lr_schedule = schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=10000,
            decay_rate=0.9)

        model = Sequential()

        model.add(Conv3D(20,
                         activation='tanh',
                         kernel_initializer='he_uniform',
                         data_format='channels_last',
                         input_shape=(joints, frames, coords, channels),
                         kernel_size=(3, 3, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 1), strides=(1,1,1), data_format='channels_last', ))
        model.add(Dropout(0.2))
        model.add(Conv3D(50, kernel_size=(2, 2, 1),  activation='tanh'))
        model.add(MaxPooling3D(pool_size=(2, 2, 1)))
        model.add(Conv3D(100, kernel_size=(3, 3, 1),  activation='tanh'))
        model.add(MaxPooling3D(pool_size=(2, 2, 1)))

        model.add(Dropout(0.2))
        model.add(Dense(300))
        model.add(Dropout(0.2))
        model.add(Dense(100))
        model.add(Flatten())

        model.add(Dense(self.label_size, activation='softmax')) # Classification
        model.compile(loss='categorical_crossentropy', optimizer=Adam(),
                      metrics=['accuracy'])

        model.summary()

        mcp_save = ModelCheckpoint(self.path + 'saved-models/bestWeights.h5', save_best_only=True, monitor='val_loss', mode='min')
        history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=(x_validation, y_validation), callbacks=[mcp_save])
        print(history.history.keys())
        print(mcp_save.best)
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()

        self.saveModel(model)
        #model.save(self.trained_model_path, 'Lstm_s')


    def saveModel(self, model):
        model_json = model.to_json()
        with open(self.path + "saved-models/model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(self.path + "saved-models/model.h5")
        print("Saved model to disk")

    def loadModel(self):
        json_file = open(self.path + 'saved-models/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(self.path + "saved-models/model.h5")
        print("Loaded model from disk")
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return loaded_model



    def predict(self, data):
        formattedData = self.format(data, False)
        shape = np.asarray([formattedData])

        score = self.model.predict(shape, verbose=0)

        print('score:')
        print(score)
        # print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
        return score