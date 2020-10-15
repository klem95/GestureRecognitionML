import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Flatten, Embedding, Dropout, Conv2D, MaxPooling2D, TimeDistributed, Conv3D, MaxPooling3D
from keras.optimizers import schedules
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import glob2
import csv
import matplotlib.pyplot as plt
from numpy import load, save


class CNN_n_LSTM:

    def __init__(self, lr, bs, e, split, f):
        self.batch_size = 2
        self.learning_rate = 0.01 if lr is None else lr
        self.epochs = 400 if e is None else e
        self.validationDataEvery = 5 if split is None else split
        self.label_size = 0
        self.dataPath = 'Data' if f is None else f
        self.trained_model_path = 'Trained_models'  # use your path
        self.time_steps = 0                                                             
        self.feature_size = 0


        self.train_dataset = []
        self.validation_dataset = []
        self.trainFiles = []
        self.validationFiles = []


    def biggestDocLength (self):
        all_files = glob2.glob(self.dataPath + "/*.csv")
        biggestRowCount = 0
        for filename in all_files:
            with open(filename, newline='') as csvfile:
                length = 0
                dataScanner = csv.reader(csvfile, delimiter=';', quotechar='|')

                for row in dataScanner:
                    length += 1
                if(length > biggestRowCount):
                    biggestRowCount = length
        return biggestRowCount


    def loadFromBuffer(self):
        try:
            npObject = load('numpy-buffers/' + self.dataPath + '-npBuffer.npy', allow_pickle=True)
            print('buffer loaded')
            return npObject
        except:
            print('no buffer')
            return False


    def bufferFile(self, npObject):
        print('saving data to buffer')
        save('numpy-buffers/' + self.dataPath + '-npBuffer.npy', npObject)


    def preprocess(self):
        all_files = glob2.glob(self.dataPath + "/*.csv")
        i = 0
        largestFrameCount = self.biggestDocLength()
        print('largestFrameCount')
        print(largestFrameCount)

        for filename in sorted(all_files):
            with open(filename, newline='') as csvfile:
                print('loading: ' + filename)
                firstLine = True
                dataScanner = csv.reader(csvfile, delimiter=';', quotechar='|')
                frames = []
                frame_count = 0
                for frame in dataScanner:
                    if firstLine:
                        firstLine = False
                        continue
                    coords = [] # x, y, z
                    for col in range(0, len(frame[:-1])):
                        if (col % 9 == 0 or col % 9 == 1 or col % 9 == 2):
                            coords.append(np.asarray( [frame[col]] ))

                    joints = np.asarray(coords).reshape(-1, 3) # produces 32 * 3
                    frames.append(np.asarray(joints).astype(float))
                    frame_count += 1

                transposed = np.transpose(np.asarray(frames), (1, 0, 2))
                print('(t, j, coords)')
                transposed = transposed.reshape((transposed.shape[0], transposed.shape[1], transposed.shape[2], 1))

                result = np.zeros((transposed.shape[0], largestFrameCount, transposed.shape[2], transposed.shape[3]))
                result[:transposed.shape[0], :transposed.shape[1], : transposed.shape[2], :transposed.shape[3]] = transposed
                print(np.asarray(result).shape)


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


    def encode_labels(self, file_names):
        mappedFileNames = []
        for filename in file_names:
            mappedFileNames.append(filename.split("_")[0])
            # print(file_names)
        integer_encoded = self.label_encoder.fit_transform(mappedFileNames)
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
            print('Preprocessing files')
            x_train = bufferedNumpy[0]
            y_train = bufferedNumpy[1]
            x_validation = bufferedNumpy[2]
            y_validation = bufferedNumpy[3]
        else:
            self.preprocess()
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


        mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

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
        model.add(MaxPooling3D(pool_size=(2, 2, 1), data_format='channels_last', ))
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

        history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=(x_validation, y_validation), callbacks=[mcp_save])
        print(history.history.keys())

        plt.title('Loss')
        print(mcp_save.best)
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='validation')
        plt.legend()
        plt.show()

        model.save(self.trained_model_path, 'Lstm_s')
