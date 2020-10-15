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


class CNN_n_LSTM:

    def __init__(self, lr, bs, e, split):
        self.batch_size = 2
        self.learning_rate = 0.01 if lr is None else lr
        self.epochs = 400 if e is None else e
        self.validationDataEvery = 5 if split is None else split
        self.label_size = 0
        self.dataPath = r'Data'
        self.trained_model_path = 'Trained_models'  # use your path
        self.time_steps = 0
        self.feature_size = 0
        self.testDataEvery = 10


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

                trsnposed = np.transpose(np.asarray(frames), (1, 0, 2))
                print('(t, j, coords)')
                trsnposed = trsnposed.reshape((trsnposed.shape[0], trsnposed.shape[1], trsnposed.shape[2], 1))
                print(trsnposed.shape)

                result = np.zeros((trsnposed.shape[0], largestFrameCount, trsnposed.shape[2], trsnposed.shape[3]))
                result[:trsnposed.shape[0], :trsnposed.shape[1], : trsnposed.shape[2], :trsnposed.shape[3]] = trsnposed


                if i % self.validationDataEvery == 0:
                    self.validation_dataset.append(result)
                    self.validationFiles.append(filename)
                else:
                    self.train_dataset.append(result)
                    self.trainFiles.append(filename)

            i += 1
        print('train_dataset set')
        print(np.asarray(self.train_dataset).shape)
        #print(np.asarray(self.trainFiles).shape)
        self.label_encoder = LabelEncoder()
        self.oneHot_encoder = OneHotEncoder(sparse=False)
        self.onehotTrainLabels = self.encode_labels(self.trainFiles)
        self.onehotValidationLabels = self.encode_labels(self.validationFiles)
        print('onehot:')
        print(np.asarray(self.onehotTrainLabels).shape)
    #   def date_evaluation(self):
    #      for sample in self.total_dataset:
    #           for row in sample:
    #              for val in row:
    #                 if type(val) is not np.float:
    # print(type(val))

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
        print('CNN')
        self.preprocess()

        x_train = np.asarray(self.train_dataset)
        y_train = np.asarray(self.onehotTrainLabels)

        x_validation = np.asarray(self.validation_dataset)
        y_validation = np.asarray(self.onehotValidationLabels)


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

        model.add(Conv3D(20, activation='relu', input_shape=(joints, frames, coords, channels), kernel_size=(3, 3, 3)))
        # model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding="valid"))
        # model.add(Dropout(0.5))
        # model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
        # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        # model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(self.label_size, activation='softmax'))  # Classification
        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr_schedule),
                      metrics=['accuracy', 'AUC'])


        history = model.fit(x_train,
                            y_train,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_data=(x_validation, y_validation),
                            callbacks=[mcp_save])

        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()


        model.save(self.trained_model_path, 'Lstm_s')
