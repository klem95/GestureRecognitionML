import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Flatten, Embedding, Dropout,Conv2D,MaxPooling2D, TimeDistributed
from keras.optimizers import schedules
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import glob2
import csv
import matplotlib.pyplot as plt


class CNN_n_LSTM:

    def __init__(self, lr, bs, e, split):
        self.batch_size = 100 if bs is None else bs
        self.learning_rate = 0.01 if lr is None else lr
        self.epochs = 400 if e is None else e
        self.validationDataEvery = 5 if split is None else split
        self.label_size = 0
        self.dataPath = r'splitRecords'
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




    def retrieve_data(self):

        all_files = glob2.glob(self.dataPath + "/*.csv")
        #print(type(all_files))

        i = 0
        count = 0

        largestRowCount = self.biggestDocLength()
        #print(largestRowCount)

        for filename in sorted(all_files):
            with open(filename, newline='') as csvfile:
                #print(filename)
                firstLine = True
                dataScanner = csv.reader(csvfile, delimiter=';', quotechar='|')
                sample = []
                row_count = 0
                for row in dataScanner:
                    row_count += 1
                    if firstLine:
                        firstLine = False
                        continue
                    float_list = [float(s.replace(',', '')) for s in row]
                    sample.append(np.asarray(float_list[:-1]).astype(float))

                #print('Length of ORIGINAL file samples: ' + str(len(sample)))


                appended = 0
                for j in range(0, largestRowCount + 1):
                    if j > row_count:
                        # print('appending zero ')
                        appended += 1
                        sample.append(np.zeros(len(float_list)).astype(float))

                #print('Length of APPENDED file samples: ' + str(len(sample)))
                #print('          APPENDED: ' + str(appended))
                # print(np.asarray(sample).shape())
                if i % self.validationDataEvery == 0:
                    self.validation_dataset.append(np.asarray(sample))  # <--- 54 is a problem...
                    self.validationFiles.append(filename)
                else:
                    self.train_dataset.append(np.asarray(sample))  # <--- 54 is a problem...
                    self.trainFiles.append(filename)

            i += 1

        print('validation files: ')
        print((self.validationFiles))
        print('train files: ')
        print((self.trainFiles))
        self.label_encoder = LabelEncoder()
        self.oneHot_encoder = OneHotEncoder(sparse=False)
        self.onehotTrainLabels = self.encode_labels(self.trainFiles)
        self.onehotValidationLabels = self.encode_labels(self.validationFiles)

    #   def date_evaluation(self):
    #      for sample in self.total_dataset:
    #           for row in sample:
    #              for val in row:
    #                 if type(val) is not np.float:
    # print(type(val))

    def encode_labels(self, file_names):
        mappedFileNames = []
        for filename in file_names:
            mappedFileNames.append(filename[5:-7])
            # print(file_names)
        integer_encoded = self.label_encoder.fit_transform(mappedFileNames)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = self.oneHot_encoder.fit_transform(integer_encoded)

        return onehot_encoded

    def train_model(self):
        self.retrieve_data()
        print('train data')
        print(np.asarray(self.train_dataset).shape)

        print('validation data')
        print(np.asarray(self.validation_dataset).shape)
        #self.date_evaluation()

        x_train = np.asarray(self.train_dataset)
        y_train = np.asarray(self.onehotTrainLabels)
        x_validation = np.asarray(self.validation_dataset)
        y_validation = np.asarray(self.onehotValidationLabels)
        print('y train: ')
        print(self.onehotValidationLabels)
        print('y validation: ')
        print(self.onehotValidationLabels)

        self.label_size = len(y_train[0])
        self.time_steps = x_train.shape[1]
        self.feature_size = x_train.shape[2]

        mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

        lr_schedule = schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=10000,
            decay_rate=0.9)

        model = Sequential()

        model.add(LSTM(100, return_sequences=True,  recurrent_dropout=0.2))
        model.add(LSTM(32, recurrent_dropout=0.2))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(self.label_size, activation='softmax'))  # Classification
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr_schedule),
                      metrics=['accuracy', 'AUC'])
        history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=(x_validation, y_validation), callbacks=[mcp_save])
        model.summary()

        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()


        model.save(self.trained_model_path, 'Lstm_s')
