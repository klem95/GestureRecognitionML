import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Flatten, Embedding, Dropout
from keras.optimizers import schedules
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import glob2
import csv
import matplotlib.pyplot as plt


class LSTM_s:

    def __init__(self):
        self.batch_size = 40
        self.epochs = 400
        self.learning_rate = 0.01

        self.label_size = 0
        self.dataPath = r'Data'
        self.trained_model_path = 'Trained_models'  # use your path
        self.time_steps = 0
        self.feature_size = 0

        self.validationDataEvery = 5

        self.train_dataset = []
        self.validation_dataset = []
        self.trainFiles = []
        self.validationFiles = []
        self.time_steps = 15


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


        timeStepLength = biggestRowCount + self.time_steps - (biggestRowCount % self.time_steps)
        print("timeStepLength:")
        print(timeStepLength)
        return timeStepLength


    def spliceTimeSteps (self, sample, row_count, largestRowCount, colCount):
        appended = 0
        for j in range(0, largestRowCount + 2):
            if j > row_count:
                # print('appending zero ')
                appended += 1
                sample.append(np.zeros(colCount).astype(float))

        print('Length of APPENDED file samples: ' + str(len(sample)))
        print('          APPENDED: ' + str(appended))
        # print(np.asarray(sample).shape())
        print(np.asarray(sample).shape[1])
        featSize = np.asarray(sample).shape[1]
        reshapedAsTimeSteps = np.asarray(sample).reshape((-1, self.time_steps, featSize))
        return reshapedAsTimeSteps

    def retrieve_data(self):

        all_files = glob2.glob(self.dataPath + "/*.csv")
        i = 0

        largestRowCount = self.biggestDocLength()
        print('largest row in files: ')
        print(largestRowCount)

        for filename in sorted(all_files):
            with open(filename, newline='') as csvfile:
                print('file: ')
                print(filename)
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
                    sample.append(np.asarray(float_list).astype(float))


                reshapedAsTimeSteps = self.spliceTimeSteps(sample, row_count, largestRowCount, len(float_list))

                for j in range(0, len(reshapedAsTimeSteps)):
                    if i % self.validationDataEvery == 0:
                        print(len(reshapedAsTimeSteps))
                        self.validation_dataset.append(np.asarray(reshapedAsTimeSteps[j]))  # <--- 54 is a problem...
                        self.validationFiles.append(filename)
                    else:
                        self.train_dataset.append(np.asarray(reshapedAsTimeSteps[j]))  # <--- 54 is a problem...
                        self.trainFiles.append(filename)

            i += 1

        print('validation files: ')
        print(len(self.validationFiles))
        print('train files: ')
        print(len(self.trainFiles))
        self.label_encoder = LabelEncoder()
        self.oneHot_encoder = OneHotEncoder(sparse=False)
        self.onehotTrainLabels = self.encode_labels(self.trainFiles)
        self.onehotValidationLabels = self.encode_labels(self.validationFiles)

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
        print('train data shape:')
        print(np.asarray(self.train_dataset).shape)

        print('validation data shape:')
        print(np.asarray(self.validation_dataset).shape)

        x_train = np.asarray(self.train_dataset)
        y_train = np.asarray(self.onehotTrainLabels)
        x_validation = np.asarray(self.validation_dataset)
        y_validation = np.asarray(self.onehotValidationLabels)
        print('y train one hot shape: ')
        print(self.onehotValidationLabels)
        print('y validation one hot shape: ')
        print(self.onehotValidationLabels)

        self.label_size = len(y_train[0])
        self.feature_size = x_train.shape[2]

        # print(self.label_size)
        # print(x_train[0].shape)
        # print(x_train[0][0].shape)

        mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

        lr_schedule = schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=10000,
            decay_rate=0.5
        )


        print(x_train.shape)

        pass

        model = Sequential()
        model.add(
            LSTM(150, return_sequences=True, recurrent_dropout=0.1, input_shape=(self.time_steps, self.feature_size)))
        model.add(LSTM(64, recurrent_dropout=0.2))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(self.label_size, activation='softmax')) # Classification
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr_schedule),
                      metrics=['accuracy'])

        model.summary()

        history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=(x_validation, y_validation), callbacks=[mcp_save])

        plt.title('Loss')
        print(mcp_save.best)
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()


        model.save(self.trained_model_path, 'Lstm_s')
