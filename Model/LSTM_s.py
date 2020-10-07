import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Flatten, Embedding, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import glob2
import csv
import matplotlib.pyplot as plt


class LSTM_s:

    def __init__(self):
        self.batch_size = 10
        self.epochs = 200
        self.learning_rate = 0.01
        self.label_size = 0
        self.trained_model_path = 'Trained_models'  # use your path

        self.validationDataEvery = 5

        self.train_dataset = []
        self.validation_dataset = []
        self.trainFiles = []
        self.validationFiles = []

    def retrieve_data(self):
        path = r'recodsZeros'  # use your path
        all_files = glob2.glob(path + "/*.csv")
        i = 0

        count = 0

        for filename in all_files:
            #            self.dfs.append(np.array(pd.read_csv(filename)))
            with open(filename, newline='') as csvfile:
                firstLine = True
                dataScanner = csv.reader(csvfile, delimiter=';', quotechar='|')
                sample = []
                for row in dataScanner:
                    if firstLine:
                        firstLine = False
                        continue
                    float_list = [float(s.replace(',', '')) for s in row]
                    sample.append(np.asarray(float_list).astype(float))
                if i % self.validationDataEvery == 0:
                    self.validation_dataset.append(np.asarray(sample))  # <--- 54 is a problem...
                    self.validationFiles.append(filename)
                else:
                    self.train_dataset.append(np.asarray(sample))  # <--- 54 is a problem...
                    self.trainFiles.append(filename)

            i += 1

        self.label_encoder = LabelEncoder()
        self.oneHot_encoder = OneHotEncoder(sparse=False)
        self.onehotTrainLabels = self.encode_labels(self.trainFiles)
        self.onehotValidationLabels = self.encode_labels(self.validationFiles)

 #   def date_evaluation(self):
  #      for sample in self.total_dataset:
 #           for row in sample:
  #              for val in row:
   #                 if type(val) is not np.float:
                        #print(type(val))

    def encode_labels(self, file_names):
        mappedFileNames = []
        for filename in file_names:
            mappedFileNames.append(filename[5:-7])
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
        self.label_size = len(y_train[0])

        print(self.label_size)
        print(x_train.shape)
        print(x_train[0].shape)
        print(x_train[0][0].shape)

        model = Sequential()
        model.add(LSTM(150, return_sequences=True, recurrent_dropout=0.2, input_shape=(60, 289)))
        model.add(LSTM(64, recurrent_dropout=0.1))
        model.add(Flatten())
        model.add(Dropout(0.1))
        model.add(Dense(self.label_size, activation='softmax'))  # Classification
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['accuracy', 'AUC'])

        model.summary()

        history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                          validation_data=(x_validation, y_validation))

        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()

        model.save(self.trained_model_path, 'Lstm_s')
