import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Flatten, Embedding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import glob2
import csv
import matplotlib.pyplot as plt


class LSTM_s:

    def __init__(self):
        self.batch_size = 10
        self.epochs = 100
        self.learning_rate = 0.01
        self.total_dataset = []
        self.label_size = 0
        self.trained_model_path = 'Trained_models'  # use your path


    def retrieve_data(self):
        path = r'recodsZeros'  # use your path
        all_files = glob2.glob(path + "/*.csv")

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

                sample = np.asarray(sample)
                self.total_dataset.append(np.asarray(sample))  # <--- 54 is a problem...

        self.label_encoder = LabelEncoder()
        self.oneHot_encoder = OneHotEncoder(sparse=False)
        self.onehotLabels = self.encode_labels(all_files)

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
        #self.date_evaluation()

        x_train = np.asarray(self.total_dataset)
        y_train = np.asarray(self.onehotLabels)
        self.label_size = len(y_train[0])
        x_validation = np.asarray(self.total_dataset)
        y_validation = np.asarray(self.onehotLabels)

        print(self.label_size)
        print(x_train.shape)
        print(x_train[0].shape)
        print(x_train[0][0].shape)

        model = Sequential()
        model.add(LSTM(100, return_sequences=True, recurrent_dropout=0.3, input_shape=(None, 289)))
        model.add(LSTM(32, recurrent_dropout=0.5))
        model.add(Flatten())
        model.add(Dense(self.label_size, activation='softmax'))  # Classification
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['accuracy', 'AUC'])

        model.summary()

        history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                           validation_data=(x_validation, y_validation))

        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()


        test = np.asarray(x_train[1]).reshape(1,len(x_train[1]),len(x_train[1][0]))
        print(test.shape)

        print(model.predict(test))

        model.save(self.trained_model_path, 'Lstm_s')
