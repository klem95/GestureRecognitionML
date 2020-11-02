import numpy as np
from numpy import load, save, genfromtxt
import glob2
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Flatten, Embedding, Dropout, Conv2D, MaxPooling2D, TimeDistributed, Conv2D, \
    Permute, Reshape
from keras.optimizers import schedules
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, model_from_json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from .tools import biggestDocLength, encode_labels
from .Skeleton_structure import Skeleton


class CNN_LSTM:
    def __init__(self, lr, bs, e, split, f, loadModel=False, path=''):
        self.path = path
        self.batch_size = 20 if bs is None else bs
        self.learning_rate = 0.01 if lr is None else lr
        self.epochs = 400 if e is None else e
        self.validationDataEvery = 5
        self.dataPath = 'Data' if f is None else f
        self.feature_pr_joint = 9
        self.largest_region = 0.0
        self.largest_frame_count = 0
        self.train_dataset = []
        self.validation_dataset = []
        self.trainFiles = []
        self.validationFiles = []
        self.feature_size = 32 * 3

        if (loadModel):
            self.model = self.loadModel()
        else:
            self.model = None

    def format(self, data, largest_frame_count):
        data = np.asarray(data)
        self.largest_region = self.get_largest_region_size()
        first_row = True
        time_steps = []
        for frame in data:
            if first_row:
                first_row = False
                continue
            total_coordinate_set = []
            col_count = 0
            for col in range(0, len(frame[:-1])):
                if col % 9 == 0 or col % 9 == 1 or col % 9 == 2:
                    channel_pr_joint = []
                    for region in Skeleton.region_look_up:
                        exist_in_region = False
                        for joint in Skeleton.region_look_up[region]:
                            if col_count == joint:
                                exist_in_region = True
                        if exist_in_region:
                            channel_pr_joint.append(frame[col])
                        else:
                            channel_pr_joint.append(0)
                    if col % 9 == 2:
                        col_count += 1
                    total_coordinate_set.append(channel_pr_joint)
            time_steps.append(total_coordinate_set)
        time_steps = np.asarray(time_steps)
        result = np.zeros((largest_frame_count, time_steps.shape[1], time_steps.shape[2]))
        print("Zero padded result: ", result.shape)

        print("Head Region:")
        for joint in Skeleton.region_look_up[0]:
            print("Joint", joint, ": x=", time_steps[0][joint * 3 + 0], "y=", time_steps[0][joint * 3 + 1], "z=",
                  time_steps[0][joint * 3 + 2])

        return result

    # print(self.largest_region)
    # print("Result: ", zero_padded_result[0])
    #  print("Result: ", zero_padded_region)

    def get_largest_region_size(self):
        largest_val = 0
        for region in Skeleton.region_look_up:
            region_size = len(Skeleton.region_look_up[region])
            if largest_val < region_size:
                largest_val = region_size

        return largest_val

    def data_preprocessing_2D_conv(self):
        all_files = glob2.glob(self.dataPath + "/*.csv")
        framed_data = []
        i = 0

        for filename in sorted(all_files):
            with open(filename, newline='') as csvfile:
                print('loading: ' + filename)
                data = genfromtxt(csvfile, delimiter=';')
                result = self.format(data, self.largest_frame_count)
                if i % self.validationDataEvery == 0:
                    self.validation_dataset.append(result)
                    self.validationFiles.append(filename)
                else:
                    self.train_dataset.append(result)
                    self.trainFiles.append(filename)
            i += 1

        self.onehotTrainLabels = encode_labels(self.trainFiles)
        self.onehotValidationLabels = encode_labels(self.validationFiles)
        print('train_dataset shape')
        print(np.asarray(self.train_dataset).shape)
        print('traning onehot shape:')
        print(np.asarray(self.onehotTrainLabels).shape)
        print(np.asarray(self.onehotTrainLabels)[0])
        print(np.asarray(self.onehotTrainLabels)[1])
        print(np.asarray(self.onehotTrainLabels)[2])

        print('validation_dataset shape')
        print(np.asarray(self.validation_dataset).shape)

    def train_model(self):

        self.largest_frame_count = biggestDocLength(self.dataPath)
        self.data_preprocessing_2D_conv()

        x_train = np.asarray(self.train_dataset)
        y_train = np.asarray(self.onehotTrainLabels)
        x_validation = np.asarray(self.validation_dataset)
        y_validation = np.asarray(self.onehotValidationLabels)

        label_size = self.onehotValidationLabels.shape[1]

        model = Sequential()
        model.add(Conv2D(filters=40, kernel_size=(5, 5), activation='tanh', data_format="channels_last",
                         input_shape=(self.largest_frame_count, self.feature_size, len(Skeleton.region_look_up))))
        model.add(Conv2D(filters=20, kernel_size=(5, 5), activation='tanh'))
        model.add(Conv2D(filters=20, kernel_size=(3, 3), activation='tanh'))
        model.add(Conv2D(filters=20, kernel_size=(2, 2), activation='tanh'))
        model.add(Conv2D(filters=10, kernel_size=(2, 2), activation='tanh'))

        convTSize = model.output_shape[1]
        model.add(Reshape((convTSize, -1)))

        model.add(LSTM(units=20, input_shape=model.output_shape, return_sequences=True, recurrent_dropout=0.2))
        model.add(LSTM(units=100, recurrent_dropout=0.1))
        model.add(Dropout(0.2))
        model.add(Dense(300))
        model.add(Dropout(0.2))
        model.add(Dense(100))
        model.add(Flatten())

        model.add(Dense(label_size, activation='softmax'))  # Classification
        model.compile(loss='categorical_crossentropy', optimizer=Adam(),
                      metrics=['accuracy'])

        model.summary()
        mcp_save = ModelCheckpoint(self.path + 'saved-models/bestWeights.h5', save_best_only=True, monitor='val_loss',
                                   mode='min')
        history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=(x_validation, y_validation), callbacks=[mcp_save])
