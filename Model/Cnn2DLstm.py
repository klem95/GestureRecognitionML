import numpy as np
from numpy import load, save, genfromtxt
import glob2
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Flatten, Embedding, Dropout, Conv2D, MaxPooling2D, TimeDistributed, Conv2D, \
    Permute, Reshape, SpatialDropout2D, Input, concatenate
from keras.optimizers import schedules
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from keras.callbacks import ModelCheckpoint
from keras.models import Model

from keras.models import Sequential, model_from_json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from .tools import biggestDocLength, encode_labels, loadModel
from .Skeleton_structure import Skeleton

class cnn2dlstm:
    def __init__(self, lr, bs, e, split, f, _loadModel=False, path=''):
        self.path = path
        self.batch_size = 20 if bs is None else bs
        self.learning_rate = 0.5 if lr is None else lr
        self.epochs = 400 if e is None else e
        self.validationDataEvery = 3
        self.dataPath = 'Data' if f is None else f
        self.feature_pr_joint = 9
        self.largest_region = 0.0
        self.largest_frame_count = 0
        self.train_dataset = []
        self.validation_dataset = []
        self.trainFiles = []
        self.validationFiles = []
        self.feature_size = 32 * 3
        self.modelType = 'cnn2dlstm'
        self.labels = []

        self.trunk_joint_count = 3
        self.upper_region_joint_count = 8
        self.lower_region_joint_count = 6
        self.head_region = 10
        self.full_body = 32


        if _loadModel:
            self.model = loadModel(self.path, self.modelType)
        else:
            self.model = None

    def format(self, data, largest_frame_count):
        data = np.asarray(data)
        self.largest_region = self.get_largest_region_size()
        first_row = True

        region_segmented_data = []
        for region in range (0, len(Skeleton.region_look_up)):
            region_segmented_data.append([None] * (len(Skeleton.region_look_up[region])*3))
            for none_element in range(0,  len(region_segmented_data[-1])):
                region_segmented_data[-1][none_element] =  []

            print("Baseline region segmented data shape: ", len(region_segmented_data[region]))



        for frame in data:
            if first_row:
                first_row = False
                continue

            col_count = 0
            for col in range(0, len(frame[:-1])):
                if col % 9 == 0 or col % 9 == 1 or col % 9 == 2:
                    coord_index = col % 9
                    for region in Skeleton.region_look_up:
                        joint_index = 0
                        for joint in Skeleton.region_look_up[region]:
                            if col_count == joint:
                                region_segmented_data[region][coord_index + joint_index * 3].append(frame[col])
                            joint_index += 1
                if col % 9 == 2:
                    col_count += 1


        for region_data in range(0,len(region_segmented_data)):
            region_segmented_data[region_data] = np.asarray(region_segmented_data[region_data])

        print("Output Data: ", len(region_segmented_data[0][1]))

        region_segmented_data = np.asarray(region_segmented_data)
        print("Type: ", type(region_segmented_data))
        print("Type: ", type(region_segmented_data[0]))
        print("Type: ", type(region_segmented_data[0][0]))


        #zero_padded_result[:time_steps.shape[1], : time_steps.shape[2]] = time_steps

        '''

        time_steps = []
        segmented_data = []
        segmented_data.append([])
        segmented_data[0].append([])

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
                        region_data = []
                        exist_in_region = False
                        for joint in Skeleton.region_look_up[region]:
                            if col_count == joint:
                                exist_in_region = True
                        if exist_in_region:
                            channel_pr_joint.append(frame[col])
                            segmented_data[0, col].append(frame[col])
                        else:
                            channel_pr_joint.append(0)
                    if col % 9 == 2:
                        col_count += 1
                    total_coordinate_set.append(channel_pr_joint)
            time_steps.append(total_coordinate_set)
        time_steps = np.asarray(time_steps)
        print(segmented_data.shape)
        result = np.zeros((largest_frame_count, time_steps.shape[1], time_steps.shape[2]))
        result[:time_steps.shape[0], :time_steps.shape[1], : time_steps.shape[2]] = time_steps
        print("Zero padded result: ", result.shape)
        '''



        return region_segmented_data

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
                # print('loading: ' + filename)
                data = genfromtxt(csvfile, delimiter=';')
                result = self.format(data, self.largest_frame_count)
                # print(result)

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


        print('validation_dataset shape')
        print(np.asarray(self.validation_dataset).shape)

    def train_model(self):

        self.largest_frame_count = biggestDocLength(self.dataPath)
        self.data_preprocessing_2D_conv()

        x_train = np.asarray(self.train_dataset)
        y_train = np.asarray(self.onehotTrainLabels)
        x_validation = np.asarray(self.validation_dataset)
        y_validation = np.asarray(self.onehotValidationLabels)

        self.labels = self.onehotValidationLabels.shape[1]

        print("Input shape (x_train): ", x_train.shape)
        print("Input shape (y_train): ", y_train.shape)
        print("Input shape (x_validation): ", x_validation.shape)
        print("Input shape (y_validation): ", y_validation.shape)

        x_train =  np.transpose(x_train)
        x_validation =  np.transpose(x_validation)
        print("Transposed", x_train.shape)
        print("Transposed", x_validation.shape)

        print(x_validation)


       #self.visualize_sample(x_train, 11)
        # print("y_train sample: ", x_train[4][4])
        # print("y_validation sample: ", y_validation[0])

        trunk_input = Input(shape=(self.largest_frame_count, self.trunk_joint_count * 3),  name="trunk")
        upper_left_input = Input(shape=(self.largest_frame_count, self.upper_region_joint_count * 3), name="upper_left")

        trunk_lstm_0 = LSTM(units=20, return_sequences=True, recurrent_dropout=0.2)(trunk_input)
        upper_left_lstm_0 = LSTM(units=20, return_sequences=True, recurrent_dropout=0.2)(upper_left_input)

        concat_layer = concatenate([trunk_lstm_0, upper_left_lstm_0])
        final_lstm_layer = LSTM(units=20, return_sequences=True, recurrent_dropout=0.2)(concat_layer)

        flatten = Flatten()(final_lstm_layer)
        output = Dense(self.labels, activation='softmax') (flatten)

        model = Model(inputs=[trunk_input,upper_left_input ], outputs=output)

        model.compile(loss = 'categorical_crossentropy', optimizer=Adam(),
                      metrics=['accuracy'])

        model.summary()
        mcp_save = ModelCheckpoint(self.path + 'saved-models/bestWeights.h5', save_best_only=True, monitor='val_loss',
                                   mode='min')
        history = model.fit([x_train[Skeleton.FULL_BODY],
                              x_train[Skeleton.UPPER_LEFT_REGION]],
                                y = y_train, epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=([x_validation[Skeleton.FULL_BODY], x_validation[Skeleton.UPPER_LEFT_REGION]],y_validation), callbacks=[mcp_save])



        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()

    def visualize_sample (self, data, sample):
        Z = data[sample, :, :, 0]
        Z1 = data[sample, :, :, 1]
        Z2 = data[sample, :, :, 2]
        Z3 = data[sample, :, :, 3]
        Z4 = data[sample, :, :, 4]
        Z5 = data[sample, :, :, 5]

        fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(6, 1)
        c = ax0.pcolor(Z)
        ax1.set_title('HEAD_REGION')
        c = ax1.pcolor(Z1)
        ax1.set_title('UPPER_RIGHT_REGION')
        c = ax2.pcolor(Z2)
        ax2.set_title('UPPER_LEFT_REGION')
        c = ax3.pcolor(Z3)
        ax3.set_title('LOWER_RIGHT_REGION')
        c = ax4.pcolor(Z4)
        ax4.set_title('LOWER_LEFT_REGION')
        c = ax5.pcolor(Z5)
        ax5.set_title('FULL_BODY')

        fig.tight_layout()
        plt.show()