import numpy as np
from numpy import load, save, genfromtxt
import glob2
from keras.models import Sequential, model_from_json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
oneHot_encoder = OneHotEncoder(sparse=False)


def format(chunk, largestFrameCount, zeroPad=True, removeFirstLine=True): # data, columnSize, zeroPad, removeFirstLine
    frames = []
    frame_count = 0
    firstLine = removeFirstLine
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

    if (zeroPad):
        print(largestFrameCount, transposed.shape)
        result = np.zeros((transposed.shape[0], largestFrameCount, transposed.shape[2], transposed.shape[3]))
        result[:transposed.shape[0], :transposed.shape[1], : transposed.shape[2], :transposed.shape[3]] = transposed
    else:
        result = transposed
    return result

def biggestDocLength (dataPath):
    all_files = glob2.glob(dataPath + "/*.csv")
    biggestRowCount = -1
    for filename in all_files:
        print(filename)
        with open(filename, newline='') as csvfile:
            data = genfromtxt(csvfile, delimiter=';')
            length = len(data.tolist())
            if(length > biggestRowCount):
                biggestRowCount = length
    print('biggest row')
    print(biggestRowCount - 1)
    return biggestRowCount - 1 # -1 assuming first column is header


def loadFromBuffer(path, dataPath):
    try:
        npObject = load(path + 'numpy-buffers/' + dataPath + '-npBuffer.npy', allow_pickle=True)
        print('buffer loaded')
        return npObject
    except:
        print('no buffer')
        return False


def bufferFile(path, dataPath, npObject):
    print('saving data to buffer')
    save(path + 'numpy-buffers/' + dataPath + '-npBuffer.npy', npObject)


def saveModel(path, model, modelType):
    model_json = model.to_json()
    with open(path + "saved-models/" + modelType + "-model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(path + "saved-models/" + modelType + "-model.h5")
    print("Saved model to disk")

def loadModel(path, modelType, weights='-bestWeights.h5'):
    json_file = open(path + 'saved-models/' + modelType + '-model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(path + "saved-models/" + modelType + weights)
    print("Loaded model from disk")
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return loaded_model


def encode_labels(file_names):
    mappedFileNames = []
    for filename in file_names:
        mappedFileNames.append(filename.split("_")[0])
    integer_encoded = label_encoder.fit_transform(mappedFileNames)
    print(integer_encoded)
    print(mappedFileNames)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = oneHot_encoder.fit_transform(integer_encoded)
    return onehot_encoded
