#from BasicLstmModel import BasicLstmModel
from Model.LSTM_s import LSTM_s
from Model.Cnn2DLstm import cnn2dlstm
from Model.CnnLstm import cnnlstm
import argparse
import numpy as np
from Model.CnnLstm import cnnlstm
from Model.Conv1d import conv1d
from Model.Conv2d import conv2d
from Model.LstmConv1d import denseConv1d

LSTM = "lstm"
CNN = "cnn"
CNNLSTM = 'cnnltsm'
CONV1D = 'conv1d'
CONV2D = 'conv2d'
LSTMCNN1D = 'lstmcnn1d'
CNN2LSTM = 'cnn2dlstm'


# CNN2d_LSTM
# LSTM_CNN1d
# CNN1d
# CNN2d
# LSTM

def main():

    parser = argparse.ArgumentParser(description="AI Model Specifications")
    parser.add_argument("-m", metavar='m', type=str, default=CNNLSTM)
    parser.add_argument("-lr", metavar='l', type=float, default=0.5)
    parser.add_argument("-bs", metavar='bs', type=int, default=400)
    parser.add_argument("-e", metavar='e', type=int, default=20)
    parser.add_argument("-f", metavar='f', type=str, default='splitRecords')
    parser.add_argument("--loadModel", metavar='loadModel', type=bool, default=False)
    parser.add_argument("-s", metavar='s', type=int, default=4)  # The data split
    args = parser.parse_args()

    if args.m == LSTM:
        lstm_model = LSTM_s(args.lr, args.bs, args.e, args.s, args.f, args.loadModel)
        lstm_model.train_model()
    elif args.m == CNNLSTM:
        model = cnnlstm(args.lr, args.bs, args.e, args.s, args.f, args.loadModel)
        if(args.loadModel == False):
            model.train_model()
        else:
            columnSize = 120
            data = np.zeros((30, 289))
            print(data.shape)
            prediction = model.predict(data, columnSize, True)
            print(prediction)
    elif args.m == CONV1D:
        model = conv1d(args.lr, args.bs, args.e, args.s, args.f, args.loadModel)
        if(args.loadModel == False):
            model.train_model()
        else:
            columnSize = 120
            data = np.zeros((30, 289))
            print(data.shape)
            prediction = model.predict(data, columnSize, True)
            print(prediction)
    elif args.m == CONV2D:
        model = conv2d(args.lr, args.bs, args.e, args.s, args.f, args.loadModel)
        if(args.loadModel == False):
            model.train_model()
        else:
            columnSize = 120
            data = np.zeros((30, 289))
            print(data.shape)
            prediction = model.predict(data, columnSize, True)
            print(prediction)
    elif args.m == LSTMCNN1D:
        model = denseConv1d(args.lr, args.bs, args.e, args.s, args.f, args.loadModel)
        if(args.loadModel == False):
            model.train_model()
        else:
            columnSize = 120
            data = np.zeros((30, 289))
            print(data.shape)
            prediction = model.predict(data, columnSize, True)
            print(prediction)


    elif args.m == CNN2LSTM:
        cnn_Lstm_model = cnn2dlstm(args.lr, args.bs, args.e, args.s, args.f, args.loadModel)
        cnn_Lstm_model.train_model()



if __name__ == "__main__":
    main()

