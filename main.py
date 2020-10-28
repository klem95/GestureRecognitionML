#from BasicLstmModel import BasicLstmModel
from Model.Cnn import cnn
from Model.LSTM_s import LSTM_s
import argparse
import numpy as np
from Model.CnnLstm import cnnlstm
LSTM = "lstm"
CNN = "cnn"
CNNLSTM = 'cnnltsm'

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
        lstm_model = LSTM_s()
        lstm_model.train_model()
    elif args.m == CNN:
        Cnn = cnn(args.lr, args.bs, args.e, args.s, args.f, args.loadModel)
        if(args.loadModel == False):
            Cnn.train_model()
        else:
            columnSize = 120
            data = np.zeros((30, 289))
            print(data.shape)
            prediction = Cnn.predict(data, True, columnSize)
            print(prediction)
    elif args.m == CNNLSTM:
        model = cnnlstm(args.lr, args.bs, args.e, args.s, args.f, args.loadModel)
        if(args.loadModel == False):
            model.train_model()
        else:
            columnSize = 120
            data = np.zeros((30, 289))
            print(data.shape)
            prediction = model.predict(data, True, columnSize)
            print(prediction)



if __name__ == "__main__":
    main()

