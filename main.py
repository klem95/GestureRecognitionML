#from BasicLstmModel import BasicLstmModel
from Model.Cnn import cnn
from Model.LSTM_s import LSTM_s
from Model.CNN_LSTM import CNN_LSTM
import argparse
import numpy as np

lstm = "lstm"
cnn = "cnn"
cnn_lstm = "cnn_lstm"


def main():

    parser = argparse.ArgumentParser(description="AI Model Specifications")
    parser.add_argument("-m", metavar='m', type=str, default=cnn_lstm)
    parser.add_argument("-lr", metavar='l', type=float, default=0.5)
    parser.add_argument("-bs", metavar='bs', type=int, default=400)
    parser.add_argument("-e", metavar='e', type=int, default=20)
    parser.add_argument("-f", metavar='f', type=str, default='splitRecords')
    parser.add_argument("--loadModel", metavar='loadModel', type=bool, default=False)
    parser.add_argument("-s", metavar='s', type=int, default=4)  # The data split
    args = parser.parse_args()

    if args.m == lstm:
        lstm_model = LSTM_s()
        lstm_model.train_model()
    elif args.m == cnn:
        Cnn = cnn(args.lr, args.bs, args.e, args.s, args.f, args.loadModel)
        if(args.loadModel == False):
            Cnn.train_model()
        else:
            columnSize = 120
            data = np.zeros((30, 289))
            print(data.shape)
            prediction = Cnn.predict(data, True, columnSize)
            print(prediction)
    elif args.m == cnn_lstm:
        cnn_Lstm_model = CNN_LSTM(args.lr, args.bs, args.e, args.s, args.f, args.loadModel)
        cnn_Lstm_model.train_model()



if __name__ == "__main__":
    main()

