#from BasicLstmModel import BasicLstmModel
from Model.CNN_n_LSTM import CNN_n_LSTM
from Model.LSTM_s import LSTM_s
import argparse

lstm = "lstm"
cnn = "cnn"

def main():

    parser = argparse.ArgumentParser(description="AI Model Specifications")
    parser.add_argument("-m", metavar='m', type=str, default=lstm)
    parser.add_argument("-lr", metavar='l', type=float, default=0.01)
    parser.add_argument("-bs", metavar='bs', type=int, default=10)
    parser.add_argument("-e", metavar='e', type=int, default=100)
    parser.add_argument("-s", metavar='s', type=int, default=5)  # The data split
    args = parser.parse_args()

    if args.m == lstm:
        lstm_model = LSTM_s()
        lstm_model.train_model()
    elif args.m == cnn:
        cnn_n_lstm = CNN_n_LSTM(args.lr, args.bs, args.e, args.s)
        cnn_n_lstm.train_model()


if __name__ == "__main__":
    main()

