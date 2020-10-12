#from BasicLstmModel import BasicLstmModel
from Model.LSTM_s import LSTM_s
import argparse

lstm = "lstm"
cnn = "cnn"

def main():

    parser = argparse.ArgumentParser(description="AI Model Specifications")
    parser.add_argument("-model", metavar='m', type=str, default=lstm)
    parser.add_argument("-lr", metavar='l', type=int, default=0.01)
    parser.add_argument("-bs", metavar='l', type=int, default=10)
    args = parser.parse_args()

    if args.model == lstm:
        lstm_model = LSTM_s()
        lstm_model.train_model()

        print("LSTM")
    elif args.model == cnn:

        print("CNN")
    #mymodel = BasicLstmModel(60,60)
    #mymodel.train_model()

if __name__ == "__main__":
    main()

