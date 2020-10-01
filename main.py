from BasicLstmModel import BasicLstmModel
from Model.LSTM_s import LSTM_s

def main():
    print("Hello World!")
    #mymodel = BasicLstmModel(60,60)
    #mymodel.train_model()
    lstm = LSTM_s()
    lstm.train_model()

if __name__ == "__main__":
    main()
