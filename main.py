from BasicLstmModel import BasicLstmModel


def main():
    print("Hello World!")
    mymodel = BasicLstmModel(60,60)
    mymodel.train_model()

if __name__ == "__main__":
    main()
