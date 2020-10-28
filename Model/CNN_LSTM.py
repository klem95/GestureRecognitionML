class CNN_LSTM:
    def __init__(self, lr, bs, e, split, f, loadModel=False, path=''):
        self.learning_rate = lr
        self.batch_size = bs
        self.epochs = e
        self.data_split = split
        self.data_path = f
        self.path = path

        if (loadModel):
            self.model = self.loadModel()
        else:
            self.model = None

    def train_model(self):

        # Data format: ()


        x_train = None
        y_train = None
        x_validate = None
        y_validate = None



