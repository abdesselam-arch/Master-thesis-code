import numpy as np

class MLPFeedfoward:
  
    def __init__(self):
        self.num_input          = 4    # Number of neurons in the input layer
        self.num_output         = 1    # Number of neurons in the output layer
        self.num_hidden_layers  = 1    # Number of hidden layers
        self.num_hidden_neurons = 4    # Number of neurons in the hidden layer
        self.sizes = [self.num_input, self.num_hidden_neurons, self.num_output]  # Number of neurons in each layer
        self.W = {}                    # Weights between each layers
        self.B = {}                    # Bias between each layers

        # Randomly initializing the weights of the layers
        # Initializing the bias as zeros
        for i in range(self.num_hidden_layers + 1):
            self.W[i + 1] = np.random.randn(self.sizes[i], self.sizes[i + 1])
            self.B[i + 1] = np.zeros((1, self.sizes[i + 1]))

    # Activation function for the output layer
    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def forward_pass(self, x):
        self.A = {}
        self.H = {}
        self.H[0] = x.reshape(1, -1)
        for i in range(self.num_hidden_layers + 1):
            self.A[i + 1] = np.matmul(self.H[i], self.W[i + 1]) + self.B[i + 1]
            self.H[i + 1] = self.sigmoid(self.A[i + 1])
        return self.H[self.num_hidden_layers + 1]

    def predict(self, X):
        y_pred = [self.forward_pass(x) for x in X]
        return np.array(y_pred).ravel()