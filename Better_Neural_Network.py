import numpy as np
import pandas as pd

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        self.RawData = 'Testing_Dataset.csv'
        self.synaptic_weights = 2 * np.random.random((3,1))-1

    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self,x):
        return x * (1-x)

    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range (training_iterations):

            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments =  np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        outputs = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return outputs

if __name__ == "__main__":

    neural_network = NeuralNetwork()
    print('Random synaptic weights: ')
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[3,2,1],
                               [4,6,5],
                               [1,2,0],
                               [0,1,2]])

    training_outputs = np.array([[2,5,3,3]]).T
    neural_network.train(training_inputs, training_outputs, 50000)

    print('Synaptic Weights after training:  ')
    print(neural_network.synaptic_weights)

    A = str(input('Input 1: '))
    B = str(input('Input 2: '))
    C = str(input('Input 3: '))

    print("New situation input data = ",A,B,C)
    print("Output Data:  ")
    print(neural_network.think(np.array([A,B,C])))






