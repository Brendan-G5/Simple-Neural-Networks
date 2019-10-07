import numpy as np
import pandas as pd

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        self.RawData = 'Testing_Dataset.csv'
        self.synaptic_weights = 2 * np.random.random((4,1))-1
        self.dataframe = self.DataFrame()
        self.training_inputs = self.set_training_inputs()
        self.training_outputs = self.set_training_outputs()

    def DataFrame(self):
        df = pd.read_csv(self.RawData)
        print(df)
        return df

    def set_training_inputs(self):
        print("Training inputs = ")
        print(self.dataframe.iloc[:,1:5])
        return self.dataframe.iloc[:,1:5]

    def set_training_outputs(self):
        print("Training outputs = ")
        print(self.dataframe.iloc[:,5])
        return self.dataframe.iloc[:,5]

    def sigmoid_derivative(self,x):
        return (x * (1-x)) * 100

    def sigmoid(self,x):
        return (1/(1 + np.exp(-x))) * 100


    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            print(training_inputs)
            print(output)
            error = training_outputs - output.T
            adjustments = np.dot(training_inputs.T, error.T * (output))
            print('THIS IS ADJUSTMENTS')
            print(adjustments)
            self.synaptic_weights = self.synaptic_weights + adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        outputs = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return outputs

if __name__ == "__main__":

    neural_network = NeuralNetwork()
    print('Random synaptic weights: ')
    print(neural_network.synaptic_weights)

    training_inputs = np.array(neural_network.training_inputs)

    training_outputs = np.array(neural_network.training_outputs)
    """
    training_inputs = np.array([[0,0,1],
                               [1,1,1],
                               [1,0,1],
                               [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T
    """
    neural_network.train(training_inputs, training_outputs, 500)

    print('Synaptic Weights after training:  ')
    print(neural_network.synaptic_weights)

    A = str(input('Grade 1: '))
    B = str(input('Grade 2: '))
    C = str(input('Grade 3: '))
    D = str(input('Grade 4: '))

    print("New situation input data = ",A,B,C,D)
    print("Output Data:  ")
    print(neural_network.think(np.array([A,B,C,D])))






