"""This is a modeule that compute neural network using numpy
"""

import numpy as np 
from typing import List, Tuple


class Network: 

    def __init__(self, sizes: List[int]):
        """Initialize the neural network with sizes
        Args:
            sizes: list of number of neurons in each layer
                    first layer is input layer, last layer is output layer
        """
        self.num_layers = len(sizes)
        self.sizes = sizes 
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, x: np.ndarray):
        """Sigmoid function : activation function
        Args:
            x: input
        Returns:
            sigmoid(x)
        """
        return 1.0/(1.0 + np.exp(-x))
    
    def feedforward(self, a: np.ndarray):
        """Feedforward the network
        Args:
            a: input (layer)
        Returns:
            output of the network
        """
        for biases, weights in zip(self.biases, self.weights):
            a = self.sigmoid(weights@a + biases)
        return a
    
    def SGD(self, training_data: List[Tuple[np.ndarray, np.ndarray]],
            epochs: int, mini_batch_size:int, eta: float,
            test_data: List[Tuple[np.ndarray, np.ndarray]]=None):
        """Stochastic Gradient Descent
        Args:
            training_data: list of tuples (x,y) where x is input and y is output
            epochs: number of epochs
            mini_batch_size: size of mini-batch
            eta: learning rate
            test_data: test data to evaluate the model
        """
        if test_data:
            n_test = len(test_data)
            n = len(training_data)
        
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f'Epoch {j}: {self.evaluate(test_data)} / {n_test}')
            else:
                print(f'Epoch {j} complete')
    
    def update_mini_batch(self, mini_batch: List,
                          eta: float):
        """Update the network weights and biases by applying
        gradient descent using backpropagation to a single mini batch
        Args:
            mini_batch: list of tuples (x,y) where x is input and y is output
            eta: learning rate
        """
        Delta_b = [np.zeros(b.shape) for b in self.biases]
        Delta_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            Delta_b = [nb+dnb for nb, dnb in zip(Delta_b, delta_nabla_b)]
            Delta_w = [nw+dnw for nw, dnw in zip(Delta_w, delta_nabla_w)]
        
        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, Delta_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, Delta_b)]

    def backprop(self, x: np.ndarray, y: np.ndarray):
        """Return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x
        Args:
            x: input
            y: output
        Returns:
            tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = w@activation + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = delta@activations[-2].T

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = (self.weights[-l+1].T@delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = delta@activations[-l-1].T
        
        return (nabla_b, nabla_w)

    def evaluate(self, test_data: List[Tuple[np.ndarray, np.ndarray]]):
        """Return the number of test inputs for which the neural network outputs the correct result
        Args:
            test_data: list of tuples (x,y) where x is input and y is output
        Returns:
            number of test inputs for which the neural network outputs the correct result
        """
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in test_data]
        return sum(int(x==y) for x, y in test_results)
    
    def cost_derivative(self, output_activations: np.ndarray, y: np.ndarray):
        """Return the vector of partial derivatives \partial C_x / \partial a for the output activations
        Args:
            output_activations: output of the network
            y: output
        Returns:
            vector of partial derivatives \partial C_x / \partial a for the output activations
        """
        return output_activations - y
    
    def sigmoid_prime(self, z: np.ndarray):
        """Derivative of the sigmoid function
        Args:
            z: input
        Returns:
            derivative of the sigmoid function
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))