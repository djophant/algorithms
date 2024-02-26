"""This module contains implementation of different scoring metrics for
    classifivation and regression problems.
"""

import numpy as np

class BinaryCrossEntropy: 
    """Binary cross Entropy function."""

    def __init__(self):
        self.epsilon = 1e-10

    def loss(self, y_true: np.ndarray, 
             y_pred: np.ndarray) -> np.float64:
        """Binary cross entropy loss function.
        Parameters:
        ----------
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.
        Returns:
        -------
            float: Binary cross entropy loss.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Length of y_true and y_pred should be the same.")
        y_pred = np.clip(y_pred, self.epsilon, 1-self.epsilon)
        return - np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient(self, y_true: np.ndarray,
                 y_pred: np.ndarray) -> np.float64:
        """Gradient of binary cross entropy loss function.
        Parameters:
        ----------
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.
        Returns:
        -------
            np.array: Gradient of binary cross entropy loss.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Length of y_true and y_pred should be the same.")
        y_pred = np.clip(y_pred, self.epsilon, 1-self.epsilon)
        return - np.mean(y_true / y_pred - (1 - y_true) / (1 - y_pred))
    
    def __call__(self, y_true: np.ndarray, 
                 y_pred: np.ndarray) -> np.float64:
        """Binary cross entropy loss function.
        Parameters:
        ----------
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.
        Returns:
        -------
            float: Binary cross entropy loss.
        """
        return self.loss(y_true, y_pred)

class CategoricalCrossEntropy:
    """Categorical cross entropy is for multi-class classification problems."""

    def __init__(self):
        self.epsilon = 1e-10

    def loss(self, y_true: np.ndarray, 
             y_pred: np.ndarray) -> np.float64:
        """Categorical cross entropy loss function.
        Parameters:
        ----------
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.
        Returns:
        -------
            float: Categorical cross entropy loss.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Length of y_true and y_pred should be the same.")
        y_pred = np.clip(y_pred, self.epsilon, 1-self.epsilon)
        return - np.mean(y_true * np.log(y_pred))
    
    def gradient(self, y_true: np.ndarray,
                    y_pred: np.ndarray) -> np.float64:
            """Gradient of categorical cross entropy loss function.
            Parameters:
            ----------
                y_true (np.array): True labels.
                y_pred (np.array): Predicted labels.
            Returns:
            -------
                np.array: Gradient of categorical cross entropy loss.
            """
            if len(y_true) != len(y_pred):
                raise ValueError("Length of y_true and y_pred should be the same.")
            y_pred = np.clip(y_pred, self.epsilon, 1-self.epsilon)
            return - np.mean(y_true / y_pred)
    
    def __call__(self, y_true: np.ndarray, 
                 y_pred: np.ndarray) -> np.float64:
        """Categorical cross entropy loss function.
        Parameters:
        ----------
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.
        Returns:
        -------
            float: Categorical cross entropy loss.
        """
        return self.loss(y_true, y_pred)
    
if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
    bce = BinaryCrossEntropy()
    print(bce(y_true, y_pred))
    print(bce.gradient(y_true, y_pred))