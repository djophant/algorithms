"""This module contains implementation of different scoring metrics for
    classifivation and regression problems.
"""
import numpy as np 
from typing import List

class BinaryCrossEntropy:

    def __init__(self):
        self.epsilon = 1e-15

    def __call__(self, 
                 y_true: np.ndarray,
                 y_pred: np.ndarray) -> np.float64:
        return self.loss(y_true, y_pred)
    
    def loss(self, y_true: np.ndarray,
             y_pred: np.ndarray) -> np.float64:
        """Compute binary cross entropy loss.
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        Returns:
            np.float64: Binary cross entropy loss.
        """
        y_pred = np.clip(y_pred, self.epsilon, 1-self.epsilon)
        return -y_true * np.log(y_pred) - (1-y_true) * np.log(1-y_pred)
    
    def gradient(self, y_true: np.ndarray,
                 y_pred: np.ndarray) -> np.float64:
        """Compute gradient of binary cross entropy loss.
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        Returns:
            np.float64: Gradient of binary cross entropy loss.
        """
        y_pred = np.clip(y_pred, self.epsilon, 1-self.epsilon)
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)
    

    if __name__ == "__main__":
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.9, 0.1, 0.8, 0.2])
        bce = BinaryCrossEntropy()
        print(bce(y_true, y_pred))