"""This module contains implementation of different scoring metrics for
    classification and regression problems.
"""
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class BinaryCrossEntropy:
    """This class computes the binary cross entropy loss for classification tasks.
    Attributes:
        num_samples (int): The number of samples processed.
        total_loss (float): The accumulated loss over all samples.
    Methods:
        update(y_true: np.ndarray, y_pred: np.ndarray) -> None:
            Updates the total loss and number of samples with new data.
        compute() -> float:
            Computes the average binary cross entropy loss over all samples.
        reset() -> None:
            Resets the total loss and number of samples to zero."""

    def __init__(self):
        self.num_samples = 0
        self.total_loss = 0.0

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        return self.update(y_true, y_pred)

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Updates the total loss and number of samples based on the true and predicted values.

        Parameters:
        y_true (np.ndarray): The ground truth binary labels.
        y_pred (np.ndarray): The predicted probabilities.

        Returns:
        None
        """
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)

        # y_pred value need to be between [epsilon, 1-epsilon]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1-epsilon)

        self.total_loss += - \
            np.sum(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
        self.num_samples += y_true.shape[0]

    def compute(self):
        """
        Computes the average loss.
        Returns:
            float: The average loss computed as total_loss divided by the number of samples.
        Raises:
            Exception: If the number of samples is zero.
        """

        if self.num_samples == 0:
            raise ValueError("The number of samples cannot zero")
        return self.total_loss / self.num_samples

    def reset(self):
        """
        Resets the internal counters for number of samples and total loss.

        This method sets the `num_samples` attribute to 0 and the `total_loss` attribute to 0.0,
        effectively clearing any accumulated data.
        """
        self.num_samples = 0
        self.total_loss = 0.0


class CategoricalCrossEntropy:
    """This class computes the categorical cross entropy loss for
       classification tasks.
    Attributes:
        num_samples (int): The number of samples processed.
        total_loss (float): The accumulated loss over all samples.
    Methods:
        update(y_true: np.ndarray, y_pred: np.ndarray) -> None:
            Updates the total loss and number of samples with new data.
        compute() -> float:
            Computes the average binary cross entropy loss over all samples.
        reset() -> None:
            Resets the total loss and number of samples to zero.
    """

    def __init__(self):
        self.num_samples = 0
        self.total_loss = 0.0

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        return self.update(y_true, y_pred)

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Updates the total loss and number of samples based on the true and predicted values.

        Parameters:
        y_true (np.ndarray): The ground truth binary labels.
        y_pred (np.ndarray): The predicted probabilities.

        Returns:
        None
        """
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)

        # y_pred value need to be between [epsilon, 1-epsilon]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1-epsilon)

        self.total_loss += - np.sum(y_true * np.log(y_pred))
        self.num_samples += y_true.shape[0]

    def compute(self):
        """
        Computes the average loss.
        Returns:
            float: The average loss computed as total_loss divided by the number of samples.
        Raises:
            Exception: If the number of samples is zero.
        """

        if self.num_samples == 0:
            raise ValueError("The number of samples cannot zero")
        return self.total_loss / self.num_samples

    def reset(self):
        """
        Resets the internal counters for number of samples and total loss.

        This method sets the `num_samples` attribute to 0 and the `total_loss` attribute to 0.0,
        effectively clearing any accumulated data.
        """
        self.num_samples = 0
        self.total_loss = 0.0


class AccuracyScore:
    """
    The goal of this class is to compute the accuracy score.
    Attributes:
        num_samples (int): the number of samples processed
        total_num_correct_pred (int) : number of correct predictions
    Methods:
        update() -> None:
            Updates the number of correct predictions and the samples
        compute(float):
            Compute the average of number of correct predictions with the number od samples
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        if not isinstance(y_true, np.ndarray):
            self.y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            self.y_pred = np.array(y_pred)
        self.y_true = y_true
        self.y_pred = y_pred
        self.num_samples = 0
        self.total_num_corect_predict = 0

    def update(self) -> None:
        """
        This method update the number of correct predicted values and the number of samples.
        Returns :
            None
        """
        self.num_samples += self.y_true.shape[0]
        self.total_num_corect_predict += np.sum(self.y_true == self.y_pred)

    def compute(self):
        """
        This method compute the average between the number of correct predictions and the samples.
        """
        if self.num_samples == 0:
            raise ValueError("The number of samples is not zero")
        return self.total_num_corect_predict / self.num_samples


class ConfusionMatrix:
    """
    This class compute a 2 x 2 Confusion matrix for binary classification.
                            | TN  FP|
                            | FN  TP|
    Attributes:
        y_true (np.ndarray): the real target 
        y_pred (np.ndarray): the predicted target base on a model.
        matrix (np.ndarray): TODO
        true_positive (int): number of reals values 1 that is predicted as 1.
        true_negative (int): number of reals values 0 that is predicted as O.
        false_positive (int): number of reals values 0 that is predicted as 1.
        false_negative (int): number of reals values 1 that is predicted as 0.
    Methods:
        update(y_true, y_pred)
            update the matrix with TP, FP, TN, FN
        compute (np.ndarray):
            update the confusion matrux with tp, fp, tn, fn.
        reset():
            Reset the matrix by 2 x 2 sero matrix.        
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        if not isinstance(y_true, np.ndarray):
            self.y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            self.y_pred = np.array(y_pred)
        self.y_true = y_true
        self.y_pred = y_pred
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        self.matrix = np.zeros((2, 2))

    def update(self) -> None:
        """
        This method update all the false and true positive 
        """
        self.true_positive += np.sum((self.y_pred == 1) & (self.y_true == 1))
        self.true_negative += np.sum((self.y_pred == 0) & (self.y_true == 0))
        self.false_positive += np.sum((self.y_pred == 1) & (self.y_true == 0))
        self.false_negative += np.sum((self.y_pred == 0) & (self.y_true == 1))

    def compute(self) -> np.ndarray:
        """
        This function compute the confusion matrix
        Return:
            2 x 2 np.ndarray
        """
        self.matrix[0][0] = self.true_negative
        self.matrix[0][1] = self.false_positive
        self.matrix[1][0] = self.false_negative
        self.matrix[1][1] = self.true_positive

        return self.matrix

    def reset(self) -> None:
        """
        Reset the matrix to get a zeros matrices
        """
        self.matrix = np.zeros((2, 2))


if __name__ == "__main__":
    # test binary cross-entropy loss
    y_true = np.array([0, 0, 1, 0, 1])
    y_pred = np.array([0.4, 0.9, 0.8, 0.0, 1])
    y_pred_bis = np.array([0, 1, 1, 1, 1])
    bce = BinaryCrossEntropy()
    bce(y_true, y_pred)
    loss = bce.compute()
    print(f"The binary cross entropy between y_true and y_pred are {loss}\n")

    cce = CategoricalCrossEntropy()
    cce(y_true, y_pred)
    loss2 = cce.compute()
    print(
        f"The categorical cross entropy between y_true and y_pred are {loss2}\n")

    cce = AccuracyScore(y_true, y_pred)
    cce.update()
    score = cce.compute()
    print(
        f"The accuracy score between y_true and y_pred are {score}\n")

    confusion_matrix = ConfusionMatrix(y_true, y_pred_bis)
    confusion_matrix.update()
    conf_matrix = confusion_matrix.compute()
    print(conf_matrix)
