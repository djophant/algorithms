import numpy as np 
from typing import Callable

class RandomVariableSimulation: 

    def __init__(self, n : int, seed : int = 11):
        """
        Parameters 
        ----------
        n : int 
            is the number of samples 
        seed : int 
            number to initialize the pseudorandom number generator (PRNG)
        """
        self.seed = seed 
        self.n = n

    def inverse_transform_sampling(self, inverse_cdf : Callable):
        """
        The inverse transform sampling help to have a sampling of a given distribution.
        Using the inverse of the cumulative distribution.

        Parameters
        ----------
        inverse_cdf : callable 
            inverse cumulative distribution function 

        Returns 
        -------
            numpy.array : Sampling of the given distribution (X1, X2, ..., Xn)
        """

        uniform_rv = np.random.uniform(low=0.0, high=1.0, size=self.n)
        f = np.vectorize(inverse_cdf)
        return f(uniform_rv)
    
    def inverse_exponential_cdf(lamda : int): 
        """
        The inverse cumulative distribution function of a exponential distribution

        parameters 
        ----------
        lamda : int 
            is the rate parameter of the distribution
        
        Returns 
        ------- 
            float : sampling of exponential random variable
         """
        
        def wrapper(p):
            """
            Parameters
            ----------
            rv : float 
            is a uniform random variable between 0 and 1
            """
            rv = np.random.uniform(0, 1)
            return -1/lamda * np.log(1.0 - rv)
        
        return wrapper 
