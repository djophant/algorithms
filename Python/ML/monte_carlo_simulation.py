import numpy as np 

class MarkovChainMonteCarlo:

    def __init__(self, size : int, random_state : int = 42):
        self.random_state = random_state 
        self.size = size

    def pi_estimation(self, radius : float):
        """
        This function give an approximation of PI using the area of a circle of radius R.
        Area = PI * R^2 => PI = Area / R^2
        Parameters
        ----------
        radius : float 
            the radius of a  circle 
        
        Returns 
        -------
            float : the approximated value of PI
        """
        np.random.seed(self.random_state)
        circle_area = np.zeros(self.size)
        radius_squared = radius**2

        for i in range(self.size):
            # two randoms variables X and Y following a uniform rv on [-R, R]
            X = np.random.uniform(-radius, radius)
            Y = np.random.uniform(-radius, radius)
            # indicator function 
            indicator = (X**2 + Y**2) <= radius_squared
            circle_area[i] = 4 * radius_squared * indicator

        circle_area_estimate = 1/self.size * np.sum(circle_area)
        pi_estimate = circle_area_estimate / radius_squared
        std_error_estimate = np.sqrt(1/(self.size-1) * np.sum(np.power(np.pi - circle_area_estimate, 2)))
        
        return pi_estimate + std_error_estimate, pi_estimate - std_error_estimate

    def binomial_tree(self, mu, sigma, S_0, N, T, step):
        """binomial tree model for option pricing

        Parameters:
        ----------
            mu (float): expected return or drift of the underlying asset's price
            sigma (float): volatility of the underlying asset's price
            S_0 (float): initial price of the underlying asset
            N (int): number of time steps or periods of the binomial tree
            T (int): total time or maturity of the option in years
            step (float): the size of each time step or period
            
        Returns:
        -------
            S_T (float): the estimated price of the option 
        """
        ratio = np.sqrt(T/N)
        ratio_step = T/step
        # up price
        u = np.exp(sigma * ratio)
        # Down price 
        d = 1/u 
        p = 0.5 + 0.5 * (mu/sigma) * ratio
        up_times = np.random.binomial(ratio_step, p, N)
        down_times = ratio_step - up_times
        S_T = S_0 * np.power(u, up_times) * np.power(d, down_times)

        return S_T

            
if __name__ == "__main__":
    mcmc = MarkovChainMonteCarlo(100000)
    print(mcmc.pi_estimation(2))

