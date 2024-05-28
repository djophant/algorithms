"""Module for running the main program."""

import time
import logging
import neural_nets as nn 
import preprocessing as pp

def main():
    """Main program."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Loading data")
    train_data, validation_data, test_data = pp.load_data_wrapper()
    
    logger.info("Initializing neural network")
    net = nn.Network([784, 30, 10])
    
    logger.info("Training neural network")
    start_time = time.time()
    net.SGD(train_data, 30, 10, 1.0, test_data=test_data)
    end_time = time.time()
    logger.info("Training complete")
    logger.info("Time taken: %s seconds", end_time - start_time)
    
    logger.info("Evaluating neural network")
    num_correct = net.evaluate(test_data)
    logger.info("Number of correct results: %s", num_correct)
    
    logger.info("Evaluating neural network")
    num_correct = net.evaluate(test_data)
    logger.info("Number of correct results: %s", num_correct)

if __name__ == "__main__":
    main()