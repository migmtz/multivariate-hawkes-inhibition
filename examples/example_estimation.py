import numpy as np
from class_and_func.hawkes_process import exp_thinning_hawkes
from class_and_func.estimator_class import loglikelihood_estimator

if __name__ == "__main__":
    
    # Set seed
    np.random.seed(7)
    
    lambda_0 = 1.2
    alpha = -0.4
    beta = 0.9
    
    # Create timestamps from exponential Hawkes process
    hawkes = exp_thinning_hawkes(lambda_0, alpha, beta, max_jumps=100)
    hawkes.simulate()
    tList = hawkes.timestamps
    
    # Estimate using the estimator class, the L-BFGS-B algorithm and the real loglikelihood
    model = loglikelihood_estimator()
    model.fit(tList)
    
    print("Estimated parameters are:", model.estimator)         # Estimated parameters are: [ 1.49351268 -0.35849889  0.43911736]
    print("With a loglikelihood of:", -model.estimated_loss)    # With a loglikelihood of: -112.56222084709412
