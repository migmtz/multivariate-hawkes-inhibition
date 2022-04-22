import numpy as np
import time
from matplotlib import pyplot as plt
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.likelihood_functions import multivariate_lstsquares_simplified, multivariate_loglikelihood_simplified
from class_and_func.estimator_class import multivariate_estimator_bfgs, multivariate_estimator_bfgs_grad
from tick.hawkes import HawkesExpKern, HawkesSumExpKern

if __name__ == "__main__":
    ### Simulation of event times
    np.random.seed(10)

    dim = 5  # 2, 3 ou 4

    mu = np.random.uniform(0,2, (dim,1))
    alpha = np.random.normal(0,1,(dim,dim))
    beta = np.random.uniform(dim,2*dim, (dim,1))
    max_jumps = 1000
    print(mu, alpha, beta)

    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)

    print("Starting simulation...")
    hawkes.simulate()
    print("Finished simulation")
    print("#"*200)
    print("My real loglikelihood", multivariate_loglikelihood_simplified((mu, alpha, beta), hawkes.timestamps))

    loglikelihood_estimation = multivariate_estimator_bfgs(dimension=dim, options={"disp":False})
    print("Starting loglikelihood...")
    start_time = time.time()
    loglikelihood_estimation.fit(hawkes.timestamps)
    end_time = time.time() - start_time

    print("Estimation through loglikelihood: ", np.round(loglikelihood_estimation.res.x, 3), "\nIn: ", end_time)

    loglikelihood_estimation_grad = multivariate_estimator_bfgs_grad(dimension=dim, options={"disp":False})
    print("Starting loglikelihood...")
    start_time = time.time()
    loglikelihood_estimation_grad.fit(hawkes.timestamps)
    end_time = time.time() - start_time

    print("Estimation through grad: ", np.round(loglikelihood_estimation_grad.res.x, 3), "\nIn: ", end_time)

