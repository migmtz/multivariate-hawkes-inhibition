import numpy as np
from matplotlib import pyplot as plt
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.likelihood_functions import multivariate_loglikelihood_simplified


if __name__ == "__main__":
    # Set seed
    np.random.seed(1)

    dim = 2  # 2, 3 ou 4

    mu = np.array([[0.5], [1.0]])
    alpha = np.array([[-1.9, 3], [1.2, 1.5]])
    beta = np.array([[5], [8]])

    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=1000)

    # Create a process with given parameters and maximal number of jumps.

    hawkes.simulate()

    hawkes.plot_intensity(plot_N=True)

    print("Original parameters: \n", multivariate_loglikelihood_simplified((mu, alpha, beta), hawkes.timestamps))
    print("Modified parameters: \n", multivariate_loglikelihood_simplified((mu+1, alpha+1, beta+1), hawkes.timestamps))

    plt.show()
