import numpy as np
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.likelihood_functions import multivariate_lstsquares_simplified


if __name__ == "__main__":
    mu = np.array([[1.0]])
    alpha = np.array([[2.0]])
    beta = np.array([[1.0]])
    dim = 1

    timestamps = [(0,0), (1.0, 1), (1.0+np.log(2),1), (1.0+np.log(2), 0)]

    real = np.log(2) - 1.5
    print("Theoretical value :", real)
    mine = multivariate_lstsquares_simplified((mu, alpha, beta), timestamps)
    print("My real least_squares", mine)
