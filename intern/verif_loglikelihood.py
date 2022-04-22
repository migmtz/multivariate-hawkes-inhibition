import numpy as np
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.likelihood_functions import multivariate_loglikelihood_simplified
from tick_estimation.hawkes import HawkesExpKern


if __name__ == "__main__":
    mu = np.array([[1.0], [1.0]])
    alpha = np.array([[2.0, 2.0], [2.0, 2.0]])
    beta = np.array([[1.0], [1.0]])
    dim = 2

    timestamps = [(0,0), (1.0, 1), (1.0+np.log(2),2), (1.0+np.log(2), 0)]

    real = np.log(2) + 4
    print("Theoretical value :", real)
    mine = multivariate_loglikelihood_simplified((mu, alpha, beta), timestamps)
    print("My real loglikelihood", mine)

    list_tick = [[np.array([t for t, m in timestamps if (m - 1) == i]) for i in range(dim)]]

    beta_his = np.double(np.c_[beta, beta])
    alpha_his = alpha/beta_his
    learnersq = HawkesExpKern(decays=beta_his, solver='bfgs')

    his = learnersq.score(list_tick, 1.0+np.log(2), baseline=mu.squeeze(), adjacency=alpha_his)

    print("His", his)

    print("difference ", mine - his, " is it : ", 2," ?")


    ######## Alternatively

    # dim = 2  # 2, 3 ou 4
    #
    # mu = np.array([[1.], [2.]])
    # alpha = np.array([[-0.5, -15 / 8], [0.2, -101 / 40]])
    # beta = np.array([[2.], [3.]])
    #
    # tList = [(0, 0), (1, 1), (1 + np.log(2), 2), (1 + np.log(6), 1)]
    #
    # multivariate_loglikelihood_simplified((mu, alpha, beta), tList)
