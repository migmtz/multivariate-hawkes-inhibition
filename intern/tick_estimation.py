import numpy as np
import time
from matplotlib import pyplot as plt
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from tick.hawkes import HawkesExpKern, HawkesSumExpKern


if __name__ == "__main__":
    ### Simulation of event times
    np.random.seed(0)

    dim = 2  # 2, 3 ou 4

    mu = np.array([[0.5], [1.0]])
    alpha = np.array([[0.9, 3], [-1.2, 1.5]])
    beta = np.array([[4], [4]])
    max_jumps = 5000

    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)

    print("Starting simulation...")
    hawkes.simulate()
    print("Finished simulation")
    print("#"*200)

    ################################## TICK

    list_tick = [[np.array([t for t, m in hawkes.timestamps if (m - 1) == i]) for i in range(dim)]]

    beta_his = np.double(np.c_[beta,beta])
    # With 'likelihood' goodness of fit, you must provide a constant decay for all kernels
    learnersq = HawkesExpKern(decays=beta_his, solver="bfgs")
    # print("His real likelihood", learnersq.score(list_tick, hawkes.timestamps[-1][0], baseline=mu.squeeze(), adjacency=alpha/beta_his))
    learnerlog = HawkesExpKern(decays=4,gofit="likelihood", solver="svrg", step=1e-1)

    learnerlog.fit(list_tick)
    learnersq.fit(list_tick)

    print("log_tick", learnerlog.adjacency*np.c_[beta,beta])
    print("log_sq", learnersq.baseline, "\n", learnersq.adjacency*np.c_[beta,beta])

    # print("attributes", learnersq.__dict__)

