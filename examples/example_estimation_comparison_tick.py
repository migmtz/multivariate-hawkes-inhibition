import numpy as np
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.streamline_tick import four_estimation, four_estimation_with_grid, plot_four
from class_and_func.estimator_class import multivariate_estimator_bfgs

from matplotlib import pyplot as plt


if __name__ == "__main__":
    # With False, the grid of beta for sumExpKern in Tick will contain only the real parameters beta
    # With True, a random grid must be provided
    with_grid = False
    beta_grid = np.array(range(1,8))
    ### Simulation of event times
    np.random.seed(0)

    dim = 2  # 2, 3 ou 4

    mu = 1*np.array([[1.5], [2.5]])
    alpha = 1*np.array([[0.0, 0.6], [-1.2, -1.5]])
    beta = 1*np.array([[1.], [2.]])
    max_jumps = 500

    ################# SIMULATION
    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)

    print("Starting simulation...")
    hawkes.simulate()
    print("Finished simulation")

    ################# ESTIMATION LOG
    loglikelihood_estimation = multivariate_estimator_bfgs(dimension=dim, options={"disp": False})
    mu_est, alpha_est, beta_est = loglikelihood_estimation.fit(hawkes.timestamps)
    print(mu_est, "\n", alpha_est, "\n", beta_est)

    ################# ESTIMATION LOG
    print("", "pen")
    loglikelihood_estimation_pen = multivariate_estimator_bfgs(dimension=dim, penalty="rlsquares", options={"disp": False})
    mu_pen, alpha_pen, beta_pen = loglikelihood_estimation_pen.fit(hawkes.timestamps)
    print(mu_pen, "\n", alpha_pen, "\n", beta_pen)

    ################# ESTIMATION TICK
    print("Tick")
    if with_grid:
        params_tick = four_estimation_with_grid(beta, beta_grid, hawkes.timestamps)
    else:
        params_tick = four_estimation(beta, hawkes.timestamps, penalty="l2")
    print(params_tick[2][1], "\n" * 3, params_tick[3][1])

    ################# PLOT

    fig, ax = plt.subplots(dim, dim)

    lim_x = np.max((1/beta)*(np.log(np.abs(alpha)+1e-10) - np.log(0.01)))

    x = np.linspace(0, lim_x, 100)

    for i in range(dim):
        for j in range(dim):
            ax[i, j].plot(x, alpha[i, j] * np.exp(-beta[i] * x), c="r", label="Real kernel")
            # ax[i, j].plot(x, alpha_est[i, j] * np.exp(-beta_est[i] * x), c="m", label="Estimated kernel")
            ax[i, j].plot(x, alpha_pen[i, j] * np.exp(-beta_pen[i] * x), c="m", label="Penalized kernel", linestyle=":")

    plot_four(params_tick, beta, ax=ax, x=x)

    plt.legend()
    plt.show()