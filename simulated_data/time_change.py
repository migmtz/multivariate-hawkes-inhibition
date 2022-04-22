import numpy as np
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from matplotlib import pyplot as plt
import seaborn as sns
from simulated_data.dictionary_parameters import dictionary as param_dict
from scipy.stats import kstest


def time_change(theta, tList):
    if isinstance(theta, np.ndarray):
        dim = int(np.sqrt(1 + theta.shape[0]) - 1)
        mu = np.array(theta[:dim]).reshape((dim, 1))
        alpha = np.array(theta[dim:dim * (dim + 1)]).reshape((dim, dim))
        beta = np.array(theta[dim * (dim + 1):]).reshape((dim, 1))
    else:
        mu, alpha, beta = (i.copy() for i in theta)
    beta = beta + 1e-10

    beta_1 = 1/beta

    counter = np.zeros((dim, 1))
    transformed_times = []
    individual_transformed_times = [[] for i in range(dim)]

    # Initialise values
    tb, mb = tList[1]
    # Compensator between beginning and first event time
    compensator = mu*(tb - tList[0][0])
    transformed_times += [np.sum(compensator)]
    individual_transformed_times[mb-1] += [compensator[mb - 1, 0]]
    # Intensity before first jump
    ic = mu + alpha[:, [mb - 1]]
    # j=1

    for tc, mc in tList[2:]:
        # First we estimate the compensator
        inside_log = (mu - np.minimum(ic, 0))/mu
        # Restart time
        t_star = tb + np.multiply(beta_1, np.log(inside_log))

        aux = 1/inside_log  # inside_log can't be equal to zero (coordinate-wise)
        #aux = np.minimum(1, aux)
        compensator = (t_star < tc)*(np.multiply(mu, tc-t_star) + np.multiply(beta_1, ic-mu)*(aux - np.exp(-beta*(tc-tb))))

        transformed_times += [np.sum(compensator)]
        counter += compensator
        individual_transformed_times[mc - 1] += [counter[mc - 1, 0]]
        counter[mc - 1] = 0

        # Then, estimation of intensity before next jump.
        if mc > 0:
            ic = mu + np.multiply((ic - mu), np.exp(-beta*(tc-tb)))
            ic += alpha[:, [mc - 1]]

        tb = tc
    #print("transformed_times", individual_transformed_times[1][0:10])
    return transformed_times, individual_transformed_times


if __name__ == "__main__":
    number = 1
    theta = param_dict[number]
    print(theta)
    dim = int(np.sqrt(1 + theta.shape[0]) - 1)

    mu = np.array(theta[:dim]).reshape((dim, 1))
    alpha = np.array(theta[dim:dim * (dim + 1)]).reshape((dim, dim))
    beta = np.array(theta[dim * (dim + 1):]).reshape((dim, 1))

    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=1000)

    hawkes.simulate()

    times, _ = time_change(theta, hawkes.timestamps)

    p_value_gen = kstest(times, cdf="expon").pvalue

    sns.set_theme()

    fig, ax = plt.subplots()
    plt.suptitle(f"pvalue = {p_value_gen}")
    ax.hist(times, density=True, bins=50)
    x = np.linspace(0,20, 1000)
    ax.plot(x, np.exp(-x))

    ###### FALSE COMPENSATOR

    timesf, _ = time_change(theta+0.05, hawkes.timestamps)

    p_value_genf = kstest(timesf, cdf="expon").pvalue

    figf, axf = plt.subplots()
    plt.suptitle(f"pvalue = {p_value_genf}")
    axf.hist(timesf, density=True, bins=50)
    x = np.linspace(0, 20, 1000)
    axf.plot(x, np.exp(-x))


    plt.show()