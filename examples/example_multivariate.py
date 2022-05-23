import numpy as np
from matplotlib import pyplot as plt
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
import seaborn as sns

from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams["mathtext.fontset"] = "dejavuserif"

if __name__ == "__main__":
    # Set seed
    np.random.seed(0)

    dim = 2  # 2, 3 ou 4

    if dim == 2:

        mu = np.array([0.8, 1.0])
        alpha = np.array([[-1.9, 3], [0.9, -0.7]])
        beta = np.array([[2, 20], [3, 2]])

    elif dim == 3:

        mu = np.array([0.5, 1.0, 1.0])
        alpha = np.array([[-1.9, 3, 0], [0, 0, 0], [1.0, 0, -0.5]])
        beta = np.array([[3, 20, 0], [0, 0, 0], [3, 0, 2]])

    elif dim == 4:

        mu = np.array([0.5, 1.0, 0.7, 0.4])
        alpha = 0.5*np.array([[-1.9, 3, -1.1, 0.5], [0.1, 0.6, 0, -1.3], [1.0, 0, -0.5, 1.7], [0.4, 0.8, 0.5, -1.0]])
        beta = 2.5*np.array([[3, 10, 2.5, 1.2], [1.7, 1.3, 0, 0.9], [3, 0, 1.4, 3.2],[1.2, 1.5, 0.8, 0.8]])

    elif dim == 5:

        mu = np.ones((5,1))
        alpha = np.zeros((5,5))
        beta = np.zeros((5,5))

    else:
        raise ValueError("Invalid dimension")

    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=15*(dim-1))

    # Create a process with given parameters and maximal number of jumps.

    hawkes.simulate()

    print(hawkes.timestamps[0], hawkes.timestamps[-1])

    sns.set_theme()

    fig, ax = plt.subplots(2, 1, sharey=True, sharex=True)

    ax[0].set_ylabel("$\lambda^1$")
    ax[1].set_ylabel("$\lambda^2$")

    ax[1].set_xlabel("$t$")
    hawkes.plot_intensity(ax=ax.T, plot_N=False)

    hawkes.plot_heatmap()

    plt.show()
