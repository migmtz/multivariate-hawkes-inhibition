import numpy as np
from matplotlib import pyplot as plt
from class_and_func.hawkes_process import exp_thinning_hawkes
import seaborn as sns

if __name__ == "__main__":

    # Set seed
    np.random.seed(0)
    sns.set_theme()

    lambda_0 = 1.05
    alpha = -0.7
    beta = 0.8
    
    # Create a process with given parameters and maximal number of jumps.
    hawkes = exp_thinning_hawkes(lambda_0=lambda_0, alpha=alpha, beta=beta, max_jumps=15)
    hawkes.simulate()
    
    # Plotting function of intensity and step functions.
    hawkes.plot_intensity(plot_N=True)

    plt.show()
