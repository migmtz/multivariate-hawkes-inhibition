import numpy as np
import csv
from ast import literal_eval as make_tuple
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from matplotlib import pyplot as plt
import seaborn as sns
from dictionary_parameters import dictionary as param_dict
from time_change import time_change
from scipy.stats import kstest


def obtain_average_estimation(file_name, number, dim, number_estimations):
    n = 0
    if file_name[0:4] == "tick":
        if file_name[5:9] == "beta":
            result = np.zeros((dim + dim * dim * dim,))
        else:
            result = np.zeros((dim + dim * dim,))
    else:
        result = np.zeros((2 * dim + dim * dim,))
    with open("estimation_"+str(number)+'_file/_estimation'+str(number)+file_name, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                result += np.array([float(i) for i in row])
                n += 1
    result /= n

    return result


if __name__ == "__main__":

    number = 9
    print("Estimation number ", str(number))
    theta = param_dict[number]
    print(theta)
    dim = int(np.sqrt(1 + theta.shape[0]) - 1)
    number_estimations = 5
    max_jumps = 5000

    mu = np.array(theta[:dim]).reshape((dim, 1))
    alpha = np.array(theta[dim:dim * (dim + 1)]).reshape((dim, dim))
    beta = np.array(theta[dim * (dim + 1):]).reshape((dim, 1))

    # plot_names = ["", "threshgrad20.0", "tick", "tick_bfgs"]

    plot_names = ["grad", "threshgrad3.0", "threshgrad10.0", "threshgrad15.0", "threshgrad20.0", "threshgrad25.0", "approx", "tick_bfgs"]
    estimations = [obtain_average_estimation(file_name, number, dim, number_estimations) for file_name in plot_names]

    #print(estimations)
    p_values = np.zeros((len(plot_names)+1, dim+1)) # As in the table
    number_simulations = number_estimations
    for i in range(number_simulations):
        np.random.seed(1000+i)
        hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)
        hawkes.simulate()
        test_times = hawkes.timestamps

        test_transformed, transformed_dimensional = time_change(theta, test_times)
        p_values[0, dim] += kstest(test_transformed, cdf="expon", mode="exact").pvalue
        for ref, i in enumerate(transformed_dimensional):
            p_values[0, ref] += kstest(i, cdf="expon", mode="exact").pvalue
        # sns.set_theme()
        #
        # fig, ax = plt.subplots()

        for ref, file_name in enumerate(plot_names):
            if file_name[0:4] == "tick":
                test_transformed, transformed_dimensional = time_change(np.concatenate((estimations[ref],beta.squeeze())), test_times)
            else:
                test_transformed, transformed_dimensional = time_change(estimations[ref], test_times)

            p_values[ref+1, dim] += kstest(test_transformed, cdf="expon", mode="exact").pvalue
            for ref_dim, i in enumerate(transformed_dimensional):
                p_values[ref+1, ref_dim] += kstest(i, cdf="expon", mode="exact").pvalue

    p_values /= number_simulations
    p_values = np.round(p_values, 3)
    print(np.mean(np.array(p_values), axis=1))
    print(np.max(np.mean(np.array(p_values), axis=1)[2:-2]))
    print("Real values p-value: ", p_values[0])
    for ref, file_name in enumerate(plot_names):
        print(file_name + " estimated values p-value: ", p_values[ref+1])

    a = p_values
    print(" \\\\\n".join([" & ".join(map(str, line)) for line in a]))

    fig,ax = plt.subplots()
    for ref,j in enumerate(p_values):
        if ref == 0:
            ax.scatter([i for i in range(dim + 1)], np.sort(j), label=str(ref))
        else:
            ax.scatter([i for i in range(dim+1)], np.sort(j),label=plot_names[ref-1])
    ax.plot([i for i in range(dim+1)], [(i*0.05)/(dim+1) for i in range(dim+1)])

    #         if file_name[0:4] == "" or file_name[0:3] == "pen":
    #             ax.scatter([i for i in range(dim)], np.sort(plist), label=file_name)
    #
    #     ax.plot([i for i in range(dim)], [0.05 for i in range(dim)], c="g", linestyle="dashed")
    #     ax.plot([i for i in range(dim)], [0.05/(dim-i) for i in range(dim)], c="b", linestyle="dashed")
    #
    # if dim > 5:
    plt.legend()
    plt.show()
