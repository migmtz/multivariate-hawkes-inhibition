import numpy as np
import csv
from ast import literal_eval as make_tuple
from class_and_func.estimator_class import multivariate_estimator_bfgs
from matplotlib import pyplot as plt
from metrics import relative_squared_loss
from dictionary_parameters import dictionary as param_dict
import seaborn as sns


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
                if file_name[5:9] == "beta":
                    print(result)
                n += 1
    result /= n

    return result


dict_names = {"":0, "grad":0, "threshgrad3.0":1, "threshgrad5.0":1, "threshgrad10.0":1, "tick":2, "tick_bfgs":3, "tick_beta":4, "tick_beta_bfgs":5, "approx":2}
styles = ["solid", "dashdot", "dashed", "dashed", "dotted", "dotted"]
colors = ["orange", "orange", "g", "b", "g", "b"]


if __name__ == "__main__":
    number = 9
    theta = param_dict[number]
    dim = int(np.sqrt(1 + theta.shape[0]) - 1)
    number_estimations = 25

    plot_names = ["grad", "threshgrad3.0", "approx", "tick_bfgs", "tick_beta_bfgs"]
    labels = ["MLE", "MLE-0.03", "Approx", "Lst-sq", "Grid-lst-sq"]
    estimations = []

    for file_name in plot_names:
        estimations += [obtain_average_estimation(file_name, number, dim, number_estimations)]

    # print("Error estimation", relative_squared_loss(theta, theta_estimated))
    # print("Error penalized", relative_squared_loss(theta, theta_pen))
    # print("Error Tick", relative_squared_loss(theta[:-dim], theta_tick))

    ####### PLOT

    sns.set_theme()

    fig, ax = plt.subplots(dim, dim)

    x = np.linspace(0, 1, 100)
    x_real = np.linspace(-0.05, 1.05, 102)

    for i in range(dim):
        for j in range(dim):
            ax[i, j].plot(x_real, theta[dim + dim*i + j] * np.exp(-theta[dim + dim*dim + i] * x_real), c="r", label="True kernel")
            for ref, estimation in enumerate(estimations):
                # print(ref)
                if plot_names[ref][0:4] == "tick":
                    if plot_names[ref][5:9] == "beta":
                        print(dim + dim * dim * i + dim * j + 0, dim + dim * dim * i + dim * j + 1)
                        ax[i, j].plot(x, np.sum([estimation[dim + dim * dim * i + dim * j + u] * np.exp(-theta[dim + dim * dim + u] * x) for u in range(dim)], axis=0),
                                      c=colors[dict_names[plot_names[ref]]],
                                      label=labels[ref], linestyle=styles[dict_names[plot_names[ref]]], alpha=0.5)
                    else:
                        ax[i, j].plot(x, estimation[dim + dim * i + j] * np.exp(-theta[dim + dim * dim + i] * x), c=colors[dict_names[plot_names[ref]]],
                                  label=labels[ref], linestyle=styles[dict_names[plot_names[ref]]], alpha=0.5)
                elif plot_names[ref][0:4] == "thre":
                    ax[i, j].plot(x, estimation[dim + dim * i + j] * np.exp(-estimation[dim + dim * dim + i] * x),
                        c=colors[dict_names[plot_names[ref]]], linestyle=styles[dict_names[plot_names[ref]]],
                        label=labels[ref], alpha=0.5, marker="X", markevery=5)
                else:
                    # print(dim + dim * i + j)
                    ax[i, j].plot(x, estimation[dim + dim * i + j] * np.exp(-estimation[dim + dim * dim + i] * x), c=colors[dict_names[plot_names[ref]]], linestyle=styles[dict_names[plot_names[ref]]],label=labels[ref], alpha=0.5)
    plt.legend()
    plt.show()