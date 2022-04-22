import numpy as np
import csv
from ast import literal_eval as make_tuple
import seaborn as sns
from dictionary_parameters import dictionary as param_dict
from matplotlib import pyplot as plt
from class_and_func.colormaps import get_continuous_cmap
from metrics import relative_squared_loss


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
    dim = 10
    number = 3
    theta = param_dict[number]
    mu = theta[:dim]
    alpha = theta[dim:-dim].reshape((dim, dim))
    beta = theta[-dim:]
    number_estimations = 1
    annot = False

    plot_names = ["","thresh25.0","thresh20.0","thresh15.0","thresh10.0", "thresh5.0", "thresh1.0", "penC1"]
    num1 = [0] +[i for i in range(len(plot_names)-2, 0, -1)] +[len(plot_names)-1]
    estimations = []

    for file_name in plot_names:
        estimations += [obtain_average_estimation(file_name, number, dim, number_estimations)]

    sns.set_theme()
    fig, ax = plt.subplots(len(plot_names), 3)
    hex_list = ['#FF3333', '#FFFFFF', '#33FF49']

    heat_matrix = alpha / beta
    sign = heat_matrix.copy()
    sign[sign != 0] = -1
    sign[sign == 0] = 1
    print(sign)
    sns.heatmap(heat_matrix, ax=ax[0][0], cmap=get_continuous_cmap(hex_list), center=0, annot=annot)

    aux_0 = []
    aux_non_0 = []

    for ref, estimation in enumerate(estimations):
        if plot_names[ref][0:4] == "tick":
            if plot_names[ref][5:9] == "beta":
                mu_est = estimation[:dim]
                alpha_est = np.mean(estimation[dim:].reshape((dim, dim, dim)), axis=0)
                beta_est = beta
            else:
                mu_est = estimation[:dim]
                alpha_est = estimation[dim:].reshape((dim, dim))
                beta_est = beta
        else:
            mu_est = estimation[:dim]
            alpha_est = estimation[dim:-dim].reshape((dim, dim))
            beta_est = estimation[-dim:]
            print("Error estimation"+plot_names[ref], relative_squared_loss(theta, estimation))
        alpha_est[np.abs(alpha_est) <=1e-16] = 0
        heat_estimated = alpha_est / beta_est

        print(num1[ref])

        sns.heatmap(heat_estimated, ax=ax[num1[ref]][1], cmap=get_continuous_cmap(hex_list), center=0,
                    annot=annot)
        aux = sign*np.abs(np.abs(np.sign(heat_matrix)) - np.abs(np.sign(heat_estimated)))
        good_non_0 = 1-np.sum(sign*np.abs(np.abs(np.sign(heat_matrix)) - np.abs(np.sign(heat_estimated))) == -1)/(np.sum(sign == -1))
        good_0 = 1-np.sum(sign*np.abs(np.abs(np.sign(heat_matrix)) - np.abs(np.sign(heat_estimated))) == 1)/(np.sum(sign == 1))
        sns.heatmap(aux,
                    ax=ax[num1[ref]][2], cmap=get_continuous_cmap(hex_list), center=0,
                    annot=annot)
        ax[num1[ref]][1].set_title(plot_names[ref])
        ax[num1[ref]][2].set_title(str(np.round(good_0,2))+" "+str(np.round(good_non_0,2)))

        if ref != 0 and ref != len(plot_names)-1:
            aux_0 += [good_0]
            aux_non_0 += [good_non_0]

    fig2, ax2 = plt.subplots()

    x = [1, 5, 10, 15, 20, 25]
    x.reverse()
    ax2.scatter(x, aux_0, label="0")
    ax2.scatter(x, aux_non_0, label="non-0")

    ax2.legend()
    plt.show()