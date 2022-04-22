import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
from class_and_func.hawkes_process import exp_thinning_hawkes
from matplotlib import rcParams
import pickle

rcParams['font.family'] = 'serif'

#########
## This script allows to obtain the boxplots that appear in our paper.
## SavedBoxplots files are generated through the error_and_pvalue.py file.
## This file requires pickle.


if __name__ == "__main__":

    prefixed = 6  # Number of parameter sets.

    box = []
    positions1 = []
    positions2 = []
    proportion_list = []
    theta_list = []

    ## Extraction of data

    for file_nb in range(prefixed):
        with open("SavedBoxplots/SavedBoxplots" + str(file_nb+1), 'rb') as saved:

            lambda_0, alpha, beta, max_jumps, iterations, box_approx, box_real, proportion = pickle.load(saved)
            box += [(box_real, box_approx)]

            ## Positions of boxplots
            positions1 += [8 * file_nb + i * 1.1 for i in range(0, prefixed, 2)]
            positions2 += [8 * file_nb + i * 1.1 + 1.0 for i in range(0, prefixed, 2)]
            theta_list += [[lambda_0, alpha, beta]]
            proportion_list += [proportion]

    ## Ordering by increasing proportion of time equal to 0.
    proportion_list = np.array(proportion_list)
    order = np.argsort(proportion_list)
    theta_list = np.array(theta_list)[order, :]
    proportion_list = proportion_list[order]
    box = np.array(box)[order]
    print("Parameter sets: ", theta_list)

    ## Box, positions, labels and colors.
    final_box = np.zeros((100, 1))
    for i in range(2):
        for j in range(len(order)):
            final_box = np.concatenate((final_box, box[j, i, :, :]), axis=1)

    final_box = final_box[:, 1:]

    positions = positions1 + positions2

    label = ["$\lambda_0^{exact}$", "$\\alpha^{exact}$", "$\\beta^{exact}$"]
    labels = [i for j in range(proportion_list.shape[0]) for i in label]
    label = ["$\lambda_0^{approx}$", "$\\alpha^{approx}$", "$\\beta^{approx}$"]
    labels += [i for j in range(proportion_list.shape[0]) for i in label]

    cmap = cm.get_cmap('tab20b')
    colors = [cmap(0.775), cmap(0.175)]

    ## Figure and boxplot

    fig = plt.figure(figsize=[1.5 * 5.3, 1.5 * 3.5], constrained_layout=True)
    gs = fig.add_gridspec(5, prefixed)

    ax = fig.add_subplot(gs[0:4, :])  # Boxplots.

    intensity_ax = []  # Intensity functions.
    for i in range(prefixed):
        intensity_ax += [fig.add_subplot(gs[4:, i])]

    box_plot = ax.boxplot(final_box, positions=positions, labels=labels, patch_artist=True)

    for i, patch in enumerate(box_plot['boxes']):
        patch.set_facecolor(colors[i // (len(box_plot['boxes']) // 2)])

    empty_string_labels = ['' for item in ax.get_xticklabels()]
    ax.set_xticklabels(empty_string_labels)
    parameters_str = ["$\lambda_0$", "$\\alpha$", "$\\beta$"]

    y_lim = ax.set_ylim(bottom=1e-5, top=1e4)

    for i in range(len(positions1)):
        ax.annotate('', xy=(positions1[i] - 0.1, 5e-6), xytext=(positions2[i] + 0.1, 5e-6),
                    arrowprops=dict(arrowstyle='|-|,widthA=0.3,widthB=0.3', facecolor='red'),
                    annotation_clip=False)  # This enables the arrow to be outside of the plot

        ax.annotate(parameters_str[i % 3], xy=(positions1[i] - 0.1, 1.5e-6), xytext=(positions2[i] - 0.8, 1.5e-6),
                    annotation_clip=False)

    legend_str = ['exact', "approx"]
    legend_elements = [Patch(facecolor=colors[i], edgecolor='k', label=legend_str[i]) for i in range(2)]
    ax.legend(handles=legend_elements, loc='best')

    ax.grid()
    ax.set_yscale('log')

    ## Plot intensity examples for each parameter set.

    for i in range(prefixed):
        np.random.seed(1)
        hawkes = exp_thinning_hawkes(theta_list[i, 0], theta_list[i, 1], theta_list[i, 2], max_jumps=max_jumps)
        hawkes.simulate()
        hawkes.plot_intensity(ax=intensity_ax[i], plot_N=False)
        intensity_ax[i].set_xlim(0, 20)
        intensity_ax[i].legend().set_visible(False)
        intensity_ax[i].set_title(str(round(100 * proportion_list[i], 4)) + "%")

    intensity_ax[0].set_ylim(0.475, 0.5001)

    ## Save image as pdf.
    # plt.savefig('intensity_boxplot_log_cropped.pdf', bbox_inches='tight', format="pdf")

    plt.show()
