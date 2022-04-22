import numpy as np
from matplotlib import pyplot as plt
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
import matplotlib
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"
import seaborn as sns


if __name__ == "__main__":
    # Set seed
    matplotlib.rcParams.update({'font.size': 14})
    sns.set_theme()

    np.random.seed(3)

    dim = 2

    mu = np.array([0.8, 1.0])
    alpha = np.array([[-1.9, 3], [0.9, -0.7]])
    beta = np.array([[2, 20], [3, 2]])

    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=15 * (dim - 1))

    # Create a process with given parameters and maximal number of jumps.

    hawkes.simulate()
    print(hawkes.timestamps[0:4])

    fig, ax = plt.subplots(2,1, figsize=[1.5*5.3, 1.3*3.5], sharey=True)
    ax[0].plot([-1, 3], [0, 0], c="k", alpha=0.75, linewidth=1)
    ax[1].plot([-1, 3], [0, 0], c="k", alpha=0.75, linewidth=1)

    # Plotting function of intensity and step functions.
    hawkes.plot_intensity(ax=ax, plot_N=False)

    ax[0].scatter([t for t,m in hawkes.timestamps[1:4]], [0,0,0], c="k", marker="x", linewidths=1)
    ax[1].scatter([t for t, m in hawkes.timestamps[1:4]], [0, 0, 0], c="k", marker="x", linewidths=1)

    ax[0].annotate("$T_{(1)}$", xy=(hawkes.timestamps[1][0], 0), xytext=(hawkes.timestamps[1][0]-0.1, -0.6),
                annotation_clip=False)
    #ax[0].annotate(f"$T_1^\star$", xy=(hawkes.timestamps[1][0], 0), xytext=(hawkes.timestamps[1][0] - 0.2, hawkes.intensity_jumps[0][1]-0.15),
                #annotation_clip=False)

    #ax[0].annotate(f"$T_2$", xy=(hawkes.timestamps[2][0], 0), xytext=(hawkes.timestamps[2][0] + 0.01, -0.6),
                #annotation_clip=False)
    #ax[0].annotate(f"$T_2^\star$", xy=(hawkes.timestamps[2][0], 0), xytext=(2.5, -0.06),
                #annotation_clip=False)

    ax[0].annotate("$T_{(3)}$", xy=(hawkes.timestamps[3][0], 0), xytext=(hawkes.timestamps[3][0] - 0.1, -0.6),
                annotation_clip=False)
    #ax[0].annotate(f"$T_3^\star$", xy=(hawkes.timestamps[3][0], 0), xytext=(6.1, -0.06),
                #annotation_clip=False)

    #ax[0].annotate(f"$T_4$", xy=(hawkes.timestamps[4][0], 0), xytext=(hawkes.timestamps[4][0] - 0.15, -0.1),
                #annotation_clip=False)
    #ax[0].annotate(f"$T_4^\star$", xy=(hawkes.timestamps[4][0], 0), xytext=(hawkes.timestamps[4][0] - 0.15, -0.15),
                #annotation_clip=False)

    #ax[1].annotate(f"$T_1$", xy=(hawkes.timestamps[1][0], 0), xytext=(hawkes.timestamps[1][0]-0.07, -0.6),
                   #annotation_clip=False)
    # ax[0].annotate(f"$T_1^\star$", xy=(hawkes.timestamps[1][0], 0), xytext=(hawkes.timestamps[1][0] - 0.2, hawkes.intensity_jumps[0][1]-0.15),
    # annotation_clip=False)

    ax[1].annotate("$T_{(2)}$", xy=(hawkes.timestamps[2][0], 0), xytext=(hawkes.timestamps[2][0]-0.03, -0.6),
                   annotation_clip=False)
    # ax[0].annotate(f"$T_2^\star$", xy=(hawkes.timestamps[2][0], 0), xytext=(2.5, -0.06),
    # annotation_clip=False)

    #ax[1].annotate(f"$T_3$", xy=(hawkes.timestamps[3][0], 0), xytext=(hawkes.timestamps[3][0]-0.07, -0.6),
                   #annotation_clip=False)


    aux = [alpha[0, m-1] for t,m in hawkes.timestamps[1:4]]
    ax[0].scatter([t for t,m in hawkes.timestamps[1:4]], np.array(hawkes.intensity_jumps[0][1:4]) - np.array(aux), linewidths=1.5, s=60,
                 facecolors='none', edgecolors='r')
    ax[0].scatter([t for t, m in hawkes.timestamps[1:4]], np.array(hawkes.intensity_jumps[0][1:4]),
                  s=15, c="r", zorder=3)
    aux = [alpha[1, m - 1] for t, m in hawkes.timestamps[1:4]]
    ax[1].scatter([t for t, m in hawkes.timestamps[1:4]], np.array(hawkes.intensity_jumps[1][1:4]) - np.array(aux),
                  linewidths=1.5, s=60,
                  facecolors='none', edgecolors='r')
    ax[1].scatter([t for t, m in hawkes.timestamps[1:4]], np.array(hawkes.intensity_jumps[1][1:4]),
                  s=15, c="r", zorder=3)
    # plt.scatter([0.2, 1.7, 5.75, 11.05, 15], np.array(hawkes.intensity_jumps[1:]), s=15, c="r", zorder=3)

    ax[0].set_xlim((0, 2))
    ax[1].set_xlim((0, 2))
    ax[0].set_ylim((-1.5, 3))

    ax[0].set_ylabel("$\lambda^1$")
    ax[1].set_ylabel("$\lambda^2$")

    ax[1].set_xlabel("$t$")

    plt.savefig('timesMarkedMulti.pdf', bbox_inches='tight', format="pdf", quality=90)

    plt.show()
