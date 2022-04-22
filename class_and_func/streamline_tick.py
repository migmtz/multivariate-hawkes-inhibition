import numpy as np
from tick.hawkes import HawkesExpKern, HawkesSumExpKern


def four_estimation(beta, tList, penalty="l2", C=[1e3 for i in range(4)]):
    """
    Estimation of 4 parameters with tick.

    First 2 using expKern both with no penalty and a given penalty (default: l2)

    Second 2 using sumExpKern in the same manner.

    For the latter, ideally it would be important to find that for the first matrix all but the first line are null, and
    so on.
    """
    dim = beta.shape[0]
    list_tick = [[np.array([t for t, m in tList if (m - 1) == i]) for i in range(dim)]]

    beta_tick = np.double(np.c_[beta, beta])

    # Estimation with exponential kernel for each interaction
    expKern = HawkesExpKern(decays=beta_tick, C=C[0])#, penalty="none")
    expKern_bfgs = HawkesExpKern(decays=beta_tick, solver="bfgs", penalty=penalty, C=C[1])

    # Estimation with sum of exponential kernel. Normally we are trying to see if by providing a grid of search for beta
    # the algorithm can find the correct parameters for each process.
    # In this case, the grid consists on the true parameters beta.
    sumexpKern = HawkesSumExpKern(decays=beta.squeeze(), C=C[2])#, penalty="elasticnet")
    sumexpKern_bfgs = HawkesSumExpKern(decays=beta.squeeze(), solver="bfgs", penalty=penalty, C=C[3])

    expKern.fit(list_tick)
    expKern_bfgs.fit(list_tick)
    sumexpKern.fit(list_tick)
    sumexpKern_bfgs.fit(list_tick)

    beta_aux = np.array([[[beta[i,0] for k in range(dim)] for j in range(dim)] for i in range(dim)])

    params = [(expKern.baseline, (expKern.adjacency)*beta),
              (expKern_bfgs.baseline, (expKern_bfgs.adjacency) * beta),
              (sumexpKern.baseline, (sumexpKern.adjacency)*beta_aux),
              (sumexpKern_bfgs.baseline, (sumexpKern_bfgs.adjacency) * beta_aux)]

    return params


def four_estimation_with_grid(beta, beta_grid, tList, penalty="l2", C=1e3):
    """
    Estimation of 4 parameters with tick.

    First 2 using expKern both with no penalty and a given penalty (default: l2)

    Second 2 using sumExpKern in the same manner.

    For the latter, ideally it would be important to find that for the first matrix all but the first line are null, and
    so on.
    """
    dim = beta.shape[0]
    list_tick = [[np.array([t for t, m in tList if (m - 1) == i]) for i in range(dim)]]

    beta_tick = np.double(np.c_[beta, beta])

    # Estimation with exponential kernel for each interaction
    expKern = HawkesExpKern(decays=beta_tick, C=10*C,penalty="elasticnet", elastic_net_ratio=0.9, random_state=10)
    expKern_bfgs = HawkesExpKern(decays=beta_tick, solver="bfgs", penalty=penalty, C=C)

    # Estimation with sum of exponential kernel. Normally we are trying to see if by providing a grid of search for beta
    # the algorithm can find the correct parameters for each process.
    # In this case, the grid consists on the true parameters beta.
    sumexpKern = HawkesSumExpKern(decays=beta_grid, C=10*C,penalty="elasticnet", elastic_net_ratio=0.9,random_state=10)
    sumexpKern_bfgs = HawkesSumExpKern(decays=beta_grid, solver="bfgs", penalty=penalty, C=C)

    expKern.fit(list_tick)
    expKern_bfgs.fit(list_tick)
    sumexpKern.fit(list_tick)
    sumexpKern_bfgs.fit(list_tick)

    beta_aux = np.array([[beta_grid for j in range(dim)] for i in range(dim)])

    params = [(expKern.baseline, (expKern.adjacency)*beta),
              (expKern_bfgs.baseline, (expKern_bfgs.adjacency) * beta),
              (sumexpKern.baseline, (sumexpKern.adjacency)*beta_aux),
              (sumexpKern_bfgs.baseline, (sumexpKern_bfgs.adjacency) * beta_aux)]

    return params


def plot_four(params, beta, ax, c=("g", "b"), x=None):
    dim = beta.shape[0]
    if x is None:
        x_div = np.linspace(0, 2, 100)
    else:
        x_div = x

    for i in range(dim):
        for j in range(dim):
            ax[i, j].plot(x_div, params[0][1][i, j]*np.exp(-beta[i]*x_div), c=c[0], label="ExpKern",
                          linestyle=(0, (5, 10)), alpha=0.5)
            ax[i, j].plot(x_div, params[1][1][i, j] * np.exp(-beta[i] * x_div), c=c[0], label="ExpKern bfgs",
                          linestyle="-", alpha=0.5)

            ax[i, j].plot(x_div, np.sum([params[2][1][i, j][u] * np.exp(-beta[u] * x_div) for u in range(dim)], axis=0),
                          c=c[1], label="sumExpKern", linestyle=(0, (3, 5, 1, 5, 1, 5)), alpha=0.5)
            ax[i, j].plot(x_div, np.sum([params[3][1][i, j][u] * np.exp(-beta[u] * x_div) for u in range(dim)], axis=0),
                          c=c[1], label="sumExpKern bfgs", linestyle="-", alpha=0.5)
