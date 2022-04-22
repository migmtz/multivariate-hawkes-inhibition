import numpy as np
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.likelihood_functions import multivariate_loglikelihood_simplified
from scipy.optimize import check_grad, approx_fprime


def grad(theta, tList, dim=None, dimensional=False):
    # mu = np.array(theta[:dim]).reshape((dim, 1))
    # alpha = np.array(theta[dim:dim * (dim + 1)]).reshape((dim, dim))
    # beta = np.array(theta[dim * (dim + 1):]).reshape((dim, 1))
    mu, alpha, beta = (i.copy() for i in theta)
    beta = beta + 1e-10

    beta_1 = 1 / beta

    timestamps = tList.copy()

    # We first check if we have the correct beginning and ending.
    if timestamps[0][1] > 0:
        timestamps = [(0, 0)] + timestamps
    if timestamps[-1][1] > 0:
        timestamps += [(timestamps[-1][0], 0)]

    # Initialise values
    tb, mb = timestamps[1]
    # Initialize grad
    grad_mu = np.zeros((dim,1))
    grad_alpha = np.zeros((dim, dim))
    grad_beta = np.zeros((dim,1))
    # For first interval/jump
    grad_mu += tb
    grad_mu[mb-1] -= 1/mu[mb-1]

    # auxiliary for C(ai, bi)
    dA = np.zeros((dim, dim))
    dB = np.zeros((dim, 1))

    dA[:, mb-1] += 1
    dB += tb*alpha[:, [mb-1]]

    ic = mu + alpha[:, [mb - 1]]
    # j=1

    for tc, mc in timestamps[2:]:
        # Restart time and auxiliaries
        inside_log = (mu - np.minimum(ic, 0)) / mu
        t_star = tb + np.multiply(beta_1, np.log(inside_log))
        exp_tpu = np.exp(-beta*(tc-tb))
        aux = 1 / inside_log  # inside_log can't be equal to zero (coordinate-wise)

        #### grad_comp for mu
        grad_mu += (t_star < tc)*(tc - t_star)

        # grad Cn wrt alpha
        grad_alpha += (t_star < tc)*(beta_1*dA*(aux - exp_tpu))

        #### grad_comp for alpha
        grad_beta += (t_star < tc) * (
                    beta_1 * (dB - tb*(ic - mu)) * (aux - exp_tpu) + beta_1 * (ic - mu) * (
                        tc - tb) * exp_tpu - (beta_1**2)*(ic-mu)*(aux-exp_tpu) + beta_1 * mu * (t_star - tb))
        # grad_beta += (t_star < tc)*(beta_1*((tb - beta_1)*(ic-mu) - dB)*(aux - exp_tpu) + beta_1*(ic-mu)*(tc-tb)*exp_tpu + beta_1*mu*(t_star-tb))

        if mc > 0:
            old_ic = ic
            ic = mu + np.multiply((ic - mu), np.exp(-beta * (tc - tb)))

            if ic[mc - 1] <= 0.0:
                pass
            #     # print("oh no")
            #     # res = 1e8
            #
            #     # rayon_spec = np.max(np.abs(np.linalg.eig(np.abs(alpha) / beta)[0]))
            #     # rayon_spec = min(rayon_spec, 0.999)
            #     # if 2*(tList[-1][0])/(1-rayon_spec) < 0:
            #     #     print("wut")
            #     # res = 2*(tList[-1][0])/(1-rayon_spec)
            #
            #     res = 1e8  # *(np.sum(mu**2) + np.sum(alpha**2) + np.sum(beta**2))
            #     return res
            else:
                grad_mu[mc-1] -= 1/ic[mc-1]
                grad_alpha[[mc-1], :] -= (dA[[mc-1], :]*exp_tpu[mc-1])/(ic[mc-1])
                grad_beta[mc-1] -= ((dB[mc-1] - tc*(old_ic[mc-1] - mu[mc-1]))*exp_tpu[mc-1])/(ic[mc-1])
            #     log_i[mc - 1] += np.log(ic[mc - 1])

            # j += 1
            # print(j, ic, 1+0.45*(1-0.5**(j-1)))
            ic += alpha[:, [mc - 1]]
            dA *= exp_tpu
            dA[:, mc-1] += 1
            dB = exp_tpu*dB + tc*alpha[:, [mc-1]]

        tb = tc
    grad_comp = np.concatenate((grad_mu, np.ravel(grad_alpha).reshape((dim*dim, 1)), grad_beta))
    return grad_comp.squeeze()


def multivariate_loglikelihood_with_grad(theta, tList, dim=None, dimensional=False):
    """

    Parameters
    ----------
    theta : tuple of array
        Tuple containing 3 arrays. First corresponds to vector of baseline intensities mu. Second is a square matrix
        corresponding to interaction matrix alpha. Last is vector of recovery rates beta.

    tList : list of tuple
        List containing tuples (t, m) where t is the time of event and m is the mark (dimension). The marks must go from
        1 to nb_of_dimensions.
        Important to note that this algorithm expects the first and last time to mark the beginning and
        the horizon of the observed process. As such, first and last marks must be equal to 0, signifying that they are
        not real event times.
        The algorithm checks by itself if this condition is respected, otherwise it sets the beginning at 0 and the end
        equal to the last time event.
    dim : int
        Number of processes only necessary if providing 1-dimensional theta. Default is None
    dimensional : bool
        Whether to return the sum of loglikelihood or decomposed in each dimension. Default is False.
Returns
    -------
    likelihood : array of float
        Value of likelihood at each process.
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """

    if isinstance(theta, np.ndarray):
        if dim is None:
            raise ValueError("Must provide dimension to unpack correctly")
        else:
            mu = np.array(theta[:dim]).reshape((dim, 1))
            alpha = np.array(theta[dim:dim * (dim + 1)]).reshape((dim, dim))
            beta = np.array(theta[dim * (dim + 1):]).reshape((dim, 1))
    else:
        mu, alpha, beta = (i.copy() for i in theta)
    beta = beta + 1e-10

    beta_1 = 1/beta

    timestamps = tList.copy()

    # We first check if we have the correct beginning and ending.
    if timestamps[0][1] > 0:
        timestamps = [(0, 0)] + timestamps
    if timestamps[-1][1] > 0:
        timestamps += [(timestamps[-1][0], 0)]

    # Initialise values
    tb, mb = timestamps[1]
    # Compensator between beginning and first event time
    compensator = mu*(tb - timestamps[0][0])
    # Intensity before first jump
    log_i = np.zeros((alpha.shape[0],1))
    log_i[mb-1] = np.log(mu[mb-1])

    ############# Gradient

    # Initialize grad
    grad_mu = np.zeros((dim, 1))
    grad_alpha = np.zeros((dim, dim))
    grad_beta = np.zeros((dim, 1))
    # For first interval/jump
    grad_mu += tb
    grad_mu[mb - 1] -= 1 / mu[mb - 1]

    # auxiliary for C(ai, bi)
    dA = np.zeros((dim, dim))
    dB = np.zeros((dim, 1))

    dA[:, mb - 1] += 1
    dB += tb * alpha[:, [mb - 1]]

    ic = mu + alpha[:, [mb - 1]]
    # j=1

    for tc, mc in timestamps[2:]:
        # First we estimate the compensator
        inside_log = (mu - np.minimum(ic, 0))/mu
        # Restart time
        t_star = tb + np.multiply(beta_1, np.log(inside_log))
        exp_tpu = np.exp(-beta * (tc - tb))
        aux = 1/inside_log  # inside_log can't be equal to zero (coordinate-wise)

        compensator += (t_star < tc)*(np.multiply(mu, tc-t_star) + np.multiply(beta_1, ic-mu)*(aux - exp_tpu))

        ############ Gradient

        #### grad_comp for mu
        grad_mu += (t_star < tc) * (tc - t_star)

        # grad Cn wrt alpha
        grad_alpha += (t_star < tc) * (beta_1 * dA * (aux - exp_tpu))

        #### grad_comp for alpha
        grad_beta += (t_star < tc) * (
                beta_1 * (dB - tb * (ic - mu)) * (aux - exp_tpu) + beta_1 * (ic - mu) * (
                tc - tb) * exp_tpu - (beta_1 ** 2) * (ic - mu) * (aux - exp_tpu) + beta_1 * mu * (t_star - tb))

        # Then, estimation of intensity before next jump.
        if mc > 0:
            old_ic = ic
            ic = mu + np.multiply((ic - mu), np.exp(-beta*(tc-tb)))

            if ic[mc - 1] <= 0.0:
                # print("oh no")
                # res = 1e8

                # rayon_spec = np.max(np.abs(np.linalg.eig(np.abs(alpha) / beta)[0]))
                # rayon_spec = min(rayon_spec, 0.999)
                # if 2*(tList[-1][0])/(1-rayon_spec) < 0:
                #     print("wut")
                # res = 2*(tList[-1][0])/(1-rayon_spec)

                res = 1e8 #*(np.sum(mu**2) + np.sum(alpha**2) + np.sum(beta**2))
                return res, np.zeros((dim*dim+2*dim, 1))
            else:
                log_i[mc-1] += np.log(ic[mc - 1])

                ######## Gradient

                grad_mu[mc - 1] -= 1 / ic[mc - 1]
                grad_alpha[[mc - 1], :] -= (dA[[mc - 1], :] * exp_tpu[mc - 1]) / (ic[mc - 1])
                grad_beta[mc - 1] -= ((dB[mc - 1] - tc * (old_ic[mc - 1] - mu[mc - 1])) * exp_tpu[mc - 1]) / (ic[mc - 1])
            # j += 1
            # print(j, ic, 1+0.45*(1-0.5**(j-1)))
            ic += alpha[:, [mc - 1]]

            ######### Gradient

            dA *= exp_tpu
            dA[:, mc - 1] += 1
            dB = exp_tpu * dB + tc * alpha[:, [mc - 1]]

        tb = tc
    likelihood = log_i - compensator
    grad_comp = np.concatenate((grad_mu, np.ravel(grad_alpha).reshape((dim * dim, 1)), grad_beta))
    if not(dimensional):
        likelihood = np.sum(likelihood)
    return -likelihood, grad_comp.squeeze()


if __name__ == "__main__":
    np.random.seed(0)

    dim = 2  # 2, 3 ou 4

    mu = np.random.uniform(0, 5, (dim, 1))
    alpha = np.random.uniform(-1, 1, (dim, dim))
    beta = np.random.uniform(0, 5, (dim, 1))
    theta = (mu, alpha, beta)
    print(mu, "\n", alpha, "\n", beta)
    max_jumps = 5000

    ################# SIMULATION
    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)

    print("Starting simulation...")
    hawkes.simulate()
    print("Finished simulation")

    ################# gard

    print(grad((np.ones((2,1)), np.ones((2,2)), np.ones((2,1))), hawkes.timestamps, dim=2))

    print("checking...")
    print(multivariate_loglikelihood_simplified(theta, hawkes.timestamps, dim))
    print(multivariate_loglikelihood_with_grad(theta, hawkes.timestamps, dim))
    print(approx_fprime(np.concatenate((mu, alpha.ravel().reshape(dim*dim,1), beta)).squeeze(), multivariate_loglikelihood_simplified, 1e-6,hawkes.timestamps, dim))
    # print(check_grad(multivariate_loglikelihood_simplified, grad, theta, hawkes.timestamps, dim))