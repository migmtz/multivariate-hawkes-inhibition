# Imports
import numpy as np
#import torch

# Functions for exponential estimation of loglikelihood.

def loglikelihood(theta, tList):
    """
    Exact computation of the loglikelihood for an exponential Hawkes process for either self-exciting or self-regulating cases. 
    Estimation for a single realization.
    
    Parameters
    ----------
    theta : tuple of float
        Tuple containing the parameters to use for estimation.
    tList : list of float
        List containing all the lists of data (event times).

    Returns
    -------
    likelihood : float
        Value of likelihood, either for 1 realization or for a batch. 
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """
    # Extract variables
    lambda0, alpha, beta = theta

    # Avoid wrong values in algorithm such as negative lambda0 or beta
    if lambda0 <= 0 or beta <= 0:
        return 1e5

    else:

        compensator_k = lambda0 * tList[1]
        lambda_avant = lambda0
        lambda_k = lambda0 + alpha

        if lambda_avant <= 0:
            return 1e5

        likelihood = np.log(lambda_avant) - compensator_k

        # Iteration
        for k in range(2, len(tList)):

            if lambda_k >= 0:
                C_k = lambda_k - lambda0
                tau_star = tList[k] - tList[k - 1]
            else:
                C_k = -lambda0
                tau_star = tList[k] - tList[k - 1] - (np.log(-(lambda_k - lambda0)) - np.log(lambda0)) / beta

            lambda_avant = lambda0 + (lambda_k - lambda0) * np.exp(-beta * (tList[k] - tList[k - 1]))
            lambda_k = lambda_avant + alpha
            compensator_k = lambda0 * tau_star + (C_k / beta) * (1 - np.exp(-beta * tau_star))

            if lambda_avant <= 0:
                return 1e5

            likelihood += np.log(lambda_avant) - compensator_k

        # We return the opposite of the likelihood in order to use minimization packages.
        return -likelihood


def likelihood_approximated(theta, tList):
    """
    Approximation method for the loglikelihood, proposed by Lemonnier.
    Estimation for a single realization.

    Parameters
    ----------
    theta : tuple of float
        Tuple containing the parameters to use for estimation.
    tList : list of float
        List containing all the lists of data (event times).

    Returns
    -------
    likelihood : float
        Value of likelihood, either for 1 realization or for a batch.
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """
    lambda0, alpha, beta = theta

    # Avoid wrong values in algorithm such as negative lambda0 or beta
    if lambda0 <= 0 or beta <= 0:
        return 1e5

    else:
        # Auxiliary values
        aux = np.log(lambda0)  # Value that will be often used

        # Set initial values and first step of iteration
        A_k_minus = 0
        Lambda_k = 0
        # likelihood = - lambda0*tList[0] + np.log(A_k_minus + lambda0)
        likelihood = - lambda0 * tList[-1] + np.log(A_k_minus + lambda0)
        tLast = tList[1]

        # Iteration
        for k in range(2, len(tList)):

            # Update A(k)
            tNext = tList[k]
            tau_k = tNext - tLast
            A_k = (A_k_minus + alpha)

            # Integral
            Lambda_k = (A_k / beta) * (1 - np.exp(-beta * tau_k))  # + lambda0*tau_k

            # Update likelihood

            A_k_minus = A_k * np.exp(-beta * tau_k)
            if A_k_minus + lambda0 <= 0:
                return 1e5
            likelihood = likelihood - Lambda_k + np.log(lambda0 + A_k_minus)

            # Update B(k) and tLast

            tLast = tNext

        # We return the opposite of the likelihood in order to use minimization packages.
        return -likelihood


def batch_likelihood(theta, nList, exact=True, penalized=False, C=1):
    """
    Wrapper function that allows to call either the exact or penalized loglikelihood functions aswell as an L2-penalization.
    
    This function works either with 1 or multiple (batch) realizations of Hawkes process.
    
    Parameters
    ----------
    theta : tuple of float
        Tuple containing the parameters to use for estimation.
    nList : list of list of float
        List containing all the lists of data (event times).
    exact : bool
        Whether to use the exact computation method (True) or the approximation by Lemonnier. Default is True.
    penalized : bool
        Whether to add an L2-penalization. Default is False.
    C : float
        Penalization factor, only used if penalized parameter is True. Default is 1.

    Returns
    -------
    batchlikelihood : float
        Value of likelihood, either for 1 realization or for a batch.
    """
    batchlikelihood = 0

    if exact:
        func = lambda x, y: loglikelihood(x, y)
    else:
        func = lambda x, y: likelihood_approximated(x, y)

    for tList in nList:
        batchlikelihood += func(theta, tList)
    batchlikelihood /= len(nList)

    if penalized:
        batchlikelihood += C * (theta[0] ** 2 + theta[1] ** 2 + theta[2] ** 2)

    return batchlikelihood


def multivariate_loglikelihood_simplified(theta, tList, dim=None, dimensional=False):
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
    ic = mu + alpha[:, [mb - 1]]
    # j=1

    for tc, mc in timestamps[2:]:
        # First we estimate the compensator
        inside_log = (mu - np.minimum(ic, 0))/mu
        # Restart time
        t_star = tb + np.multiply(beta_1, np.log(inside_log))

        aux = 1/inside_log  # inside_log can't be equal to zero (coordinate-wise)
        aux = np.minimum(1, aux)
        compensator += (t_star < tc)*(np.multiply(mu, tc-t_star) + np.multiply(beta_1, ic-mu)*(aux - np.exp(-beta*(tc-tb))))

        # Then, estimation of intensity before next jump.
        if mc > 0:
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
                return res
            else:
                log_i[mc-1] += np.log(ic[mc - 1])
            # j += 1
            # print(j, ic, 1+0.45*(1-0.5**(j-1)))
            ic += alpha[:, [mc - 1]]

        tb = tc
    likelihood = log_i - compensator
    if not(dimensional):
        likelihood = np.sum(likelihood)
    return -likelihood


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

def multivariate_loglikelihood_with_grad_pen(theta, tList, eps, dim=None, eta=None, C=1.0):
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

    if eta is None:
        eta_p = np.ones((dim,dim))
    else:
        eta_p = eta.reshape((dim,dim))

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
    grad_alpha += C * np.sum(alpha/eta_p)
    grad_comp = np.concatenate((grad_mu, np.ravel(grad_alpha).reshape((dim * dim, 1)), grad_beta))
    likelihood = np.sum(likelihood) + 0.5 * C * np.sum((alpha** 2 + eps) / eta_p) + 0.5 * C * np.sum(eta_p)
    return -likelihood, grad_comp.squeeze()


def multivariate_loglikelihood_jit(theta, timestamps):
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

    dim = 2

    mu = np.zeros((dim,1))
    mu[:,0] = theta[0:dim]
    alpha = np.zeros((dim,dim))
    for i in range(dim):
        alpha[i,:] = theta[dim*(i+1):dim*(i+2)]
    alphaT = alpha.T.copy()
    beta = np.zeros((dim,1))
    beta[:, 0] = theta[-dim:]

    beta_1 = 1/(beta + 1e-10)

    # Initialise values
    tb, mb = timestamps[1]
    # Compensator between beginning and first event time
    compensator = mu * (tb - timestamps[0][0])
    # Intensity before first jump
    log_i = np.zeros((dim, 1))
    log_i[mb - 1] = np.log(mu[mb - 1])
    ic = mu + alphaT[mb - 1,:].reshape((dim, 1))
    # j=1

    for i in range(2, len(timestamps)):
        tc, mc = timestamps[i]
        # First we estimate the compensator
        inside_log = (mu - np.minimum(ic, 0)) / mu
        # Restart time
        t_star = tb + np.multiply(beta_1, np.log(inside_log))

        aux = 1 / inside_log  # inside_log can't be equal to zero (coordinate-wise)
        aux = np.minimum(1, aux)
        compensator += (t_star < tc) * (
                    np.multiply(mu, tc - t_star) + np.multiply(beta_1, ic - mu) * (aux - np.exp(-beta * (tc - tb))))

        # Then, estimation of intensity before next jump.
        if mc > 0:
            ic = mu + np.multiply((ic - mu), np.exp(-beta * (tc - tb)))

            if ic[mc - 1] <= 0.0:
                # print("oh no")
                # res = 1e8

                # rayon_spec = np.max(np.abs(np.linalg.eig(np.abs(alpha) / beta)[0]))
                # rayon_spec = min(rayon_spec, 0.999)
                # if 2*(tList[-1][0])/(1-rayon_spec) < 0:
                #     print("wut")
                # res = 2*(tList[-1][0])/(1-rayon_spec)

                res = 1e8  # *(np.sum(mu**2) + np.sum(alpha**2) + np.sum(beta**2))
                return res
            else:
                log_i[mc - 1] += np.log(ic[mc - 1])
            # j += 1
            # print(j, ic, 1+0.45*(1-0.5**(j-1)))
            ic += alphaT[mc - 1, :].reshape((dim,1))

        tb = tc
    likelihood = log_i - compensator
    return -np.sum(likelihood)


def multivariate_lstsquares_simplified(theta, tList, dim=None):
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
Returns
    -------
    least_squares_error : array of float
        Value of least-squares error at each process.
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """
    # import scipy.integrate as integrate
    if isinstance(theta, np.ndarray):
        if dim is None:
            raise ValueError("Must provide dimension to unpack correctly")
        else:
            mu = np.array(theta[:dim]).reshape((dim, 1))
            alpha = np.array(theta[dim:dim * (dim + 1)]).reshape((dim, dim))
            beta = np.array(theta[dim * (dim + 1):]).reshape((dim, 1))
    else:
        mu, alpha, beta = (i.copy() for i in theta)
    beta = beta + 1e-16

    beta_1 = 1/beta
    beta_sum = beta + beta.T

    timestamps = tList.copy()

    # We first check if we have the correct beginning and ending.
    if timestamps[0][1] > 0:
        timestamps = [(0, 0)] + timestamps
    if timestamps[-1][1] > 0:
        timestamps += [(timestamps[-1][0], 0)]

    # Initialise values
    tb, mb = timestamps[1]
    # Compensator between beginning and first event time
    compensator_sq = (tb - timestamps[0][0])*np.sum(mu**2)
    # Intensity before first jump
    lambda_i = np.zeros((alpha.shape[0],1))
    lambda_i[mb-1] = lambda_i[mb-1] + mu[mb-1]
    ic = mu + alpha[:, [mb - 1]]
    # j=1
    for tc, mc in timestamps[2:]:
        # First we estimate the compensator
        inside_log = (mu - np.minimum(ic, 0))/mu
        # Restart time
        t_star = tb + np.multiply(beta_1, np.log(inside_log))
        # As we consider the cross-interactions in the integral.
        t_star_ij = np.maximum(t_star, t_star.T)

        first_term = (mu*mu.T)*(tc - t_star_ij)
        middle_term = (mu*((ic - mu)*beta_1).T)*(np.exp(-beta*(t_star_ij - tb)) - np.exp(-beta*(tc - tb))).T
        middle_term = middle_term + middle_term.T
        last_term = (((ic - mu)*(ic - mu).T)/beta_sum)*(np.exp(-beta_sum*(t_star_ij - tb)) - np.exp(-beta_sum*(tc - tb)))
        aux = (t_star_ij < tc)*(first_term + middle_term + last_term)

        # for i in range(mu.shape[0]):
        #     for j in range(mu.shape[0]):
        #         function = lambda x: (mu[i] + (ic[i] - mu[i]) * np.exp(-beta[i] * (x - tb)))*(mu[j] + (ic[j] - mu[j]) * np.exp(-beta[j] * (x - tb)))
        #         if t_star_ij[i,j] < tc:
        #             approx = integrate.quad(function, t_star_ij[i,j], tc)[0]
        #             print("Approximation: ", approx, "    Real: ", aux[i,j])
        #         else:
        #             approx = 0
        #             print("Approximation: ", approx, "    Real: ", aux[i,j])
        #         if np.round(approx, 5) != np.round(aux[i,j], 5):
        #             print("WTFFFFFFFFFFFFFFFFFF")
        # print("")

        compensator_sq += np.sum(aux)
        # Then, estimation of intensity before next jump.
        if mc > 0:
            ic = mu + np.multiply((ic - mu), np.exp(-beta*(tc-tb)))

            lambda_i[mc-1] += np.maximum(ic[mc - 1], 0)
            # j += 1
            # print(j, ic, 1+0.45*(1-0.5**(j-1)))
            ic += alpha[:, [mc - 1]]

        tb = tc
    least_squares_error = compensator_sq - (2)*np.sum(lambda_i)
    return least_squares_error


def lstsquares_(theta, tList, ax=None, dim=None):

    if isinstance(theta, np.ndarray):
        if dim is None:
            raise ValueError("Must provide dimension to unpack correctly")
        else:
            mu = np.array(theta[:dim]).reshape((dim, 1))
            alpha = np.array(theta[dim:dim * (dim + 1)]).reshape((dim, dim))
            beta = np.array(theta[dim * (dim + 1):]).reshape((dim, 1))
    else:
        mu, alpha, beta = (i.copy() for i in theta)
    beta[beta == 0] = 1

    beta_1 = 1/beta
    beta_sum = beta + beta.T

    timestamps = tList.copy()

    # We first check if we have the correct beginning and ending.
    if timestamps[0][1] > 0:
        timestamps = [(0, 0)] + timestamps
    if timestamps[-1][1] > 0:
        timestamps += [(timestamps[-1][0], 0)]

    # Initialise values
    tb, mb = timestamps[1]
    # Compensator between beginning and first event time
    compensator_sq = (tb - timestamps[0][0])*np.sum(mu**2)
    # Intensity before first jump
    lambda_i = np.zeros((alpha.shape[0],1))
    # for i in range(dim):
    #     ax[0,i].scatter([tb], [lambda_i[i] + mu[i]], c="g")
    lambda_i[mb-1] = lambda_i[mb-1] + mu[mb-1]
    ic = mu + alpha[:, [mb - 1]]
    # j=1

    for tc, mc in timestamps[2:]:
        # First we estimate the compensator
        inside_log = (mu - np.minimum(ic, 0))/mu
        # Restart time
        t_star = np.zeros((dim,1))
        for i in range(dim):
            t_star[i] = tb + beta_1[i]*np.log(inside_log[i])
        # As we consider the cross-interactions in the integral.
        t_star_ij = np.maximum(t_star, t_star.T)
        first_term = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                first_term[i,j] = mu[i]*mu[j] * (tc - t_star_ij[i,j])
        middle_term = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                middle_term[i,j] = (mu[i]*(ic[j] - mu[j])/beta[j])*(np.exp(-beta[j]*(t_star_ij[i,j] - tb)) - np.exp(-beta[j]*(tc-tb)))
        aux2 = middle_term.copy()
        for i in range(dim):
            for j in range(dim):
                middle_term[i,j] = aux2[i,j] + aux2[j,i]
        last_term = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                last_term[i,j] = ((ic[i] - mu[i])*(ic[j] - mu[j])/(beta[i] + beta[j]))*(np.exp(-(beta[i] + beta[j])*(t_star_ij[i,j] - tb)) - np.exp(-(beta[i] + beta[j])*(tc-tb)))

        aux = np.zeros((dim, dim))

        for i in range(dim):
            for j in range(dim):
                if t_star_ij[i,j] < tc:
                    aux[i,j] = first_term[i,j] + middle_term[i,j] + last_term[i,j]

        for i in range(dim):
            for j in range(dim):
                compensator_sq += aux[i,j]

        # Then, estimation of intensity before next jump.
        if mc > 0:
            ic = mu + np.multiply((ic - mu), np.exp(-beta*(tc-tb)))
            # for i in range(dim):
            #     ax[0,i].scatter([tc], [ic[i]], c="g")

            lambda_i[mc-1] += np.maximum(ic[mc - 1], 0)
            # j += 1
            # print(j, ic, 1+0.45*(1-0.5**(j-1)))
            ic += alpha[:, [mc - 1]]

        tb = tc
    least_squares_error = compensator_sq - (2/timestamps[-1][0])*np.sum(lambda_i)
    return least_squares_error


def multivariate_loglikelihood_approximated(theta, tList, dim=None, dimensional=False):
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
    compensator = mu*(timestamps[-1][0] - timestamps[0][0])
    # Intensity before first jump
    log_i = np.zeros((alpha.shape[0],1))
    log_i[mb-1] = np.log(mu[mb-1])
    ic = mu + alpha[:, [mb - 1]]
    # j=1

    for tc, mc in timestamps[2:]:

        compensator += np.multiply(beta_1, ic-mu)*(1 - np.exp(-beta*(tc-tb)))

        # Then, estimation of intensity before next jump.
        if mc > 0:
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
                return res
            else:
                log_i[mc-1] += np.log(ic[mc - 1])
            # j += 1
            # print(j, ic, 1+0.45*(1-0.5**(j-1)))
            ic += alpha[:, [mc - 1]]

        tb = tc
    likelihood = log_i - compensator
    if not(dimensional):
        likelihood = np.sum(likelihood)
    return -likelihood