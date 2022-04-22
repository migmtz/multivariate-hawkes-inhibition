import numpy as np
import scipy.integrate as integrate
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes


def verify_compensator(theta, tList, dim=None, dimensional=False):
    """
    Function used to verify computation of compensator.
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
        blu = (t_star < tc)*(np.multiply(mu, tc-t_star) + np.multiply(beta_1, ic-mu)*(aux - np.exp(-beta*(tc-tb))))
        for i in range(mu.shape[0]):
            function = lambda x: mu[i] + (ic[i] - mu[i])*np.exp(-beta[i]*(x-tb))
            if t_star[i] < tc:
                approx = integrate.quad(function, t_star[i], tc)[0]
                print("Approximation: ", approx, "    Real: ", blu[i])
            else:
                approx = 0
                print("Approximation: ", approx, "    Real: ", blu[i])
            if np.round(approx, 5) != np.round(blu[i], 5):
                print("error")
        print("")
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

                res = 1e8*(np.sum(mu**2) + np.sum(alpha**2) + np.sum(beta**2))
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


if __name__ == "__main__":
    ### Simulation of event times
    # np.random.seed(0)

    dim = 2  # 2, 3 ou 4

    mu = np.random.uniform(0, 1, (2,1))
    alpha = np.random.normal(0, 1, (2,2))
    beta = np.random.uniform(0, 1, (2,1))
    max_jumps = 100

    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)

    hawkes.simulate()

    verify_compensator((mu, alpha, beta), hawkes.timestamps)
