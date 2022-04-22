import numpy as np
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.estimator_class import multivariate_estimator_bfgs, multivariate_estimator_bfgs_grad
from scipy.optimize import check_grad, approx_fprime
import time


def grad(theta, tList, dim=None, dimensional=False):
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


if __name__ == "__main__":
    np.random.seed(0)

    dim = 2  # 2, 3 ou 4

    mu = np.array([[1.5], [2.5]])
    alpha = np.array([[0.0, 0.3], [-1.2, -1.5]])
    beta = np.array([[1.], [2.]])
    theta = (mu, alpha, beta)
    print(mu, "\n", alpha, "\n", beta)
    max_jumps = 1000

    ################# SIMULATION
    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)

    print("Starting simulation...")
    hawkes.simulate()
    print("Finished simulation")

    ################# gard

    loglikelihood_estimation = multivariate_estimator_bfgs(dimension=dim, options={"disp": False})
    print("Starting loglikelihood...")
    start_time = time.time()
    loglikelihood_estimation.fit(hawkes.timestamps)
    end_time = time.time() - start_time

    print("Estimation through approx loglikelihood: ", np.round(loglikelihood_estimation.res.x, 3), "\nIn: ", end_time)

    loglikelihood_estimation = multivariate_estimator_bfgs_grad(grad=grad, dimension=dim, options={"disp": False})
    print("Starting loglikelihood...")
    start_time = time.time()
    loglikelihood_estimation.fit(hawkes.timestamps)
    end_time = time.time() - start_time

    print("Estimation through grad loglikelihood: ", np.round(loglikelihood_estimation.res.x, 3), "\nIn: ", end_time)

    loglikelihood_estimation = multivariate_estimator_bfgs_grad(grad=True, dimension=dim, options={"disp": False})
    print("Starting loglikelihood...")
    start_time = time.time()
    loglikelihood_estimation.fit(hawkes.timestamps)
    end_time = time.time() - start_time

    print("Estimation through grad in same loglikelihood: ", np.round(loglikelihood_estimation.res.x, 3), "\nIn: ", end_time)

