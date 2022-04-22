import numpy as np


def relative_squared_loss(theta, est):
    theta_low = est.copy()
    dim = int(np.sqrt(1 +theta.shape[0]) - 1)
    theta_low[-dim:][theta[-dim:] == 0] = theta[-dim:][theta[-dim:] == 0]

    num = (theta - theta_low) ** 2
    theta_low = theta**2

    mu_error = np.sqrt(np.sum(num[0:dim])/np.sum(theta_low[0:dim]))
    alpha_error = np.sqrt(np.sum(num[dim:dim+dim*dim]) / np.sum(theta_low[dim:dim+dim*dim]))
    beta_error = np.sqrt(np.sum(num[-dim:]) / np.sum(theta_low[-dim:]))

    full_error = np.sqrt(np.sum(num)/np.sum(theta_low))

    return mu_error, alpha_error, beta_error, full_error


if __name__ == "__main__":
    a_reel = np.array([1,2,0,-0.015841911122878687,-0.3248656721048282,-0.19476233068062582,10,12])
    b = np.array([1,2,0,0,-0.3,-0.2,10,12])

    print(relative_squared_loss(a_reel,b))