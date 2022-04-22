import numpy as np
from scipy import stats
from scipy.optimize import minimize
import csv
from ast import literal_eval as make_tuple
import seaborn as sns
from dictionary_parameters import dictionary as param_dict
from matplotlib import pyplot as plt
from class_and_func.colormaps import get_continuous_cmap
from metrics import relative_squared_loss
from class_and_func.likelihood_functions import *


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


class multivariate_estimator_bfgs_non_penalized(object):
    """
    Estimator class for Exponential Hawkes process obtained through minimizaton of a loss using the L-BFGS-B algorithm.

    Attributes
    ----------
    res : OptimizeResult
        Result from minimization.
    """

    def __init__(self, loss=multivariate_loglikelihood_simplified, dimension=None, initial_guess="random", options=None,
                 penalty=False, C=1, eps=1e-6):
        """
        Parameters
        ----------
        loss : {loglikelihood, likelihood_approximated} or callable.
            Function to minimize. Default is loglikelihood.
        dimension : int
            Dimension of problem to optimize. Default is None.
        initial_guess : str or ndarray.
            Initial guess for estimated vector. Either random initialization, or given vector of dimension (2*dimension + dimension**2,). Default is "random".
        options : dict
            Options to pass to the minimization method. Default is {'disp': False}.

        Attributes
        ----------
        bounds :
        """
        if dimension is None:
            raise ValueError("Dimension is necessary for initialization.")
        self.dim = dimension
        self.penalty = penalty
        if penalty == "l2":
            self.loss = lambda x, y, z: loss(x, y, z) + C * np.linalg.norm(x[-dimension:])
        elif penalty == "rlsquares":
            self.eps = eps
            self.loss = lambda x, y, z, eta, eps: loss(x, y, z) + 0.5 * C * np.sum(
                (x[self.dim: self.dim + self.dim ** 2] ** 2 + eps) / eta) + 0.5 * C * np.sum(eta)
        else:
            self.loss = loss

        self.bounds = [(1e-12, None) for i in range(self.dim)] + [(None, None) for i in range(self.dim * self.dim)] + [
            (1e-12, None) for i in range(self.dim)]
        if isinstance(initial_guess, str) and initial_guess == "random":
            self.initial_guess = np.concatenate(
                (np.concatenate((np.ones(self.dim), np.ones(self.dim * self.dim))), np.ones(self.dim)))
        if options is None:
            self.options = {'disp': False}
        else:
            self.options = options

    def fit(self, timestamps, threshold=0.01, limit=1000, maxiter=15):
        """
        Parameters
        ----------
        timestamps : list of tuple.
            Ordered list containing event times and marks.
        """

        if self.penalty != "rlsquares":
            self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",
                                args=(timestamps, self.dim), bounds=self.bounds,
                                options=self.options)
        else:
            self.options['iprint'] = 0
            self.loss = multivariate_loglikelihood_simplified
            print("loss", self.loss)
            eps = 1
            print("first")
            self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",
                                args=(timestamps, self.dim), bounds=self.bounds,
                                options=self.options)
            self.initial_guess = self.res.x

            alpha = np.abs(self.res.x[self.dim:-self.dim])
            ordered_alpha = np.sort(alpha)
            norm = np.sum(ordered_alpha)
            aux, i = 0, 0
            while aux <= threshold:
                aux += ordered_alpha[i] / norm
                i += 1
            i -= 1
            thresh = ordered_alpha[i]  # We erase those STRICTLY lower
            self.bounds = [(1e-12, None) for i in range(self.dim)] + [(None, None) if i >= thresh else (0, 1e-16)
                                                                      for i in alpha] + [
                              (1e-12, None) for i in range(self.dim)]
            print("second")
            self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",
                                args=(timestamps, self.dim), bounds=self.bounds,
                                options=self.options)

            print(self.res.x)

        self.mu_estim = np.array(self.res.x[0: self.dim])
        self.alpha_estim = np.array(self.res.x[self.dim: -self.dim]).reshape((self.dim, self.dim))
        self.beta_estim = np.array(self.res.x[-self.dim:])

        return self.mu_estim, self.alpha_estim, self.beta_estim


if __name__ == "__main__":
    np.random.seed(0)
    number = 1
    theta = param_dict[number]
    dim = int(np.sqrt(1 + theta.shape[0]) - 1)

    first = 1
    C = 50
    stop_criteria = 1e-4
    threshold = 0.05
    with open("estimation_" + str(number) + '_file/_simulation' + str(number), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        with open("estimation_" + str(number) + '_file/_estimation' + str(number) + 'thresh'+str(threshold*100), 'w', newline='') as myfile:
            i = 1
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for row in csv_reader:
                if i <= first:
                    print("# ", i)
                    tList = [make_tuple(i) for i in row]

                    loglikelihood_estimation_pen = multivariate_estimator_bfgs_non_penalized(dimension=dim, penalty="rlsquares", C=C,
                                                                               eps=stop_criteria,
                                                                               options={"disp": False})
                    res = loglikelihood_estimation_pen.fit(tList, threshold=threshold)
                    print(loglikelihood_estimation_pen.res.x)
                    wr.writerow(loglikelihood_estimation_pen.res.x.tolist())
                    i += 1
                else:
                    break

    before = 1
    until = 25
    with open("estimation_"+str(number)+'_file/_simulation'+str(number), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        with open("estimation_"+str(number)+'_file/_estimation'+str(number) + 'thresh'+str(threshold*100), 'a', newline='') as myfile:
            i = 1
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for row in csv_reader:
                if i <= before:
                    i += 1
                elif i > until:
                    break
                else:
                    print("# ", i)
                    tList = [make_tuple(i) for i in row]
                    # print(stop_criteria)
                    loglikelihood_estimation_pen = multivariate_estimator_bfgs_non_penalized(dimension=dim,
                                                                                             penalty="rlsquares", C=C,
                                                                                             eps=stop_criteria,
                                                                                             options={"disp": False})
                    res = loglikelihood_estimation_pen.fit(tList, threshold=threshold)
                    print(loglikelihood_estimation_pen.res.x)
                    wr.writerow(loglikelihood_estimation_pen.res.x.tolist())
                    i += 1

