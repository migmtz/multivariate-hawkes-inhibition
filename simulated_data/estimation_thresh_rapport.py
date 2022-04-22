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

    def __init__(self, loss=multivariate_loglikelihood_simplified, grad=True, dimension=None, initial_guess="random",
                 options=None):
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

        if isinstance(grad, bool) and grad:
            self.loss = multivariate_loglikelihood_with_grad
        else:
            self.loss = loss
        self.grad = grad

        self.bounds = [(1e-12, None) for i in range(self.dim)] + [(None, None) for i in range(self.dim * self.dim)] + [
            (1e-12, None) for i in range(self.dim)]
        if isinstance(initial_guess, str) and initial_guess == "random":
            self.initial_guess = np.concatenate(
                (np.concatenate((np.ones(self.dim), np.ones(self.dim * self.dim))), np.ones(self.dim)))
        if options is None:
            self.options = {'disp': False}
        else:
            self.options = options

    def fit(self, timestamps, initial, threshold=0.01):
        """
        Parameters
        ----------
        timestamps : list of tuple.
            Ordered list containing event times and marks.
        """

        self.options['iprint'] = 0
        print("loss", self.loss)
        print("first")
        self.initial_guess = initial

        alpha = np.abs(self.initial_guess[self.dim:-self.dim]).reshape((self.dim, self.dim))
        beta = np.abs(self.initial_guess[-self.dim:]).reshape((self.dim, 1))

        print(alpha[0,0], beta[0,0])
        alpha = alpha/beta
        alpha = np.ravel(alpha)
        print(alpha[0])

        ordered_alpha = np.sort(alpha)
        norm = np.sum(ordered_alpha)
        aux, i = 0, 0
        while aux <= threshold:
            aux += ordered_alpha[i] / norm
            i += 1
        i -= 1
        thresh = ordered_alpha[i]  # We erase those STRICTLY lower
        self.initial_guess = np.concatenate(
                (np.concatenate((np.ones(self.dim), np.zeros(self.dim * self.dim))), np.ones(self.dim)))
        self.bounds = [(1e-12, None) for i in range(self.dim)] + [(None, None) if i >= thresh else (0, 1e-16)
                                                                  for i in alpha] + [
                          (1e-12, None) for i in range(self.dim)]
        print("second")
        self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",jac=self.grad,
                            args=(timestamps, self.dim), bounds=self.bounds,
                            options=self.options)

        print(self.res.x)

        self.mu_estim = np.array(self.res.x[0: self.dim])
        self.alpha_estim = np.array(self.res.x[self.dim: -self.dim]).reshape((self.dim, self.dim))
        self.beta_estim = np.array(self.res.x[-self.dim:])

        return self.mu_estim, self.alpha_estim, self.beta_estim


if __name__ == "__main__":
    np.random.seed(0)
    number = 7
    theta = param_dict[number]
    dim = int(np.sqrt(1 + theta.shape[0]) - 1)

    file_name = "grad"

    threshold = 0.35

    with open("estimation_"+str(number)+'_file/_estimation'+str(number)+file_name, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for z, row in enumerate(csv_reader):
            result = np.array([float(i) for i in row])
            print("z", z)
            if z == 0:
                with open("estimation_" + str(number) + '_file/_simulation' + str(number), 'r') as read_obj:
                    csv_reader = csv.reader(read_obj)
                    with open("estimation_" + str(number) + '_file/_estimation' + str(number) + 'threshrapport'+str(threshold*100), 'w', newline='') as myfile:
                        i = 1
                        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                        for rows in csv_reader:
                            if i <= 1:
                                print("# ", i)
                                tList = [make_tuple(i) for i in rows]

                                loglikelihood_estimation_pen = multivariate_estimator_bfgs_non_penalized(dimension=dim,
                                                                                           options={"disp": False})
                                res = loglikelihood_estimation_pen.fit(tList, initial=result, threshold=threshold)
                                print(loglikelihood_estimation_pen.res.x)
                                wr.writerow(loglikelihood_estimation_pen.res.x.tolist())
                                i += 1
                            else:
                                break
            else:
                with open("estimation_"+str(number)+'_file/_simulation'+str(number), 'r') as read_obj:
                    csv_reader = csv.reader(read_obj)
                    for i, rows in enumerate(csv_reader):
                        with open("estimation_"+str(number)+'_file/_estimation'+str(number)+'threshrapport'+str(threshold*100), 'a', newline='') as myfile:
                            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                            if z == i:
                                print("# ", z, i+1)
                                tList = [make_tuple(i) for i in rows]
                                # print(stop_criteria)
                                loglikelihood_estimation_pen = multivariate_estimator_bfgs_non_penalized(dimension=dim,
                                                                                                         options={"disp": False})
                                res = loglikelihood_estimation_pen.fit(tList, initial=result, threshold=threshold)
                                print(loglikelihood_estimation_pen.res.x)
                                wr.writerow(loglikelihood_estimation_pen.res.x.tolist())
                                #i += 1

