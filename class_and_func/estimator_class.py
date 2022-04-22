import numpy as np
from scipy import stats
from scipy.optimize import minimize
from class_and_func.hawkes_process import exp_thinning_hawkes
from class_and_func.likelihood_functions import *
import time


class loglikelihood_estimator(object):
    """
    Estimator class for Exponential Hawkes process obtained through minimizaton of a loss.
    Contemplated losses are functions from likelihood_functions, either loglikelihood or likelihood_approximated, or a callable.

    Attributes
    ----------
    estimator : array
        Array containing estimated parameters.
    estimated_loss : float
        Value of loss at estimated parameters.
    model : object "exp_thinning_hawkes"
        Class containing the estimated parameters, timestamps and corresponding intensities. Exists only if return_model is set to True.
    """

    def __init__(self, loss=loglikelihood, solver="L-BFGS-B", C=None, initial_guess=np.array((1.0, 0.0, 1.0)),
                 simplex=True, bounds=[(0.0, None), (None, None), (0.0, None)], return_model=False,
                 options={'disp': False}):
        """
        Parameters
        ----------
        loss : {loglikelihood, likelihood_approximated}
            Function to minimize. Default is loglikelihood.
        solver : {"L-BFGS-B", "nelder-mead"}
            Solver used in function minimize from scipy.optimize. 
            "L-BFGS-B" uses the bounds argument and "nelder-mead" the simplex argument.
            Default is "L-BFGS-B".
        C : float
            Penalty constant. Only taken in account if not None. Default is None.
        initial_guess : array of float.
            Initial guess for estimated vector. When using "nelder-mead", it is only used if simplex is False. Default is np.array((1.0, 0.0, 1.0)).
        simplex : bool
            Whether if initialize the solver with a simplex. 
            The simplex is then initialized randomly in four points where the loglikelihood is finite. Default is True.
        bounds : list.
            Bounds to set for the algorithm. By default, only bounds are for lambda_0 and beta to be non-negative.
            If method appears to be unstable, a bounds like ((epsilon, None), (None, None), (epsilon, None)) with epsilon = 1e-10 is recommended.
            Default is ((0.0, None), (None, None), (0.0, None)).
        return_model : bool
            Whether to create an object "exp_thinning_hawkes" with obtained estimation. 
            The class has its corresponding intensity function.
            
        """
        if C is not None:
            self.loss = lambda theta, timestamps: loss(theta, timestamps) + C * (
                        theta[0] ** 2 + theta[1] ** 2 + theta[2] ** 2)
            self.C = C
        else:
            self.loss = loss
        self.solver = solver
        self.initial_guess = initial_guess
        self.simplex = simplex
        self.bounds = bounds
        self.return_model = return_model
        self.options = options

        if solver == "L-BFGS-B":
            self._estimator = loglikelihood_estimator_bfgs(loss=self.loss, bounds=self.bounds,
                                                           initial_guess=self.initial_guess, options=self.options)
        elif solver == "nelder-mead":
            self._estimator = loglikelihood_estimator_nelder(loss=self.loss, simplex=self.simplex,
                                                             initial_guess=self.initial_guess, options=self.options)
        else:
            raise ValueError('Unknown solver %s' % solver)

    def fit(self, timestamps):
        """
        Parameters
        ----------
        timestamps : list of float
            Ordered list containing event times.
        """

        self.res = self._estimator.fit(timestamps)

        self.estimator = self.res.x
        self.estimated_loss = self.res.fun

        if self.return_model:
            self.model = exp_thinning_hawkes(self.estimator[0], self.estimator[1], self.estimator[2])
            self.model.set_time_intensity(timestamps)


class loglikelihood_estimator_bfgs(object):
    """
    Estimator class for Exponential Hawkes process obtained through minimizaton of a loss using the L-BFGS-B algorithm.

    Attributes
    ----------
    res : OptimizeResult
        Result from minimization.
    """

    def __init__(self, loss=loglikelihood, bounds=[(0.0, None), (None, None), (0.0, None)],
                 initial_guess=np.array((1.0, 0.0, 1.0)), options={'disp': False}):
        """
        Parameters
        ----------
        loss : {loglikelihood, likelihood_approximated} or callable.
            Function to minimize. Default is loglikelihood.
        bounds : list.
            Bounds to set for the algorithm. By default, only bounds are for lambda_0 and beta to be non-negative.
            If method appears to be unstable, a bounds like ((epsilon, None), (None, None), (epsilon, None)) with epsilon = 1e-10 is recommended.
            Default is ((0.0, None), (None, None), (0.0, None)).
        initial_guess : array of float.
            Initial guess for estimated vector. Default is np.array((1.0, 0.0, 1.0)).
        options : dict
            Options to pass to the minimization method. Default is {'disp': False}.
        """
        self.loss = loss
        self.bounds = bounds
        self.initial_guess = np.array((1.0, 0.0, 1.0))
        self.options = options

    def fit(self, timestamps):
        """
        Parameters
        ----------
        timestamps : list of float
            Ordered list containing event times.
        """

        self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",
                            args=timestamps, bounds=self.bounds,
                            options=self.options)

        return (self.res)


class loglikelihood_estimator_nelder(object):  #### A FINIR
    """
    Estimator class for Exponential Hawkes process obtained through minimizaton of a loss using the Nelder-Mead simplex algorithm.
    
    Attributes
    ----------
    res : OptimizeResult
        Result from minimization.
    """

    def __init__(self, loss=loglikelihood, simplex=True, initial_guess=np.array((1.0, 0.0, 1.0)),
                 options={'disp': False}):
        """
        Parameters
        ----------
        loss : {loglikelihood, likelihood_approximated} or callable.
            Function to minimize. Default is loglikelihood.
        simplex : bool
            Whether if initialize the solver with a simplex. 
            The simplex is then initialized randomly in four points where the loglikelihood is finite. Default is True.
        initial_guess : array of float.
            Initial guess for estimated vector. Used only if simples is False. Default is np.array((1.0, 0.0, 1.0)).
        options : dict
            Options to pass to the minimization method. Default is {'disp': False}.
        """
        self.loss = loss
        self.simplex = simplex
        self.initial_guess = initial_guess
        self.options = options

    def fit(self, timestamps):
        """
        Parameters
        ----------
        timestamps : list of float
            Ordered list containing event times.
        """
        if self.simplex:
            x_simplex = []

            while len(x_simplex) != 4:
                candidate = np.array([np.random.normal(0, 1), np.random.normal(0, 1), np.random.normal(0, 1)])
                like = self.loss(candidate, timestamps)
                if like < 1e10:
                    x_simplex += [candidate]

            self.options['initial_simplex'] = x_simplex

        self.res = minimize(self.loss, self.initial_guess, method="nelder-mead",
                            args=timestamps, options=self.options)

        return self.res


class multivariate_estimator_bfgs(object):
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
            eps = 1
            self.et = np.abs(self.initial_guess[self.dim: self.dim + self.dim ** 2])
            # start_time = time.time()
            self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",
                                args=(timestamps, self.dim, self.et, eps), bounds=self.bounds,
                                options=self.options)
            # print(time.time()-start_time)
            self.old_et =self.et
            self.et = np.sqrt(np.array(self.res.x[self.dim: self.dim + self.dim ** 2]) ** 2 + eps)
            self.initial_guess = self.res.x
            acc = 1
            eps *= 1/2
            # self.options['maxiter'] = maxiter
            while acc < limit and np.linalg.norm(self.et - self.old_et) > self.eps:
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

                # start_time = time.time()
                self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",
                                    args=(timestamps, self.dim, self.et, eps), bounds=self.bounds,
                                    options=self.options)
                # print(time.time() - start_time, end=" ")
                self.old_et = self.et
                self.et = np.sqrt(np.array(self.res.x[self.dim: -self.dim]) ** 2 + eps)

                # print(self.res.x, np.linalg.norm(self.et - self.old_et))
                acc += 1
                eps *= 1 / 2
                self.initial_guess = self.res.x

            # print(acc, "   ", np.linalg.norm(self.et - self.old_et))

            alpha = np.abs(self.res.x[self.dim:-self.dim])
            ordered_alpha = np.sort(alpha)
            norm = np.sum(ordered_alpha)
            aux, i = 0, 0
            while aux <= threshold:
                aux += ordered_alpha[i]/norm
                i += 1
            i -= 1
            thresh = ordered_alpha[i] # We erase those STRICTLY lower
            self.bounds = [(1e-12, None) for i in range(self.dim)] + [(None, None) if i >= thresh else (0, 1e-16) for i in alpha] + [
                              (1e-12, None) for i in range(self.dim)]
            self.res = minimize(multivariate_loglikelihood_simplified, self.initial_guess, method="L-BFGS-B",
                                args=(timestamps, self.dim), bounds=self.bounds,
                                options=self.options)

            print(self.res.x, "   ", acc)

        self.mu_estim = np.array(self.res.x[0: self.dim])
        self.alpha_estim = np.array(self.res.x[self.dim: -self.dim]).reshape((self.dim, self.dim))
        self.beta_estim = np.array(self.res.x[-self.dim:])

        return self.mu_estim, self.alpha_estim, self.beta_estim


class multivariate_estimator_bfgs_grad(object):
    """
    Estimator class for Exponential Hawkes process obtained through minimizaton of a loss using the L-BFGS-B algorithm.

    Attributes
    ----------
    res : OptimizeResult
        Result from minimization.
    """

    def __init__(self, loss=multivariate_loglikelihood_simplified, grad=True, dimension=None, initial_guess="random",
                 options=None, penalty=False, C=1, eps=1e-6):
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
            if isinstance(grad, bool) and grad:
                self.loss = multivariate_loglikelihood_with_grad_pen
                self.grad = True
            else:
                self.loss = lambda x, y, z, eta, eps: loss(x, y, z) + 0.5 * C * np.sum(
                    (x[self.dim: self.dim + self.dim ** 2] ** 2 + eps) / eta) + 0.5 * C * np.sum(eta)
                self.grad = False

        else:
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

    def fit(self, timestamps, limit=1000):
        """
        Parameters
        ----------
        timestamps : list of tuple.
            Ordered list containing event times and marks.
        """

        if self.penalty != "rlsquares":
            self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B", jac=self.grad,
                                args=(timestamps, self.dim), bounds=self.bounds,
                                options=self.options)
        else:
            if self.grad:
                self.et = np.ones((self.dim*self.dim))
                self.old_et = self.et + 2 * self.eps
                acc = 1
                eps = 1
                while acc < limit and np.linalg.norm(self.et - self.old_et) > self.eps:
                    print(acc, "   ", self.initial_guess)
                    self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B", jac=self.grad,
                                        args=(timestamps, eps, self.dim, self.et, eps), bounds=self.bounds,
                                        options=self.options)
                    self.old_et = self.et
                    self.et = np.sqrt(np.array(self.res.x[self.dim: self.dim + self.dim ** 2]) ** 2 + eps)
                    print(self.old_et, self.et)
                    acc += 1
                    eps *= 1 / 2
                    self.initial_guess = self.res.x
            else:
                self.et = np.abs(self.initial_guess[self.dim: self.dim + self.dim ** 2])
                self.old_et = self.et + 2 * self.eps
                acc = 1
                eps = 1
                while acc < limit and np.linalg.norm(self.et - self.old_et) > self.eps:
                    print(acc, "   ", np.linalg.norm(self.et - self.old_et))
                    self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",
                                        args=(timestamps, self.dim, self.et, eps), bounds=self.bounds,
                                        options=self.options)
                    self.old_et = self.et
                    self.et = np.sqrt(np.array(self.res.x[self.dim: self.dim + self.dim ** 2]) ** 2 + eps)
                    print(self.old_et, self.et)
                    acc += 1
                    eps *= 1 / 2
                    self.initial_guess = self.res.x

        self.mu_estim = np.array(self.res.x[0: self.dim])
        self.alpha_estim = np.array(self.res.x[self.dim: self.dim + self.dim ** 2]).reshape((self.dim, self.dim))
        self.beta_estim = np.array(self.res.x[-self.dim:])

        return self.mu_estim, self.alpha_estim, self.beta_estim


class multivariate_estimator_jit(object):
    """
    Estimator class for Exponential Hawkes process obtained through minimizaton of a loss using the L-BFGS-B algorithm.

    Attributes
    ----------
    res : OptimizeResult
        Result from minimization.
    """

    def __init__(self, loss=multivariate_loglikelihood_simplified, dimension=None, initial_guess="random", options=None,
                 penalty=False, C=1):
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
        if penalty:
            self.loss = lambda x, y, z: loss(x, y, z) + C * np.linalg.norm(x[-dim:])
        else:
            self.loss = loss
        self.bounds = [(1e-12, None) for i in range(self.dim)] + [(None, None) for i in range(self.dim * self.dim)] + [
            (1e-12, None) for i in range(self.dim)]
        print(len(self.bounds))
        if isinstance(initial_guess, str) and initial_guess == "random":
            self.initial_guess = (np.ones((self.dim, 1)), np.zeros((self.dim, self.dim)), np.ones((self.dim, 1)))
        if options is None:
            self.options = {'disp': False}
        else:
            self.options = options

    def fit(self, timestamps):
        """
        Parameters
        ----------
        timestamps : list of tuple.
            Ordered list containing event times and marks.
        """

        self.initial_guess = np.ones((8,))

        self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",
                            args=(timestamps), bounds=self.bounds,
                            options=self.options)

        self.mu_estim = np.array(self.res.x[0: self.dim])
        self.alpha_estim = np.array(self.res.x[self.dim: self.dim + self.dim ** 2]).reshape((self.dim, self.dim))
        self.beta_estim = np.array(self.res.x[-self.dim:])

        return (self.mu_estim, self.alpha_estim, self.beta_estim)
