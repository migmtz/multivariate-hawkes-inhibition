import numpy as np
from class_and_func.hawkes_process import exp_thinning_hawkes
from class_and_func.estimator_class import loglikelihood_estimator
from class_and_func.likelihood_functions import loglikelihood, likelihood_approximated
from scipy.stats import kstest

#########
## This allows to save our parameters in the SavedBoxplots files used in boxplots_errors.py.
# import pickle

#########
## Uncomment if numba is available. This package allows to significantly accelerate the estimations.
# from numba import njit

#########
## This script allows to estimate the average estimators and average p-values as they appear in our paper.
## Values are printed for each set of parameters using both the exact and approximated loglikelihoods.
## Simultaneously, it allows to create the SavedBoxplots files.


if __name__ == "__main__":

    ## Define losses in case numba is used
    try:
        exact_loss = njit(loglikelihood)
        approx_loss = njit(likelihood_approximated)
    except NameError:
        exact_loss = loglikelihood
        approx_loss = likelihood_approximated

    ## All parameter sets considered.
    thetaTests = [
        (0.5, -0.001, 0.4),
        (0.5, -0.2, 0.4),
        (1.05, -0.75, 0.8),
        (2.43, -0.98, 0.4),
        (2.85, -2.5, 1.8),
        (1.6, -0.75, 0.1)
    ]

    max_jumps = 200
    iterations = 100

    for t, theta in enumerate(thetaTests):
        print("Parameters: ", theta)
        lambda_0, alpha, beta = theta
        theta_array = np.array(theta)

        ## Array containing the absolute error for each estimation.
        box_real = np.empty((iterations, 3), np.float64)
        box_approx = np.empty((iterations, 3), np.float64)

        ## Array containing the estimators found at each estimation.
        estimation_real = np.empty((iterations, 3), np.float64)
        estimation_approx = np.empty((iterations, 3), np.float64)

        ## p-values and proportion of time equal to 0.
        kstestExact = 0
        kstestApprox = 0
        proportion = 0

        for iter in range(iterations):
            np.random.seed(7 * iter)

            ## Simulation
            hawkes = exp_thinning_hawkes(lambda_0, alpha, beta, max_jumps=max_jumps)
            hawkes.simulate()
            tList = np.array(hawkes.timestamps)

            ## Proportion of time equal to 0
            negIntervals = 0

            for i, lambda_k in enumerate(hawkes.intensity_jumps):
                if lambda_k < 0:
                    negIntervals += np.log(1 - lambda_k/lambda_0)/beta

            proportion += negIntervals/tList[-1]

            ## Exact estimation
            model_exact = loglikelihood_estimator(loss=exact_loss, return_model=True)
            np.random.seed(7 * iter)
            model_exact.fit(tList)

            process = model_exact.model
            process.compensator_transform()
            kstestExact += kstest(process.intervals_transformed, cdf="expon").pvalue

            exact_error = np.abs(model_exact.estimator - theta_array)/np.abs(theta_array)
            estimation_real[iter, :] = model_exact.estimator
            box_real[iter, :] = exact_error

            ## Approximated estimation
            model_approx = loglikelihood_estimator(loss=approx_loss, return_model=True)
            np.random.seed(7 * iter)
            model_approx.fit(tList)

            process = model_approx.model
            process.compensator_transform()
            kstestApprox += kstest(process.intervals_transformed, cdf="expon").pvalue

            approx_error = np.abs(model_approx.estimator - theta_array)/np.abs(theta_array)
            estimation_approx[iter, :] = model_approx.estimator
            box_approx[iter, :] = approx_error

        proportion /= iterations
        kstestExact /= iterations
        kstestApprox /= iterations

        print(f"theta = {theta} estimation = {np.mean(estimation_real, axis=0)} \n Exact p-value = {kstestExact}")
        print(f"theta = {theta} estimation = {np.mean(estimation_approx, axis=0)} \n Approximate p-value = {kstestApprox}")
        print("")

        ## Uncomment to create the SavedBoxplots files through pickle.
        # toSave = [lambda_0, alpha, beta, max_jumps, iterations, box_approx, box_real, proportion]
        #
        # with open('SavedBoxplots/SavedBoxplots' + str(t+1), 'wb') as f:
        #     pickle.dump(toSave, f)
