import csv
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
import numpy as np
from dictionary_parameters import dictionary as param_dict
# from ast import literal_eval as make_tuple
from matplotlib import pyplot as plt

if __name__ == "__main__":
    number = 9
    theta = param_dict[number]
    dim = int(np.sqrt(1 + theta.shape[0]) - 1)
    mu = param_dict[number][:dim].reshape(dim, 1)
    alpha = param_dict[number][dim:-dim].reshape((dim, dim))
    beta = param_dict[number][-dim:].reshape(dim, 1)
    number_repetitions = 25
    with open("estimation_"+str(number)+'_file/_simulation'+str(number), 'w', newline='') as myfile:
        max_jumps = 5000

        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)

        trtt = np.zeros((dim,))
        for i in range(number_repetitions):
            np.random.seed(i)

            hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)
            hawkes.simulate()

            tList = hawkes.timestamps
            wr.writerow(tList)

            for l in tList:
                if l[1] > 0:
                    trtt[l[1]-1] += 1
        print(trtt/number_repetitions)
        hawkes.plot_intensity()
        plt.show()

    # with open('_simulation0', 'r') as read_obj:
    #     csv_reader = csv.reader(read_obj)
    #     # Iterate over each row in the csv using reader object
    #     for row in csv_reader:
    #         # row variable is a list that represents a row in csv
    #         print([make_tuple(i) for i in row[0:2]])