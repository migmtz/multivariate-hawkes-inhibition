import numpy as np
import csv
from ast import literal_eval as make_tuple
from class_and_func.estimator_class import multivariate_estimator_bfgs


if __name__ == "__main__":
    np.random.seed(0)
    dim = 10
    number = 3
    first = 1
    with open("estimation_"+str(number)+'_file/_simulation'+str(number), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        with open("estimation_"+str(number)+'_file/_estimation'+str(number), 'w', newline='') as myfile:
            i = 1
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for row in csv_reader:
              while i <= first:
                print("# ", i)
                tList = [make_tuple(i) for i in row]

                loglikelihood_estimation = multivariate_estimator_bfgs(dimension=dim, options={"disp": False})
                res = loglikelihood_estimation.fit(tList)
                # print(loglikelihood_estimation.res.x)
                wr.writerow(loglikelihood_estimation.res.x.tolist())
                i += 1

    before = 1
    until = 5
    with open("estimation_"+str(number)+'_file/_simulation'+str(number), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        with open("estimation_"+str(number)+'_file/_estimation'+str(number), 'a', newline='') as myfile:
            i = 1
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for row in csv_reader:
                if i <= before:
                    i += 1
                elif i > before and i <= until:
                    print("# ", i)
                    tList = [make_tuple(i) for i in row]

                    loglikelihood_estimation = multivariate_estimator_bfgs(dimension=dim, options={"disp": False})
                    res = loglikelihood_estimation.fit(tList)
                    # print(loglikelihood_estimation.res.x)
                    wr.writerow(loglikelihood_estimation.res.x.tolist())
                    i += 1