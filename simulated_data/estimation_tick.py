import numpy as np
import csv
from ast import literal_eval as make_tuple
from class_and_func.streamline_tick import four_estimation
from dictionary_parameters import dictionary as param_dict
from metrics import relative_squared_loss

# Note to myself, first two estimations contain dim + dim**2 parameters corresponding to
# mu and alpha
# 3rd and 4th contain dim + dim**3

if __name__ == "__main__":
    number = 9
    theta = param_dict[number]
    dim = int(np.sqrt(1 + theta.shape[0]) - 1)
    beta = theta[-dim:].reshape((dim,1)) + 1e-16

    until = 25
    C_grid = [1, 10, 100, 1000, 10000, 100000, 1000000]
    C_def, error = np.zeros((until, 4)), np.array([[np.inf for i in range(4)] for j in range(until)])
    for C in C_grid:
        with open("estimation_" + str(number) + '_file/_simulation' + str(number), 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            i = 1
            for row in csv_reader:
                if i <= until:
                    tList = [make_tuple(i) for i in row]

                    params_tick = four_estimation(beta, tList, penalty="l2", C=[C for i in range(4)])
                    for k in range(4):
                        if k > 1:
                            estimation = np.concatenate((params_tick[k][0], np.mean(params_tick[k][1], axis=2).ravel())).tolist()
                        else:
                            estimation = np.concatenate((params_tick[k][0], params_tick[k][1].ravel())).tolist()
                        err = relative_squared_loss(theta, np.concatenate((estimation, beta.squeeze())))[3]
                        if err < np.inf and err < error[i-1, k] and err != 0.0:
                            C_def[i-1, k], error[i-1, k] = C, err
                    i += 1
    print(C_def)
    # C_def = 380
    with open("estimation_"+str(number)+'_file/_simulation'+str(number), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        file_simple = open("estimation_"+str(number)+'_file/_estimation'+str(number)+"tick", 'w', newline='')
        file_pen = open("estimation_"+str(number)+'_file/_estimation'+str(number)+"tick_bfgs", 'w', newline='')
        file_beta_simple = open("estimation_"+str(number)+'_file/_estimation'+str(number)+"tick_beta", 'w', newline='')
        file_beta_pen = open("estimation_"+str(number)+'_file/_estimation'+str(number)+"tick_beta_bfgs", 'w', newline='')
        i = 1
        ws = csv.writer(file_simple, quoting=csv.QUOTE_ALL)
        wp = csv.writer(file_pen, quoting=csv.QUOTE_ALL)
        wbs = csv.writer(file_beta_simple, quoting=csv.QUOTE_ALL)
        wbp = csv.writer(file_beta_pen, quoting=csv.QUOTE_ALL)

        for row in csv_reader:
            if i <= until:
                print("# ", i)
                tList = [make_tuple(i) for i in row]

                params_tick = four_estimation(beta, tList, penalty="l2", C=C_def[i-1, :])
                ws.writerow(np.concatenate((params_tick[0][0], params_tick[0][1].ravel())).tolist())
                wp.writerow(np.concatenate((params_tick[1][0], params_tick[1][1].ravel())).tolist())
                wbs.writerow(np.concatenate((params_tick[2][0], params_tick[2][1].ravel())).tolist())
                wbp.writerow(np.concatenate((params_tick[3][0], params_tick[3][1].ravel())).tolist())
                i += 1

        file_simple.close()
        file_pen.close()
        file_beta_simple.close()
        file_beta_pen.close()