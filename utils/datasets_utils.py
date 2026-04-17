import numpy as np



import pickle
import os
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import scipy
from scipy.stats import norm



def mut_format_conversion(muts):
    muts = muts.split(';')
    out = []
    for mut in muts:
        mut = mut.split(':')
        chain = mut[0]
        original_res = mut[1][0]
        pos = mut[1][1: -1]
        after_res = mut[1][-1]
        mut = original_res + chain + pos + after_res
        out.append(mut)
    return out


def get_known_AbAg_list(S645, SARS):
    out_list = []
    for line in S645:
        if line[0] == '#': continue
        line = line.split(',')
        out_list.append(line[0] + '_' + line[1])
    out_list.append('6M0J_A_E')
    return list(set(out_list))











































































































































def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def load_variable(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


def load_data(data_path):
    with open(data_path, 'r', encoding='UTF-8') as f:
        result_dict = {}
        ddg = []
        pred = []
        name = []
        pre_xiao0 = 0
        tru_xiao0 = 0
        lines = f.readlines()
        for line in lines:
            if ':' in line:
                if 'Rp' == line.strip().split(':')[0]:
                    result_dict['Rp'] = line.strip().split(':')[1]
                elif 'MSE' == line.strip().split(':')[0]:
                    result_dict['MSE'] = line.strip().split(':')[1]
                elif 'RMSE' == line.strip().split(':')[0]:
                    result_dict['RMSE'] = line.strip().split(':')[1]
                elif 'MAE' == line.strip().split(':')[0]:
                    result_dict['MAE'] = line.strip().split(':')[1]
                elif 'R2' == line.strip().split(':')[0]:
                    result_dict['R2'] = line.strip().split(':')[1]
                elif 'Rp_confidence_interval' == line.strip().split(':')[0]:
                    result_dict['Rp_confidence_interval'] = line.strip().split(':')[1]
                elif 'P_value' == line.strip().split(':')[0]:
                    result_dict['P-value'] = line.strip().split(':')[1]

                continue


            if line.strip() == 'name;yddg;yhat': continue
            if line.strip() == 'yddg;yhat': continue
            if line.strip().split(',')[0] == 'name': continue
            if line.strip().split(';')[0] == 'name': continue
            a = line.split(";")



            ddg.append(a[1])
            pred.append(a[2].split(',')[0])
            name.append(a[0])


            if float(a[2].split(',')[0]) < -0.1:
                pre_xiao0 += 1
                if float(a[1]) <= 0:
                    tru_xiao0 += 1


        ddg = np.array(ddg, dtype='float')
        pred = np.array(pred, dtype='float')
        try:
            result_dict['Optimisation ratio'] = float(tru_xiao0) / float(pre_xiao0)
        except ZeroDivisionError:
            result_dict['Optimisation ratio'] = 'None'
    return list(ddg), list(pred), name, result_dict


def load_data_addType(data_path):
    with open(data_path, 'r', encoding='UTF-8') as f:
        result_dict = {}
        ddg = []
        pred = []
        name = []
        types = []
        pre_xiao0 = 0
        tru_xiao0 = 0
        lines = f.readlines()
        for line in lines:
            if ':' in line:
                if 'Rp' == line.strip().split(':')[0]:
                    result_dict['Rp'] = line.strip().split(':')[1]
                elif 'MSE' == line.strip().split(':')[0]:
                    result_dict['MSE'] = line.strip().split(':')[1]
                elif 'RMSE' == line.strip().split(':')[0]:
                    result_dict['RMSE'] = line.strip().split(':')[1]
                elif 'MAE' == line.strip().split(':')[0]:
                    result_dict['MAE'] = line.strip().split(':')[1]
                elif 'R2' == line.strip().split(':')[0]:
                    result_dict['R2'] = line.strip().split(':')[1]
                elif 'Rp_confidence_interval' == line.strip().split(':')[0]:
                    result_dict['Rp_confidence_interval'] = line.strip().split(':')[1]
                elif 'P_value' == line.strip().split(':')[0]:
                    result_dict['P-value'] = line.strip().split(':')[1]

                continue


            if line.strip() == 'name;yddg;yhat': continue
            if line.strip() == 'yddg;yhat': continue
            if line.strip().split(',')[0] == 'name': continue
            if line.strip().split(';')[0] == 'name': continue
            a = line.split(";")



            ddg.append(a[1])
            pred.append(a[2].split(',')[0])
            name.append(a[0])
            types.append(a[3].strip())


            if float(a[2].split(',')[0]) < -0.1:
                pre_xiao0 += 1
                if float(a[1]) <= 0:
                    tru_xiao0 += 1


        ddg = np.array(ddg, dtype='float')
        pred = np.array(pred, dtype='float')
        try:
            result_dict['Optimisation ratio'] = float(tru_xiao0) / float(pre_xiao0)
        except ZeroDivisionError:
            result_dict['Optimisation ratio'] = 'None'
    return list(ddg), list(pred), list(types), name, result_dict


def get_Mut_to_cleanedMut_dict(S1, S2):
    out_dict = {}
    for line in S1:
        if line[0] == '#':continue
        line = line.split(';')
        out_dict[line[0].strip('"') + '_' + line[1].strip('"')] = line[2].strip('"')
    for line in S2:
        if line[0] == '#':continue
        line = line.split(';')
        out_dict[line[0] + '_' + line[1]] = line[2]
    return out_dict


def get_cleanedMut(pdb_mut, cleanedMut_to_Mut_dict):
    try:
        out = cleanedMut_to_Mut_dict[pdb_mut]
    except KeyError:
        out = pdb_mut.split('_')[-1]
    return out




























def cal_Rp_confidence_interval(Rp, n):

    z = 0.5 * np.log((1 + Rp) / (1 - Rp))


    SE = 1 / np.sqrt(n - 3)


    z_upper = norm.ppf(0.975)
    z_lower = norm.ppf(0.025)


    CI_lower = z - z_upper * SE
    CI_upper = z - z_lower * SE


    CI_lower_back = (np.exp(2 * CI_lower) - 1) / (np.exp(2 * CI_lower) + 1)
    CI_upper_back = (np.exp(2 * CI_upper) - 1) / (np.exp(2 * CI_upper) + 1)

    print("皮尔逊相关系数的95%置信区间:", CI_lower_back, CI_upper_back)
    return CI_lower_back, CI_upper_back



















def cal_evaluation_index(file):
    try:
        yddg, yhat, _, _ = load_data(file)
        if len(yddg) == 0:return None
        mse = mean_squared_error(yddg, yhat)
        rmse = sqrt(mse)
        mae = mean_absolute_error(yddg, yhat)
        r2 = r2_score(yddg, yhat)
        Rp, P_v = scipy.stats.pearsonr(yddg, yhat)
        Rp_lower_back, Rp_upper_back = cal_Rp_confidence_interval(Rp, len(yddg))

        out_dict = {'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'Rp': Rp,
                    'Rp_confidence_interval': [Rp_lower_back, Rp_upper_back],
                    'P_value': P_v}
    except ValueError:
        out_dict = {'MSE': None,
                    'RMSE': None,
                    'MAE': None,
                    'R2': None,
                    'Rp': None,
                    'Rp_confidence_interval': [None, None],
                    'P_value': None}
    return out_dict


def yddg_yhat_write_evaluation_index(yddg_yhat_path):
    evaluation_index_dict = cal_evaluation_index(yddg_yhat_path)
    if evaluation_index_dict == None:return
    data = open(yddg_yhat_path, 'r').readlines()
    out_f = open(yddg_yhat_path, 'w')

    for k in evaluation_index_dict.keys():
        print(k, ':', evaluation_index_dict[k])
        out_f.write(k + ':' + str(evaluation_index_dict[k]) + '\n')

    out_f.write('name;yddg;yhat' + '\n')
    for line in data:
        if ':' in line: continue
        if line.strip() == 'name;yddg;yhat': continue
        if line.strip() == 'yddg;yhat': continue
        out_f.write(line)

    out_f.close()

