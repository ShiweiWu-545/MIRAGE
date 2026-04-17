



import os
import sys

import matplotlib.pyplot as plt

from models.predictor import DDGPredictor

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append('..')
from models.train import *
from data.my_config import config
import torch.nn as nn


def predict(config, py_name, train_best_model, label, load_result_weight=False, load_model='validation_model'):
    result_weight, config, weight_name, py_name, logger = load_default_settings(config, py_name, label, validation_flag=config.observer.validation_Rp, predict_flag=load_result_weight, load_model=load_model)




    train_dataset, validation_dataset, test_dataset, contact_area_dict = load_model_data(config,
                                                                                         validation_flag=False,
                                                                                         label=label
                                                                                         )
    train_dataset, validation_dataset, test_dataset = get_batch(train_dataset, validation_dataset, test_dataset, config)

    if load_result_weight:
        model = DDGPredictor(config).to(config.model.device)
        model.load_state_dict(load_weight(result_weight), strict=True)
    else:
        weight, config = load_variable(config.model.my_model.path + py_name.split('.')[0] + '/train_model/' + train_best_model)
        model = DDGPredictor(config).to(config.model.device)
        model.load_state_dict(load_weight(weight), strict=True)



    loss_fn = nn.MSELoss(reduction='mean')
    run = run_model(config)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.train.optimizer.lr,
        betas=(config.train.optimizer.beta1, config.train.optimizer.beta2),
        weight_decay=config.train.optimizer.weight_decay)


    test_result_dict = run.make_validation_step(model, test_dataset, contact_area_dict, loss_fn)
    Rp = scipy.stats.pearsonr(test_result_dict.yhat, test_result_dict.yddg)[0]
    print(Rp)

    if config.observer.validation_Rp:
        out_path = '../result/' + label + '/validation_result/' + py_name.split('.')[0] + '/'
    else:
        out_path = '../result/' + label + '/' + py_name.split('.')[0] + '/'
    out_path = Path(out_path)
    print('out_path:%s' % out_path)
    if load_result_weight is False:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        else:
            shutil.rmtree(out_path)
            os.makedirs(out_path)


    if load_result_weight is False:
        shutil.copy(config.model.my_model.path + py_name.split('.')[0] + '/train_model/' + train_best_model, out_path)


    with open(out_path / 'yddg_yhat.csv', 'w') as file:
        file.write('RP:' + str(Rp) + '\n')
        file.write('yddg;yhat' + '\n')
        for i in range(len(test_result_dict.yhat)):

            file.write(test_result_dict.pdb_name[i] + ';' + str(test_result_dict.yddg[i]) + ';' + str(test_result_dict.yhat[i]) + '\n')

def load_model_and_index(config, load_result_weight=False, py_name_list=None):
    if load_result_weight is False:
        if config.observer.validation_Rp:
            path_index = '../debug/validation_best_model_index/max_index_list.pt'
            path_model = '../debug/validation_best_model_index/model_list.pt'
        else:
            path_index = '../debug/max_index_list.pt'
            path_model = '../debug/model_list.pt'
        py_name_list = load_variable(path_model)
        model_list = load_variable(path_index)
    else:
        if not isinstance(py_name_list, list):
            py_name_list = [py_name_list]
        model_list = [''] * len(py_name_list)
    return py_name_list, model_list


def run_predict(config, load_result_weight=False, load_model='validation_model', label='', py_name=''):
    config = load_terminal_paras(config, load_result_weight=load_result_weight, label=label)
    py_name_list, model_list = load_model_and_index(config, load_result_weight=load_result_weight, py_name_list=py_name)

    for py_name, train_best_model in zip(py_name_list, model_list):
        predict(config, py_name, train_best_model, label=config.datasets.label, load_result_weight=load_result_weight, load_model=load_model)
        print('已完成:', py_name, train_best_model)


if __name__ == '__main__':





    load_result_weight = False
    py_name = ['fold1_model.py']
    label = 'test2_DG_blind'

    run_predict(config, load_result_weight=load_result_weight,
                label=label,
                py_name=py_name,
                load_model='validation_model')
