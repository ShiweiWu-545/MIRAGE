import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


sys.path.append('..')

import scipy
from utils.datasets import *
import re
from easydict import EasyDict as edict
from pathlib import Path












class run_model():

    def __init__(self, config):
        super().__init__()

        self.py_name = config.model.fold_name
        self.config = config


        if not os.path.exists("./tensorboard_log"):
            os.makedirs("./tensorboard_log")
        timestamp = time.strftime("/%Y%m%d-%H%M%S" + "/")
        self.writer = SummaryWriter("./tensorboard_log/" + self.py_name.split('.')[0] + timestamp)

    def leaning_rate_config(self, train_result_dict,
                            train_losses_epoch,
                            epoch, val_flag, lr):


        loss_min = 0
        if epoch != 0 and epoch % self.config.train.val_freq == 0:
            loss_min = min(train_losses_epoch)
            val_flag = 1

        if 0 < val_flag <= self.config.train.val_num:
            if loss_min > train_result_dict.cur_train_loss:
                val_flag = 0
                return val_flag, lr
            else:
                val_flag += 1

            if val_flag == 10 + 1:
                lr = lr / 2
                val_flag = 0
                return val_flag, lr

            return val_flag, lr
        else:
            return val_flag, lr

    def make_train_step(self, model,
                        train_dataset, contact_area_dict,
                        loss_fn, optimizer, lr,
                        cur_epoch):
        train_yhat = []
        train_yddg = []
        train_losses_all = 0
        train_loss = 0
        counter = 0
        for x_train, y_train in tqdm(train_dataset):
            counter += 1
            if self.config.feature.contact_area[0]: x_train = batch_contact_area(self.config, x_train, contact_area_dict)

            x_train = recursive_to(x_train, self.config.model.device)
            y_train = y_recursive_to(y_train, self.config.model.device)

            model.train()

            if self.config.model.mission == 'DDG':
                if self.config.feature.contact_area[0]: yhat = model(x_train['wt'], x_train['mut'], x_train['contact_area'])
                else: yhat = model(x_train['wt'], x_train['mut'])
            else:
                if self.config.feature.contact_area[0]: yhat = model(x_train, None, x_train['contact_area'])
                else: yhat = model(x_train, None)

            train_loss = loss_fn(y_train, yhat)
            train_losses_all += train_loss.item()


            train_loss.backward()

            optimizer.defaults['lr'] = lr
            optimizer.step()
            optimizer.zero_grad()


            for i, j in zip(yhat, y_train):
                train_yhat.append(i.item())
                train_yddg.append(j.item())



        return edict({'yhat': train_yhat,
                      'yddg': train_yddg,
                      'losses_all': train_losses_all,
                      'cur_train_loss': train_loss.item()})

    def make_validation_step(self, model,
                             validation_datasets, contact_area_dict,
                             loss_fn):
        if not validation_datasets: return None


        with torch.no_grad():
            validation_yhat = []
            validation_yddg = []
            validation_pdbName = []
            Rp = 0
            validation_losses_all = 0
            for x_validation, y_validation in tqdm(validation_datasets):

                if self.config.feature.contact_area[0]: x_validation = batch_contact_area(self.config, x_validation, contact_area_dict)
                x_validation = recursive_to(x_validation, self.config.model.device)
                y_validation = y_recursive_to(y_validation, self.config.model.device)

                model.eval()

                if self.config.model.mission == 'DDG':
                    if self.config.feature.contact_area[0]: yhat = model(x_validation['wt'], x_validation['mut'], x_validation['contact_area'])
                    else: yhat = model(x_validation['wt'], x_validation['mut'])
                elif self.config.model.mission == 'DG':
                    if self.config.feature.contact_area[0]: yhat = model(x_validation, None, x_validation['contact_area'])
                    else: yhat = model(x_validation, None)

                validation_loss = loss_fn(y_validation, yhat)
                validation_losses_all += validation_loss.item()





                name_num = 0
                for i, j in zip(yhat, y_validation):



                    i = i.item()
                    j = j.item()

                    validation_yhat.append(i)
                    validation_yddg.append(j)
                    validation_pdbName.append(x_validation['pdb_name'][name_num])
                    name_num += 1

            return edict({'yhat': validation_yhat,
                          'yddg': validation_yddg,
                          'losses_all': validation_losses_all,
                          'pdb_name': validation_pdbName,
                          'cur_train_loss': validation_loss.item()})

    def processing_result(self, result_dict, cur_epoch,
                          Rp_list, losses_epoch_list,
                          logger, py_name, process_name, index='Rp'):
        if not result_dict: return

        if index == 'Rp':
            Rp = scipy.stats.pearsonr(result_dict.yhat, result_dict.yddg)[0]
            Rp_list.append(Rp)

            loss = result_dict.losses_all / math.ceil(
                len(result_dict.yddg) / self.config.train.batch_size)
            logger.info(py_name.split('.')[0] + ' ' + str(cur_epoch) + '轮loss：' + str(loss))
            losses_epoch_list.append(loss)
            logger.info(py_name.split('.')[0] + ' datasets Rp：' + str(Rp))
            self.tensorboard_write(py_name, process_name, Rp, loss, cur_epoch)

    def tensorboard_write(self, py_name, process_name, data_Rp, data_Loss, epoch):

        self.writer.add_scalar('Rp/' + py_name.split('.')[0] + process_name,
                               data_Rp,
                               epoch)
        self.writer.add_scalar('Loss/' + py_name.split('.')[0] + process_name,
                               data_Loss,
                               epoch)


def save_model(state_dict, config, epoch, test_Rp, py_name, test_result_dict=None, continue_train=False):
    os.makedirs(config.model.my_model.path + py_name.split('.')[0] + '/train_model', exist_ok=True)





    save_variable((state_dict, config), config.model.my_model.path + py_name.split('.')[0] + '/' + 'latest_model.pt')

    if not test_Rp:
        return

    best_model_path = config.model.my_model.path + py_name.split('.')[0] + '/' + 'latest_model.pt'
    best_rp = test_Rp[-1]


    before_best_model_path = ''
    before_best_rp = 0
    historical_best_rp = 0
    result_dir = Path('../result') / py_name.split('.')[0]
    result_dir.mkdir(parents=True, exist_ok=True)
    result_model_path = result_dir / 'train_model.pt'
    result_csv_path = result_dir / 'result.csv'
    if result_csv_path.exists():
        first_line = result_csv_path.read_text().splitlines()[0]
        if first_line.startswith('RP:'):
            historical_best_rp = float(first_line.split(':', 1)[1])

    for model in os.listdir(config.model.my_model.path + py_name.split('.')[0] + '/'):
        result = re.match('^train_model_best_(\d{1}\.\d+).*$', model)
        resultf = re.match('^train_model_best_-(\d{1}\.\d+).*$', model)

        if result:
            before_best_model_path = config.model.my_model.path + py_name.split('.')[0] + '/' + model
            before_best_rp = result.group(1)
        elif resultf:
            before_best_model_path = config.model.my_model.path + py_name.split('.')[0] + '/' + model
            before_best_rp = resultf.group(1)


    if best_rp > historical_best_rp:
        shutil.copy(best_model_path, result_model_path)
        with open(result_csv_path, 'w') as file:
            file.write('RP:' + str(best_rp) + '\n')
            file.write('yddg;yhat' + '\n')
            for i in range(len(test_result_dict.yhat)):

                file.write(test_result_dict.pdb_name[i] + ';' + str(test_result_dict.yddg[i]) + ';' + str(
                    test_result_dict.yhat[i]) + '\n')


    loc = {}
    exec('flag = ' + str(best_rp) + '>' + str(before_best_rp), globals(), loc)
    flag = loc['flag']
    if continue_train:
        epoch = 1
    if flag or epoch == 0:
        try:
            os.remove(before_best_model_path)
        except FileNotFoundError:
            pass
        shutil.copy(best_model_path, config.model.my_model.path + py_name.split('.')[0] + '/' + 'train_model_best_' + str(best_rp) + '.pt')


        out_path = config.model.my_model.path + py_name.split('.')[0]
        with open(out_path + '/' + 'train_model_best_yddg_yhat.csv', 'w') as file:
            file.write('RP:' + str(best_rp) + '\n')
            file.write('yddg;yhat' + '\n')
            for i in range(len(test_result_dict.yhat)):

                file.write(test_result_dict.pdb_name[i] + ';' + str(test_result_dict.yddg[i]) + ';' + str(
                    test_result_dict.yhat[i]) + '\n')
        return


def train_init(label, py_name, weight_name=None, validation_flag=False, continue_train=True):


    if validation_flag:
        f_path = config.observer.parameter.parameter_path + '/' + label + '/validation_curve/parameter_' + py_name.split('.')[0]
    else:
        f_path = config.observer.parameter.parameter_path + '/' + label + '/parameter_' + py_name.split('.')[0]


    if continue_train is False:
        try:
            shutil.rmtree(f_path)
        except FileNotFoundError:
            pass


    if continue_train is False:
        try:
            shutil.rmtree(config.model.my_model.path + py_name.split('.')[0] + '/train_model')
        except FileNotFoundError:
            pass


    train_losses_epoch = []
    train_Rp = []

    validation_losses_epoch = []
    validation_Rp = []

    test_losses_epoch = []
    test_Rp = []

    continue_train_epoch = 0
    if continue_train:
        new_f_path, f_name = get_new_f(f_path)
        parameter_f = open(new_f_path, 'r').readlines()
        loc = {}
        for line in parameter_f:
            if line.strip() == 'train_Rp':
                flag = 'train_Rp'
                continue
            elif line.strip() == 'train_losses_epoch':
                flag = 'train_losses_epoch'
                continue
            elif line.strip() == 'validation_Rp':
                flag = 'validation_Rp'
                continue
            elif line.strip() == 'validation_losses_epoch':
                flag = 'validation_losses_epoch'
                continue
            elif line.strip() == 'test_Rp':
                flag = 'test_Rp'
                continue
            elif line.strip() == 'test_losses_epoch':
                flag = 'test_losses_epoch'
                continue

            exec(flag + ' = ' + line, globals(), loc)

        max_Rp_index = loc['test_Rp'].index(max(loc['test_Rp']))
        continue_train_epoch = max_Rp_index + 1
        train_losses_epoch = loc['train_losses_epoch'][:max_Rp_index + 1]
        train_Rp = loc['train_Rp'][:max_Rp_index + 1]

        validation_losses_epoch = loc['validation_losses_epoch'][:max_Rp_index + 1]
        validation_Rp = loc['validation_Rp'][:max_Rp_index + 1]

        test_losses_epoch = loc['test_losses_epoch'][:max_Rp_index + 1]
        test_Rp = loc['test_Rp'][:max_Rp_index + 1]

    if weight_name:

        path = '../procedure_parameter/' + label
        if validation_flag:
            path = path + '/validation_curve/parameter_fold' + str(py_name.split('_')[0][-1]) + '_model'
        else:
            path = path + '/parameter_fold' + str(py_name.split('_')[0][-1]) + '_model'

        for curDir, dirs, files in os.walk(path):
            break
        for f in files:
            if weight_name.split('_')[-1].split('.')[0] == f.split('_')[0]:
                break
        assert int(f.split('_')[0]) == int(weight_name.split('_')[-1].split('.')[0])
        path = path + '/' + f


        parameter_f = open(path, 'r').readlines()
        loc = {}
        for line in parameter_f:
            if line.strip() == 'train_Rp':
                flag = 'train_Rp'
                continue
            elif line.strip() == 'train_losses_epoch':
                flag = 'train_losses_epoch'
                continue
            elif line.strip() == 'validation_Rp':
                flag = 'validation_Rp'
                continue
            elif line.strip() == 'validation_losses_epoch':
                flag = 'validation_losses_epoch'
                continue
            elif line.strip() == 'test_Rp':
                flag = 'test_Rp'
                continue
            elif line.strip() == 'test_losses_epoch':
                flag = 'test_losses_epoch'
                continue

            exec(flag + ' = ' + line, globals(), loc)

        train_losses_epoch = loc['train_losses_epoch']
        train_Rp = loc['train_Rp']

        validation_losses_epoch = loc['validation_losses_epoch']
        validation_Rp = loc['validation_Rp']

        test_losses_epoch = loc['test_losses_epoch']
        test_Rp = loc['test_Rp']


    return train_losses_epoch, train_Rp, validation_losses_epoch, validation_Rp, test_losses_epoch, test_Rp, continue_train_epoch
