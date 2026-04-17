from utils.datasets import *


if __name__ == '__main__':

    no_AbAg_flag = 0
    predict_flag = 0
    total_flag = 0
    dataset_XXXX_flag = 0
    S1131_flag = 1
    AF3_S1131_flag = 0
    M1707_flag = 0
    S645_flag = 0
    S645_no27_flag = 0
    C3684_flag = 0
    S4169_flag = 0
    S8338_flag = 0
    M1101_flag = 0

    from data.my_config import *


    path = os.getcwd().split('recurrence')[0]


    flag = 1
    if flag and dataset_XXXX_flag:
        x_data_available = '/data_4T/dataset/Skempi2_ddg_useful'
        x_data_dataset_del_M1707 = '../datasets/skempi_v2_del_M1707.csv'
        x_data_dataset_del_S1131 = '../datasets/skempi_v2_del_S1131.csv'
        y_data = '../datasets/total_ddg_S1_S2_realT.csv'

        for x_data, name in zip([x_data_dataset_del_S1131, x_data_dataset_del_M1707],
                                ['dataset-S1131', 'dataset-M1707']):


            dataset = CustomDataset(config, x_data_available, x_data, y_data)

            save_variable(dataset, '../datasets/' + config.feature.choice_residue + '/' + name + '_' +
                          str(config.feature.Residue_selection_based_on_core.max_rnum) + '_' +
                          str(config.feature.Residue_selection_based_on_core.mut_lr) + '_' +
                          str(config.feature.Residue_selection_based_on_core.core_ratio) + '.pt')
        print('dataset-XXXX计算完毕，数据集完成生成！！！')

    flag = 1
    if flag and S1131_flag:
        x_data_available = '/data_4T/dataset/Skempi2_ddg_useful'
        x_data = '../datasets/S1131.csv'
        y_data = '../datasets/total_ddg_S1_S2_realT.csv'

        dataset = CustomDataset(config, x_data_available, x_data, y_data, S1131=True)

        save_variable(dataset, '../datasets/' + config.feature.choice_residue + '/' + 'S1131' + '_' +
                      str(config.feature.Residue_selection_based_on_core.max_rnum) + '_' +
                      str(config.feature.Residue_selection_based_on_core.mut_lr) + '_' +
                      str(config.feature.Residue_selection_based_on_core.core_ratio) + '.pt')
        print('计算完毕，数据集完成生成！！！')



    flag = 1
    if flag and AF3_S1131_flag:
        x_data_available = '/data_4T/dataset/AF3_mutDataset/S2_S1131_AF3_Cddg_useful'
        x_data = '../datasets/S1131.csv'
        y_data = '../datasets/total_ddg_S1_S2_realT.csv'

        dataset = CustomDataset(config, x_data_available, x_data, y_data, S1131=True)

        save_variable(dataset, '../datasets/' + config.feature.choice_residue + '/' + 'AF3S1131' + '_' +
                      str(config.feature.Residue_selection_based_on_core.max_rnum) + '_' +
                      str(config.feature.Residue_selection_based_on_core.mut_lr) + '_' +
                      str(config.feature.Residue_selection_based_on_core.core_ratio) + '.pt')
        print('计算完毕，数据集完成生成！！！')


    flag = 1
    if flag and M1707_flag:
        x_data_available = '/data_4T/dataset/Skempi2_ddg_useful'
        x_data = '../datasets/M1707.csv'
        y_data = '../datasets/total_ddg_S1_S2_realT.csv'

        dataset = CustomDataset(config, x_data_available, x_data, y_data, M1707=True)

        save_variable(dataset, '../datasets/' + config.feature.choice_residue + '/' + 'M1707' + '_' +
                      str(config.feature.Residue_selection_based_on_core.max_rnum) + '_' +
                      str(config.feature.Residue_selection_based_on_core.mut_lr) + '_' +
                      str(config.feature.Residue_selection_based_on_core.core_ratio) + '.pt')
        print('计算完毕，数据集完成生成！！！')

    flag = 1
    if flag and S4169_flag:
        x_data_available = '/data_4T/dataset/Skempi2_ddg_useful'
        x_data = '../datasets/S4169.csv'
        y_data = '../datasets/S4169.csv'

        dataset = CustomDataset(config, x_data_available, x_data, y_data, S4169=True)
        save_variable(dataset, '../datasets/' + config.feature.choice_residue + '/' + 'S4169' + '_' +
                      str(config.feature.Residue_selection_based_on_core.max_rnum) + '_' +
                      str(config.feature.Residue_selection_based_on_core.mut_lr) + '_' +
                      str(config.feature.Residue_selection_based_on_core.core_ratio) + '.pt')
        print('计算完毕，数据集完成生成！！！')

    flag = 1
    if flag and S8338_flag:
        x_data_available = '/data_4T/dataset/Skempi2_ddg_useful'
        x_data = '../datasets/S4169.csv'
        y_data = '../datasets/S4169.csv'

        dataset = CustomDataset(config, x_data_available, x_data, y_data, S8338=True)
        save_variable(dataset, '../datasets/' + config.feature.choice_residue + '/' + 'S8338' + '_' +
                      str(config.feature.Residue_selection_based_on_core.max_rnum) + '_' +
                      str(config.feature.Residue_selection_based_on_core.mut_lr) + '_' +
                      str(config.feature.Residue_selection_based_on_core.core_ratio) + '.pt')
        print('计算完毕，数据集完成生成！！！')


    flag = 1
    if flag and total_flag:
        x_data_available = path + 'recurrence/datasets/SKEMPI2_PDBs/treatment/Cddg/ddg_useful'
        y_data = path + 'recurrence/datasets/total_ddg_S1_S2_realT.csv'

        x_S1131 = path + 'recurrence/datasets/S1131/S1131.csv'
        x_data_dataset_del_S1131 = path + 'recurrence/datasets/SKEMPI2_PDBs/SKEMPI2_datasets/csv/skempi_v2_del_S1131.csv'
        x_M1707 = path + 'recurrence/datasets/M1707/M1707.csv'
        x_data_dataset_del_M1707 = path + 'recurrence/datasets/SKEMPI2_PDBs/SKEMPI2_datasets/csv/skempi_v2_del_M1707.csv'

        x_data = [x_S1131, x_data_dataset_del_S1131, x_M1707, x_data_dataset_del_M1707]

        dataset = CustomDataset(config, x_data_available, x_data, y_data, total=True)
        save_variable(dataset, '../datasets/' + config.feature.choice_residue + '/' + 'total_datasets.pt')
        print('计算完毕，数据集完成生成！！！')

    flag = 1
    if flag and S645_flag:
        x_data_available = '/data_4T/dataset/AB-bind/AB-bind_ddg_useful'
        x_data = '/data_4T/dataset/S645_S1131_S4947_S4169_S8338_S787.xlsx'
        y_data = '/data_4T/dataset/S645_S1131_S4947_S4169_S8338_S787.xlsx'

        dataset = CustomDataset(config, x_data_available, x_data, y_data, S645=True)

        save_variable(dataset, '../datasets/' + config.feature.choice_residue + '/' + 'S645' + '_' +
                      str(config.feature.Residue_selection_based_on_core.max_rnum) + '_' +
                      str(config.feature.Residue_selection_based_on_core.mut_lr) + '_' +
                      str(config.feature.Residue_selection_based_on_core.core_ratio) + '.pt')
        print('计算完毕，数据集完成生成！！！')

    flag = 1
    if flag and S645_no27_flag:
        x_data_available = '/data_4T/dataset/AB-bind/AB-bind_ddg_useful'
        x_data = '/data_4T/dataset/S645_S1131_S4947_S4169_S8338_S787.xlsx'
        y_data = '/data_4T/dataset/S645_S1131_S4947_S4169_S8338_S787.xlsx'

        dataset = CustomDataset(config, x_data_available, x_data, y_data, S645_no27=True)

        save_variable(dataset, '../datasets/' + config.feature.choice_residue + '/' + 'S645_no27' + '_' +
                      str(config.feature.Residue_selection_based_on_core.max_rnum) + '_' +
                      str(config.feature.Residue_selection_based_on_core.mut_lr) + '_' +
                      str(config.feature.Residue_selection_based_on_core.core_ratio) + '.pt')
        print('计算完毕，数据集完成生成！！！')

    flag = 1
    if flag and predict_flag:
        predict_data_path = '/media/wsw/wsw/突变对蛋白质亲和力影响预测/Rosetta_cal/生物实验4K3J_1_2_4_8_20/突变/8突变/Cddg_8突变'

        dataset = CustomDataset(config, predict_data_path, predict_data_path, predict_data_path, predict=True)
        save_variable(dataset, '../datasets/' + config.feature.choice_residue + '/' + 'predict_data.pt')
        print('计算完毕，数据集完成生成！！！')

    flag = 1
    if flag and C3684_flag:


        x_data_available = r'D:\Desktop\literature\portein\Deep learning guided optimization of human antibody against SARS_CoV_2 variants with broad neutralization\recurrence\datasets\SARS-CoV-2 spike protein\Cddg\Cddg_result\Cddg_useful'
        x_data = r'D:\Desktop\literature\portein\Deep learning guided optimization of human antibody against SARS_CoV_2 variants with broad neutralization\recurrence\datasets\SARS-CoV-2 spike protein\C3684.csv'
        y_data = x_data

        dataset = CustomDataset(config, x_data_available, x_data, y_data, C3684=True)
        save_variable(dataset, '../datasets/' + config.feature.choice_residue + '/' + 'C3684.pt')
        print('计算完毕，数据集完成生成！！！')

    flag = 1
    if flag and no_AbAg_flag:
        x_data_available = '/data_4T/dataset/Skempi2_ddg_useful'
        x_data = '../datasets/no_AbAg_datasets.csv'
        y_data = '../datasets/total_ddg_S1_S2_realT.csv'

        dataset = CustomDataset(config, x_data_available, x_data, y_data, no_AbAg=True)

        save_variable(dataset, '../datasets/' + config.feature.choice_residue + '/' + 'no_AbAg' + '_' +
                      str(config.feature.Residue_selection_based_on_core.max_rnum) + '_' +
                      str(config.feature.Residue_selection_based_on_core.mut_lr) + '_' +
                      str(config.feature.Residue_selection_based_on_core.core_ratio) + '.pt')
        print('计算完毕，数据集完成生成！！！')

    flag = 0
    if flag:

        SKEMPI2_PDBs = path + 'recurrence/datasets/SKEMPI2_PDBs/SKEMPI2_datasets/csv/skempi_v2.csv'
        S1131 = path + 'recurrence/datasets/S1131/all.xlsx'

        object_f = open('./skempi_v2-S1131.csv', 'w')
        SKEMPI2_PDBs_f = open(SKEMPI2_PDBs, 'r')
        S1131_f = openpyxl.load_workbook(S1131)
        S1131_f = S1131_f.active

        S1131_dict = {}
        for num, line in enumerate(S1131_f.rows):
            if line[0].value == 'pdbID':
                continue
            if num >= 1403:
                break
            if line[6].value != 1:
                continue
            li = [line[0].value + '_' + line[1].value + '_' + line[2].value, '', line[3].value]
            S1131_dict[CustomDataset.pdb_neme_toAB(li)] = [None, None, line[5].value]

        for line in SKEMPI2_PDBs_f:
            if line[0] == '#':
                object_f.write(line)
                continue
            li = line.strip().split(';')
            ne = CustomDataset.pdb_neme_toAB([li[0], li[1], li[2]])
            try:
                print(S1131_dict[ne])
            except KeyError:
                object_f.write(line)

        object_f.close()
        SKEMPI2_PDBs_f.close()


    flag = 0
    if flag:
        S1131 = path + 'recurrence/datasets/S1131/all.xlsx'
        dataset_del_S1131 = path + 'recurrence/datasets/SKEMPI2_PDBs/SKEMPI2_datasets/csv/最终数据集/skempi_v2_去重复_去问题-S1131.csv'

        dataset_del_S1131_f = open(dataset_del_S1131, 'r')
        S1131_f = openpyxl.load_workbook(S1131)
        S1131_f = S1131_f.active

        S1131_compound = []
        dataset_del_S1131_compound = []

        for line in dataset_del_S1131_f:
            if line[0] == '#':
                continue
            li = line.strip().split(';')
            dataset_del_S1131_compound.append(li[0])

        for num, line in enumerate(S1131_f.rows):
            if line[0].value == 'pdbID':
                continue
            if num >= 1403:
                break
            if line[6].value != 1:
                continue
            S1131_compound.append(line[0].value + '_' + line[1].value + '_' + line[2].value)

        S1131_compound = set(S1131_compound)
        dataset_del_S1131_compound = set(dataset_del_S1131_compound)

        compound = S1131_compound | dataset_del_S1131_compound

        if len(compound) == (len(S1131_compound) + len(dataset_del_S1131_compound)):
            print('S1131和dataset-S1131之间无重复复合物')
        else:
            print('S1131和dataset-S1131之间有重复复合物')

        dataset_del_S1131_f.close()



    flag = 0
    if flag:
        fold_num = 5
        dataset_del_S1131 = path + 'recurrence/datasets/SKEMPI2_PDBs/SKEMPI2_datasets/csv/最终数据集/skempi_v2_去重复_去问题-S1131.csv'
        dataset_del_S1131_f = open(dataset_del_S1131, 'r')

        compound, compound_name, len_dataset_del_S1131 = get_compound_class_data(dataset_del_S1131_f)


        total_compound = len(list(compound.keys()))
        total_data = len_dataset_del_S1131
        compound_name = list(set(compound_name))
        random.seed(2021)
        random.shuffle(compound_name)
        num = len_dataset_del_S1131 // fold_num














        header = get_header(dataset_del_S1131_f)






















        surplus_compound = set(list(compound.keys()))
        surplus_data = {}

        data_num = {}
        path = '../'
        used_compound = 0
        used_data = 0
        for curDir, dirs, files in os.walk(path):
            for i in files:
                if re.match('.*_\d{1}\.csv$', i):
                    datasets = open(path + '/' + i)
                    cur_compound, _, cur_len_num = get_compound_class_data(datasets)
                    data_num[i] = {'compound': len(cur_compound), 'num': cur_len_num}

                    surplus_compound -= set(list(cur_compound.keys()))
                    used_compound += len(list(compound.keys()))
                    used_data += cur_len_num
                    datasets.close()
            break

        for i in list(surplus_compound):
            surplus_data[i] = compound[i]



        surplus_compound = list(surplus_compound)
        all_possibly = 0

        for i in surplus_compound:
            pass

        print('lalala')
