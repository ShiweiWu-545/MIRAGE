from pathlib import Path
import os
import pandas as pd
import numpy as np

from utils.datasets import CustomDataset


def seriesStr_to_str(x):


    return ','.join(x.tolist())
def serInt_to_str(x):


    return ','.join(list(map(str, x)))
def str_diff(x, y):
    x = x.split(',')
    y = y.split(',')
    set1 = set(x)
    set2 = set(y)
    return ','.join(list(set1.symmetric_difference(set2)))
def num_diff(x, y):
    x = x.split(',')
    y = y.split(',')
    return len(y) - len(x)
def res_num(x):
    x = x.split(',')
    return len(x)
def quote_text(val):
    return f'"{val}"' if isinstance(val, str) else val

def find_matching_rows(df, total_df, **kwargs):

    results = []
    out_df = []
    out_list = []


    for index, row in df.iterrows():
        cur_set = row['set']

        try:
            pdb_value = row['PDB']
        except KeyError:
            pdb_value = row['name']
        condition = (total_df['name'] == pdb_value.upper())

        condition &= (total_df['set'] != 'noready')
        condition &= (total_df['set'] == cur_set)
        debug0 = total_df[condition]


        Mutations = row['Mutations']
        if not isinstance(Mutations, str):
            if np.isnan(Mutations):
                condition &= (total_df['Mutations'].isna())
            else:
                assert 'Mutations weizhi' == 0
        else:
            condition &= (total_df['Mutations'] == Mutations)
        debug_mut = total_df[condition]

        if row['set'] == 'test3':
            Mutations = row['Mutations']
            if not isinstance(Mutations, str):
                if np.isnan(Mutations):
                    condition &= (total_df['Mutations'].isna())
            else:
                condition &= (total_df['Mutations'] == Mutations)
            debug_mut = total_df[condition]
            kwargs['Mutations'] = 1

        if len(df) != 1741:
            try:
                Pairwise_interaction = row['Pairwise interaction']
            except KeyError:
                Pairwise_interaction = row['protein_pair'].strip('()').split("'")
                Pairwise_interaction_0 = Pairwise_interaction[1].replace(', ', '')
                Pairwise_interaction_1 = Pairwise_interaction[3].replace(', ', '')
                Pairwise_interaction = Pairwise_interaction_0 + ';' + Pairwise_interaction_1

            proteinA = ', '.join(Pairwise_interaction.strip(';').split(';')[0])
            proteinB = ', '.join(Pairwise_interaction.strip(';').split(';')[1])

            condition &= (total_df['proteinA'].isin([proteinA.strip(' ,'), proteinB.strip(' ,')]))
            debug1 = total_df[condition]
            condition &= (total_df['proteinB'].isin([proteinA.strip(' ,'), proteinB.strip(' ,')]))
            debug2 = total_df[condition]

        for key, value in kwargs.items():
            if key == 'Mutations' and value == 'nan':
                condition &= total_df['Mutations'].isna()
                debug3 = total_df[condition]
            elif key == 'set' and isinstance(value, (list, tuple, str)):
                if isinstance(value, str):
                    condition &= (total_df[key] == value)
                else:
                    condition &= (total_df[key].isin(value))
                debug4 = total_df[condition]
            elif key != 'Mutations':
                condition &= (total_df[key] == value)
                debug5 = total_df[condition]


        filtered_rows = total_df[condition]


        if not filtered_rows.empty:

            result_row = filtered_rows.iloc[-1]
            result_index = result_row.name
            results.append((result_row, result_index))
            out_df.append(result_row)
            out_list.append(result_index)
        else:

            print('lalala')
    out_df = pd.DataFrame(out_df)
    return out_df, out_list


if __name__ == '__main__':
    path_root = Path('/data_4T/dataset/readyPDB/')
    for dirpath, dirnames, filenames in os.walk(path_root):
        break
    set_list = dirnames
    region_analyse_table = pd.DataFrame()





    columns_data = ['pdb_name', 'mut', 'mut_num', 'chains', 'dg', 'mut_region(wt)', 'mut_region(mut)',
                    'wt_cor', 'mut_cor', 'wt_rim', 'mut_rim', 'wt_sup', 'mut_sup',
                    'wt_int', 'mut_int', 'wt_sur', 'mut_sur']
    columns_result = ['mut_region_change',
                      'cor_change(pos)', 'cor_change(num)',
                      'rim_change(pos)', 'rim_change(num)',
                      'sup_change(pos)', 'sup_change(num)',
                      'int_change(pos)', 'int_change(num)',
                      'sur_change(pos)', 'sur_change(num)',
                      'interface_res_num_change', 'set']
    columns = columns_data + columns_result

    columns = columns_data + columns_result
    for c in columns:region_analyse_table[c] = []


    total_df = pd.read_excel('../datasets/PPB-Affinity.xlsx')

    for cur_set in set_list:
        if cur_set == 'AF':continue
        sample_list = os.listdir(path_root / cur_set)

        for pdb_name in sample_list:
            df = pd.DataFrame({
                'PDB': [pdb_name.split('_')[0]],
                'set': [cur_set],
                'Mutations': [np.nan],
                'Pairwise interaction': [';'.join(pdb_name.split('_')[1:])]
            })

            new_row = {}


            new_row['pdb_name'] = pdb_name
            new_row['set'] = cur_set

            try:
                new_row['dg'] = float("{:.2f}".format(find_matching_rows(df, total_df)[0]['dG'].iloc[0]))
            except KeyError:
                continue

            pdb_path = path_root / cur_set / pdb_name
            path_region = pdb_path / (''.join(pdb_name.rsplit('_', 1)) + '.region')

            region_f = pd.read_csv(path_region)


            proteinA = region_f[region_f['protein'] == 0]['chain'].unique()
            proteinB = region_f[region_f['protein'] == 1]['chain'].unique()
            new_row['chains'] = seriesStr_to_str(proteinA).replace(',', '') + '_' + seriesStr_to_str(proteinB).replace(',', '')


            tem_list = region_f[region_f['region'] == 'COR']['pos'].tolist()
            new_row['wt_cor'] = serInt_to_str(tem_list)

            tem_list = region_f[region_f['region'] == 'RIM']['pos'].tolist()
            new_row['wt_rim'] = serInt_to_str(tem_list)

            tem_list = region_f[region_f['region'] == 'INT']['pos'].tolist()
            new_row['wt_int'] = serInt_to_str(tem_list)

            tem_list = region_f[region_f['region'] == 'SUP']['pos'].tolist()
            new_row['wt_sup'] = serInt_to_str(tem_list)

            tem_list = region_f[region_f['region'] == 'SUR']['pos'].tolist()
            new_row['wt_sur'] = serInt_to_str(tem_list)





            region_analyse_table = pd.concat([region_analyse_table, pd.DataFrame([new_row])], ignore_index=True)

    region_analyse_table.to_csv('../datasets/region_analyse_DG.csv', index=False, quoting=1)
    region_analyse_table.to_excel('../datasets/region_analyse_DG.xlsx', index=False, engine='openpyxl')











