import numpy as np
import pandas as pd
import os


def process_samples(test1_path, test2_path, test3_path, s1131_path, ppb_affinity_path, output_dir):

    ppb_affinity = pd.read_excel(ppb_affinity_path)



    ppb_affinity['protein_pair'] = ppb_affinity.apply(
        lambda row: tuple(sorted([row['proteinA'], row['proteinB']])), axis=1
    )


    unique_samples = ppb_affinity[['name', 'Mutations', 'proteinA', 'proteinB', 'protein_pair']].drop_duplicates(
        subset=['name', 'protein_pair', 'Mutations'])


    unique_samples_path = f'{output_dir}/unique_samples_updated.xlsx'
    unique_samples.to_excel(unique_samples_path, index=False)


    test1 = pd.read_excel(test1_path)
    test2 = pd.read_excel(test2_path)
    test3 = pd.read_excel(test3_path)
    s1131 = pd.read_csv(s1131_path)


    def name_formate(x):
        return x.split('_')[0]

    vectorized_upper = np.vectorize(str.upper)
    vectorized_format = np.vectorize(name_formate)
    names_test1 = test1['PDB'].dropna().unique()
    names_test2 = vectorized_upper(test2['PDB'].dropna().unique())
    names_test3 = vectorized_format(test3['#Pdb'].dropna().unique())
    names_s1131 = s1131['#protein'].dropna().unique()


    all_names_to_remove = set(names_test1) | set(names_test2) | set(names_test3) | set(names_s1131)


    filtered_samples = unique_samples[~unique_samples['name'].isin(all_names_to_remove)]


    filtered_samples_path = f'{output_dir}/PPB-Affinity_del_test123_S1131.xlsx'
    filtered_samples.to_excel(filtered_samples_path, index=False)

    return unique_samples_path, filtered_samples_path


os.chdir('../')
unique_samples_path, filtered_samples_path = process_samples(
    './datasets/test1.xlsx',
    './datasets/test2.xlsx',
    './datasets/test3.xlsx',
    './datasets/S1131.csv',
    './datasets/PPB-Affinity.xlsx',
    'debug')
