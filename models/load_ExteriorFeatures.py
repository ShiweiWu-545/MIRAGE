import h5py
from torch import cdist
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import pairwise_distances
from Bio import PDB
import os




ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4


def get_contact_dis_mat(pos_a, pos_b, mask_a, mask_b):
    l1, n_atom = pos_a.shape[:2]
    l2, n_atom = pos_b.shape[:2]

    mask_a = mask_a.astype('bool')
    mask_b = mask_b.astype('bool')
    p1 = pos_a[mask_a]
    p2 = pos_b[mask_b]
    dis_mat = cdist(p1, p2)


    index_a = np.tile(np.arange(l1)[:, None], reps=[1, n_atom])[mask_a]
    index_b = np.tile(np.arange(l2)[:, None], reps=[1, n_atom])[mask_b]


    contact_dis_mat = dis_mat

    groups = index_a
    _ndx = np.argsort(groups)
    _id, _pos, g_count = np.unique(groups[_ndx], return_index=True, return_counts=True)
    contact_dis_mat = np.minimum.reduceat(contact_dis_mat[_ndx], _pos, axis=0)

    groups = index_b
    _ndx = np.argsort(groups)
    _id, _pos, g_count = np.unique(groups[_ndx], return_index=True, return_counts=True)
    contact_dis_mat = np.minimum.reduceat(contact_dis_mat[:, _ndx], _pos, axis=1)

    return contact_dis_mat


def single_mutation_parsing(mutation):
    aatype_from = mutation[0]
    chainid = mutation[1]
    position = int(mutation[2:-1])

    aatype_to = mutation[-1]
    return chainid, position, aatype_from, aatype_to


def mutation_parsing_to_frame(mutation):
    cols = 'chainid, position, aatype_from, aatype_to'.split(', ')
    mutations = []
    for m in mutation.split(';'):
        mutations += [single_mutation_parsing(m)]
    df_mut = pd.DataFrame(mutations, columns=cols)
    df_mut['position'] = df_mut['position'].astype('int')
    return df_mut


def get_contact_node(adjacency_matrix, max_len, limit_length):

    edges = []
    num_nodes = len(adjacency_matrix)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i, j] > 0:
                edges.append((i, j, adjacency_matrix[i, j]))


    edges.sort(key=lambda x: x[2])


    selected_nodes = set()
    for edge in edges:
        u, v, weight = edge
        if weight == 10000:
            break
        if weight > limit_length:
            break
        selected_nodes.add(u)
        if len(selected_nodes) >= max_len:
            break
        selected_nodes.add(v)


        if len(selected_nodes) >= max_len:
            break
    return selected_nodes


def get_knn_dis_index(pos37, mask37, chainids, limit_length, n_neighbor, max_len=6000):
    pos37 = pos37[:max_len]
    mask37 = mask37[:max_len]
    chainid = chainids[:max_len]



    ca_wt = pos37[:, 1]


    contact_dis_mat = pairwise_distances(ca_wt, ca_wt)


    contact_dis_mat[chainid[None, :] == chainid[:, None]] = 10000






    inter_res_index = get_contact_node(contact_dis_mat, n_neighbor, limit_length)









    return list(inter_res_index)


def get_knn_dis_index_mat(pos14, n_neighbor=None):
    if n_neighbor == None:
        return np.arange(pos14.shape[0])
    ca_pos = pos14[:, ATOM_CA]

    dis_mat = pairwise_distances(ca_pos, ca_pos)
    dis_mat_zero_mask = dis_mat == 0
    dis_mat[dis_mat_zero_mask] = np.inf
    dis_min_arg = dis_mat.argsort(axis=-1)
    inter_res_index = dis_min_arg[:, :n_neighbor]
    return inter_res_index


def energy_filter(energyfile):
    del_labels = ['SCORE:', 'pose_id', "fa_intra_rep", 'fa_intra_sol_xover4', 'lk_ball_wtd', 'pro_close',
                  'hbond_sr_bb', 'dslf_fa13', 'omega', 'fa_dun', 'p_aa_pp', 'yhh_planarity',
                  'ref', 'rama_prepro', 'total', 'description']
    energyfile = energyfile[energyfile['restype2'] != 'onebody']
    energyfile = energyfile.drop(del_labels, axis=1)
    energyfile[['resi1', 'resi2']] = energyfile[['resi1', 'resi2']].astype(int)

    len_res = max(energyfile['resi2'])
    energy_matrix = np.zeros((len_res, len_res, 7))
    for _, row in energyfile.iterrows():
        energy_matrix[row['resi1'] - 1][row['resi2'] - 1] = row['fa_atr':]
        energy_matrix[row['resi2'] - 1][row['resi1'] - 1] = row['fa_atr':]
    return energy_matrix




class ComplexLoader(object):
    def __init__(self, data_root=None):
        super(ComplexLoader, self).__init__()



    def get_protein(self, paths, chains, datasets):
        reverse_flag = False
        HM_flag = False
        if paths.split('_', 1)[0] == 'reverse':
            reverse_flag = True
            paths = paths.split('_', 1)[1]
        elif paths.split('_', 1)[0] == 'HM':
            HM_flag = True

        feature_dict = {}
        if datasets == 'Skempi2':
            data_root = Path("/data_4T/dataset/Skempi2_ddg_useful")
            region_dir = Path("/data_4T/dataset/S2_relax/region")
            h5_path_root = data_root
            name = paths.split('_')
            region_name = name[0] + '_' + name[1]
            region_path = region_dir / region_name / (region_name + ".region")

            mut_path = data_root / paths / ('MUT_' + paths + ".pdb")
            wt_path = data_root / paths / ('WT_' + paths + ".pdb")
            mut_energy_path = data_root / paths / "mut_energy.sc"
            wt_energy_path = data_root / paths / "wt_energy.sc"
            mut_h5_path = data_root / paths / ('MUT_' + paths + ".h5")
            wt_h5_path = data_root / paths / ('WT_' + paths + ".h5")

        elif datasets == 'ABbind':
            data_root = Path("/data_4T/dataset/AB-bind_ddg_useful")

            h5_path_root = data_root
            name = paths.split('_')
            if HM_flag:
                region_name = name[0] + '_' + name[1] + '_' + name[2]
            else:
                region_name = name[0] + '_' + name[1]


            mut_path = data_root / paths / ('MUT_' + paths + ".pdb")
            wt_path = data_root / paths / ('WT_' + paths + ".pdb")
            mut_energy_path = data_root / paths / "mut_energy.sc"
            wt_energy_path = data_root / paths / "wt_energy.sc"
            mut_h5_path = data_root / paths / ('MUT_' + paths + ".h5")
            wt_h5_path = data_root / paths / ('WT_' + paths + ".h5")


        feature_dict_path = data_root / paths / ('ExteriorFeatures_' + paths + '.feature')
        if os.path.exists(feature_dict_path):

            from utils.datasets import load_variable
            return load_variable(feature_dict_path)












        wt_prottrans_ori = h5py.File(wt_h5_path, 'r')
        mut_prottrans_ori = h5py.File(mut_h5_path, 'r')

        chains = self.get_chains(wt_path)
        if reverse_flag:
            wt_prottrans = np.concatenate([mut_prottrans_ori['PDB|' + k][:] for k in chains], axis=0)



            mut_prottrans = np.concatenate([wt_prottrans_ori['PDB|' + k][:] for k in chains], axis=0)
        else:
            wt_prottrans = np.concatenate([wt_prottrans_ori['PDB|' + k][:] for k in chains], axis=0)
            mut_prottrans = np.concatenate([mut_prottrans_ori['PDB|' + k][:] for k in chains], axis=0)

        feature_dict = {



            'wt_prottrans': wt_prottrans,
            'mut_prottrans': mut_prottrans,
        }

        from utils.datasets import save_variable
        save_variable(feature_dict, feature_dict_path)
        return feature_dict

    def get_chains(self, path):
        result = []

        parser = PDB.PDBParser()


        structure = parser.get_structure('0', path)
        model = structure[0]


        for chain in model:
            result.append(chain.id)
        return result



    def load(self, paths_pdb, proteinA, proteinB, max_length, limit_length, set):

        if isinstance(paths_pdb, str):
            paths_wt = paths_pdb.split(',')


        pdbs, energy, region, prottrans = self.get_protein(paths_pdb, set)

        index_cor = region[region["region"] == "COR"]['pos'].tolist()
        index_rim = region[region["region"] == "RIM"]['pos'].tolist()
        index_sup = region[region["region"] == "SUP"]['pos'].tolist()
        conbine = index_cor + index_sup + index_rim
        conbine = conbine[:128]
        inter_region_index = [x - 1 for x in conbine]

        chains = list(pdbs.keys())


        chains_id = []
        for chain in chains:
            pdbs[chain] = pdbs[chain].mask_select(pdbs[chain].mask == 1)
            if chain in proteinA:
                chains_id.append(0)
            elif chain in proteinB:
                chains_id.append(1)
            else:
                raise ValueError(chain + ' not in ' + paths_pdb + ''.join(proteinA) + ''.join(proteinB))
























        if ("XXXX|" + chains[0]) in prottrans.keys():
            res_prottrans = np.concatenate([prottrans['XXXX|' + x][:] for x in chains], axis=0)
        elif ("PDB|" + chains[0]) in prottrans.keys():
            res_prottrans = np.concatenate([prottrans['PDB|' + x][:] for x in chains], axis=0)


































        return energy, res_prottrans




class UnibindDataset(Dataset):
    def __init__(self, df_path, data_root, max_length=256,
                 col_pdb='name', col_proteinA='proteinA', col_proteinB='proteinB', col_allchains='all_chains',
                 set_c='set', cols_label=None,
                 n_neighbors=None, limit_length=None,
                 train=True, diskcache=None, ):

        if cols_label is None:
            cols_label = ['dG']

        self.data_loader = ComplexLoader(data_root)

        if isinstance(df_path, pd.DataFrame):
            self.df = df_path.copy()
        else:
            self.df = pd.read_csv(df_path, low_memory=False)





        self.max_length = max_length
        self.n_neighbors = n_neighbors
        self.limit_length = limit_length
        self.col_pdb = col_pdb
        self.col_proteinA = col_proteinA
        self.col_proteinB = col_proteinB
        self.col_allchains = col_allchains
        self.cols_label = cols_label
        self.set_c = set_c
        self.train = train
        self.diskcache = diskcache


        for col in self.cols_label:
            self.df[col] = self.df.get(col, np.nan)
        self.num_classes = len(self.cols_label)

    def __len__(self):
        return len(self.df)


    def load_data(self, paths_pdb, proteinA, proteinB, max_length, limit_length, allchains, cache, set):
        if '_' not in paths_pdb:
            paths_pdb = paths_pdb + '_' + allchains
        key = paths_pdb, proteinA, proteinB, max_length, limit_length, set

        if cache is None:
            values = self.data_loader.load(*key)
            return values





        values = self.data_loader.load(*key)
        cache[key] = values
        return values


    def get_data(self, idx):
        case = self.df.iloc[idx]
        paths_pdb = case[self.col_pdb]

        proteinA = case[self.col_proteinA]
        proteinB = case[self.col_proteinB]
        proteinA = [it for it in proteinA]
        proteinB = [it for it in proteinB]
        allchains = case[self.col_allchains]
        set_c = case[self.set_c]
        cache = self.diskcache




        values = self.load_data(paths_pdb, proteinA, proteinB, self.max_length, self.limit_length, allchains, cache,
                                set_c)
        p14 = values[0]
        energy_wt = values[1]
        res_prottrans = values[2]


        labels = case[self.cols_label]
        labels_valid_mask = True







        complex_all = {
            'pos14': p14.atom_positions.astype('float32'),
            'pos14_mask': p14.atom_mask.astype('bool'),
            'aa': p14.aatype.astype('int'),
            'seq': p14.residue_index.astype('int'),
            'chain_seq': p14.chainid.astype('int'),
            'res_energy': energy_wt.astype('float32'),



            'res_prottrans': res_prottrans.astype('float32'),
        }








        complex_all['neighbors'] = get_knn_dis_index_mat(complex_all['pos14'],
                                                         len(complex_all['pos14']))


        label = (labels.values.astype('float32'), labels_valid_mask, set_c)

        return complex_all, label

    def __getitem__(self, idx):
        try:
            return self.get_data(idx)
        except Exception as e:
            case = self.df.iloc[idx]
            paths_wt = case[self.col_pdb]
            print(case, paths_wt)
            print(e)
            raise e
