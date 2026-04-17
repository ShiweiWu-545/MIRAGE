import math
import torch
from torch.utils.data._utils.collate import default_collate

from .protein import ATOM_CA, parse_pdb
from data.my_config import config
from .misc import recursive_to
import numpy as np
import pandas as pd
import copy
from pathlib import Path



class PaddingCollate(object):

    def __init__(self, length_ref_key='mutation_mask', pad_values={'aa': 20, 'pos14': float('999'), 'icode': ' ', 'chain_id': '-'}, donot_pad={'foldx'}, eight=False):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.donot_pad = donot_pad
        self.eight = eight

    def _pad_last(self, x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            if len(x.size()) > 2:
                if x.size(0) == x.size(1):
                    cha = n - x.size(0)
                    x = np.pad(x.numpy(), ((0, cha), (0, cha), (0, 0)), 'constant')
                    x = torch.from_numpy(x)
                    return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            if value == 0:
                return x
            pad = [value] * (n - len(x))
            return x + pad
        elif isinstance(x, str):
            if value == 0:
                return x
            pad = value * (n - len(x))
            return x + pad
        elif isinstance(x, dict):
            padded = {}
            for k, v in x.items():
                if k in self.donot_pad:
                    padded[k] = v
                else:
                    padded[k] = self._pad_last(v, n, value=self._get_pad_value(k))
            return padded
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):


        max_length = 128
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:




            if len(data['aa']) > 128:
                print('lalala')
            data_padded = {}


            for k, v in data.items():

                    data_padded[k] = self._pad_last(v, max_length, value=self._get_pad_value(k))


            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)



def _mask_list(l, mask):
    return [l[i] for i in range(len(l)) if mask[i]]


def _mask_string(s, mask):


    return ''.join([s[i] for i in range(len(s)) if mask[i]])


def _mask_dict_recursively(d, mask):

    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and v.size(0) == mask.size(0):
            try:
                if v.size(0) == v.size(1):
                    out[k] = v[mask][:, mask]
                else:
                    out[k] = v[mask]
            except IndexError:
                out[k] = v[mask]
        elif isinstance(v, list) and len(v) == mask.size(0):
            out[k] = _mask_list(v, mask)
        elif isinstance(v, str) and len(v) == mask.size(0):
            out[k] = _mask_string(v, mask)
        elif isinstance(v, dict):
            out[k] = _mask_dict_recursively(v, mask)
        else:
            out[k] = v
    return out


class mult_128_KnnResidue(object):

    def __init__(self, num_neighbors=128):
        super().__init__()
        self.num_neighbors = num_neighbors

    def __call__(self, data):
        pos_CA = data['wt']['pos14'][:, ATOM_CA]
        pos_CA_mut = pos_CA[data['mutation_mask']]
        diff = pos_CA_mut.view(1, -1, 3) - pos_CA.view(-1, 1, 3)
        dist = torch.linalg.norm(diff, dim=-1)

        try:
            mask = torch.zeros([dist.size(0), dist.size(1)], dtype=torch.bool)
            mask[dist.argsort(axis=0)[:self.num_neighbors]] = True
        except IndexError as e:
            print(data)
            raise e


        return _mask_dict_recursively(data, mask)


class KnnResidue(object):

    def __init__(self, num_neighbors=config.feature.nearby_residues,
                 max_rnum=config.feature.Residue_selection_based_on_core.max_rnum,
                 mut_lr=config.feature.Residue_selection_based_on_core.mut_lr,
                 core_ratio=config.feature.Residue_selection_based_on_core.core_ratio):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.max_rnum = max_rnum
        self.mut_lr = mut_lr
        self.core_ratio = core_ratio

    def get_mut_around_res(self, pos_CA, pos_CA_mut):
        diff = pos_CA_mut.view(1, -1, 3) - pos_CA.view(-1, 1, 3)
        dist = torch.linalg.norm(diff, dim=-1)
        try:
            index_list = dist.min(dim=1)[0].argsort().tolist()
        except IndexError as e:
            raise e
        return index_list

    def get_center(self, data):

        num_points = data.size(0)

        sum_of_points = torch.sum(data, dim=0)

        center = sum_of_points / num_points
        return center

    def translate_point(self, point1, point2, distance):

        translation_vector = point2 - point1

        vector_length = torch.norm(translation_vector)

        unit_translation_vector = translation_vector / vector_length

        scaled_translation_vector = unit_translation_vector * distance

        translated_point1 = point1 + scaled_translation_vector
        return translated_point1

    def __call__(self, data, core_index, lock_file_csv=None):

        core_mask = torch.zeros([data['mutation_mask'].size(0)], dtype=torch.bool)
        core_mask[core_index] = True


        pos_CA = data['wt']['pos14'][:, ATOM_CA]
        pos_CA_mut = pos_CA[data['mutation_mask']]


        if core_index is None:
            index_list = self.get_mut_around_res(pos_CA, pos_CA_mut)[:self.num_neighbors]
            mask = torch.zeros([data['mutation_mask'].size(0)], dtype=torch.bool)
            mask[index_list] = True
            return _mask_dict_recursively(data, mask), mask


        index_list_old = self.get_mut_around_res(pos_CA, pos_CA_mut)[:self.num_neighbors]
        index_list = copy.deepcopy(index_list_old)
        core_pos_CA_center = self.get_center(pos_CA[core_mask])
        rnum = 0

        if config.feature.Residue_selection_based_on_core.dynamic_mut:

            missing_core_ratio = (len(set(core_index) - (set(index_list))) / len(set(core_index)))

            new_df = pd.DataFrame()
            new_df['core_ratio'] = [1 - missing_core_ratio]
            new_df['distance(core center-mut)'] = (torch.norm(core_pos_CA_center - pos_CA_mut)).item()
            if lock_file_csv != None:
                lock_file_csv.acquire()
                new_df.to_csv('../error_record/core_ratio.csv', mode='a', header=False, index=False)
                lock_file_csv.release()
            else:
                new_df.to_csv('../error_record/core_ratio.csv', mode='a', header=False, index=False)

            while (len(set(core_index) - (set(index_list))) / len(set(core_index))) > (1 - self.core_ratio):
                missing_core_ratio = (len(set(core_index) - (set(index_list))) / len(set(core_index)))
                print('不在所选残基中的core占总core的比例：', missing_core_ratio)
                pos_CA_mut = self.translate_point(pos_CA_mut, core_pos_CA_center, self.mut_lr)
                index_list = self.get_mut_around_res(pos_CA, pos_CA_mut)[:self.num_neighbors]
                rnum += 1
                if rnum >= self.max_rnum:
                    break
        else:
            index_list = [x for x in index_list if x not in core_index]
            index_list = list(core_index) + index_list[:(self.num_neighbors - len(list(core_index)))]

        if config.feature.Residue_selection_based_on_core.Residues_outside_the_regulatory_capacity:
            mask = torch.zeros([data['mutation_mask'].size(0)], dtype=torch.bool)
            mask[index_list] = True
        else:
            if rnum >= self.max_rnum:
                mask = torch.zeros([data['mutation_mask'].size(0)], dtype=torch.bool)
                mask[index_list_old] = True
            else:
                mask = torch.zeros([data['mutation_mask'].size(0)], dtype=torch.bool)
                mask[index_list] = True


        return data, mask


def y_reverse(str_y):
    if str_y[0] == '-':
        str_y = str_y[1:]
    else:
        str_y = '-' + str_y
    return str_y


def get_mutant_nearby_residues(data_wt, data_mut, pdb_name, core_index, lock_file_csv=None):
    transform = KnnResidue()
    mutation_mask = (data_wt['aa'] != data_mut['aa'])
    batch, mask = transform({'wt': data_wt, 'mut': data_mut, 'mutation_mask': mutation_mask}, core_index, lock_file_csv)
    return batch, mask


def get_protein_divide(pdb_name):

    chains = []
    tem_data = open('../datasets/skempi_v2.csv').readlines()
    for line in tem_data:
        if line[0] == '#': continue
        name = line.split(';', 1)[0]
        if pdb_neme_toAB(name) == pdb_name.split('_')[0] + '_' + pdb_name.split('_')[1]:
            chains = name.split('_', 1)[1].split('_')
    return chains


def pdb_neme_toAB(i):
    mut = None
    if isinstance(i, list):
        mut = i[2]
        i = i[0]
    tem = i.split('_', 1)
    tem[1] = tem[1].replace('_', '')
    tem = tem[0] + '_' + tem[1]

    if mut != None:
        mut = mut.replace(',', '_')
        pdb = tem + '_' + mut
    else:
        pdb = tem
    return pdb


def get_rASA_surface_residues(config, data_wt, data_mut=None, pdb_name=None):
    if config.model.mission == 'DDG':
        mutation_mask = (data_wt['aa'] != data_mut['aa'])
        if True not in mutation_mask:
            raise IndexError

        chains = get_protein_divide(pdb_name)
    else:
        chains = pdb_name.split('_')[1:]

    mask = data_wt['interface_label'].bool()

    if config.model.mission == 'DDG':
        batch = _mask_dict_recursively({'wt': data_wt, 'mut': data_mut, 'mutation_mask': mutation_mask}, mask)
    else:
        batch = _mask_dict_recursively(data_wt, mask)
    return batch, mask



    print('lalala')



def get_junction_surface_residues(config, data_wt, data_mut=None, pdb_name=None):
    if config.model.mission == 'DDG':
        mutation_mask = (data_wt['aa'] != data_mut['aa'])
        if True not in mutation_mask:
            raise IndexError

        chains = get_protein_divide(pdb_name)
    else:
        chains = pdb_name.split('_')[1:]


    pos_CA = data_wt['pos14'][:, ATOM_CA]
    chain0_index = []
    for i in list(data_wt['chain_id']):
        res_chain_flag = False
        for n in chains[0]:
            if i == n:
                chain0_index.append(True)
                res_chain_flag = True
                break
        if not res_chain_flag:
            for n in chains[1]:
                if i == n:
                    chain0_index.append(False)
                    res_chain_flag = True
                    break
        assert res_chain_flag
    _, chain0 = portion_to_portion_minimum_distance(data_wt['pos14'][chain0_index, :], data_wt['pos14'][reverse_bool(chain0_index), :])
    _, chain1 = portion_to_portion_minimum_distance(data_wt['pos14'][reverse_bool(chain0_index), :], data_wt['pos14'][chain0_index, :])


    distance_index = []
    num0, num1 = 0, 0
    for bool_index in chain0_index:
        if bool_index:
            distance_index.append(int(chain0[num0]))
            num0 += 1
        else:
            distance_index.append(int(chain1[num1]))
            num1 += 1


    mask = torch.zeros([len(distance_index)], dtype=torch.bool)
    for mask_num, i in enumerate(distance_index):
        if i < config.feature.surface:
            mask[mask_num] = True


    if config.model.mission == 'DDG':
        batch = _mask_dict_recursively({'wt': data_wt, 'mut': data_mut, 'mutation_mask': mutation_mask}, mask)
    else:
        batch = _mask_dict_recursively(data_wt, mask)
    return batch, mask


def get_region(config, pdb_name):
    if config.model.mission == 'DDG':
        if pdb_name.split('_')[0] == 'reverse':
            pdb = '_'.join(pdb_name.split('_')[1: 3])
        elif pdb_name.split('_')[0] == 'HM':
            pdb = '_'.join(pdb_name.split('_')[0: 3])
        else:
            pdb = '_'.join(pdb_name.split('_')[0: 2])
        region_path = Path(config.datasets.region_path)
        chain_A_B = get_protein_divide(pdb_name)
    else:
        root_path = Path(config.datasets_DG.readyPDB)
        paths_pdb = pdb_name[0].split('_')[0]
        proteinA = pdb_name[0].split('_')[1]
        proteinB = pdb_name[0].split('_')[2]
        set_name = pdb_name[1]
        chain_A_B = [proteinA, proteinB]
        pdb = paths_pdb + '_' + proteinA + proteinB
        pdb_name = pdb_name[0]

        if set_name == "SKEMPI_v2.0":
            data_root = Path(config.datasets_DG.Skempi2)


            pdb_sp = paths_pdb.split('_')
            if len(pdb_sp) > 2:
                dir = data_root / paths_pdb
                path = dir / ('MUT_' + paths_pdb + ".pdb")
                energy_path = dir / "mut_energy.sc"
                h5_path = dir / ('MUT_' + paths_pdb + ".h5")

                region_path = dir
                pdb = 'MUT_' + paths_pdb
            else:
                file = list(data_root.glob(f"{paths_pdb}*"))[0]
                name = str(file.name)
                dir = data_root / file
                path = dir / ('WT_' + name + ".pdb")
                energy_path = dir / "wt_energy.sc"
                h5_path = dir / ('WT_' + name + ".h5")

                region_path = dir
                pdb = 'WT_' + name
        else:
            dir = root_path / set_name / (paths_pdb + '_' + proteinA + '_' + proteinB)
            path = dir / (paths_pdb + '_' + proteinA + proteinB + ".pdb")
            energy_path = dir / "wt_energy.sc"

            region_path = dir
            h5_path = dir / (paths_pdb + '_' + proteinA + proteinB + ".h5")


    return_dict = {}

    region_dict = {'COR': [],
                   'RIM': [],
                   'INT': [],
                   'SUP': [],
                   'SUR': [], }

    region_dict_proteinA = {'COR': [],
                            'RIM': [],
                            'INT': [],
                            'SUP': [],
                            'SUR': [], }

    region_dict_proteinB = {'COR': [],
                            'RIM': [],
                            'INT': [],
                            'SUP': [],
                            'SUR': [], }


    try:
        if pdb_name.split('_')[0] == 'reverse':
            region_f = pd.read_csv(region_path / (pdb.split('_', 1)[1] + '.region'))
        else:
            region_f = pd.read_csv(region_path / (pdb + '.region'))
    except FileNotFoundError:

        from utils.cal_rasa import get_region_ske
        get_region_ske(region_path / (pdb + '.pdb'), chain_A_B[0], chain_A_B[1])


        try:
            if pdb_name.split('_')[0] == 'reverse':
                region_f = pd.read_csv(region_path / (pdb.split('_', 1)[1] + '.region'))
            else:
                region_f = pd.read_csv(region_path / (pdb + '.region'))
        except FileNotFoundError:
            print('FileNotFoundError', pdb)
            return None



    return_dict['complex'] = region_dict
    return_dict['protein' + chain_A_B[0]] = region_dict_proteinA
    return_dict['protein' + chain_A_B[1]] = region_dict_proteinB

    for index, line in region_f.iterrows():


        pos = str(line['pos'])
        res = line['res']
        chain = line['chain']
        reg = line['region']


        region_dict[reg].append(pos)

        if chain in chain_A_B[0]:
            region_dict_proteinA[reg].append(pos)

        elif chain in chain_A_B[1]:
            region_dict_proteinB[reg].append(pos)


    return_dict['complex'] = region_dict
    return_dict['protein' + chain_A_B[0]] = region_dict_proteinA
    return_dict['protein' + chain_A_B[1]] = region_dict_proteinB

    return return_dict


def get_core_residues(config, data_wt, data_mut, pdb_name):
    mutation_mask = (data_wt['aa'] != data_mut['aa'])
    data = {'wt': data_wt, 'mut': data_mut, 'mutation_mask': mutation_mask}
    if True not in mutation_mask:
        raise IndexError


    region_dict = get_region(pdb_name)
    if region_dict is None:return None, None

    core_resseq = region_dict['complex']['COR']

    mask = torch.zeros([data_wt['resseq'].size(0),], dtype=torch.bool)
    indices = [index for index, value in enumerate(data_wt['resseq'].tolist()) if str(value) in core_resseq]

    mask[indices] = True
    return _mask_dict_recursively(data, mask), mask


def get_interface_residues(config, data_wt, pdb_name):






    region_dict = get_region(config, pdb_name)
    if region_dict is None:return None, None

    interface_resseq = region_dict['complex']['COR'] + region_dict['complex']['RIM'] + region_dict['complex']['SUP']

    mask = torch.zeros([data_wt['resseq'].size(0),], dtype=torch.bool)
    indices = [index for index, value in enumerate(data_wt['resseq'].tolist()) if str(value) in interface_resseq]

    mask[indices] = True
    return '', mask


def choice_residue_mission_navigation(config, data_wt, data_mut, pdb_name, lock_file_csv=None):
    batch = {}
    if config.model.mission == 'DDG':
        batch = DDG_choice_residue(config, data_wt, data_mut, pdb_name, lock_file_csv=None)
    elif config.model.mission == 'DG':
        batch = DG_choice_residue(config, data_wt, pdb_name, lock_file_csv=None)
    return batch


def DG_choice_residue(config, data_dg, pdb_name, lock_file_csv=None):
    batch = {}
    if config.feature.DGchoice_residue == 'Junction surface residues':
        batch, mask = get_junction_surface_residues(config, data_dg, pdb_name=pdb_name)
    elif config.feature.DGchoice_residue == 'rASA surface':
        batch, mask = get_rASA_surface_residues(config, data_dg, pdb_name=pdb_name)

    return batch


def DDG_choice_residue(config, data_wt, data_mut, pdb_name, lock_file_csv=None):

    core_index = None
    if config.feature.Residue_selection_based_on_core.label:
        _, core_mask = get_core_residues(config, data_wt, data_mut, pdb_name)
        if core_mask is None:
            core_index = None
        else:
            core_index = torch.nonzero(core_mask).squeeze().tolist()

    _, interface_mask = get_interface_residues(config, data_wt, pdb_name)


    batch = {}
    if config.feature.choice_residue == 'Mutant nearby residues':
        data, mask = get_mutant_nearby_residues(data_wt, data_mut, pdb_name, core_index, lock_file_csv)
    elif (config.feature.choice_residue == 'Junction surface residues_CA') or\
            (config.feature.choice_residue == 'Junction surface residues_14'):
        data, mask = get_junction_surface_residues(data_wt, data_mut, pdb_name)
    elif (config.feature.choice_residue == 'The intersection of A and B_14') or\
            (config.feature.choice_residue == 'The intersection of A and B_CA'):
        mutation_mask = (data_wt['aa'] != data_mut['aa'])
        if True not in mutation_mask: raise IndexError
        _, mut_mask = get_mutant_nearby_residues(data_wt, data_mut, pdb_name)
        _, junction_mask = get_junction_surface_residues(data_wt, data_mut, pdb_name)

        mask = mut_mask & junction_mask

        data = {'wt': data_wt, 'mut': data_mut, 'mutation_mask': mutation_mask}


    data['interface_label'] = interface_mask.int()
    batch = _mask_dict_recursively(data, mask)
    return batch


def choice_residue(config, data_wt, data_mut=None, pdb_name=None, lock_file_csv=None):
    batch = choice_residue_mission_navigation(config, data_wt, data_mut, pdb_name, lock_file_csv=None)
    return batch





def residues_fill(pdb_name, batch):

    residues_fill_flag = False


    chains = get_protein_divide(pdb_name)
    chains_flag = [False, False]


    chains_num = [0, 0]
    for i in batch['wt']['chain_id']:
        if i in chains[0]: chains_num[0] += 1
        elif i in chains[1]: chains_num[1] += 1
    if chains_num[0] < config.feature.surface or chains_num[1] < config.feature.surface: residues_fill_flag = True


    cur_num = 0
    for index, i in enumerate(chains_num):
        if i < config.feature.surface:
            cur_num += config.feature.surface
            chains_flag[index] = True
        else:
            cur_num += i


    if cur_num >= config.feature.nearby_residues:
        if not residues_fill_flag: return batch
        for i, cha, num in zip(chains_flag, chains, chains_num):
            if not i: continue
            cha = list(cha)[-1]
            batch = run_residues_fill(batch, cha, config.feature.surface - num)
    else:
        if not residues_fill_flag:
            cha = list(chains[-1])[-1]
            batch = run_residues_fill(batch, cha, config.feature.nearby_residues - sum(chains_num))
        else:



            for i, cha, num in zip(chains_flag, chains, chains_num):
                if not i: continue
                cha = list(cha)[-1]
                batch = run_residues_fill(batch, cha, config.feature.nearby_residues - sum(chains_num))

    return batch


def run_residues_fill(batch, chain, difference_value):




    insert_point = 0


    tem_variate = True
    for j, i in enumerate(batch['wt']['chain_id']):
        if i != chain and tem_variate:
            continue
        elif i == chain:
            tem_variate = False
            continue
        tem_variate = True
        insert_point = j
        break
    if not tem_variate:
        insert_point = j + 1


    num = 0
    last = ''
    for i in batch['wt']['chain_id']:
        if last != i:
            last = i
            num += 1
        if i == chain:
            break
    batch['wt']['chain_seq'] = tensor_insert(batch['wt']['chain_seq'], insert_point, num, difference_value)
    batch['mut']['chain_seq'] = tensor_insert(batch['mut']['chain_seq'], insert_point, num, difference_value)


    batch['wt']['chain_id'] = str_insert(batch['wt']['chain_id'], insert_point, chain, difference_value)
    batch['mut']['chain_id'] = str_insert(batch['mut']['chain_id'], insert_point, chain, difference_value)


    batch['wt']['aa'] = tensor_insert(batch['wt']['aa'], insert_point, 20, difference_value)
    batch['mut']['aa'] = tensor_insert(batch['mut']['aa'], insert_point, 20, difference_value)


    batch['wt']['resseq'] = tensor_insert(batch['wt']['resseq'], insert_point, 0, difference_value)
    batch['mut']['resseq'] = tensor_insert(batch['mut']['resseq'], insert_point, 0, difference_value)


    batch['wt']['icode'] = str_insert(batch['wt']['icode'], insert_point, ' ', difference_value)
    batch['mut']['icode'] = str_insert(batch['mut']['icode'], insert_point, ' ', difference_value)


    batch['wt']['seq'] = tensor_insert(batch['wt']['seq'], insert_point, 99999, difference_value)
    batch['mut']['seq'] = tensor_insert(batch['mut']['seq'], insert_point, 99999, difference_value)


    batch['wt']['pos14'] = tensor_insert(batch['wt']['pos14'], insert_point, 99999, difference_value)
    batch['mut']['pos14'] = tensor_insert(batch['mut']['pos14'], insert_point, 99999, difference_value)


    batch['wt']['pos14_mask'] = tensor_insert(batch['wt']['pos14_mask'], insert_point, False, difference_value)
    batch['mut']['pos14_mask'] = tensor_insert(batch['mut']['pos14_mask'], insert_point, False, difference_value)

    batch['mutation_mask'] = (batch['wt']['aa'] != batch['mut']['aa'])

    return batch


def str_insert(str_data, insert_point, insert_data, num):
    B = insert_data * num
    A = str_data[: insert_point]
    C = str_data[insert_point:]
    out = A + B + C

    return out


def tensor_insert(tensor_data, insert_point, insert_data, num):

    out = torch.zeros((0, ))
    if len(tensor_data.size()) == 1:
        B = torch.full((num,), insert_data, dtype=tensor_data.dtype).unsqueeze(1)
        tensor_data = tensor_data.unsqueeze(1)
        A = tensor_data[: insert_point]
        C = tensor_data[insert_point:]
        out = torch.cat((A, B), dim=0)
        out = torch.cat((out, C), dim=0).squeeze(-1)
    elif len(tensor_data.size()) == 3:
        B = torch.full((num, config.feature.Atom[1], 3), insert_data, dtype=tensor_data.dtype)
        A = tensor_data[: insert_point]
        C = tensor_data[insert_point:]
        out = torch.cat((A, B), dim=0)
        out = torch.cat((out, C), dim=0)

    return out

def filtered_999atom(point):
    filtered_point_list = []
    for i in range(point.shape[0]):
        residue = point[i]
        filtered_residue = residue[~torch.all(residue > 999, dim=1)]
        filtered_point_list.append(filtered_residue)



    filtered_tensor = point[~point.all(point > 999, dim=2)]


    out_list = []





    return out_list


def portion_to_portion_minimum_distance(initial_point, termination_point):




    if torch.cuda.is_available():
        initial_point = recursive_to(initial_point, config.model.device)
        termination_point = recursive_to(termination_point, config.model.device)

    if len(initial_point.shape) == 3:
        distance = termination_point[None, :, None, :, :] - initial_point[:, None, :, None, :]
        distance = torch.linalg.norm(distance, dim=-1)
        distance = distance.min(dim=3)[0]


        padding = torch.full(distance.shape, 99999, dtype=distance.dtype).to(distance)
        distance = torch.where(~distance.bool(), padding, distance)

        distance = distance.min(dim=2)[0]
        minimum_distance_index = distance.min(dim=1)[0].argsort()
        minimum_distance = distance.min(dim=1)[0]

    else:
        distance = termination_point.view(1, -1, 3) - initial_point.view(-1, 1, 3)
        distance = torch.linalg.norm(distance, dim=-1)
        minimum_distance_index = distance.min(dim=1)[0].argsort()
        minimum_distance = distance.min(dim=1)[0]

    return minimum_distance, minimum_distance_index


def reverse_bool(bool_list):
    out = []
    for i in bool_list: out.append(not i)
    return out


def load_wt_mut_pdb_pair(x, y, error,
                         wt_path, mut_path,
                         pdb_name, label,
                         ddg,
                         finished_num,
                         lock=None, lock_file=None, lock_file_csv=None):
    if label == '':

        try:
            data_wt = parse_pdb(wt_path, pdb_name)
            data_mut = parse_pdb(mut_path, pdb_name)
            batch = choice_residue(data_wt, data_mut, pdb_name, lock_file_csv)
            batch['pdb_name'] = pdb_name
            if len(batch['mutation_mask']) == 0: ddg = '0.00'
            if str(batch['wt']) == str(batch['mut']): ddg = '0.00'

            if lock != None:
                lock.acquire()
                x.append(batch)
                y.append(ddg)
                lock.release()
            else:
                x.append(batch)
                y.append(ddg)
                return x, y, error
        except IndexError:
            if lock != None:
                lock_file.acquire()
                error.append(label + pdb_name)
                lock_file.release()
            else:
                error.append(label + pdb_name)
                return x, y, error

        print('finished:', finished_num)

    else:
        try:
            data_wt = parse_pdb(mut_path, pdb_name)
            data_mut = parse_pdb(wt_path, pdb_name)
            batch = choice_residue(data_wt, data_mut, pdb_name)
            batch['pdb_name'] = label + pdb_name
            ddg = y_reverse(ddg)
            if len(batch['mutation_mask']) == 0: ddg = '0.00'
            if str(batch['wt']) == str(batch['mut']): ddg = '0.00'

            if lock != None:
                lock.acquire()
                x.append(batch)
                y.append(ddg)
                lock.release()
            else:
                x.append(batch)
                y.append(ddg)
                return x, y, error
        except IndexError:
            if lock != None:
                lock_file.acquire()
                error.append(label + pdb_name)
                lock_file.release()
            else:
                error.append(label + pdb_name)
                return x, y, error

        print('finished:', finished_num)
