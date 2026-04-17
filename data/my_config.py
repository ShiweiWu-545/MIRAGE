from easydict import EasyDict as edict


config = edict({
    'debug': {},
    'feature': {
        'Atom': ['All non-hydrogen atoms', 14],
        'Atom_list': [['All non-hydrogen atoms', 14],
                      ['All atoms', 24],
                      ['Trunk atoms', 4],
                      ['All trunk atoms', 7],
                      ''],

        'Spatial_distance': 'CB',
        'Spatial_distance_list': ['CA',
                                  'CB'],

        'surface': 64,
        'nearby_residues': 128,
        'DG_surface_num': 128,

        'choice_residue': 'Mutant nearby residues',
        'choice_residue_list': ['Mutant nearby residues',
                                'Junction surface residues_14',
                                'Junction surface residues_CA',
                                'The intersection of A and B_14',
                                'The intersection of A and B_CA'],
        'DGchoice_residue': 'rASA surface',
        'DGchoice_residue_list': [
                                'Junction surface residues',
                                'rASA surface'],

        'contact_area': [False, 'Residue level'],
        'contact_area_list': [[False, 'Residue level'],
                              [True, 'Compound level (MLP)'],
                              [True, 'Compound level (Manual design)']],

        'datasets_extend_reverse': True,
        'Recycling': [False, 3],
        'Dropout': [False, 0.5],
        'relpos_sparse': False,
        'Side_chain_geometry': True,




        'LayerNorm': False,
        'weight': {'SENet': False,
                   'Attention': True},
        'diff_Attention': True,
        'pass0': True,
        'pass1': True,
        'pass2': False,
        'pass3': False,
        'Gate': {'gate0': False,
                 'gate1': False,
                 'gate2': False,
                 'gate3': False,
                 'gate4': False,
                 'gate5': False,
                 },
        'freeze_parameter': False,
        'res_prottrans': True,
        'CrossAttention': False,


        'chains_to_two': False,
        'Residue_selection_based_on_core': {'label': True,
                                            'dynamic_mut': True,
                                            'Residues_outside_the_regulatory_capacity': True,
                                            'max_rnum': 20,
                                            'mut_lr': 0.5,
                                            'core_ratio': 0.75},

        },


    'model': {
        'mission': 'DG',
        'pool': False,
        'save_model': True,
        'model': '../data/model.pt',
        'my_model': {
            'path': '../data/mymodel/',

            'model_best': '../data/mymodel/mymodel.pt',
            'train_model': '../data/mymodel/train_model/',
            'train_model_best': '../data/mymodel/'},
        'device': 'cuda',
        'node_feat_dim': 128,
        'pair_sequence_feat_dim': 64,
        'max_relpos': 32,

        'geomattn': {
            'num_layers': 3,
            'spatial_attn_mode': 'CB'}},


    'train': {
            'loss_weights': {
                'ddG': 1.0},

            'max_iters': 200,
            'val_freq': 10,
            'val_num': 2,
            'batch_size': 32,
            'seed': 2021,
            'max_grad_norm': 50.0,

            'optimizer': {
                'type': 'adam',
                'lr': 0.00005,
                'weight_decay': 0,
                'beta1': 0.9,
                'beta2': 0.999},

            'scheduler': {
                'type': 'plateau',
                'factor': 0.5,
                'patience': 10,
                'min_lr': 1e-06}},


    'datasets': {
        'total': '../datasets/ /total_datasets.pt',
        'train': {
            'dataset_del_S1131': {
                'CSV': '../datasets/skempi_v2_del_S1131.csv',
                'PT': '../datasets/ /dataset-S1131.pt'},
            'dataset_del_M1707': {
                'CSV': '../datasets/skempi_v2_del_M1707.csv',
                'PT': '../datasets/ /dataset-M1707.pt'},
            'no_AbAg': {'CSV': '',
                        'PT': '../datasets/ /no_AbAg.pt'}},

        'val': {
            'dataset_path': ''},
        'test': {
            'S1131': '../datasets/ /S1131.pt',
            'AF3S1131': '../datasets/ /AF3S1131_20_0.5_0.75.pt',
            'M1707': '../datasets/ /M1707.pt_20_0.5_0.75_False.pt',
            'S645': '../datasets/ /S645_20_0.5_0.75_targetC.pt',
            'S645_no27': '../datasets/ /S645_no27.pt',
            'S4169': '../datasets/ /S4169_128_20_0.5_0.75.pt',
            'S8338': '../datasets/ /S8338_True_False_True_20_0.5_0.75.pt',
            'mixed': '../datasets/ /F_20_2_75_mixed.pt',
            'predict': '../datasets/ /predict_data.pt',
            'C3684': '../datasets/ /C3684.pt',
            'HIV_78': '../datasets/ /HIV_78_ddg.pt',
            'AB_228': '../datasets/ /AB_228.pt'},
        'contact_area': '../datasets/contact_area.surface',
        'region_path': '/data_4T/dataset/AF3_mutDataset/S2_AF3_relax_pdbs',
        },
    'datasets_DG': {
        'readyPDB': '/data_4T/dataset/readyPDB',
        'DGdatasets': '../datasets/PPB-Affinity.xlsx',
        'Skempi2': '/data_4T/dataset/Skempi2_6193/Skempi2_ddg_useful',

    },


    'observer': {
        'parameter': {
            'parameter_path': '../procedure_parameter',
            'parameter_validation_path': '../procedure_parameter/validation_curve/parameter'},
        'train_losses_epoch': True,
        'train_Rp': True,
        'validation_Rp': False,
        'validation_losses_epoch': False,
        'test_Rp': True,
        'test_losses_epoch': True}})
