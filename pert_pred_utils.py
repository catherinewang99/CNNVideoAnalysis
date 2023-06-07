'''
Adaptedfrom choice_pred_utils.py.
'''


import os, sys, time, argparse, pickle, math
sys.path.append('NeuralPopulationAnalysisUtils/Code/Python')
from neupop import Session as Session
from neupop import PopulationAnalysis as PA
from neupop import MultiSessionAnalysis as MS
from neupop import decoders as ld
from neupop import session_parsers as parsers

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from alm_datasets import ALMDataset

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from collections import defaultdict

cat = np.concatenate

n_trial_types_list = range(2)

def load_sess_by_task_dict():

    save_path = os.path.join('pert_pred_utils_results','load_sess_by_task_dict')
    os.makedirs(save_path, exist_ok=True)

    if os.path.isfile(os.path.join(save_path, 'sess_by_task_dict.pkl')):
        with open(os.path.join(save_path, 'sess_by_task_dict.pkl'), 'rb') as f:
            sess_by_task_dict = pickle.load(f)

    else:
        # Copied from NeuronalData5/cd_vel_pred_switch_linear4_full_cv_main.py
        filehandler = open('sorted_filenames.obj', 'rb')
        filenames = pickle.load(filehandler)


        filehandler = open('four_laser_filenames.obj', 'rb')
        four_laser_filenames = pickle.load(filehandler)

        mice_by_task_dict = {}

        mice_by_task_dict['original'] = [40, 44, 46, 48, 52]
        #mice_by_task_dict['full_reverse'] = [79, 80, 81, 83, 84, 85, 88, 89, 90, 94]
        #mice_by_task_dict['rule_reverse'] = [61, 66, 69, 71, 74, 78]
        #mice_by_task_dict['pole_reverse'] = [91, 92, 96, 98, 99, 100]

        sess_by_task_dict = {}

        for k in mice_by_task_dict.keys():
            sess_by_task_dict[k] = []
            sess_by_task_dict[k+'_kilosort'] = []
            sess_by_task_dict[k+'_manual'] = []

            sess_by_task_dict[k+'_guang_one_laser'] = []
            sess_by_task_dict[k+'_guang_one_laser_sel'] = []

            sess_by_task_dict[k+'_four_laser'] = []
            sess_by_task_dict[k+'_one_laser'] = []

            sess_by_task_dict[k+'_kilosort_four_laser'] = []
            sess_by_task_dict[k+'_kilosort_one_laser'] = []

            sess_by_task_dict[k+'_manual_four_laser'] = []
            sess_by_task_dict[k+'_manual_one_laser'] = []


        '''
        Guang's selective and one-laser session.
        '''
        import scipy.io as spio
        mat_dict = spio.loadmat('Si_goodsession_laser.mat', squeeze_me=True)
        sess_names = np.array(mat_dict['SessionID'])

        sel_sess_idxs = np.array(mat_dict['goodsession']-1).astype(int)
        sel_sess_mask = np.zeros(len(sess_names)).astype(bool)
        sel_sess_mask[sel_sess_idxs] = True

        one_laser_mask = (np.array(mat_dict['spot_allsession']).astype(int) == 1)

        # 1. standard, 2. reversed contingency, 3. fully reversed, 4. reversed side. length same as SessionID.
        guang_task_idxs = np.array(mat_dict['task_allsession']).astype(int)
        task_idx_to_str_dict = {}
        task_idx_to_str_dict[1] = 'original'
        task_idx_to_str_dict[2] = 'rule_reverse'
        task_idx_to_str_dict[3] = 'full_reverse'
        task_idx_to_str_dict[4] = 'pole_reverse'


        guang_one_laser_sess_names = sess_names[one_laser_mask]
        guang_one_laser_sel_sess_names = sess_names[one_laser_mask*sel_sess_mask]



        for sess_iter, filename in enumerate(filenames):
            sess_idx = sess_iter + 1
            cur_sess_name = filename.split('\\')[1][:-4].replace('kilosort','') # BAYLORGC100_2020_01_21

            if cur_sess_name in guang_one_laser_sess_names:
                cur_task_idx = guang_task_idxs[sess_names==cur_sess_name][0] # [0] inverts array to int
                k = task_idx_to_str_dict[cur_task_idx]
                sess_by_task_dict[k+'_guang_one_laser'].append(sess_idx)

            if cur_sess_name in guang_one_laser_sel_sess_names:
                cur_task_idx = guang_task_idxs[sess_names==cur_sess_name][0]  # [0] inverts array to int
                k = task_idx_to_str_dict[cur_task_idx]
                k = task_idx_to_str_dict[cur_task_idx]
                sess_by_task_dict[k+'_guang_one_laser_sel'].append(sess_idx)


        for sess_iter, filename in enumerate(filenames):
            sess_idx = sess_iter + 1
            
            mice_number = filename.split('\\')[1][:-4].split('_')[0][8:] # e.g. 100kilosort
            
            if 'kilosort' in mice_number:
                mice_number = mice_number[:-8]
                kilosort = True
            else:
                kilosort = False

            if filename in four_laser_filenames:
                four_laser = True
            else:
                four_laser = False

            mice_number = int(mice_number)

            for k, v in mice_by_task_dict.items():
                if mice_number in v:
                    sess_by_task_dict[k].append(sess_idx)
                    if kilosort:
                        sess_by_task_dict[k+'_kilosort'].append(sess_idx)
                    else:
                        sess_by_task_dict[k+'_manual'].append(sess_idx)

                    if four_laser:
                        sess_by_task_dict[k+'_four_laser'].append(sess_idx)
                        if kilosort:
                            sess_by_task_dict[k+'_kilosort_four_laser'].append(sess_idx)
                        else:
                            sess_by_task_dict[k+'_manual_four_laser'].append(sess_idx)

                    else:
                        sess_by_task_dict[k+'_one_laser'].append(sess_idx)
                        if kilosort:
                            sess_by_task_dict[k+'_kilosort_one_laser'].append(sess_idx)
                        else:
                            sess_by_task_dict[k+'_manual_one_laser'].append(sess_idx)

                    break


        for k in sess_by_task_dict.keys():
            sess_by_task_dict[k].sort()

        # Save
        with open(os.path.join(save_path, 'sess_by_task_dict.pkl'), 'wb') as f:
            pickle.dump(sess_by_task_dict, f)


    return sess_by_task_dict


def load_no_stim_i_good_trials(choice_type, prep_root_path, f1, f2):

    data_prefix = 'Trial_Ordered_Long_Prep_'
    # filename: e.g. NeuronalData/BAYLORGC95kilosort_2019_12_19.mat
    filename = os.path.join('NeuronalData', '_'.join([f1,f2])+'.mat')
    if not os.path.exists(filename):
        filename = os.path.join('NeuronalData', '_'.join([f1+'kilosort',f2])+'.mat')


    target_root_path = os.path.join('pert_pred_utils_results','choice_type_{}'.format(choice_type),'load_no_stim_i_good_trials')
    target_save_path = os.path.join(target_root_path, data_prefix + 'NeuronalData', f1, f2)
    if 'kilosort' in filename.split('\\')[1]:
        target_save_path = os.path.join(target_root_path, data_prefix + 'NeuronalData', f1+'kilosort', f2)

    os.makedirs(target_save_path, exist_ok=True)

    i_good_trials_save_name = 'no_stim_i_good_trials.npy'

    if os.path.isfile(os.path.join(target_save_path, i_good_trials_save_name)):

        no_stim_i_good_trials = np.load(os.path.join(target_save_path, i_good_trials_save_name))

    else:

        prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1,f2]))
        if not os.path.exists(prep_save_path):
            prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1+'kilosort',f2]))


        assert os.path.isfile(os.path.join(prep_save_path, 'culled_session.obj'))

        filehandler = open(os.path.join(prep_save_path, 'culled_session.obj'), 'rb')
        sess = pickle.load(filehandler)
        i_good_trials = np.array(sess.i_good_trials.copy())


        # Load no_stim_mask.
        no_stim_mask_path = os.path.join(prep_save_path, 'no_stim', 'stim_mask.npy')
        no_stim_mask = np.load(no_stim_mask_path)

        no_stim_i_good_trials_old = i_good_trials[no_stim_mask]

        # filename: e.g. NeuronalData/BAYLORGC95kilosort_2019_12_19.mat
        filename = os.path.join('NeuronalData', '_'.join([f1,f2])+'.mat')
        if not os.path.exists(filename):
            filename = os.path.join('NeuronalData', '_'.join([f1+'kilosort',f2])+'.mat')

        neural_unit_location = sess.neural_unit_location.copy()
        bin_centers = np.load(os.path.join(prep_save_path, 'bin_centers.npy'))


        no_stim_train_set = ALMDataset(filename, root_path=prep_root_path, data_prefix=data_prefix, stim_type='no_stim', data_type='train', neural_unit_location=neural_unit_location, loc_name='both', bin_min=0, bin_max=len(bin_centers)-1)
        no_stim_test_set = ALMDataset(filename, root_path=prep_root_path, data_prefix=data_prefix, stim_type='no_stim', data_type='test', neural_unit_location=neural_unit_location, loc_name='both', bin_min=0, bin_max=len(bin_centers)-1)
        
        if choice_type == 'instructed':
            no_stim_train_trial_type_labels = no_stim_train_set.trial_type_labels
            no_stim_test_trial_type_labels = no_stim_test_set.trial_type_labels
            no_stim_choice_labels = np.concatenate([no_stim_train_trial_type_labels, no_stim_test_trial_type_labels], 0)
        else:
            no_stim_train_labels = no_stim_train_set.labels
            no_stim_test_labels = no_stim_test_set.labels
            no_stim_choice_labels = np.concatenate([no_stim_train_labels, no_stim_test_labels], 0)



        no_stim_i_good_trials = np.zeros(2, dtype=object)
        for i in n_trial_types_list:
            no_stim_i_good_trials[i] = no_stim_i_good_trials_old[(no_stim_choice_labels==i)]

        np.save(os.path.join(target_save_path, i_good_trials_save_name), no_stim_i_good_trials)

    return no_stim_i_good_trials




def load_no_stim_i_good_trials_v2(inst_trial_type, prep_root_path, f1, f2):
    '''
    The difference from load_no_stim_i_good_trials:
    We condition no_stim_i_good_trials on inst_trial_type and also the index i for it
    refers to whether the trial is an error trial.
    '''

    data_prefix = 'All_Pert_Normal_Trial_Ordered_Long_Prep_'
    # filename: e.g. NeuronalData/BAYLORGC95kilosort_2019_12_19.mat
    filename = os.path.join('NeuronalData', '_'.join([f1,f2])+'.mat')
    if not os.path.exists(filename):
        filename = os.path.join('NeuronalData', '_'.join([f1+'kilosort',f2])+'.mat')


    target_root_path = os.path.join('pert_pred_utils_results','inst_trial_type_{}'.format(inst_trial_type),'load_no_stim_i_good_trials_v2')
    target_save_path = os.path.join(target_root_path, data_prefix + 'NeuronalData', f1, f2)
    if 'kilosort' in filename.split('\\')[1]:
        target_save_path = os.path.join(target_root_path, data_prefix + 'NeuronalData', f1+'kilosort', f2)

    os.makedirs(target_save_path, exist_ok=True)

    i_good_trials_save_name = 'no_stim_i_good_trials.npy'

    if os.path.isfile(os.path.join(target_save_path, i_good_trials_save_name)):
        no_stim_i_good_trials = np.load(os.path.join(target_save_path, i_good_trials_save_name))

    else:
        print('computing no_stim_i_good_trials...')
        prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1,f2]))
        if not os.path.exists(prep_save_path):
            prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1+'kilosort',f2]))


        assert os.path.isfile(os.path.join(prep_save_path, 'culled_session.obj'))

        filehandler = open(os.path.join(prep_save_path, 'culled_session.obj'), 'rb')
        sess = pickle.load(filehandler)
        i_good_trials = np.array(sess.i_good_trials.copy())


        # Load no_stim_mask.
        no_stim_mask_path = os.path.join(prep_save_path, 'no_stim', 'stim_mask.npy')
        no_stim_mask = np.load(no_stim_mask_path)

        no_stim_i_good_trials_old = i_good_trials[no_stim_mask]

        # filename: e.g. NeuronalData/BAYLORGC95kilosort_2019_12_19.mat
        filename = os.path.join('NeuronalData', '_'.join([f1,f2])+'.mat')
        if not os.path.exists(filename):
            filename = os.path.join('NeuronalData', '_'.join([f1+'kilosort',f2])+'.mat')

        neural_unit_location = sess.neural_unit_location.copy()
        bin_centers = np.load(os.path.join(prep_save_path, 'bin_centers.npy'))


        no_stim_train_set = ALMDataset(filename, root_path=prep_root_path, data_prefix=data_prefix, stim_type='no_stim', data_type='train', neural_unit_location=neural_unit_location, loc_name='both', bin_min=0, bin_max=len(bin_centers)-1)
        no_stim_test_set = ALMDataset(filename, root_path=prep_root_path, data_prefix=data_prefix, stim_type='no_stim', data_type='test', neural_unit_location=neural_unit_location, loc_name='both', bin_min=0, bin_max=len(bin_centers)-1)
        
        no_stim_train_trial_type_labels = no_stim_train_set.trial_type_labels
        no_stim_test_trial_type_labels = no_stim_test_set.trial_type_labels
        no_stim_trial_type_labels = np.concatenate([no_stim_train_trial_type_labels, no_stim_test_trial_type_labels], 0)

        no_stim_train_labels = no_stim_train_set.labels
        no_stim_test_labels = no_stim_test_set.labels
        no_stim_labels = np.concatenate([no_stim_train_labels, no_stim_test_labels], 0)

        no_stim_error_labels = (no_stim_trial_type_labels != no_stim_labels).astype(int)

        no_stim_i_good_trials = np.zeros(2, dtype=object)
        for i in n_trial_types_list:
            no_stim_i_good_trials[i] = no_stim_i_good_trials_old[(no_stim_trial_type_labels==inst_trial_type)*(no_stim_error_labels==i)]

        np.save(os.path.join(target_save_path, i_good_trials_save_name), no_stim_i_good_trials)

    return no_stim_i_good_trials





def load_no_stim_i_good_trials_v3(prep_root_path, f1, f2):
    '''
    Differences from load_no_stim_i_good_trials_v2:
    1. We don't condition on inst_trial_type.
    '''
    data_prefix = 'All_Pert_Normal_Trial_Ordered_Long_Prep_'
    # filename: e.g. NeuronalData/BAYLORGC95kilosort_2019_12_19.mat
    filename = os.path.join('NeuronalData', '_'.join([f1,f2])+'.mat')
    if not os.path.exists(filename):
        filename = os.path.join('NeuronalData', '_'.join([f1+'kilosort',f2])+'.mat')


    target_root_path = os.path.join('pert_pred_utils_results', 'load_no_stim_i_good_trials_v3')
    target_save_path = os.path.join(target_root_path, data_prefix + 'NeuronalData', f1, f2)
    if 'kilosort' in filename.split('\\')[1]:
        target_save_path = os.path.join(target_root_path, data_prefix + 'NeuronalData', f1+'kilosort', f2)

    os.makedirs(target_save_path, exist_ok=True)

    i_good_trials_save_name = 'no_stim_i_good_trials.npy'

    if os.path.isfile(os.path.join(target_save_path, i_good_trials_save_name)):
        no_stim_i_good_trials = np.load(os.path.join(target_save_path, i_good_trials_save_name))

    else:
        print('computing no_stim_i_good_trials...')
        prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1,f2]))
        if not os.path.exists(prep_save_path):
            prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1+'kilosort',f2]))


        assert os.path.isfile(os.path.join(prep_save_path, 'culled_session.obj'))

        filehandler = open(os.path.join(prep_save_path, 'culled_session.obj'), 'rb')
        sess = pickle.load(filehandler)
        i_good_trials = np.array(sess.i_good_trials.copy())


        # Load no_stim_mask.
        no_stim_mask_path = os.path.join(prep_save_path, 'no_stim', 'stim_mask.npy')
        no_stim_mask = np.load(no_stim_mask_path)

        no_stim_i_good_trials_old = i_good_trials[no_stim_mask]

        # filename: e.g. NeuronalData/BAYLORGC95kilosort_2019_12_19.mat
        filename = os.path.join('NeuronalData', '_'.join([f1,f2])+'.mat')
        if not os.path.exists(filename):
            filename = os.path.join('NeuronalData', '_'.join([f1+'kilosort',f2])+'.mat')

        neural_unit_location = sess.neural_unit_location.copy()
        bin_centers = np.load(os.path.join(prep_save_path, 'bin_centers.npy'))


        no_stim_train_set = ALMDataset(filename, root_path=prep_root_path, data_prefix=data_prefix, stim_type='no_stim', data_type='train', neural_unit_location=neural_unit_location, loc_name='both', bin_min=0, bin_max=len(bin_centers)-1)
        no_stim_test_set = ALMDataset(filename, root_path=prep_root_path, data_prefix=data_prefix, stim_type='no_stim', data_type='test', neural_unit_location=neural_unit_location, loc_name='both', bin_min=0, bin_max=len(bin_centers)-1)
        
        no_stim_train_trial_type_labels = no_stim_train_set.trial_type_labels
        no_stim_test_trial_type_labels = no_stim_test_set.trial_type_labels
        no_stim_trial_type_labels = np.concatenate([no_stim_train_trial_type_labels, no_stim_test_trial_type_labels], 0)

        no_stim_train_labels = no_stim_train_set.labels
        no_stim_test_labels = no_stim_test_set.labels
        no_stim_labels = np.concatenate([no_stim_train_labels, no_stim_test_labels], 0)

        no_stim_error_labels = (no_stim_trial_type_labels != no_stim_labels).astype(int)

        no_stim_i_good_trials = np.zeros(2, dtype=object)
        for i in n_trial_types_list:
            no_stim_i_good_trials[i] = no_stim_i_good_trials_old[(no_stim_error_labels==i)]

        np.save(os.path.join(target_save_path, i_good_trials_save_name), no_stim_i_good_trials)

    return no_stim_i_good_trials





def load_no_stim_i_good_trials_for_pert_pred(inst_trial_type, prep_root_path, f1, f2):
    '''
    The difference from load_no_stim_i_good_trials_v2:
    We do not separate error and correct trials.

    '''

    data_prefix = 'Trial_Ordered_Long_Prep_'
    # filename: e.g. NeuronalData/BAYLORGC95kilosort_2019_12_19.mat
    filename = os.path.join('NeuronalData', '_'.join([f1,f2])+'.mat')
    if not os.path.exists(filename):
        filename = os.path.join('NeuronalData', '_'.join([f1+'kilosort',f2])+'.mat')


    target_root_path = os.path.join('pert_pred_utils_results','inst_trial_type_{}'.format(inst_trial_type),'load_no_stim_i_good_trials_for_pert_pred')
    target_save_path = os.path.join(target_root_path, data_prefix + 'NeuronalData', f1, f2)
    if 'kilosort' in filename.split('\\')[1]:
        target_save_path = os.path.join(target_root_path, data_prefix + 'NeuronalData', f1+'kilosort', f2)

    os.makedirs(target_save_path, exist_ok=True)

    i_good_trials_save_name = 'no_stim_i_good_trials.npy'

    if os.path.isfile(os.path.join(target_save_path, i_good_trials_save_name)):
        no_stim_i_good_trials = np.load(os.path.join(target_save_path, i_good_trials_save_name))

    else:
        print('computing no_stim_i_good_trials...')
        prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1,f2]))
        if not os.path.exists(prep_save_path):
            prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1+'kilosort',f2]))


        assert os.path.isfile(os.path.join(prep_save_path, 'culled_session.obj'))

        filehandler = open(os.path.join(prep_save_path, 'culled_session.obj'), 'rb')
        sess = pickle.load(filehandler)
        i_good_trials = np.array(sess.i_good_trials.copy())


        # Load no_stim_mask.
        no_stim_mask_path = os.path.join(prep_save_path, 'no_stim', 'stim_mask.npy')
        no_stim_mask = np.load(no_stim_mask_path)

        no_stim_i_good_trials_old = i_good_trials[no_stim_mask]

        # filename: e.g. NeuronalData/BAYLORGC95kilosort_2019_12_19.mat
        filename = os.path.join('NeuronalData', '_'.join([f1,f2])+'.mat')
        if not os.path.exists(filename):
            filename = os.path.join('NeuronalData', '_'.join([f1+'kilosort',f2])+'.mat')

        neural_unit_location = sess.neural_unit_location.copy()
        bin_centers = np.load(os.path.join(prep_save_path, 'bin_centers.npy'))


        no_stim_train_set = ALMDataset(filename, root_path=prep_root_path, data_prefix=data_prefix, stim_type='no_stim', data_type='train', neural_unit_location=neural_unit_location, loc_name='both', bin_min=0, bin_max=len(bin_centers)-1)
        no_stim_test_set = ALMDataset(filename, root_path=prep_root_path, data_prefix=data_prefix, stim_type='no_stim', data_type='test', neural_unit_location=neural_unit_location, loc_name='both', bin_min=0, bin_max=len(bin_centers)-1)
        
        no_stim_train_trial_type_labels = no_stim_train_set.trial_type_labels
        no_stim_test_trial_type_labels = no_stim_test_set.trial_type_labels
        no_stim_trial_type_labels = np.concatenate([no_stim_train_trial_type_labels, no_stim_test_trial_type_labels], 0)

        no_stim_train_labels = no_stim_train_set.labels
        no_stim_test_labels = no_stim_test_set.labels
        no_stim_labels = np.concatenate([no_stim_train_labels, no_stim_test_labels], 0)

        no_stim_error_labels = (no_stim_trial_type_labels != no_stim_labels).astype(int)

        no_stim_i_good_trials = no_stim_i_good_trials_old[(no_stim_trial_type_labels==inst_trial_type)]

        np.save(os.path.join(target_save_path, i_good_trials_save_name), no_stim_i_good_trials)

    return no_stim_i_good_trials




def load_stim_i_good_trials_for_pert_pred(pert_type, inst_trial_type, prep_root_path, f1, f2):
    '''
    The difference from load_stim_i_good_trials_v2:
    We do not separate error and correct trials.

    '''

    data_prefix = 'Trial_Ordered_Long_Prep_'
    # filename: e.g. NeuronalData/BAYLORGC95kilosort_2019_12_19.mat
    filename = os.path.join('NeuronalData', '_'.join([f1,f2])+'.mat')
    if not os.path.exists(filename):
        filename = os.path.join('NeuronalData', '_'.join([f1+'kilosort',f2])+'.mat')


    target_root_path = os.path.join('pert_pred_utils_results', 'pert_type_{}'.format(pert_type), \
        'inst_trial_type_{}'.format(inst_trial_type),'load_stim_i_good_trials_for_pert_pred')
    target_save_path = os.path.join(target_root_path, data_prefix + 'NeuronalData', f1, f2)
    if 'kilosort' in filename.split('\\')[1]:
        target_save_path = os.path.join(target_root_path, data_prefix + 'NeuronalData', f1+'kilosort', f2)

    os.makedirs(target_save_path, exist_ok=True)

    i_good_trials_save_name = 'stim_i_good_trials.npy'

    if os.path.isfile(os.path.join(target_save_path, i_good_trials_save_name)):
        stim_i_good_trials = np.load(os.path.join(target_save_path, i_good_trials_save_name))

    else:
        print('computing stim_i_good_trials...')
        prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1,f2]))
        if not os.path.exists(prep_save_path):
            prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1+'kilosort',f2]))


        assert os.path.isfile(os.path.join(prep_save_path, 'culled_session.obj'))

        filehandler = open(os.path.join(prep_save_path, 'culled_session.obj'), 'rb')
        sess = pickle.load(filehandler)
        i_good_trials = np.array(sess.i_good_trials.copy())


        # Load stim_mask.
        stim_mask_path = os.path.join(prep_save_path, pert_type, 'stim_mask.npy')
        stim_mask = np.load(stim_mask_path)

        stim_i_good_trials_old = i_good_trials[stim_mask]

        # filename: e.g. NeuronalData/BAYLORGC95kilosort_2019_12_19.mat
        filename = os.path.join('NeuronalData', '_'.join([f1,f2])+'.mat')
        if not os.path.exists(filename):
            filename = os.path.join('NeuronalData', '_'.join([f1+'kilosort',f2])+'.mat')

        neural_unit_location = sess.neural_unit_location.copy()
        bin_centers = np.load(os.path.join(prep_save_path, 'bin_centers.npy'))


        stim_train_set = ALMDataset(filename, root_path=prep_root_path, data_prefix=data_prefix, stim_type=pert_type, data_type='train', neural_unit_location=neural_unit_location, loc_name='both', bin_min=0, bin_max=len(bin_centers)-1)
        stim_test_set = ALMDataset(filename, root_path=prep_root_path, data_prefix=data_prefix, stim_type=pert_type, data_type='test', neural_unit_location=neural_unit_location, loc_name='both', bin_min=0, bin_max=len(bin_centers)-1)
        
        stim_train_trial_type_labels = stim_train_set.trial_type_labels
        stim_test_trial_type_labels = stim_test_set.trial_type_labels
        stim_trial_type_labels = np.concatenate([stim_train_trial_type_labels, stim_test_trial_type_labels], 0)

        stim_train_labels = stim_train_set.labels
        stim_test_labels = stim_test_set.labels
        stim_labels = np.concatenate([stim_train_labels, stim_test_labels], 0)

        stim_error_labels = (stim_trial_type_labels != stim_labels).astype(int)

        stim_i_good_trials = stim_i_good_trials_old[(stim_trial_type_labels==inst_trial_type)]

        np.save(os.path.join(target_save_path, i_good_trials_save_name), stim_i_good_trials)

    return stim_i_good_trials








def load_bin_centers(prep_root_path, f1,f2):

    data_prefix = 'Trial_Ordered_Long_Prep_states_2'
    # filename: e.g. NeuronalData/BAYLORGC95kilosort_2019_12_19.mat
    filename = os.path.join('NeuronalData', '_'.join([f1,f2])+'.mat')
    if not os.path.exists(filename):
        filename = os.path.join('NeuronalData', '_'.join([f1+'kilosort',f2])+'.mat')


    target_root_path = os.path.join('pert_pred_utils_results','load_bin_centers')
    target_save_path = os.path.join(target_root_path, data_prefix + 'NeuronalData', f1, f2)
    if 'kilosort' in filename.split('\\')[1]:
        target_save_path = os.path.join(target_root_path, data_prefix + 'NeuronalData', f1+'kilosort', f2)

    os.makedirs(target_save_path, exist_ok=True)

    bin_centers_save_name = 'bin_centers.npy'

    if os.path.isfile(os.path.join(target_save_path, bin_centers_save_name)):

        bin_centers = np.load(os.path.join(target_save_path, bin_centers_save_name))

    else:

        prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1,f2]))
        if not os.path.exists(prep_save_path):
            prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1+'kilosort',f2]))


        bin_centers = np.load(os.path.join(prep_save_path, 'bin_centers.npy'))

        np.save(os.path.join(target_save_path, bin_centers_save_name), bin_centers)

    return bin_centers



def map_f3_to_trial_idx(f1,f2,f3):
    '''
    The mismatchs in GC50_2018_12_13, GC83_2019_08_15, GC84_2019_07_24 are taken care
    of at the level of split_frames. That is, we map those avi files to the correct trial idxs
    when we split frames for them.
    '''

    raw_avi_trial_idx = int(f3.split('-')[3])
    if f1 == 'BAYLORGC86' and f2  == '2019_07_29':
        '''
        Ignore the very last trial.
        '''
        if raw_avi_trial_idx == 486:
            return None
        else:
            return raw_avi_trial_idx

    elif f1 == 'BAYLORGC90' and f2  == '2019_11_21':
        '''
        Only use the first 588 trials.
        '''
        if raw_avi_trial_idx > 588:
            return None
        else:
            return raw_avi_trial_idx

    else:
        return raw_avi_trial_idx






def compute_selected_frames_and_pred_timesteps(configs, pred_times, begin_frame, end_frame, skip_frame, go_cue_time,\
    return_selected_frame_times=False):
    # selected_frames include both begin and end_frame.
    n_frames = int(np.rint((end_frame - begin_frame)/skip_frame)) + 1
    selected_frames = (begin_frame + skip_frame*np.arange(n_frames))


    # Determine pred_timesteps
    selected_frame_times = selected_frames/200.0 - go_cue_time

    pred_timesteps = []
    for pred_time in pred_times:
        pred_timesteps.append(np.argmin(np.abs(selected_frame_times - pred_time)))

    pred_timesteps = np.array(pred_timesteps)


    # To deal with lrcn_neureg_bn_static2 etc.
    if '_input_channel' in configs.keys() and configs['_input_channel'] != 1:
        n_input_frames = configs['_input_channel']

        n_pre_frames = (n_input_frames-1)//2
        n_post_frames = (n_input_frames-1)//2

        if (n_input_frames-1)%2 != 0:
            n_pre_frames += 1

        # out of range for early times.
        if (pred_timesteps[0] - n_pre_frames) < 0:
            new_begin_frame = begin_frame - skip_frame*n_pre_frames
        else:
            new_begin_frame = begin_frame

        # out of range for late times.
        if (pred_timesteps[-1] + n_post_frames) >= n_frames:
            new_end_frame = end_frame + skip_frame*n_post_frames
        else:
            new_end_frame = end_frame


        n_frames = int(np.rint((new_end_frame - new_begin_frame)/skip_frame)) + 1
        selected_frames = (new_begin_frame + skip_frame*np.arange(n_frames))

        # Determine pred_timesteps
        selected_frame_times = selected_frames/200.0 - go_cue_time

        pred_timesteps = []
        for pred_time in pred_times:
            pred_timesteps.append(np.argmin(np.abs(selected_frame_times - pred_time)))

        pred_timesteps = np.array(pred_timesteps)


    if return_selected_frame_times:
        return selected_frames, pred_timesteps, selected_frame_times

    else:
        return selected_frames, pred_timesteps


def load_no_stim_i_good_trials_for_state_pred(inst_trial_type, prep_root_path, f1, f2):
    '''
    The difference from load_stim_i_good_trials_for_pert_pred:
    We are loading state predictions instead of perturbation predictions

    '''

    data_prefix = 'Trial_Ordered_Long_Prep_'
    # filename: e.g. NeuronalData/BAYLORGC95kilosort_2019_12_19.mat
    filename = os.path.join('NeuronalData', '_'.join([f1,f2])+'.mat')
    if not os.path.exists(filename):
        filename = os.path.join('NeuronalData', '_'.join([f1+'kilosort',f2])+'.mat')


    target_root_path = os.path.join('pert_pred_utils_results','inst_trial_type_{}'.format(inst_trial_type),'load_no_stim_i_good_trials_for_pert_pred')
    target_save_path = os.path.join(target_root_path, data_prefix + 'NeuronalData', f1, f2)
    if 'kilosort' in filename.split('\\')[1]:
        target_save_path = os.path.join(target_root_path, data_prefix + 'NeuronalData', f1+'kilosort', f2)

    os.makedirs(target_save_path, exist_ok=True)

    i_good_trials_save_name = 'no_stim_i_good_trials.npy'

    if os.path.isfile(os.path.join(target_save_path, i_good_trials_save_name)):
        no_stim_i_good_trials = np.load(os.path.join(target_save_path, i_good_trials_save_name))

    else:
        print('computing no_stim_i_good_trials...')
        prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1,f2]))
        if not os.path.exists(prep_save_path):
            prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1+'kilosort',f2]))


        assert os.path.isfile(os.path.join(prep_save_path, 'culled_session.obj'))

        filehandler = open(os.path.join(prep_save_path, 'culled_session.obj'), 'rb')
        sess = pickle.load(filehandler)
        i_good_trials = np.array(sess.i_good_trials.copy())


        # Load no_stim_mask.
        no_stim_mask_path = os.path.join(prep_save_path, 'no_stim', 'stim_mask.npy')
        no_stim_mask = np.load(no_stim_mask_path)

        no_stim_i_good_trials_old = i_good_trials[no_stim_mask]

        # filename: e.g. NeuronalData/BAYLORGC95kilosort_2019_12_19.mat
        filename = os.path.join('NeuronalData', '_'.join([f1,f2])+'.mat')
        if not os.path.exists(filename):
            filename = os.path.join('NeuronalData', '_'.join([f1+'kilosort',f2])+'.mat')

        neural_unit_location = sess.neural_unit_location.copy()
        bin_centers = np.load(os.path.join(prep_save_path, 'bin_centers.npy'))


        no_stim_train_set = ALMDataset(filename, root_path=prep_root_path, data_prefix=data_prefix, stim_type='no_stim', data_type='train', neural_unit_location=neural_unit_location, loc_name='both', bin_min=0, bin_max=len(bin_centers)-1)
        no_stim_test_set = ALMDataset(filename, root_path=prep_root_path, data_prefix=data_prefix, stim_type='no_stim', data_type='test', neural_unit_location=neural_unit_location, loc_name='both', bin_min=0, bin_max=len(bin_centers)-1)
        
        no_stim_train_trial_type_labels = no_stim_train_set.trial_type_labels
        no_stim_test_trial_type_labels = no_stim_test_set.trial_type_labels
        no_stim_trial_type_labels = np.concatenate([no_stim_train_trial_type_labels, no_stim_test_trial_type_labels], 0)

        no_stim_train_labels = no_stim_train_set.labels
        no_stim_test_labels = no_stim_test_set.labels
        no_stim_labels = np.concatenate([no_stim_train_labels, no_stim_test_labels], 0)

        no_stim_error_labels = (no_stim_trial_type_labels != no_stim_labels).astype(int)

        no_stim_i_good_trials = no_stim_i_good_trials_old[(no_stim_trial_type_labels==inst_trial_type)]

        np.save(os.path.join(target_save_path, i_good_trials_save_name), no_stim_i_good_trials)

    return no_stim_i_good_trials




def load_stim_i_good_trials_for_state_pred(pert_type, inst_trial_type, prep_root_path, f1, f2):
    '''
    The difference from load_stim_i_good_trials_v2:
    We do not separate error and correct trials.

    '''

    data_prefix = 'Trial_Ordered_Long_Prep_states_2'
    # filename: e.g. NeuronalData/BAYLORGC95kilosort_2019_12_19.mat
    filename = os.path.join('NeuronalData', '_'.join([f1,f2])+'.mat')
    if not os.path.exists(filename):
        filename = os.path.join('NeuronalData', '_'.join([f1+'kilosort',f2])+'.mat')


    target_root_path = os.path.join('pert_pred_utils_results', 'pert_type_{}'.format(pert_type), \
        'inst_trial_type_{}'.format(inst_trial_type),'load_stim_i_good_trials_for_pert_pred')
    target_save_path = os.path.join(target_root_path, data_prefix + 'NeuronalData', f1, f2)
    if 'kilosort' in filename.split('\\')[1]:
        target_save_path = os.path.join(target_root_path, data_prefix + 'NeuronalData', f1+'kilosort', f2)

    os.makedirs(target_save_path, exist_ok=True)

    i_good_trials_save_name = 'stim_i_good_trials.npy'

    if os.path.isfile(os.path.join(target_save_path, i_good_trials_save_name)):
        stim_i_good_trials = np.load(os.path.join(target_save_path, i_good_trials_save_name))

    else:
        print('computing stim_i_good_trials...')
        prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1,f2]))
        if not os.path.exists(prep_save_path):
            prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1+'kilosort',f2]))


        assert os.path.isfile(os.path.join(prep_save_path, 'culled_session.obj'))

        filehandler = open(os.path.join(prep_save_path, 'culled_session.obj'), 'rb')
        sess = pickle.load(filehandler)
        i_good_trials = np.array(sess.i_good_trials.copy())


        # Load stim_mask.
        stim_mask_path = os.path.join(prep_save_path, pert_type, 'stim_mask.npy')
        stim_mask = np.load(stim_mask_path)

        stim_i_good_trials_old = i_good_trials[stim_mask]

        # filename: e.g. NeuronalData/BAYLORGC95kilosort_2019_12_19.mat
        filename = os.path.join('NeuronalData', '_'.join([f1,f2])+'.mat')
        if not os.path.exists(filename):
            filename = os.path.join('NeuronalData', '_'.join([f1+'kilosort',f2])+'.mat')


        stim_train_set = ALMDataset(filename, root_path=prep_root_path, data_prefix=data_prefix, stim_type=pert_type, data_type='train', neural_unit_location=0, loc_name='both', bin_min=0, bin_max=0)
        stim_test_set = ALMDataset(filename, root_path=prep_root_path, data_prefix=data_prefix, stim_type=pert_type, data_type='test', neural_unit_location=0, loc_name='both', bin_min=0, bin_max=0)
        
        stim_train_trial_type_labels = stim_train_set.trial_type_labels
        stim_test_trial_type_labels = stim_test_set.trial_type_labels
        stim_trial_type_labels = np.concatenate([stim_train_trial_type_labels, stim_test_trial_type_labels], 0)

        stim_train_labels = stim_train_set.labels
        stim_test_labels = stim_test_set.labels
        stim_labels = np.concatenate([stim_train_labels, stim_test_labels], 0)

        stim_error_labels = (stim_trial_type_labels != stim_labels).astype(int)

        stim_i_good_trials = stim_i_good_trials_old[(stim_trial_type_labels[0]==inst_trial_type)]

        np.save(os.path.join(target_save_path, i_good_trials_save_name), stim_i_good_trials)

    return stim_i_good_trials







