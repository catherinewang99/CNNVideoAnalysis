'''
One key difference from cd_vel_dataset is that the cd is derived from trial_type_labels, not labels.
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
from alm_datasets import ALMDataset, ALMDatasetSimple, ALMDatasetTest

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from collections import defaultdict

cat = np.concatenate

filehandler = open('sorted_filenames.obj', 'rb')
filenames = pickle.load(filehandler)


def nestdict():
    def the_nestdict():
        return defaultdict(the_nestdict)

    return the_nestdict()


def nestdict_to_dict(nestdict):
    if isinstance(nestdict, defaultdict):
        temp = dict(nestdict)
        for key in temp.keys():
            temp[key] = nestdict_to_dict(temp[key])
        return temp
    else:
        return nestdict


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--generate_dataset', action='store_true', default=True)
    parser.add_argument('--fr_percentile', type=float, default=0)

    parser.add_argument('--results_dir', type=str, default='all_neurons_time_traces_results')
    parser.add_argument('--data_prefix', type=str, default='Trial_Ordered_Long_Prep_')

    return parser.parse_args()


def main():
    args = parse_args()
    assert 'Trial_Ordered' in args.data_prefix

    print(args.generate_dataset)
    if args.generate_dataset:
        for file_iter, filename in enumerate(filenames):
            args.filename = filename
            print('')
            print('Session {}'.format(file_iter + 1))
            print('filename: {}'.format(filename))
            print('')
            #if args.filename == 'NeuronalData\\BAYLORAT44_20201001.mat': # For debugging purposes
            generate_dataset(args)


def generate_dataset(args):
    '''
    cd_vel[i][loc_name]: (T, n_trials, 1). Note that cd_vel at the last time point is zero, because there is no cd_ipsi beyond the last time point.
    cd_ipsi[i][loc_name]: (T, n_trials, 1)
    cd_contra[i][loc_name]: (T, n_trials, 1)

    x_ipsi[i][loc_name]: (T, n_trials, n_neurons_ipsi)
    x_contra[i][loc_name]: (T, n_trials, n_neurons_contra)
    x_total[i][loc_name]: (T, n_trials, n_neurons_total)
    '''
    n_trial_types_list = range(2)
    #loc_name_list = ['left_ALM', 'right_ALM']

    # Get bin_max and neural_unit_location.
    split_filename = args.filename.split('\\')
    prep_save_path = os.path.join(args.data_prefix + split_filename[0], split_filename[1][:-4])

    assert os.path.isfile(os.path.join(prep_save_path, 'culled_session.obj'))

    filehandler = open(os.path.join(prep_save_path, 'culled_session.obj'), 'rb')
    sess = pickle.load(filehandler)

    bin_centers = np.load(os.path.join(prep_save_path, 'bin_centers.npy'))
    if 'Bin0.1' in args.data_prefix:
        bin_width = 0.1
    else:
        bin_width = 0.4

    samp_bin_min = np.abs((bin_centers - bin_width / 2) - sess.task_pole_on_time.min()).argmin()
    if not np.isclose(((bin_centers - bin_width / 2) - sess.task_pole_on_time.min())[samp_bin_min], 0, atol=1e-2) and \
            ((bin_centers - bin_width / 2) - sess.task_pole_on_time.min())[samp_bin_min] < 0:
        print(
            'warning: np.isclose(((bin_centers-bin_width/2)-sess.task_pole_off_time.min())[bin_min], 0, atol=1e-2) is False.')
        samp_bin_min = samp_bin_min + 1

    samp_bin_max = np.abs((bin_centers + bin_width / 2) - sess.task_pole_off_time.min()).argmin()
    if not np.isclose(((bin_centers + bin_width / 2) - sess.task_pole_off_time.min())[samp_bin_max], 0, atol=1e-2) and \
            ((bin_centers + bin_width / 2) - sess.task_pole_off_time.min())[samp_bin_max] > 0:
        print(
            'warning: np.isclose(((bin_centers+bin_width/2)-sess.task_cue_on_time.min())[bin_max], 0, atol=1e-2) is False.')
        samp_bin_max = samp_bin_max - 1

    bin_min = np.abs((bin_centers - bin_width / 2) - sess.task_pole_off_time.min()).argmin()
    if not np.isclose(((bin_centers - bin_width / 2) - sess.task_pole_off_time.min())[bin_min], 0, atol=1e-2) and \
            ((bin_centers - bin_width / 2) - sess.task_pole_off_time.min())[bin_min] < 0:
        print(
            'warning: np.isclose(((bin_centers-bin_width/2)-sess.task_pole_off_time.min())[bin_min], 0, atol=1e-2) is False.')
        bin_min = bin_min + 1

    bin_max = np.abs((bin_centers + bin_width / 2) - sess.task_cue_on_time.min()).argmin()
    if not np.isclose(((bin_centers + bin_width / 2) - sess.task_cue_on_time.min())[bin_max], 0, atol=1e-2) and \
            ((bin_centers + bin_width / 2) - sess.task_cue_on_time.min())[bin_max] > 0:
        print(
            'warning: np.isclose(((bin_centers+bin_width/2)-sess.task_cue_on_time.min())[bin_max], 0, atol=1e-2) is False.')
        bin_max = bin_max - 1

    neural_unit_location = sess.neural_unit_location.copy()

    i_good_trials = np.array(sess.i_good_trials.copy())

    print('')
    print('i_good_trials')
    print(i_good_trials)
    print('sorted?: {}'.format((i_good_trials == sorted(i_good_trials)).all()))
    assert (i_good_trials == np.sort(i_good_trials)).all()

    #no_stim_train_set = ALMDataset(args.filename, data_prefix=args.data_prefix, stim_type=['no_stim', 'left_stim', 'right_stim'], data_type='train',
    #                               neural_unit_location=neural_unit_location, loc_name='both', bin_min=0,
    #                               bin_max=len(bin_centers) - 1)
    #no_stim_test_set = ALMDataset(args.filename, data_prefix=args.data_prefix, stim_type=['no_stim', 'left_stim', 'right_stim'], data_type='test',
    #                              neural_unit_location=neural_unit_location, loc_name='both', bin_min=0,
    #                              bin_max=len(bin_centers) - 1)

    #START OF THE NORMAL TRAIN SPLIT

    all_stim_set = ALMDataset(args.filename, data_prefix=args.data_prefix, stim_type='test', data_type='test',
                                  neural_unit_location=neural_unit_location, loc_name='both', bin_min=0,
                                  bin_max=len(bin_centers) - 1)


    stim_rates = all_stim_set.rates
    stim_trial_type_labels = all_stim_set.trial_type_labels
    stim_labels = all_stim_set.labels
    
    stim_suc_labels = (stim_trial_type_labels == stim_labels).astype(int)

    #no_stim_train_rates = no_stim_train_set.rates
    #no_stim_test_rates = no_stim_test_set.rates
    #no_stim_rates = np.concatenate([no_stim_train_rates, no_stim_test_rates], 1)
    #np.save('stimrates.npy', no_stim_rates)


    #no_stim_train_trial_type_labels = no_stim_train_set.trial_type_labels
    #no_stim_test_trial_type_labels = no_stim_test_set.trial_type_labels
    #no_stim_trial_type_labels = np.concatenate([no_stim_train_trial_type_labels, no_stim_test_trial_type_labels], 0)
    #np.save('triallabels.npy', no_stim_trial_type_labels)

    #no_stim_train_labels = no_stim_train_set.labels
    #no_stim_test_labels = no_stim_test_set.labels
    #no_stim_labels = np.concatenate([no_stim_train_labels, no_stim_test_labels], 0)

    #no_stim_suc_labels = (no_stim_trial_type_labels == no_stim_labels).astype(int)
    #np.save('no_stim_suc_labels.npy', no_stim_suc_labels)

    all_neurons = nestdict()

    for i in n_trial_types_list:

        #switch = True
        #for j in range(len(stim_labels)):
        #    if stim_trial_type_labels[j] == i and stim_suc_labels[j] == 1:
        #        if switch:
                    
        #            rates = stim_rates[:, j:j+1, :]

        #            switch = False

        #        else:
        #            rates = np.concatenate([rates, stim_rates[:, j:j+1, :]], 1)

        #all_neurons[i] = rates
        
        all_neurons[i] = stim_rates[:,
                         (stim_trial_type_labels == i)]  # (T, n_trials, n_neurons)

        #np.save('allnn.npy', all_neurons[i])

    full_trial_average = cat([all_neurons[i] for i in n_trial_types_list], 1).mean(0).mean(0)  # (n_neurons)

    perc = args.fr_percentile

    fr_thr = np.percentile(full_trial_average, args.fr_percentile)
    print('n_neurons before thresholding: {}'.format(all_neurons[0].shape[2]))
    print('full_trial_average ({} percentile): {:.2f}'.format(args.fr_percentile, fr_thr))

    # Only select those neurons that are above the fr_percentile.
    neuron_mask = full_trial_average >= fr_thr # all True

    for i in n_trial_types_list:
        all_neurons[i] = all_neurons[i][..., neuron_mask]

        #np.save('alln.npy', all_neurons[i])


    print('')
    print('n_neurons after thresholding: {}'.format(all_neurons[0].shape[2]))

    # Save.

    all_neurons = nestdict_to_dict(all_neurons)

    split_filename = args.filename.split('\\')
    subsplit_idx = split_filename[1].find('_')

    sub_dir = args.data_prefix + 'NeuronalData'

    save_path = os.path.join(args.results_dir, sub_dir, split_filename[1][:subsplit_idx],
                             split_filename[1][subsplit_idx + 1:-4])
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, 'all_neurons_{}.obj'.format(int(args.fr_percentile))), 'wb') as f:
        pickle.dump(all_neurons, f)

    # We want to save no_stim_correct_i_good_trials.

    '''
    We save i_good_trials.
    '''
    i_good_trials_save_name = 'i_good_trials.npy'
    np.save(os.path.join(save_path, i_good_trials_save_name), i_good_trials)

    # Load no_stim_mask. CW: commented out because we actually want to use all stim types
    #stim_type = ['no_stim', 'left_stim', 'right_stim']
    #no_stim_i_good_trials = None
    #for st in stim_type:

    #    no_stim_mask_path = os.path.join(prep_save_path, st, 'stim_mask.npy')
    #    no_stim_mask = np.load(no_stim_mask_path)

    #    if no_stim_i_good_trials is None:
    #        no_stim_i_good_trials = i_good_trials[no_stim_mask]
    #    else:
    #        # Need to reorganize this into train/test the way it is done with rates
    #        no_stim_i_good_trials = np.concatenate([no_stim_i_good_trials, i_good_trials[no_stim_mask]], 0)


    no_stim_i_good_trials = i_good_trials

    print('')
    print('no_stim_i_good_trials')
    print(no_stim_i_good_trials)

    print('')
    print('no_stim_i_correct_good_trials')
    print(no_stim_i_good_trials[stim_suc_labels == 1])

    no_stim_correct_i_good_trials = np.zeros(2, dtype=object)
    for i in n_trial_types_list:
        no_stim_correct_i_good_trials[i] = no_stim_i_good_trials[
            (stim_trial_type_labels == i) * (stim_suc_labels == 1)]

    no_stim_correct_i_good_trials_save_name = 'no_stim_correct_i_good_trials.npy'
    np.save(os.path.join(save_path, no_stim_correct_i_good_trials_save_name), no_stim_correct_i_good_trials)

    '''
    We also save no_stim_i_good_trials.
    '''
    no_stim_i_good_trials_old = no_stim_i_good_trials.copy()
    no_stim_i_good_trials = np.zeros(2, dtype=object)
    for i in n_trial_types_list:
        no_stim_i_good_trials[i] = no_stim_i_good_trials_old[(stim_trial_type_labels == i)]

    no_stim_i_good_trials_save_name = 'no_stim_i_good_trials.npy'
    np.save(os.path.join(save_path, no_stim_i_good_trials_save_name), no_stim_i_good_trials)

    # Save bin_centers. Needed for figuring out begin and end bin corresponding to begin and end frames.
    bin_center_save_name = 'bin_centers.npy'
    np.save(os.path.join(save_path, bin_center_save_name), bin_centers)




    ### 5 FOLD CV START

    #for trial in [0]:

    #    all_stim_set = ALMDatasetSimple(args.filename, data_prefix=args.data_prefix, stim_type=trial, data_type='train',
    #                                  neural_unit_location=neural_unit_location, loc_name='both', bin_min=0,
    #                                  bin_max=len(bin_centers) - 1)

    #    stim_rates = all_stim_set.rates
    #    stim_trial_type_labels = all_stim_set.trial_type_labels
    #    stim_labels = all_stim_set.labels
    #    i_good = all_stim_set.i_good
    
    #    stim_suc_labels = (stim_trial_type_labels == stim_labels).astype(int)


    #    all_neurons = nestdict()

    #    for i in n_trial_types_list:
        
    #        all_neurons[i] = stim_rates[:,
    #                         (stim_trial_type_labels == i)]  # (T, n_trials, n_neurons)

    #    full_trial_average = cat([all_neurons[i] for i in n_trial_types_list], 1).mean(0).mean(0)  # (n_neurons)

    #    perc = args.fr_percentile

    #    fr_thr = np.percentile(full_trial_average, args.fr_percentile)
    #    print('n_neurons before thresholding: {}'.format(all_neurons[0].shape[2]))
    #    print('full_trial_average ({} percentile): {:.2f}'.format(args.fr_percentile, fr_thr))

    #    # Only select those neurons that are above the fr_percentile.
    #    neuron_mask = full_trial_average >= fr_thr # all True

    #    for i in n_trial_types_list:
    #        all_neurons[i] = all_neurons[i][..., neuron_mask]


    #    print('')
    #    print('n_neurons after thresholding: {}'.format(all_neurons[0].shape[2]))

    #    # Save.

    #    all_neurons = nestdict_to_dict(all_neurons)

    #    split_filename = args.filename.split('\\')
    #    subsplit_idx = split_filename[1].find('_')

    #    sub_dir = args.data_prefix + 'NeuronalData'

    #    save_path = os.path.join(args.results_dir, sub_dir, split_filename[1][:subsplit_idx],
    #                             split_filename[1][subsplit_idx + 1:-4])
    #    os.makedirs(save_path, exist_ok=True)

    #    with open(os.path.join(save_path, 'all_neurons_{}.obj'.format(int(args.fr_percentile))), 'wb') as f:
    #        pickle.dump(all_neurons, f)

    #    # We want to save no_stim_correct_i_good_trials.

    #    '''
    #    We save i_good_trials.
    #    '''
    #    i_good_trials = i_good
    #    i_good_trials_save_name = 'i_good_trials.npy'
    #    np.save(os.path.join(save_path, i_good_trials_save_name), i_good_trials)

    #    # Load no_stim_mask. CW: commented out because we actually want to use all stim types
    #    #stim_type = ['no_stim', 'left_stim', 'right_stim']
    #    #no_stim_i_good_trials = None
    #    #for st in stim_type:

    #    #    no_stim_mask_path = os.path.join(prep_save_path, st, 'stim_mask.npy')
    #    #    no_stim_mask = np.load(no_stim_mask_path)

    #    #    if no_stim_i_good_trials is None:
    #    #        no_stim_i_good_trials = i_good_trials[no_stim_mask]
    #    #    else:
    #    #        # Need to reorganize this into train/test the way it is done with rates
    #    #        no_stim_i_good_trials = np.concatenate([no_stim_i_good_trials, i_good_trials[no_stim_mask]], 0)


    #    no_stim_i_good_trials = i_good_trials

    #    print('')
    #    print('no_stim_i_good_trials')
    #    print(no_stim_i_good_trials)

    #    print('')
    #    print('no_stim_i_correct_good_trials')
    #    print(no_stim_i_good_trials[stim_suc_labels == 1])

    #    no_stim_correct_i_good_trials = np.zeros(2, dtype=object)
    #    for i in n_trial_types_list:
    #        no_stim_correct_i_good_trials[i] = no_stim_i_good_trials[
    #            (stim_trial_type_labels == i) * (stim_suc_labels == 1)]

    #    no_stim_correct_i_good_trials_save_name = 'no_stim_correct_i_good_trials.npy'
    #    np.save(os.path.join(save_path, no_stim_correct_i_good_trials_save_name), no_stim_correct_i_good_trials)

    #    '''
    #    We also save no_stim_i_good_trials.
    #    '''
    #    no_stim_i_good_trials_old = no_stim_i_good_trials.copy()
    #    no_stim_i_good_trials = np.zeros(2, dtype=object)
    #    for i in n_trial_types_list:
    #        no_stim_i_good_trials[i] = no_stim_i_good_trials_old[(stim_trial_type_labels == i)]

    #    no_stim_i_good_trials_save_name = 'no_stim_i_good_trials.npy'
    #    np.save(os.path.join(save_path, no_stim_i_good_trials_save_name), no_stim_i_good_trials)

    #    # Save bin_centers. Needed for figuring out begin and end bin corresponding to begin and end frames.
    #    bin_center_save_name = 'bin_centers.npy'
    #    np.save(os.path.join(save_path, bin_center_save_name), bin_centers)


    #    ### TEST SET ###

    #    all_stim_set = ALMDatasetTest(args.filename, data_prefix=args.data_prefix, stim_type=trial, data_type='train',
    #                                  neural_unit_location=neural_unit_location, loc_name='both', bin_min=0,
    #                                  bin_max=len(bin_centers) - 1)

    #    stim_rates = all_stim_set.rates
    #    stim_trial_type_labels = all_stim_set.trial_type_labels
    #    stim_labels = all_stim_set.labels
    #    i_good = all_stim_set.i_good
    
    #    stim_suc_labels = (stim_trial_type_labels == stim_labels).astype(int)


    #    all_neurons = nestdict()

    #    for i in n_trial_types_list:
        
    #        all_neurons[i] = stim_rates[:,
    #                         (stim_trial_type_labels == i)]  # (T, n_trials, n_neurons)

    #    full_trial_average = cat([all_neurons[i] for i in n_trial_types_list], 1).mean(0).mean(0)  # (n_neurons)

    #    perc = args.fr_percentile

    #    fr_thr = np.percentile(full_trial_average, args.fr_percentile)
    #    print('n_neurons before thresholding: {}'.format(all_neurons[0].shape[2]))
    #    print('full_trial_average ({} percentile): {:.2f}'.format(args.fr_percentile, fr_thr))

    #    # Only select those neurons that are above the fr_percentile.
    #    neuron_mask = full_trial_average >= fr_thr # all True

    #    for i in n_trial_types_list:
    #        all_neurons[i] = all_neurons[i][..., neuron_mask]


    #    print('')
    #    print('n_neurons after thresholding: {}'.format(all_neurons[0].shape[2]))

    #    # Save.

    #    all_neurons = nestdict_to_dict(all_neurons)

    #    split_filename = args.filename.split('\\')
    #    subsplit_idx = split_filename[1].find('_')

    #    sub_dir = args.data_prefix + 'NeuronalData'

    #    save_path = os.path.join(args.results_dir, sub_dir, split_filename[1][:subsplit_idx],
    #                             split_filename[1][subsplit_idx + 1:-4])
    #    os.makedirs(save_path, exist_ok=True)

    #    with open(os.path.join(save_path, 'testall_neurons_{}.obj'.format(int(args.fr_percentile))), 'wb') as f:
    #        pickle.dump(all_neurons, f)

    #    # We want to save no_stim_correct_i_good_trials.

    #    '''
    #    We save i_good_trials.
    #    '''
    #    i_good_trials = i_good
    #    i_good_trials_save_name = 'testi_good_trials.npy'
    #    np.save(os.path.join(save_path, i_good_trials_save_name), i_good_trials)

    #    # Load no_stim_mask. CW: commented out because we actually want to use all stim types
    #    #stim_type = ['no_stim', 'left_stim', 'right_stim']
    #    #no_stim_i_good_trials = None
    #    #for st in stim_type:

    #    #    no_stim_mask_path = os.path.join(prep_save_path, st, 'stim_mask.npy')
    #    #    no_stim_mask = np.load(no_stim_mask_path)

    #    #    if no_stim_i_good_trials is None:
    #    #        no_stim_i_good_trials = i_good_trials[no_stim_mask]
    #    #    else:
    #    #        # Need to reorganize this into train/test the way it is done with rates
    #    #        no_stim_i_good_trials = np.concatenate([no_stim_i_good_trials, i_good_trials[no_stim_mask]], 0)


    #    no_stim_i_good_trials = i_good_trials

    #    print('')
    #    print('no_stim_i_good_trials')
    #    print(no_stim_i_good_trials)

    #    print('')
    #    print('no_stim_i_correct_good_trials')
    #    print(no_stim_i_good_trials[stim_suc_labels == 1])

    #    no_stim_correct_i_good_trials = np.zeros(2, dtype=object)
    #    for i in n_trial_types_list:
    #        no_stim_correct_i_good_trials[i] = no_stim_i_good_trials[
    #            (stim_trial_type_labels == i) * (stim_suc_labels == 1)]

    #    no_stim_correct_i_good_trials_save_name = 'testno_stim_correct_i_good_trials.npy'
    #    np.save(os.path.join(save_path, no_stim_correct_i_good_trials_save_name), no_stim_correct_i_good_trials)

    #    '''
    #    We also save no_stim_i_good_trials.
    #    '''
    #    no_stim_i_good_trials_old = no_stim_i_good_trials.copy()
    #    no_stim_i_good_trials = np.zeros(2, dtype=object)
    #    for i in n_trial_types_list:
    #        no_stim_i_good_trials[i] = no_stim_i_good_trials_old[(stim_trial_type_labels == i)]

    #    no_stim_i_good_trials_save_name = 'testno_stim_i_good_trials.npy'
    #    np.save(os.path.join(save_path, no_stim_i_good_trials_save_name), no_stim_i_good_trials)

    #    # Save bin_centers. Needed for figuring out begin and end bin corresponding to begin and end frames.
    #    bin_center_save_name = 'testbin_centers.npy'
    #    np.save(os.path.join(save_path, bin_center_save_name), bin_centers)

if __name__ == '__main__':
    main()

