import os, sys, time, pickle, argparse, math

sys.path.append('NeuralPopulationAnalysisUtils/Code/Python')

from neupop import Session as Session
from neupop import PopulationAnalysis as PA
from neupop import MultiSessionAnalysis as MS
from neupop import decoders as ld
from neupop import session_parsers as parsers

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import json


# Sorted according to session numbers.

def mouse_sort(x):
    # e.g. x = 'AT[#]'
    if 'kilosort' in x:
        return int(x[2:-8])
    else:
        if x == 'GC17':
            return 18

        elif x == 'GC18':
            return 17

        else:
            return int(x[2:])


parser = argparse.ArgumentParser()
parser.add_argument('--train_prop', type=float, default=0.8, help='Fraction of neurons to train on')
parser.add_argument('--filename', type=str, default='')

args = parser.parse_args()

filelist = filter(lambda x: '.mat' in x, os.listdir('NeuronalData'))
filedict = {}

for filename in filelist:
    split_filename = filename.split('_')
    mouse_label = split_filename[0][6:] # Ex.: AT40

    '''
    BAYLORGC78kilosort_2019_07_01.mat: no right and bi stim trials, and also no pert_normal neurons at all!
    '''
    if filename in ['BAYLORAT40_20201015_behavior.mat']:  # These four are sessions sorted in both ways.
        continue

    if mouse_label in filedict:
        filedict[mouse_label].append(os.path.join('NeuronalData', filename))
    else:
        filedict[mouse_label] = [os.path.join('NeuronalData', filename)]

for key in filedict.keys():
    filedict[key].sort()



# Compile filenames from the neural data directory
filenames = []
for key in sorted(filedict.keys(), key=mouse_sort):
    filenames.extend(filedict[key])



filehandler = open('sorted_filenames.obj', 'wb')
pickle.dump(filenames, filehandler)

filehandler = open('sorted_filenames.obj', 'rb')
filenames = pickle.load(filehandler)

print('')
print('All filenames:')
for filename in filenames:
    print(filename)


###
#filehandler = open('neuronal_data4_sorted_filenames.obj', 'rb')
#prev_filenames = pickle.load(filehandler)

#new_filenames = []
#print('')
#print('New filenames:')
#for filename in filenames:
#    if filename not in prev_filenames:
#        print(filename)
#        new_filenames.append(filename)

#temp1 = set(filenames)
#temp2 = set(prev_filenames)

#intersect = temp1.intersection(temp2)

#print(len(temp1))
#print(len(temp2))
#print(len(intersect))
#print(temp2.difference(temp1))

###

def sliding_histogram_bk(spikeTimes, begin_time, end_time, bin_width, stride, rate=True):
    '''
    Calculates the number of spikes for each unit in each sliding bin of width bin_width, strided by stride.
    begin_time and end_time are treated as the lower- and upper- bounds for bin centers.
    '''
    bin_begin_time = begin_time + bin_width / 2
    bin_end_time = end_time - bin_width / 2
    # This is to deal with cases where for e.g. (bin_end_time-bin_begin_time)/stride is actually 43 but evaluates to 42.999999... due to numerical errors in floating-point arithmetic.
    if np.allclose((bin_end_time - bin_begin_time) / stride, math.floor((bin_end_time - bin_begin_time) / stride) + 1):
        n_bins = math.floor((bin_end_time - bin_begin_time) / stride) + 2
    else:
        n_bins = math.floor((bin_end_time - bin_begin_time) / stride) + 1
    binCenters = bin_begin_time + np.arange(n_bins) * stride

    binIntervals = np.vstack((binCenters - bin_width / 2, binCenters + bin_width / 2)).T

    #numneuron = 1
    # Count number of spikes for each sliding bin
    #binSpikes = []
    #for unit in spikeTimes:

    #    trialspikes = []
    #    numtrial = 0
    #    for trial in unit:
    #        trialspikes += [np.sum(np.all([trial >= binInt[0], trial < binInt[1]], axis=0)) for binInt in binIntervals]
    #        currspikes = [np.sum(np.all([trial >= binInt[0], trial < binInt[1]], axis=0)) for binInt in binIntervals]
    #        if numneuron == 7 and numtrial == 50:
    #            np.save("D:\\currspikes.npy", np.asarray(currspikes))
    #            np.save("D:\\bincenters.npy", binCenters)

    #            bug = 1
    #        numtrial += 1

    #    binSpikes += [trialspikes]


    #    numneuron += 1

    #binSpikes = np.asarray(binSpikes).swapaxes(0, -1)

        
    binSpikes = np.asarray([[[np.sum(np.all([trial >= binInt[0], trial < binInt[1]], axis=0)) for binInt in
                                  binIntervals] for trial in unit]
                                for unit in spikeTimes]).swapaxes(0, -1)

    if rate:
        return (binCenters, binSpikes / float(bin_width))
    return (binCenters, binSpikes)


def get_bin_centers(begin_time, end_time, bin_width, stride, rate=True):
    '''
    Calculates the number of spikes for each unit in each sliding bin of width bin_width, strided by stride.
    begin_time and end_time are treated as the lower- and upper- bounds for bin centers.
    '''
    bin_begin_time = begin_time + bin_width / 2
    bin_end_time = end_time - bin_width / 2
    # This is to deal with cases where for e.g. (bin_end_time-bin_begin_time)/stride is actually 43 but evaluates to 42.999999... due to numerical errors in floating-point arithmetic.
    if np.allclose((bin_end_time - bin_begin_time) / stride, math.floor((bin_end_time - bin_begin_time) / stride) + 1):
        n_bins = math.floor((bin_end_time - bin_begin_time) / stride) + 2
    else:
        n_bins = math.floor((bin_end_time - bin_begin_time) / stride) + 1
    binCenters = bin_begin_time + np.arange(n_bins) * stride
    
    return binCenters

def preprocess_bk(session, trial_mask=None, unit_mask=None, align_to=None, begin_time=-3.05, end_time=1.35,
                  bin_width=0.1, stride=0.1, both_labels=False, label_by_report=True):
    '''
    Pert_Normal_Bin0.1_Prep_ALMData: align_to=None, begin_time=-3.05, end_time=1.35, bin_width=0.1, stride=0.1
    Pert_Normal_Prep_ALMData: align_to=None, begin_time=-3.2, end_time=1.5, bin_width=0.4, stride=0.1
    Prep_ALMData: align_to=None, begin_time=-3.2, end_time=1.5, bin_width=0.4, stride=0.1
    Long_Prep_ALMdata: align_to=None, begin_time=-4.6, end_time=1.5, bin_width=0.4, stride=0.1
    '''
    # Align time
    if align_to is None:
        session.align_time(None)
    else:
        raise ValueError('align_to must be None (align to cue on time).')

    if both_labels:
        label_str = session.behavior_report_type
        trial_type_label_str = session.task_trial_type
    else:
        if label_by_report:
            label_str = session.behavior_report_type
        else:
            label_str = session.task_trial_type

    bin_centers, rates = None, None
    # Compute spike rate for each sliding bin
    #bin_centers, rates = \
    #    sliding_histogram_bk(session.neural_data[unit_mask][:, trial_mask], begin_time, end_time, bin_width, stride)

    bin_centers = get_bin_centers(begin_time, end_time, bin_width, stride)


    # Store label as 0s (l) and 1s (r)
    if both_labels:
        labels = (label_str == 'r').astype(int)
        labels = labels[trial_mask]
        trial_type_labels = (trial_type_label_str == 'r').astype(int)
        trial_type_labels = trial_type_labels[trial_mask]
    else:
        labels = (label_str == 'r').astype(int)
        labels = labels[trial_mask]

    if both_labels:
        return bin_centers, rates, labels, trial_type_labels
    else:
        return bin_centers, rates, labels


def train_test_split(rates, labels, trial_type_labels, train_prop, shuffle=False):
    '''
    rates: (T, n_trials, n_neurons)
    labels: (n_trials,)
    '''
    n_trials = rates.shape[1]
    if shuffle:
        np.random.seed(123)
        trial_idx = np.random.permutation(n_trials)
    else:
        trial_idx = np.arange(n_trials)

    train_idx = trial_idx[:int(n_trials * train_prop)]
    test_idx = trial_idx[int(n_trials * train_prop):]

    train_rates = rates[:, train_idx].copy()
    test_rates = rates[:, test_idx].copy()

    train_labels = labels[train_idx].copy()
    test_labels = labels[test_idx].copy()

    train_trial_type_labels = trial_type_labels[train_idx].copy()
    test_trial_type_labels = trial_type_labels[test_idx].copy()

    return train_rates, train_labels, train_trial_type_labels, test_rates, test_labels, test_trial_type_labels

def train_test_split_states(labels, trial_type_labels, train_prop, shuffle=False):
    '''
    rates: (T, n_trials, n_neurons)
    labels: (n_trials,)
    '''
    n_trials = labels.shape[1]
    if shuffle:
        np.random.seed(123)
        trial_idx = np.random.permutation(n_trials)
    else:
        trial_idx = np.arange(n_trials)

    train_idx = trial_idx[:int(n_trials * train_prop)]
    test_idx = trial_idx[int(n_trials * train_prop):]

    if labels.shape[0] == len(train_idx):
        train_labels = labels[train_idx].copy()
        test_labels = labels[test_idx].copy()

        train_trial_type_labels = trial_type_labels[train_idx].copy()
        test_trial_type_labels = trial_type_labels[test_idx].copy()

    else:
        train_labels = labels.T[train_idx].copy()
        test_labels = labels.T[test_idx].copy()

        train_trial_type_labels = trial_type_labels.T[train_idx].copy()
        test_trial_type_labels = trial_type_labels.T[test_idx].copy()

    return train_labels, train_trial_type_labels, test_labels, test_trial_type_labels

def train_test_split_5fold(rates, labels, trial_type_labels, i_good_trials, shuffle=False):
    '''
    rates: (T, n_trials, n_neurons) x 5
    labels: (n_trials,) x 5
    '''
    train_prop = 0.2 # 5 fold 

    n_trials = rates.shape[1]
    if shuffle:
        np.random.seed(123)
        trial_idx = np.random.permutation(n_trials)
    else:
        trial_idx = np.arange(n_trials)

    train_rates = []
    train_labels = []
    train_trial_type_labels = []
    train_i_good = []

    for i in range(5):
        prop = train_prop * i
        train_idx = trial_idx[int(n_trials * prop) : int(n_trials * (prop + 0.2))]
        
        train_rates += [rates[:, train_idx].copy()]

        train_labels += [labels[train_idx].copy()]

        train_trial_type_labels += [trial_type_labels[train_idx].copy()]

        train_i_good += [i_good_trials[train_idx].copy()]

    return train_rates, train_labels, train_trial_type_labels, train_i_good


def save_by_stim_type(rates, labels, trial_type_labels, mask, prep_save_path, stim_type, train_prop, shuffle=True):
    stim_save_path = os.path.join(prep_save_path, stim_type)
    #stim_rates = rates[:, mask]
    if labels.shape[0] == 1:
        stim_labels = labels.T[mask]
        stim_trial_type_labels = trial_type_labels.T[mask]
    else:
        stim_labels = labels[mask]
        stim_trial_type_labels = trial_type_labels[mask]

    train_labels, train_trial_type_labels, test_labels, test_trial_type_labels = train_test_split_states(
        stim_labels, stim_trial_type_labels, train_prop, shuffle)
    #train_rates, train_labels, train_trial_type_labels, test_rates, test_labels, test_trial_type_labels = train_test_split(
    #    stim_rates, stim_labels, stim_trial_type_labels, train_prop, shuffle)

    train_stim_save_path = os.path.join(stim_save_path, 'train')
    test_stim_save_path = os.path.join(stim_save_path, 'test')
    if not os.path.exists(train_stim_save_path):
        os.makedirs(train_stim_save_path)
    if not os.path.exists(test_stim_save_path):
        os.makedirs(test_stim_save_path)

    #np.save(os.path.join(train_stim_save_path, 'rates.npy'), train_rates)
    np.save(os.path.join(train_stim_save_path, 'labels.npy'), train_labels)
    np.save(os.path.join(train_stim_save_path, 'trial_type_labels.npy'), train_trial_type_labels)

    #np.save(os.path.join(test_stim_save_path, 'rates.npy'), test_rates)
    np.save(os.path.join(test_stim_save_path, 'labels.npy'), test_labels)
    np.save(os.path.join(test_stim_save_path, 'trial_type_labels.npy'), test_trial_type_labels)



def find_pert_normal_neuron_mask(no_stim_rates, left_stim_rates, right_stim_rates, bi_stim_rates, stim_bin_min,
                                 stim_bin_max, neural_unit_location):
    # We exclude neurons that have higher trial-averaged firing rates during the perturbation period (defined to be time bins that are fully within the perturbation period) in left, right, or bi_stim than in no_stim.
    n_neurons = no_stim_rates.shape[2]
    no_stim_rates = no_stim_rates[stim_bin_min:stim_bin_max + 1].reshape(-1, n_neurons).mean(0)
    left_stim_rates = left_stim_rates[stim_bin_min:stim_bin_max + 1].reshape(-1, n_neurons).mean(0)
    right_stim_rates = right_stim_rates[stim_bin_min:stim_bin_max + 1].reshape(-1, n_neurons).mean(0)
    bi_stim_rates = bi_stim_rates[stim_bin_min:stim_bin_max + 1].reshape(-1, n_neurons).mean(0)
    no_left_mask = (no_stim_rates >= left_stim_rates)
    no_left_mask[neural_unit_location == 'right_ALM'] = True
    no_right_mask = (no_stim_rates >= right_stim_rates)
    no_right_mask[neural_unit_location == 'left_ALM'] = True
    no_bi_mask = (no_stim_rates >= bi_stim_rates)

    return no_left_mask * no_right_mask * no_bi_mask


def pert_normal_screening(bin_centers, rates, sess, bin_width, pyramidal_only):
    neural_unit_location = sess.neural_unit_location.copy()
    stim_site = sess.stim_site.copy()
    neural_unit_type = sess.neural_unit_type.copy()

    # Now separately for each stim type
    no_stim_mask = (stim_site == sess.stim_site_str2num_dict['no stim'])
    left_stim_mask = (stim_site == sess.stim_site_str2num_dict['left ALM'])
    right_stim_mask = (stim_site == sess.stim_site_str2num_dict['right ALM'])
    bi_stim_mask = (stim_site == sess.stim_site_str2num_dict['bi ALM'])

    no_stim_rates = rates[:, no_stim_mask]
    left_stim_rates = rates[:, left_stim_mask]
    right_stim_rates = rates[:, right_stim_mask]
    bi_stim_rates = rates[:, bi_stim_mask]

    stim_bin_min = np.abs((bin_centers - bin_width / 2) - np.nanmin(sess.stim_on_time)).argmin()
    if not np.isclose(((bin_centers - bin_width / 2) - np.nanmin(sess.stim_on_time))[stim_bin_min], 0, atol=1e-2) and \
            ((bin_centers - bin_width / 2) - np.nanmin(sess.stim_on_time))[stim_bin_min] < 0:
        print(
            'warning: np.isclose(((bin_centers-bin_width/2)-np.nanmin(sess.stim_on_time))[stim_bin_min], 0, atol=1e-2) is False.')
        stim_bin_min = stim_bin_min + 1

    stim_bin_max = np.abs((bin_centers + bin_width / 2) - np.nanmin(sess.stim_off_time)).argmin()
    if not np.isclose(((bin_centers + bin_width / 2) - np.nanmin(sess.stim_off_time))[stim_bin_max], 0, atol=1e-2) and \
            ((bin_centers + bin_width / 2) - np.nanmin(sess.stim_off_time))[stim_bin_max] > 0:
        print(
            'warning: np.isclose(((bin_centers+bin_width/2)-np.nanmin(sess.stim_off_time))[stim_bin_max], 0, atol=1e-2) is False.')
        stim_bin_max = stim_bin_max - 1

    pert_normal_neuron_mask = find_pert_normal_neuron_mask(no_stim_rates, left_stim_rates, right_stim_rates,
                                                           bi_stim_rates, stim_bin_min, stim_bin_max,
                                                           neural_unit_location)

    if pyramidal_only:
        # As Guang told me, I need to only include putative pyramidal neurons.
        new_pert_normal_neuron_mask = pert_normal_neuron_mask * (neural_unit_type == 'putative_pyramidal')

    else:
        new_pert_normal_neuron_mask = pert_normal_neuron_mask

    print('')
    print('original n_neurons: ')
    print(len(pert_normal_neuron_mask))
    print('pert normal n_neurons: ')
    print(len(np.nonzero(pert_normal_neuron_mask)[0]))
    print('new pert normal n_neurons: ')
    print(len(np.nonzero(new_pert_normal_neuron_mask)[0]))

    rates = rates[..., new_pert_normal_neuron_mask]
    neural_unit_location = neural_unit_location[new_pert_normal_neuron_mask]
    neural_unit_type = neural_unit_type[new_pert_normal_neuron_mask]
    sess.select_units(new_pert_normal_neuron_mask)

    return rates, neural_unit_location, neural_unit_type


def main(args):
    split_filename = args.filename.split('\\')

    with open('state_pred_configs.json','r') as read_file:
        configs = json.load(read_file)

    data_prefix = configs['data_prefix']  # Pert_Normal_Bin0.1_Prep_, Pert_Normal_Prep_, Pert_Normal_Long_Prep_.

    prep_save_path = os.path.join(data_prefix + split_filename[0], split_filename[1][:-4])
    if not os.path.exists(prep_save_path):
        os.makedirs(prep_save_path)

    # # key is filename, the elements of filenames list above, and the value is the session number Guang defined.
    # filehandler = open('sess_num_guang_map.obj', 'rb')
    # sess_num_map = pickle.load(filehandler)

    # print('')
    # print('Session {}'.format(sess_num_map[args.filename]))
    # sess = Session(args.filename, parser=parsers.ParserNuoLiDualALM)
    # print('')
    # return

    if data_prefix == 'Pert_Normal_Bin0.1_Prep_':
        prep_kwargs = {'align_to': None, 'begin_time': -3.05, 'end_time': 1.35, 'bin_width': 0.1, 'stride': 0.1}

    elif data_prefix == 'Pert_Normal_Prep_':
        prep_kwargs = {'align_to': None, 'begin_time': -3.2, 'end_time': 1.5, 'bin_width': 0.4, 'stride': 0.1}

    elif data_prefix in ['Pert_Normal_Long_Prep_', 'All_Pert_Normal_Long_Prep_', 'Long_Prep_']:
        prep_kwargs = {'align_to': None, 'begin_time': -4.6, 'end_time': 1.5, 'bin_width': 0.4, 'stride': 0.1}

    elif data_prefix == 'Pert_Normal_Trial_Ordered_Prep_':
        prep_kwargs = {'align_to': None, 'begin_time': -3.2, 'end_time': 1.5, 'bin_width': 0.4, 'stride': 0.1}

    elif data_prefix == 'All_Pert_Normal_Bin0.1_Long_Prep_':
        prep_kwargs = {'align_to': None, 'begin_time': -4.45, 'end_time': 1.35, 'bin_width': 0.1, 'stride': 0.1}

    elif data_prefix == 'All_Pert_Normal_Bin0.2_Long_Prep_':
        prep_kwargs = {'align_to': None, 'begin_time': -4.5, 'end_time': 1.4, 'bin_width': 0.2, 'stride': 0.2}

    elif data_prefix == 'All_Pert_Normal_Bin0.05_Long_Prep_':
        prep_kwargs = {'align_to': None, 'begin_time': -4.425, 'end_time': 1.325, 'bin_width': 0.05, 'stride': 0.05}

    elif data_prefix == 'All_Pert_Normal_Bin0.2_Stride0.1_Long_Prep_':
        prep_kwargs = {'align_to': None, 'begin_time': -4.5, 'end_time': 1.4, 'bin_width': 0.2, 'stride': 0.1}

    elif data_prefix == 'All_Pert_Normal_Bin0.01_Stride0.01_Long_Prep_':
        prep_kwargs = {'align_to': None, 'begin_time': -4.405, 'end_time': 1.305, 'bin_width': 0.01, 'stride': 0.01}

    elif data_prefix == 'All_Pert_Normal_Stride0.01_Long_Prep_':
        prep_kwargs = {'align_to': None, 'begin_time': -4.6, 'end_time': 1.5, 'bin_width': 0.4, 'stride': 0.01}
    elif data_prefix == 'All_Pert_Normal_Bin0.1_Stride0.01_Long_Prep_':
        prep_kwargs = {'align_to': None, 'begin_time': -4.45, 'end_time': 1.35, 'bin_width': 0.1, 'stride': 0.01}

    elif data_prefix in ['Pert_Normal_Trial_Ordered_Long_Prep_', 'All_Pert_Normal_Trial_Ordered_Long_Prep_']:
        prep_kwargs = {'align_to': None, 'begin_time': -4.6, 'end_time': 1.5, 'bin_width': 0.4, 'stride': 0.1}

    elif data_prefix == 'Trial_Ordered_Long_Prep_':
        prep_kwargs = {'align_to': None, 'begin_time': -4.6, 'end_time': 3.3, 'bin_width': 0.4, 'stride': 0.1}

    elif data_prefix == 'Trial_Ordered_Long_Prep_states_2':
        prep_kwargs = {'align_to': None, 'begin_time': -4.6, 'end_time': 3.3, 'bin_width': 0.4, 'stride': 0.1}


    if 'All_Pert' in data_prefix:
        pyramidal_only = False

    else:
        pyramidal_only = True

    # First save the entire data including all stim types

    # if trial_type_labels.npy already exists
    if os.path.isfile(os.path.join(prep_save_path, 'trial_type_labels.npy')):
        #rates = np.load(os.path.join(prep_save_path, 'rates.npy'))
        labels = np.load(os.path.join(prep_save_path, 'labels.npy'))
        trial_type_labels = np.load(os.path.join(prep_save_path, 'trial_type_labels.npy'))
        bin_centers = np.load(os.path.join(prep_save_path, 'bin_centers.npy'))
        filehandler = open(os.path.join(prep_save_path, 'culled_session.obj'), 'rb')
        sess = pickle.load(filehandler)

    else:
        sess = Session(args.filename, parser=parsers.ParserNuoLiDualALM_states)



        # temp

        # Bin_centers and rates will be set to None
        bin_centers, rates, labels, trial_type_labels = preprocess_bk(session=sess, both_labels=True, **prep_kwargs)
        #if 'Pert_Normal' in data_prefix:
        #    rates, neural_unit_location, neural_unit_type = pert_normal_screening(bin_centers, rates, sess,
        #                                                                          bin_width=prep_kwargs['bin_width'],
        #                                                                          pyramidal_only=pyramidal_only)
        #else:
        #    neural_unit_location = sess.neural_unit_location.copy()
        #    neural_unit_type = sess.neural_unit_type.copy()

        #assert (neural_unit_location == sess.neural_unit_location).all()
        #assert (neural_unit_type == sess.neural_unit_type).all()

        np.save(os.path.join(prep_save_path, 'rates.npy'), rates)
        np.save(os.path.join(prep_save_path, 'labels.npy'), labels)
        np.save(os.path.join(prep_save_path, 'trial_type_labels.npy'), trial_type_labels)
        np.save(os.path.join(prep_save_path, 'bin_centers.npy'), bin_centers)



        # Save sess after preprocess_bk so that align_to is updated.
        filehandler = open(os.path.join(prep_save_path, 'culled_session.obj'), 'wb')
        pickle.dump(sess, filehandler)

    print(bin_centers)

    print('task_pole_on_time: min {:.1f} max {:.1f}'.format(sess.task_pole_on_time.min(), sess.task_pole_on_time.max()))
    print('task_pole_off_time: min {:.1f} max {:.1f}'.format(sess.task_pole_off_time.min(),
                                                             sess.task_pole_off_time.max()))
    print('stim_on_time: min {:.1f} max {:.1f}'.format(np.nanmin(sess.stim_on_time), np.nanmax(sess.stim_on_time)))
    print('stim_off_time: min {:.1f} max {:.1f}'.format(np.nanmin(sess.stim_off_time), np.nanmax(sess.stim_off_time)))
    print('task_cue_on_time: min {:.1f} max {:.1f}'.format(sess.task_cue_on_time.min(), sess.task_cue_on_time.max()))
    print('task_cue_off_time: min {:.1f} max {:.1f}'.format(sess.task_cue_off_time.min(), sess.task_cue_off_time.max()))

    #neural_unit_location = sess.neural_unit_location.copy()
    #stim_site = sess.stim_site.copy()

    if 'states' in data_prefix:
        train_labels, train_trial_type_labels, test_labels, test_trial_type_labels = train_test_split_states(
            labels, trial_type_labels, args.train_prop, shuffle=False)
    elif 'Trial_Ordered' in data_prefix:

        train_rates, train_labels, train_trial_type_labels, test_rates, test_labels, test_trial_type_labels = train_test_split(
            rates, labels, trial_type_labels, args.train_prop, shuffle=False)
    else:
        train_rates, train_labels, train_trial_type_labels, test_rates, test_labels, test_trial_type_labels = train_test_split(
            rates, labels, trial_type_labels, args.train_prop, shuffle=True)

    train_prep_save_path = os.path.join(prep_save_path, 'train')
    test_prep_save_path = os.path.join(prep_save_path, 'test')
    if not os.path.exists(train_prep_save_path):
        os.makedirs(train_prep_save_path)
    if not os.path.exists(test_prep_save_path):
        os.makedirs(test_prep_save_path)

    #np.save(os.path.join(train_prep_save_path, 'rates.npy'), train_rates)
    np.save(os.path.join(train_prep_save_path, 'labels.npy'), train_labels)
    np.save(os.path.join(train_prep_save_path, 'trial_type_labels.npy'), train_trial_type_labels) 

    #np.save(os.path.join(test_prep_save_path, 'rates.npy'), test_rates)
    np.save(os.path.join(test_prep_save_path, 'labels.npy'), test_labels)
    np.save(os.path.join(test_prep_save_path, 'trial_type_labels.npy'), test_trial_type_labels)

    #i_good_trials = np.array(sess.i_good_trials.copy())

    #train_rates, train_labels, train_trial_type_labels, train_i_good_trials = train_test_split_5fold(
    #        rates, labels, trial_type_labels, i_good_trials, shuffle=True)

    #for i in range(5):

    #    train_prep_save_path = os.path.join(prep_save_path, 'train' + str(i))
    #    if not os.path.exists(train_prep_save_path):
    #        os.makedirs(train_prep_save_path)

    #    np.save(os.path.join(train_prep_save_path, 'rates.npy'), train_rates[i])
    #    np.save(os.path.join(train_prep_save_path, 'labels.npy'), train_labels[i])
    #    np.save(os.path.join(train_prep_save_path, 'trial_type_labels.npy'), train_trial_type_labels[i])
    #    np.save(os.path.join(train_prep_save_path, 'i_good.npy'), train_i_good_trials[i])



    state_var = sess.state.copy()


    # Separately for each engagement type
    state_1_mask = (state_var == 0)
    state_2_mask = (state_var == 1)
    state_3_mask = (state_var == 2)
    
    print('')
    print('state 1 n_trials: ', len(np.nonzero(state_1_mask)[0]))
    print('state 2 n_trials: ', len(np.nonzero(state_2_mask)[0]))
    print('state 3 n_trials: ', len(np.nonzero(state_3_mask)[0]))

    # Save stim masks.
    stim_save_path = os.path.join(prep_save_path, 'state_1')
    os.makedirs(stim_save_path, exist_ok=True)
    np.save(os.path.join(stim_save_path, 'stim_mask.npy'), state_1_mask)

    stim_save_path = os.path.join(prep_save_path, 'state_2')
    os.makedirs(stim_save_path, exist_ok=True)
    np.save(os.path.join(stim_save_path, 'stim_mask.npy'), state_2_mask)

    # Only include for three state situation

    if 'states_3' in data_prefix:
        stim_save_path = os.path.join(prep_save_path, 'state_3')
        os.makedirs(stim_save_path, exist_ok=True)
        np.save(os.path.join(stim_save_path, 'stim_mask.npy'), state_3_mask)

    # Save by stim type

    if 'states_2' in data_prefix:
        save_by_stim_type(rates, labels, trial_type_labels, state_1_mask, prep_save_path, 'state_1', args.train_prop,
                          shuffle=False)
        save_by_stim_type(rates, labels, trial_type_labels, state_2_mask, prep_save_path, 'state_2',
                          args.train_prop, shuffle=False)
    elif 'Trial_Ordered' in data_prefix:
        save_by_stim_type(rates, labels, trial_type_labels, state_1_mask, prep_save_path, 'state_1', args.train_prop,
                          shuffle=False)
        save_by_stim_type(rates, labels, trial_type_labels, state_2_mask, prep_save_path, 'state_2',
                          args.train_prop, shuffle=False)
        save_by_stim_type(rates, labels, trial_type_labels, state_3_mask, prep_save_path, 'state_3',
                          args.train_prop, shuffle=False)


    else:
        save_by_stim_type(rates, labels, trial_type_labels, state_1_mask, prep_save_path, 'state_1', args.train_prop,
                          shuffle=True)
        save_by_stim_type(rates, labels, trial_type_labels, state_2_mask, prep_save_path, 'state_2',
                          args.train_prop, shuffle=True)
        save_by_stim_type(rates, labels, trial_type_labels, state_3_mask, prep_save_path, 'state_3',
                          args.train_prop, shuffle=True)


    ## Now separately for each stim type
    #no_stim_mask = (stim_site == sess.stim_site_str2num_dict['no stim'])
    #left_stim_mask = (stim_site == sess.stim_site_str2num_dict['left ALM'])
    #right_stim_mask = (stim_site == sess.stim_site_str2num_dict['right ALM'])
    #bi_stim_mask = (stim_site == sess.stim_site_str2num_dict['bi ALM'])


    #print('')
    #print('no_stim n_trials: ', len(np.nonzero(no_stim_mask)[0]))
    #print('left_stim n_trials: ', len(np.nonzero(left_stim_mask)[0]))
    #print('right_stim n_trials: ', len(np.nonzero(right_stim_mask)[0]))
    #print('bi_stim n_trials: ', len(np.nonzero(bi_stim_mask)[0]))


    ## Save stim masks.
    #stim_save_path = os.path.join(prep_save_path, 'no_stim')
    #os.makedirs(stim_save_path, exist_ok=True)
    #np.save(os.path.join(stim_save_path, 'stim_mask.npy'), no_stim_mask)

    #stim_save_path = os.path.join(prep_save_path, 'left_stim')
    #os.makedirs(stim_save_path, exist_ok=True)
    #np.save(os.path.join(stim_save_path, 'stim_mask.npy'), left_stim_mask)

    #stim_save_path = os.path.join(prep_save_path, 'right_stim')
    #os.makedirs(stim_save_path, exist_ok=True)
    #np.save(os.path.join(stim_save_path, 'stim_mask.npy'), right_stim_mask)

    #stim_save_path = os.path.join(prep_save_path, 'bi_stim')
    #os.makedirs(stim_save_path, exist_ok=True)
    #np.save(os.path.join(stim_save_path, 'stim_mask.npy'), bi_stim_mask)

    #if 'Trial_Ordered' in data_prefix:
    #    save_by_stim_type(rates, labels, trial_type_labels, no_stim_mask, prep_save_path, 'no_stim', args.train_prop,
    #                      shuffle=False)
    #    save_by_stim_type(rates, labels, trial_type_labels, left_stim_mask, prep_save_path, 'left_stim',
    #                      args.train_prop, shuffle=False)
    #    save_by_stim_type(rates, labels, trial_type_labels, right_stim_mask, prep_save_path, 'right_stim',
    #                      args.train_prop, shuffle=False)
    #    save_by_stim_type(rates, labels, trial_type_labels, bi_stim_mask, prep_save_path, 'bi_stim', args.train_prop,
    #                      shuffle=False)

    #else:
    #    save_by_stim_type(rates, labels, trial_type_labels, no_stim_mask, prep_save_path, 'no_stim', args.train_prop,
    #                      shuffle=True)
    #    save_by_stim_type(rates, labels, trial_type_labels, left_stim_mask, prep_save_path, 'left_stim',
    #                      args.train_prop, shuffle=True)
    #    save_by_stim_type(rates, labels, trial_type_labels, right_stim_mask, prep_save_path, 'right_stim',
    #                      args.train_prop, shuffle=True)
    #    save_by_stim_type(rates, labels, trial_type_labels, bi_stim_mask, prep_save_path, 'bi_stim', args.train_prop,
    #                      shuffle=True)


if __name__ == '__main__':
    if args.filename:
        print('filename: {}'.format(args.filename))
        main(args)
    else:
        # print('n_sess: {}'.format(len(filenames)))
        # for sess_iter, filename in enumerate(filenames):
        #     args.filename = filename
        #     print('')
        #     print('Session {}'.format(sess_iter+1))
        #     print('filename: {}'.format(filename))

        #     main(args)

        print('n_sess: {}'.format(len(filenames)))
        for sess_iter, filename in enumerate(filenames):
            args.filename = filename
            print('')
            print('Session {}'.format(sess_iter + 1))
            print('filename: {}'.format(filename))

            main(args)
