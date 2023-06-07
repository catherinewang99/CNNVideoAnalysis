from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torch.utils.data as data
import os


class ALMDataset(data.Dataset):
    def __init__(self, filename, stim_type, data_type, neural_unit_location, loc_name, bin_min, bin_max, root_path=None, save_path=None, data_prefix='Pert_Normal_Prep_', label_cond=None, fluct_type=None, fluct_trial_type=None):
        if save_path is None:
            split_filename = filename.split('\\')
            save_path = os.path.join(data_prefix + split_filename[0], split_filename[1][:-4])
        if root_path is not None:
            save_path = os.path.join(root_path, save_path)
        self.stim_type = stim_type
        self.data_type = data_type
        self.fluct_type = fluct_type
        self.fluct_trial_type = fluct_trial_type
        self.neural_unit_location = neural_unit_location
        self.loc_name = loc_name
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.label_cond = label_cond

        if loc_name != 'both':
            self.unit_mask = (neural_unit_location==loc_name)

        if stim_type == 'test':
            self.rates = np.load(os.path.join(save_path, 'rates.npy'))
            self.labels = np.load(os.path.join(save_path, 'labels.npy'))
            self.trial_type_labels = np.load(os.path.join(save_path, 'trial_type_labels.npy'))

        elif isinstance(stim_type, list):
            self.rates = None
            self.labels = None
            self.trial_type_labels = None
            for st in stim_type:
                data_dir = os.path.join(save_path, st, data_type)
                if loc_name != 'both':
                    rates = np.load(os.path.join(data_dir, 'rates.npy'))[bin_min:bin_max+1,:,self.unit_mask].astype(np.float32)
                else:
                    rates = np.load(os.path.join(data_dir, 'rates.npy'))[bin_min:bin_max+1].astype(np.float32)
                labels = np.load(os.path.join(data_dir, 'labels.npy'))
                trial_type_labels = np.load(os.path.join(data_dir, 'trial_type_labels.npy'))
                if self.rates is None:
                    self.rates = rates
                    self.labels = labels
                    self.trial_type_labels = trial_type_labels
                else:
                    self.rates = np.concatenate([self.rates, rates], 1)
                    self.labels = np.concatenate([self.labels, labels], 0)
                    self.trial_type_labels = np.concatenate([self.trial_type_labels, trial_type_labels], 0)
        elif 'states' in data_prefix:
            data_dir = os.path.join(save_path, stim_type, data_type) # stim_type: no_stim etc. data_type: train or test.

            self.labels = np.load(os.path.join(data_dir, 'labels.npy'))
            self.trial_type_labels = np.load(os.path.join(data_dir, 'trial_type_labels.npy'))

        else:
            data_dir = os.path.join(save_path, stim_type, data_type) # stim_type: no_stim etc. data_type: train or test.
            if loc_name != 'both':
                self.rates = np.load(os.path.join(data_dir, 'rates.npy'))[bin_min:bin_max+1,:,self.unit_mask].astype(np.float32)
            else:
                self.rates = np.load(os.path.join(data_dir, 'rates.npy'))[bin_min:bin_max+1].astype(np.float32)
            self.labels = np.load(os.path.join(data_dir, 'labels.npy'))
            self.trial_type_labels = np.load(os.path.join(data_dir, 'trial_type_labels.npy'))

        if self.fluct_type is not None and 'cond_fluct' in self.fluct_type:
            # We want to estimate conditioned-trial average trajectory from the train and test set combined.
            assert stim_type == 'no_stim'
            train_data_dir = os.path.join(save_path, stim_type, 'train')
            test_data_dir = os.path.join(save_path, stim_type, 'test')
            if loc_name != 'both':
                train_rates = np.load(os.path.join(train_data_dir, 'rates.npy'))[bin_min:bin_max+1,:,self.unit_mask].astype(np.float32)
                test_rates = np.load(os.path.join(test_data_dir, 'rates.npy'))[bin_min:bin_max+1,:,self.unit_mask].astype(np.float32)
            else:
                train_rates = np.load(os.path.join(train_data_dir, 'rates.npy'))[bin_min:bin_max+1].astype(np.float32)
                test_rates = np.load(os.path.join(test_data_dir, 'rates.npy'))[bin_min:bin_max+1].astype(np.float32)

            full_rates = np.concatenate([train_rates, test_rates], 1)

            if self.fluct_type == 'behave_cond_fluct':
                train_labels = np.load(os.path.join(train_data_dir, 'labels.npy'))
                test_labels = np.load(os.path.join(test_data_dir, 'labels.npy'))
                full_labels = np.concatenate([train_labels, test_labels], 0)
                if self.fluct_trial_type == 'left':
                    self.fluct = self.rates[:,(self.labels==0)] - full_rates[:,(full_labels==0)].mean(1, keepdims=True)
                elif self.fluct_trial_type == 'right':
                    self.fluct = self.rates[:,(self.labels==1)] - full_rates[:,(full_labels==1)].mean(1, keepdims=True)
            elif self.fluct_type == 'trial_cond_fluct':
                train_trial_type_labels = np.load(os.path.join(train_data_dir, 'trial_type_labels.npy'))
                test_trial_type_labels = np.load(os.path.join(test_data_dir, 'trial_type_labels.npy'))
                full_trial_type_labels = np.concatenate([train_trial_type_labels, test_trial_type_labels], 0)
                if self.fluct_trial_type == 'left':
                    self.fluct = self.rates[:,(self.trial_type_labels==0)] - full_rates[:,(full_trial_type_labels==0)].mean(1, keepdims=True)
                elif self.fluct_trial_type == 'right':
                    self.fluct = self.rates[:,(self.trial_type_labels==1)] - full_rates[:,(full_trial_type_labels==1)].mean(1, keepdims=True)

        if label_cond is not None:
            assert label_cond in [0, 1]
            self.trial_mask = self.labels==label_cond
            self.rates = self.rates[:,self.trial_mask]
            self.labels = self.labels[self.trial_mask]
            self.trial_type_labels = self.trial_type_labels[self.trial_mask]


    def __getitem__(self, i):
        if self.fluct_type is None:
            return self.rates[:,i], self.labels[i], self.trial_type_labels[i]
        else:
            return self.fluct[:,i], self.fluct[:,i], self.fluct[:,i]

    def __len__(self):
        if self.fluct_type is None:
            return self.rates.shape[1]
        else:
            return self.fluct.shape[1]

class ALMDatasetTest(data.Dataset):
    def __init__(self, filename, data_prefix, stim_type, data_type, root_path=None, neural_unit_location=None, loc_name='both', bin_min=None, bin_max=None):
        split_filename = filename.split('\\')
        save_path = os.path.join(data_prefix + split_filename[0], split_filename[1][:-4])

        if root_path is not None:
            save_path = os.path.join(root_path, save_path)

        self.stim_type = stim_type
        self.data_type = data_type
        self.neural_unit_location = neural_unit_location
        self.loc_name = loc_name
        self.bin_min = bin_min
        self.bin_max = bin_max

        if loc_name != 'both':
            assert self.neural_unit_location is not None
            self.unit_mask = (neural_unit_location==loc_name)


        
        self.rates = None
        self.labels = None
        self.trial_type_labels = None
        self.i_good = None


        i = stim_type
        data_dir = os.path.join(save_path, data_type + str(i)) # data_type: train or test. stim_type: 0 thru 4

        if bin_min is not None and bin_max is not None:
            rates = np.load(os.path.join(data_dir, 'rates.npy'))[bin_min:bin_max+1].astype(np.float32)

        elif bin_min is not None and bin_max is None:
            rates = np.load(os.path.join(data_dir, 'rates.npy'))[bin_min:].astype(np.float32)

        elif bin_min is None and bin_max is not None:
            rates = np.load(os.path.join(data_dir, 'rates.npy'))[:bin_max+1].astype(np.float32)

        elif bin_min is None and bin_max is None:
            rates = np.load(os.path.join(data_dir, 'rates.npy')).astype(np.float32)


        if loc_name != 'both':
            rates = rates[...,self.unit_mask]
            
        labels = np.load(os.path.join(data_dir, 'labels.npy'))
        trial_type_labels = np.load(os.path.join(data_dir, 'trial_type_labels.npy'))
        i_good = np.load(os.path.join(data_dir, 'i_good.npy'))


        if self.rates is None:
            self.rates = rates
            self.labels = labels
            self.trial_type_labels = trial_type_labels
            self.i_good = i_good
        else:
            self.rates = np.concatenate([self.rates, rates], 1)
            self.labels = np.concatenate([self.labels, labels], 0)
            self.trial_type_labels = np.concatenate([self.trial_type_labels, trial_type_labels], 0)
            self.i_good = np.concatenate([self.i_good, i_good], 0)



    def __len__(self):
        return self.rates.shape[1]

    def __getitem__(self, i):
        return self.rates[:,i], self.labels[i], self.trial_type_labels[i]


class ALMDatasetSimple(data.Dataset):
    def __init__(self, filename, data_prefix, stim_type, data_type, root_path=None, neural_unit_location=None, loc_name='both', bin_min=None, bin_max=None):
        split_filename = filename.split('\\')
        save_path = os.path.join(data_prefix + split_filename[0], split_filename[1][:-4])

        if root_path is not None:
            save_path = os.path.join(root_path, save_path)

        self.stim_type = stim_type
        self.data_type = data_type
        self.neural_unit_location = neural_unit_location
        self.loc_name = loc_name
        self.bin_min = bin_min
        self.bin_max = bin_max

        if loc_name != 'both':
            assert self.neural_unit_location is not None
            self.unit_mask = (neural_unit_location==loc_name)

        trainlist = list(range(5))
        trainlist.remove(stim_type)
        
        self.rates = None
        self.labels = None
        self.trial_type_labels = None
        self.i_good = None

        for i in trainlist:


            data_dir = os.path.join(save_path, data_type + str(i)) # data_type: train or test. stim_type: 0 thru 4

            if bin_min is not None and bin_max is not None:
                rates = np.load(os.path.join(data_dir, 'rates.npy'))[bin_min:bin_max+1].astype(np.float32)

            elif bin_min is not None and bin_max is None:
                rates = np.load(os.path.join(data_dir, 'rates.npy'))[bin_min:].astype(np.float32)

            elif bin_min is None and bin_max is not None:
                rates = np.load(os.path.join(data_dir, 'rates.npy'))[:bin_max+1].astype(np.float32)

            elif bin_min is None and bin_max is None:
                rates = np.load(os.path.join(data_dir, 'rates.npy')).astype(np.float32)


            if loc_name != 'both':
                rates = rates[...,self.unit_mask]
            
            labels = np.load(os.path.join(data_dir, 'labels.npy'))
            trial_type_labels = np.load(os.path.join(data_dir, 'trial_type_labels.npy'))
            i_good = np.load(os.path.join(data_dir, 'i_good.npy'))


            if self.rates is None:
                self.rates = rates
                self.labels = labels
                self.trial_type_labels = trial_type_labels
                self.i_good = i_good
            else:
                self.rates = np.concatenate([self.rates, rates], 1)
                self.labels = np.concatenate([self.labels, labels], 0)
                self.trial_type_labels = np.concatenate([self.trial_type_labels, trial_type_labels], 0)
                self.i_good = np.concatenate([self.i_good, i_good], 0)



    def __len__(self):
        return self.rates.shape[1]

    def __getitem__(self, i):
        return self.rates[:,i], self.labels[i], self.trial_type_labels[i]

