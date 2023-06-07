from multiprocessing import allow_connection_pickling
import os, sys, time, argparse, pickle, math
sys.path.append('NeuralPopulationAnalysisUtils/Code/Python')
from neupop import Session as Session
from neupop import PopulationAnalysis as PA
from neupop import MultiSessionAnalysis as MS
from neupop import decoders as ld
from neupop import session_parsers as parsers

from alm_datasets import ALMDataset


import numpy as np
import numpy.lib.npyio as npyio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable

import matplotlib.pyplot as plt
from matplotlib import gridspec
from all_neurons_video_pred_functions import *
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
import pickle
import json
from collections import defaultdict

from scipy import stats

cat = np.concatenate

# key is filename, the elements of filenames list above, and the value is the session number Guang defined.
#filehandler = open('sess_num_guang_map.obj', 'rb')
#sess_num_map = pickle.load(filehandler)


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

def mouse_sort(x):
    # e.g. x = 'BAYLORGC[#]'
    return int(x[8:])




# Define CropOutPoles.
class CropOutPoles(object):
    
    def __init__(self):
        pass

    def __call__(self, img):

        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        w, h = img.size


        # Following Aiden's process_image in MiceBehavior/data.py

        return img.crop((0, int(h/6)+1, w, h))

    def __repr__(self):
        return self.__class__.__name__






class AllNeuronsRegFromVideosReconCvExp(object):
    def __init__(self):
        # Load pred_self.configs
        with open('all_neurons_reg_recon_configs.json','r') as read_file:
            self.configs = json.load(read_file)

        self.sess_type = self.configs['sess_type']

        self.n_cv = self.configs['n_cv']

        self.n_trial_types_list = range(2)
        self.n_loc_names_list = ['left_ALM', 'right_ALM']

        self.n_trial_types = 2


        # Model save path
        cnn_channel_str = '_'.join([str(i) for i in self.configs['_cnn_channel_list']])
        fc_layer_str = '_'.join([str(self.configs['_ln1_out']), str(self.configs['_ln2_out']), str(self.configs['_lstmHidden'])])

        if self.configs['img_type'] == 'jpg':
            img_type_str = ''
        elif self.configs['img_type'] == 'png':
            img_type_str = 'png_frames'

        if self.configs['neural_begin_time'] == -1.4 and self.configs['neural_end_time'] == -0.2 and self.configs['neural_skip_time'] == 0.2:
            neural_time_str = ''
        else:
            neural_time_str = 'begin_{:.1f}_end_{:.1f}_skip_{:.1f}'.format(self.configs['neural_begin_time'], self.configs['neural_end_time'], self.configs['neural_skip_time'])



        self.model_save_path = os.path.join(self.configs['models_cv_dir'], self.sess_type, 'fr_percentile_{}'.format(int(self.configs['fr_percentile'])),\
         img_type_str, neural_time_str, 'view_type_{}'.format(self.configs['view_type'][0]), 'model_name_{}'.format(self.configs['model_name']), \
            'cnn_channel_list_{}'.format(cnn_channel_str), 'fc_layer_{}'.format(fc_layer_str))



        # recon_save_path.
        if len(self.configs['models_cv_dir'].split('\\')) != 1:
            recon_dir = self.configs['models_cv_dir'].split('\\')
            recon_dir[-1] = 'all_neurons_reg_from_videos_recon_cv_results'
            recon_dir = '\\'.join(recon_dir)
        else:
            recon_dir = 'all_neurons_reg_from_videos_recon_cv_results'

        self.recon_save_path = os.path.join(recon_dir, self.sess_type, 'fr_percentile_{}'.format(int(self.configs['fr_percentile'])),\
         img_type_str, neural_time_str, 'view_type_{}'.format(self.configs['view_type'][0]), 'model_name_{}'.format(self.configs['model_name']), \
            'cnn_channel_list_{}'.format(cnn_channel_str), 'fc_layer_{}'.format(fc_layer_str))

        # define f1_list
        video_root_path = os.path.join(self.configs['video_data_dir'], 'png_frames')

        if self.sess_type == 'standard':
            self.f1_list = list(filter(lambda x:int(x[8:])<=50, os.listdir(video_root_path)))
        elif self.sess_type == 'full_reverse':
            self.f1_list = list(filter(lambda x:int(x[8:])>=79, os.listdir(video_root_path)))
        elif self.sess_type == 'rule_reverse':
            self.f1_list = list(filter(lambda x:int(x[8:])>=61 and int(x[8:])<=71, os.listdir(video_root_path)))
        elif self.sess_type == 'full_delay':
            self.f1_list = list(os.listdir(video_root_path))
        else:
            raise ValueError('invalid sess type.')


    def cor_mat(self):
        '''
        For each session, we order neurons in terms of their r2, and calculate their correlation matrix.
        the pairwise correlation is calculated as follows:
        1. For each trial type, calculate across-trial correlation at each time point, and then average over time.
        2. Then average correlation over trial types.
        We do this for control trials and bi stim trials.
        '''
        save_path = os.path.join(self.recon_save_path, 'plots', 'cor_mat')
        os.makedirs(save_path, exist_ok=True)

        sess_map, n_sess = self.get_sess_map()

        '''
        Load r2 to determine which neurons to sample
        '''
        r2_array = self.compute_r2()

        for sess_idx, (f1, f2) in enumerate(sess_map):

            print('')
            print('[{}/{}]-th Session'.format(sess_idx+1, n_sess))
            print('f1: {}'.format(f1))
            print('f2: {}'.format(f2))

            self.cor_mat_helper(f1, f2, r2_array[sess_idx], save_path)




    def cor_mat_helper(self, f1, f2, r2_array, save_path):
        '''
        r2_array[i]: (n_neurons)
        target(pred)[i]: (n_trials, T, n_neurons)
        '''
        plot_save_path = os.path.join(save_path, '_'.join([f1, f2]))
        os.makedirs(plot_save_path, exist_ok=True)



        '''
        We select up to top and bottom 10 percentile neurons.
        '''
        r2 = r2_array.mean(0)
        sorted_neuron_inds = np.argsort(r2)[::-1]


        '''
        Load control and perturbation trial neural traces.
        '''
        filename = os.path.join('NeuronalData', '_'.join([f1, f2]) + '.mat')
        '''
        neural_data_list[i] = (rates, trial_type_labels) for i's stim type. stim_type = ['no_stim', 'left', 'right', 'bi']
        '''

        # Choose timesteps to plot
        start_time = -1.6
        end_time = -0.2
        T = int((end_time-start_time)/0.1) + 1

        start_time_bi_stim = -0.7
        T_bi_stim = int(round((end_time-start_time_bi_stim)/0.1)) + 1

        start_bin_bi_stim = int(round((start_time_bi_stim - start_time)/0.1))

        neural_data_list, bin_centers = self.get_neural_data_from_filename(self.configs['data_prefix'], filename, start_time, end_time)

        n_stim_types = 4

        n_neurons = neural_data_list[0][0].shape[2]

        no_stim_cor_mat = np.zeros((n_neurons, n_neurons, self.n_trial_types, T))
        bi_stim_cor_mat = np.zeros((n_neurons, n_neurons, self.n_trial_types, T_bi_stim))


        for stim_idx, (rates, trial_type_labels) in enumerate(neural_data_list):

            if stim_idx == 0:
                rates = rates[...,sorted_neuron_inds]
                for i in self.n_trial_types_list:
                    no_stim_cor_mat[:,:,i] = self.compute_cor_mat(rates[:,(trial_type_labels==i)]) # (n_neurons, n_neurons, T)

            elif stim_idx == 3:
                rates = rates[...,sorted_neuron_inds]
                for i in self.n_trial_types_list:
                    bi_stim_cor_mat[:,:,i] = self.compute_cor_mat(rates[start_bin_bi_stim:,(trial_type_labels==i)]) # (n_neurons, n_neurons, T)

            else:
                continue


        '''
        Plot
        '''
        fig = plt.figure(figsize=(40, 20))

        gs = gridspec.GridSpec(1,2, wspace=0.4, hspace=0.5)

        no_stim_cor_mat_avg_abs = np.abs(no_stim_cor_mat.mean(-1).mean(-1)) # (n_neurons, n_neurons)
        bi_stim_cor_mat_avg_abs = np.abs(bi_stim_cor_mat.mean(-1).mean(-1)) # (n_neurons, n_neurons)

        ax = fig.add_subplot(gs[0])
        ax.set_title('Control trials', fontsize=22)
        ax.imshow(no_stim_cor_mat_avg_abs, vmin=0, vmax=1, cmap='Reds')

        ax = fig.add_subplot(gs[1])
        ax.set_title('Bi stim trials', fontsize=22)
        ax.imshow(bi_stim_cor_mat_avg_abs, vmin=0, vmax=1, cmap='Reds')

        fig.savefig(os.path.join(plot_save_path, 'cor_mat.png'))


    def compute_cor_mat(self, rates):
        '''
        Args:
        rates: (T, n_trials, n_neurons)

        Return:
        cor_mat: (n_neurons, n_neurons, T)
        '''
        T, n_trials, n_neurons = rates.shape

        cor_mat = np.zeros((n_neurons, n_neurons, T))

        for t in range(T):
            cur_rates = rates[t]
            '''
            For some pairs of neurons, the std of either neuron is zero. In this case, corr is nan, and we want to remove such neurons.
            '''
            temp = np.corrcoef(cur_rates.T)
            temp[np.isnan(temp)] = 0
            cor_mat[...,t] = temp


        return cor_mat




    def sample_neural_traces(self):
        '''
        For each session, we take N sample neurons, and for each of them, we have one panel comparing target and predicted neural time traces,
        and three panels comparing control-trial and perturbation-trial trajectories.
        '''
        save_path = os.path.join(self.recon_save_path, 'plots', 'sample_neural_traces')
        os.makedirs(save_path, exist_ok=True)


        with open(os.path.join(self.recon_save_path, 'compute_pred', 'cv_agg', 'cv_agg_sess_trial_type_wise_target.obj'), 'rb') as f:
            cv_agg_sess_trial_type_wise_target = pickle.load(f)

        with open(os.path.join(self.recon_save_path, 'compute_pred', 'cv_agg', 'cv_agg_sess_trial_type_wise_pred.obj'), 'rb') as f:
            cv_agg_sess_trial_type_wise_pred = pickle.load(f)

        sess_map, n_sess = self.get_sess_map()

        '''
        Load r2 to determine which neurons to sample
        '''
        r2_array = self.compute_r2()

        for sess_idx, (f1, f2) in enumerate(sess_map):

            print('')
            print('[{}/{}]-th Session'.format(sess_idx+1, n_sess))
            print('f1: {}'.format(f1))
            print('f2: {}'.format(f2))

            self.sample_neural_traces_helper(f1, f2, r2_array[sess_idx], \
                cv_agg_sess_trial_type_wise_target[f1][f2], cv_agg_sess_trial_type_wise_pred[f1][f2], save_path)



    def sample_neural_traces_helper(self, f1, f2, r2_array, target, pred, save_path):
        '''
        r2_array[i]: (n_neurons)
        target(pred)[i]: (n_trials, T, n_neurons)
        '''
        plot_save_path = os.path.join(save_path, '_'.join([f1, f2]))
        os.makedirs(plot_save_path, exist_ok=True)



        '''
        We select up to top and bottom 10 percentile neurons.
        '''
        r2 = r2_array.mean(0)

        top_mask = (r2 >= np.percentile(r2, q=90)) # bottom 10 percentile
        bottom_mask = (r2 <= np.percentile(r2, q=10)) # bottom 10 percentile

        top_inds = np.nonzero(top_mask)[0]
        bottom_inds = np.nonzero(bottom_mask)[0]

        # Sort top and bottom inds by r2. top_inds are sorted in descending order, bottom_inds in ascending order.
        top_inds = top_inds[np.argsort(r2[top_inds])[::-1]]
        bottom_inds = bottom_inds[np.argsort(r2[bottom_inds])]


        n_top_neurons = len(top_inds)
        n_bottom_neurons = len(bottom_inds)

        n_sample_neurons = n_top_neurons + n_bottom_neurons


        '''
        Load control and perturbation trial neural traces.
        '''
        filename = os.path.join('NeuronalData', '_'.join([f1, f2]) + '.mat')
        '''
        neural_data_list[i] = (rates, trial_type_labels) for i's stim type. stim_type = ['no_stim', 'left', 'right', 'bi']
        '''

        # Choose timesteps to plot
        start_time = -1.6
        end_time = -0.2
        T = int((end_time-start_time)/0.1) + 1


        neural_data_list, bin_centers = self.get_neural_data_from_filename(self.configs['data_prefix'], filename, start_time, end_time)

        n_stim_types = 4


        neural_traces_mean = np.zeros((n_sample_neurons, n_stim_types, self.n_trial_types, T))
        neural_traces_std = np.zeros((n_sample_neurons, n_stim_types, self.n_trial_types, T))

        for stim_idx, (rates, trial_type_labels) in enumerate(neural_data_list):

            # top neurons
            top_rates = rates[...,top_inds]
            for i in self.n_trial_types_list:
                neural_traces_mean[:n_top_neurons, stim_idx, i] = top_rates[:,(trial_type_labels==i)].mean(1).T 
                neural_traces_std[:n_top_neurons, stim_idx, i] = top_rates[:,(trial_type_labels==i)].std(1).T 


            # bottom neurons
            bottom_rates = rates[...,bottom_inds]
            for i in self.n_trial_types_list:
                neural_traces_mean[n_top_neurons:, stim_idx, i] = bottom_rates[:,(trial_type_labels==i)].mean(1).T 
                neural_traces_std[n_top_neurons:, stim_idx, i] = bottom_rates[:,(trial_type_labels==i)].std(1).T 



        '''
        Select top and bottom neurons in target and pred.
        '''
        target_mean = np.zeros((n_sample_neurons, self.n_trial_types, T))
        target_std = np.zeros((n_sample_neurons, self.n_trial_types, T))

        pred_mean = np.zeros((n_sample_neurons, self.n_trial_types, T))
        pred_std = np.zeros((n_sample_neurons, self.n_trial_types, T))


        for i in self.n_trial_types_list:

            # top neurons
            top_target = target[i][...,top_inds]
            target_mean[:n_top_neurons,i] = top_target.mean(0).T
            target_std[:n_top_neurons,i] = top_target.std(0).T

            top_pred = pred[i][...,top_inds]
            pred_mean[:n_top_neurons,i] = top_pred.mean(0).T
            pred_std[:n_top_neurons,i] = top_pred.std(0).T


            # bottom neurons
            bottom_target = target[i][...,bottom_inds]
            target_mean[n_top_neurons:,i] = bottom_target.mean(0).T
            target_std[n_top_neurons:,i] = bottom_target.std(0).T

            bottom_pred = pred[i][...,bottom_inds]
            pred_mean[n_top_neurons:,i]  = bottom_pred.mean(0).T
            pred_std[n_top_neurons:,i]  = bottom_pred.std(0).T




        '''
        Load neural_unit_location
        '''
        neural_unit_location = self.get_neural_unit_location_from_filename(self.configs['data_prefix'], filename)
        neural_unit_location_sample_neurons = np.zeros(n_sample_neurons, dtype=object)
        neural_unit_location_sample_neurons[:n_top_neurons] = neural_unit_location[top_inds]
        neural_unit_location_sample_neurons[n_top_neurons:] = neural_unit_location[bottom_inds]




        fig = plt.figure(figsize=(50,25))

        n_row = int(np.floor(np.sqrt(n_top_neurons))) + 1 if not (int(np.sqrt(n_top_neurons)))**2 == n_top_neurons else int(np.sqrt(n_top_neurons))
        n_col = n_row

        color = ['r', 'b']

        # For each of top and bottom neurons.
        gs = gridspec.GridSpec(1, 2, wspace=0.4, hspace=0.5)

        for k, neuron_type in enumerate(['highly video-predictable neurons', 'hardly video-predictable neurons']):

            ax = fig.add_subplot(gs[k])
            ax.set_title(neuron_type, fontsize=20)
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])

            sub_gs = gridspec.GridSpecFromSubplotSpec(n_row, n_col, subplot_spec=gs[k], wspace=0.4, hspace=0.5)

            if k == 0:
                cur_target_mean = target_mean[:n_top_neurons]
                cur_target_std = target_std[:n_top_neurons]
                
                cur_pred_mean = pred_mean[:n_top_neurons]
                cur_pred_std = pred_std[:n_top_neurons]

                cur_neural_traces_mean = neural_traces_mean[:n_top_neurons]
                cur_neural_traces_std = neural_traces_std[:n_top_neurons]
                cur_neural_unit_location = neural_unit_location_sample_neurons[:n_top_neurons]
                cur_n_neurons = n_top_neurons

            else:
                cur_target_mean = target_mean[n_top_neurons:]
                cur_target_std = target_std[n_top_neurons:]
                
                cur_pred_mean = pred_mean[n_top_neurons:]
                cur_pred_std = pred_std[n_top_neurons:]

                cur_neural_traces_mean = neural_traces_mean[n_top_neurons:]
                cur_neural_traces_std = neural_traces_std[n_top_neurons:]
                cur_neural_unit_location = neural_unit_location_sample_neurons[n_top_neurons:]
                cur_n_neurons = n_bottom_neurons

            for neuron_idx in range(cur_n_neurons):

                ax = fig.add_subplot(sub_gs[neuron_idx])
                if k == 0:
                    ax.set_title('top {} neuron / hemi: {}'.format(neuron_idx+1, cur_neural_unit_location[neuron_idx]))
                else:
                    ax.set_title('bottom {} neuron / hemi: {}'.format(neuron_idx+1, cur_neural_unit_location[neuron_idx]))

                ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])


                ssub_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=sub_gs[neuron_idx], wspace=0.4, hspace=0.5)

                for s in range(n_stim_types):

                    ax = fig.add_subplot(ssub_gs[s])

                    if s == 0:
                        '''
                        Plot time traces of target and pred in control trials.
                        '''
                        for i in self.n_trial_types_list:
                            cur_trace_mean0 = cur_target_mean[neuron_idx,i]
                            
                            cur_trace_mean1 = cur_pred_mean[neuron_idx,i]
                            cur_trace_std1 = cur_pred_std[neuron_idx,i]

                            ax.plot(bin_centers, cur_trace_mean0, color=color[i], ls='--')
                            ax.plot(bin_centers, cur_trace_mean1, color=color[i], ls='-')
                            ax.fill_between(bin_centers, cur_trace_mean1-cur_trace_std1, cur_trace_mean1+cur_trace_std1, color=color[i], alpha=0.3)
                    else:
                        '''
                        Plot time traces of control and pert trials.
                        '''
                        for i in self.n_trial_types_list:
                            cur_trace_mean0 = cur_neural_traces_mean[neuron_idx,0,i]
                            
                            cur_trace_mean1 = cur_neural_traces_mean[neuron_idx,s,i]
                            cur_trace_std1 = cur_neural_traces_std[neuron_idx,s,i]

                            ax.plot(bin_centers, cur_trace_mean0, color=color[i], ls='--')
                            ax.plot(bin_centers, cur_trace_mean1, color=color[i], ls='-')
                            ax.fill_between(bin_centers, cur_trace_mean1-cur_trace_std1, cur_trace_mean1+cur_trace_std1, color=color[i], alpha=0.3)

        fig.savefig(os.path.join(plot_save_path, 'sample_neural_traces.png'))






    def error_svd(self):
        '''
        We take the error = target - pred, and then perform SVD (not PCA, 
        because we are interested in variance as measured from origin (which is the error by defn), not from mean) on it.
        '''



        error_svd_save_path = os.path.join(self.recon_save_path, 'error_svd')
        os.makedirs(error_svd_save_path, exist_ok=True)


        '''
        Load target and pred.
        cv_agg_sess_trial_type_wise_target(or pred)[f1][f2][i]: (n_trials, T, n_neurons)
        cv_agg_sess_trial_type_wise_trial_idx[f1][f2][i]: (n_trials)
        '''


        with open(os.path.join(self.recon_save_path, 'compute_pred', 'cv_agg', 'cv_agg_sess_trial_type_wise_target.obj'), 'rb') as f:
            cv_agg_sess_trial_type_wise_target = pickle.load(f)

        with open(os.path.join(self.recon_save_path, 'compute_pred', 'cv_agg', 'cv_agg_sess_trial_type_wise_pred.obj'), 'rb') as f:
            cv_agg_sess_trial_type_wise_pred = pickle.load(f)



        sess_map, n_sess = self.get_sess_map()

        '''
        Do Error SVD.
        '''


        '''
        sess_trial_type_wise_error_svd_weight[sess_idx,i]: (n_neurons, n_comp)
        sess_trial_type_wise_error_svd_cor[sess_idx,i]: (n_comp), where n_comp = n_neurons

        '''
        if os.path.isfile(os.path.join(error_svd_save_path, 'sess_trial_type_wise_error_svd_weight.npy')) and True:

            sess_trial_type_wise_error_svd_weight = np.load(os.path.join(error_svd_save_path, 'sess_trial_type_wise_error_svd_weight.npy'), allow_picke=True)
            sess_trial_type_wise_error_svd_r2 = np.load(os.path.join(error_svd_save_path, 'sess_trial_type_wise_error_svd_r2.npy'), allow_picke=True)

            sess_trial_type_wise_error_svd_target_proj = np.load(os.path.join(error_svd_save_path, 'sess_trial_type_wise_error_svd_target_proj.npy'), allow_picke=True)
            sess_trial_type_wise_error_svd_pred_proj = np.load(os.path.join(error_svd_save_path, 'sess_trial_type_wise_error_svd_pred_proj.npy'), allow_picke=True)


        else:

            sess_trial_type_wise_error_svd_weight = np.zeros((n_sess, self.n_trial_types), dtype=object)
            sess_trial_type_wise_error_svd_r2 = np.zeros((n_sess, self.n_trial_types), dtype=object)

            sess_trial_type_wise_error_svd_target_proj = np.zeros((n_sess, self.n_trial_types), dtype=object)
            sess_trial_type_wise_error_svd_pred_proj = np.zeros((n_sess, self.n_trial_types), dtype=object)


            for sess_idx, (f1, f2) in enumerate(sess_map):

                for i in self.n_trial_types_list:

                    print('')
                    print('[{}/{}]-th Session'.format(sess_idx+1, n_sess))
                    print('f1: {}'.format(f1))
                    print('f2: {}'.format(f2))
                    print('i: {}'.format(i))
                    print('')

                    cur_target = cv_agg_sess_trial_type_wise_target[f1][f2][i] # (n_trials, T, n_neurons)
                    cur_pred = cv_agg_sess_trial_type_wise_pred[f1][f2][i]

                    n_neurons = cur_target.shape[2]

                    cur_target = cur_target.reshape(-1, n_neurons)
                    cur_pred = cur_pred.reshape(-1, n_neurons)

                    cur_error = cur_target - cur_pred 

                    # Row vectors of vh
                    u, s, vh = np.linalg.svd(cur_error, full_matrices=False)

                    assert (s == np.sort(s)[::-1]).all()

                    '''
                    Store error_svd_weights.
                    We reverse the order so that the first component is the minimum error direction and so on.
                    '''
                    sess_trial_type_wise_error_svd_weight[sess_idx,i] = vh.T[:,::-1] # (n_neurons, n_comp)

                    sess_trial_type_wise_error_svd_target_proj[sess_idx,i] = cur_target.dot(vh.T[:,::-1]) # (n_samples, n_comp)
                    sess_trial_type_wise_error_svd_pred_proj[sess_idx,i] = cur_pred.dot(vh.T[:,::-1]) # (n_samples, n_comp)

                    '''
                    Compute r2.
                    '''
                    sess_trial_type_wise_error_svd_r2[sess_idx,i] = \
                    r2_score(sess_trial_type_wise_error_svd_target_proj[sess_idx,i], sess_trial_type_wise_error_svd_pred_proj[sess_idx,i], multioutput='raw_values') #(n_comp)

                    '''
                    r2_score gives 1 if y_true and y_pred are identical, and y_true has zero std. We set r2 to 0 in such cases.
                    '''

                    zero_r2_mask = np.isclose(np.var(sess_trial_type_wise_error_svd_target_proj[sess_idx,i], 0), 0, rtol=0, atol=1e-5) # (n_comp)
                    sess_trial_type_wise_error_svd_r2[sess_idx,i][zero_r2_mask] = 0



            np.save(os.path.join(error_svd_save_path, 'sess_trial_type_wise_error_svd_weight.npy'), sess_trial_type_wise_error_svd_weight)
            np.save(os.path.join(error_svd_save_path, 'sess_trial_type_wise_error_svd_r2.npy'), sess_trial_type_wise_error_svd_r2)

            np.save(os.path.join(error_svd_save_path, 'sess_trial_type_wise_error_svd_target_proj.npy'), sess_trial_type_wise_error_svd_target_proj)
            np.save(os.path.join(error_svd_save_path, 'sess_trial_type_wise_error_svd_pred_proj.npy'), sess_trial_type_wise_error_svd_pred_proj)


        '''
        Plot
        '''
        print('')
        print('')
        print('-----------------')
        print('Start plotting...')
        print('-----------------')
        print('')
        print('')




        if False:
            '''
            Bar graph of r2.
            '''
            fig = plt.figure(figsize=(30,30))
            n_row = int(np.floor(np.sqrt(n_sess))) + 1 if not (int(np.sqrt(n_sess)))**2 == n_sess else int(np.sqrt(n_sess))
            n_col = n_row

            gs = gridspec.GridSpec(n_row, n_col, wspace=0.4, hspace=0.5)


            for sess_idx, (f1, f2) in enumerate(sess_map):

                ax = fig.add_subplot(gs[sess_idx])
                ax.set_title('{}/{}'.format(f1, f2))

                cur_r2 = sess_trial_type_wise_error_svd_r2[sess_idx].mean(0)

                n_comp = len(cur_r2)
                ax.bar(np.arange(n_comp), cur_r2, width=0.5)
                ax.set_ylim(None,1)
                ax.axhline(0, ls='--', c='k')

            fig.savefig(os.path.join(error_svd_save_path, 'error_svd_r2_bar.png'))






        if True:
            '''
            Bar graph of r2.
            '''
            fig = plt.figure(figsize=(30,30))
            n_row = int(np.floor(np.sqrt(n_sess))) + 1 if not (int(np.sqrt(n_sess)))**2 == n_sess else int(np.sqrt(n_sess))
            n_col = n_row

            gs = gridspec.GridSpec(n_row, n_col, wspace=0.4, hspace=0.5)


            for sess_idx, (f1, f2) in enumerate(sess_map):

                ax = fig.add_subplot(gs[sess_idx])
                ax.set_title('{}/{}'.format(f1, f2))
                ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])

                sub_gs = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs[sess_idx], wspace=0.4, hspace=0.5)

                for i in self.n_trial_types_list:
                    # First col: train
                    ax = fig.add_subplot(sub_gs[i])
                    ax.set_title('trial type: {}'.format('L' if i == 0 else 'R'))

                    cur_target_proj = sess_trial_type_wise_error_svd_target_proj[sess_idx,i] # (n_samples, n_comp)
                    cur_pred_proj = sess_trial_type_wise_error_svd_pred_proj[sess_idx,i] # (n_samples, n_comp)

                    cur_mse = ((cur_target_proj-cur_pred_proj)**2).mean(0) # (n_comp)

                    n_comp = len(cur_mse)
                    ax.bar(np.arange(n_comp), cur_mse, width=0.5)

            fig.savefig(os.path.join(error_svd_save_path, 'error_svd_mse_bar.png'))




    def cca(self):
        train_frac = self.configs['cca_train_frac']


        cca_save_path = os.path.join(self.recon_save_path, 'cca', 'cca_train_frac_{:.2f}'.format(train_frac))
        os.makedirs(cca_save_path, exist_ok=True)


        '''
        Load target and pred.
        cv_agg_sess_trial_type_wise_target(or pred)[f1][f2][i]: (n_trials, T, n_neurons)
        cv_agg_sess_trial_type_wise_trial_idx[f1][f2][i]: (n_trials)
        '''


        with open(os.path.join(self.recon_save_path, 'compute_pred', 'cv_agg', 'cv_agg_sess_trial_type_wise_target.obj'), 'rb') as f:
            cv_agg_sess_trial_type_wise_target = pickle.load(f)

        with open(os.path.join(self.recon_save_path, 'compute_pred', 'cv_agg', 'cv_agg_sess_trial_type_wise_pred.obj'), 'rb') as f:
            cv_agg_sess_trial_type_wise_pred = pickle.load(f)

        with open(os.path.join(self.recon_save_path, 'compute_pred', 'cv_agg', 'cv_agg_sess_trial_type_wise_trial_idx.obj'), 'rb') as f:
            '''
            This is the index number appearing in f3: e.g. E3-BAYLORGC25-2018_09_17-13
            trial_idx = int(f3.split('-')[3])
            '''
            cv_agg_sess_trial_type_wise_trial_idx = pickle.load(f)


        '''
        Do CCA.
        '''
        n_sess = 0
        for f1 in sorted(cv_agg_sess_trial_type_wise_target.keys(), key=mouse_sort):
            # f1: BAYLORGC[#]
            for f2 in sorted(cv_agg_sess_trial_type_wise_target[f1].keys()):
                # f2: [yyyy]_[mm]_[dd]
                n_sess += 1

        sess_map = np.zeros(n_sess, dtype=object)

        sess_idx = 0

        for f1 in sorted(cv_agg_sess_trial_type_wise_target.keys(), key=mouse_sort):
            # f1: BAYLORGC[#]
            for f2 in sorted(cv_agg_sess_trial_type_wise_target[f1].keys()):
                # f2: [yyyy]_[mm]_[dd]

                sess_map[sess_idx] = (f1,f2)

                sess_idx += 1


        '''
        sess_trial_type_wise_cca_weight[sess_idx,i]: (n_neurons, n_comp)
        sess_trial_type_wise_cca_cor[sess_idx,i]: (n_comp), where n_comp = n_neurons

        '''
        if os.path.isfile(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_weight_target.npy')) and True:

            sess_trial_type_wise_cca_weight_target = np.load(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_weight_target.npy'), allow_picke=True)
            sess_trial_type_wise_cca_weight_pred = np.load(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_weight_pred.npy'), allow_picke=True)
            sess_trial_type_wise_cca_cor_train = np.load(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_cor_train.npy'), allow_picke=True)
            sess_trial_type_wise_cca_cor_test = np.load(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_cor_test.npy'), allow_picke=True)
            sess_trial_type_wise_cca_r2_train = np.load(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_r2_train.npy'), allow_picke=True)
            sess_trial_type_wise_cca_r2_test = np.load(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_r2_test.npy'), allow_picke=True)

            sess_trial_type_wise_cca_target_proj_train = np.load(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_target_proj_train.npy'), allow_picke=True)
            sess_trial_type_wise_cca_pred_proj_train = np.load(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_pred_proj_train.npy'), allow_picke=True)

            sess_trial_type_wise_cca_target_proj_test = np.load(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_target_proj_test.npy'), allow_picke=True)
            sess_trial_type_wise_cca_pred_proj_test = np.load(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_pred_proj_test.npy'), allow_picke=True)


        else:

            sess_trial_type_wise_cca_weight_target = np.zeros((n_sess, self.n_trial_types), dtype=object)
            sess_trial_type_wise_cca_weight_pred = np.zeros((n_sess, self.n_trial_types), dtype=object)
            sess_trial_type_wise_cca_cor_train = np.zeros((n_sess, self.n_trial_types), dtype=object)
            sess_trial_type_wise_cca_cor_test = np.zeros((n_sess, self.n_trial_types), dtype=object)
            # r2 along each cca component
            sess_trial_type_wise_cca_r2_train = np.zeros((n_sess, self.n_trial_types), dtype=object)
            sess_trial_type_wise_cca_r2_test = np.zeros((n_sess, self.n_trial_types), dtype=object)

            sess_trial_type_wise_cca_target_proj_train = np.zeros((n_sess, self.n_trial_types), dtype=object)
            sess_trial_type_wise_cca_pred_proj_train = np.zeros((n_sess, self.n_trial_types), dtype=object)

            sess_trial_type_wise_cca_target_proj_test = np.zeros((n_sess, self.n_trial_types), dtype=object)
            sess_trial_type_wise_cca_pred_proj_test = np.zeros((n_sess, self.n_trial_types), dtype=object)


            from sklearn.cross_decomposition import CCA
            from scipy.stats import pearsonr


            np.random.seed(123)

            for sess_idx, (f1, f2) in enumerate(sess_map):

                for i in self.n_trial_types_list:

                    print('')
                    print('[{}/{}]-th Session'.format(sess_idx+1, n_sess))
                    print('f1: {}'.format(f1))
                    print('f2: {}'.format(f2))
                    print('i: {}'.format(i))
                    print('')

                    cur_target = cv_agg_sess_trial_type_wise_target[f1][f2][i] # (n_trials, T, n_neurons)
                    cur_pred = cv_agg_sess_trial_type_wise_pred[f1][f2][i]

                    n_comp = cur_target.shape[2]

                    cur_target = cur_target.reshape(-1, n_comp)
                    cur_pred = cur_pred.reshape(-1, n_comp)

                    shuffled_inds = np.random.permutation(len(cur_target))
                    train_inds = shuffled_inds[:int(len(cur_target)*train_frac)]
                    test_inds = shuffled_inds[int(len(cur_target)*train_frac):]

                    cur_target_train = cur_target[train_inds]
                    cur_target_test = cur_target[test_inds]

                    cur_pred_train = cur_pred[train_inds]
                    cur_pred_test = cur_pred[test_inds]


                    cca = CCA(n_components=n_comp, scale=False)

                    cca.fit(cur_target_train, cur_pred_train)


                    '''
                    Store cca_weights.
                    '''
                    sess_trial_type_wise_cca_weight_target[sess_idx,i] = cca.x_rotations_ # (n_neurons, n_comp)
                    sess_trial_type_wise_cca_weight_pred[sess_idx,i] = cca.y_rotations_ # (n_neurons, n_comp)


                    # Note that this projs are centered (by mean of train_X and train_Y)!
                    # target_proj_train, pred_proj_train = cca.transform(cur_target_train, cur_pred_train) # (n_samples, n_comp)
                    # target_proj_test, pred_proj_test = cca.transform(cur_target_test, cur_pred_test) # (n_samples, n_comp)

                    '''
                    Since cca.transform returns centered (centered by mean of train_X and train_Y) projections, 
                    we will instead simply take the dot product of cur_target(pred) and cca._x(y)_rotations_
                    '''
                    target_proj_train, pred_proj_train = np.dot(cur_target_train, cca.x_rotations_), np.dot(cur_pred_train, cca.y_rotations_) # (n_samples, n_comp)
                    target_proj_test, pred_proj_test = np.dot(cur_target_test, cca.x_rotations_), np.dot(cur_pred_test, cca.y_rotations_) # (n_samples, n_comp)

                    sess_trial_type_wise_cca_target_proj_train[sess_idx,i] = target_proj_train
                    sess_trial_type_wise_cca_pred_proj_train[sess_idx,i] = pred_proj_train

                    sess_trial_type_wise_cca_target_proj_test[sess_idx,i] = target_proj_test
                    sess_trial_type_wise_cca_pred_proj_test[sess_idx,i] = pred_proj_test


                    # temp
                    # target_proj_train2, pred_proj_train2 = cca.transform(cur_target_train, cur_pred_train) # (n_samples, n_comp)
                    # target_proj_test2, pred_proj_test2 = cca.transform(cur_target_test, cur_pred_test) # (n_samples, n_comp)



                    # print('np.abs((target_proj_train - target_proj_train.mean(0)) - target_proj_train2).max()')
                    # print('{:.3e}'.format(np.abs((target_proj_train - target_proj_train.mean(0)) - target_proj_train2).max()))

                    # print('np.abs((pred_proj_train - pred_proj_train.mean(0)) - pred_proj_train2).max()')
                    # print('{:.3e}'.format(np.abs((pred_proj_train - pred_proj_train.mean(0)) - pred_proj_train2).max()))

                    # print('np.abs((target_proj_test - target_proj_train.mean(0)) - target_proj_test2).max()')
                    # print('{:.3e}'.format(np.abs((target_proj_test - target_proj_train.mean(0)) - target_proj_test2).max()))

                    # print('np.abs((pred_proj_test - pred_proj_train.mean(0)) - pred_proj_test2).max()')
                    # print('{:.3e}'.format(np.abs((pred_proj_test - pred_proj_train.mean(0)) - pred_proj_test2).max()))


                    # assert np.allclose(target_proj_train - target_proj_train.mean(0), target_proj_train2, atol=1e-5, rtol=1e-5)
                    # assert np.allclose(pred_proj_train - pred_proj_train.mean(0), pred_proj_train2, atol=1e-5, rtol=1e-5)
                    # # Note that cca.transform center X and Y by the mean of X_train and Y_train.
                    # assert np.allclose(target_proj_test - target_proj_train.mean(0), target_proj_test2, atol=1e-5, rtol=1e-5)
                    # assert np.allclose(pred_proj_test - pred_proj_train.mean(0), pred_proj_test2, atol=1e-5, rtol=1e-5)
                    # print('')
                    # print('assert all passed!')
                    # print('')
                    # temp


                    '''
                    Compute cca_cor.
                    '''
                    cca_cor_train = np.zeros(n_comp)
                    cca_cor_test = np.zeros(n_comp)
                    for k in range(n_comp):

                        # if np.allclose(target_proj_train[:,k].std(), 0):

                        #     print('')
                        #     print('component idx: {}'.format(k))
                        #     print('')
                        #     print('target_proj_train[:,k].std() is zero.')
                        #     print('')

                        # if np.allclose(pred_proj_train[:,k].std(), 0):

                        #     print('')
                        #     print('component idx: {}'.format(k))
                        #     print('')
                        #     print('pred_proj_train[:,k].std() is zero.')
                        #     print('')

                        # if np.allclose(target_proj_test[:,k].std(), 0):
                        #     print('')
                        #     print('component idx: {}'.format(k))
                        #     print('')
                        #     print('target_proj_test[:,k].std() is zero.')
                        #     print('')
                        
                        # if np.allclose(pred_proj_test[:,k].std(), 0):
                        #     print('')
                        #     print('component idx: {}'.format(k))
                        #     print('')
                        #     print('pred_proj_test[:,k].std() is zero.')
                        #     print('')


                        cca_cor_train[k], _ = pearsonr(target_proj_train[:,k], pred_proj_train[:,k])
                        cca_cor_test[k], _ = pearsonr(target_proj_test[:,k], pred_proj_test[:,k])


                    sess_trial_type_wise_cca_cor_train[sess_idx,i] = cca_cor_train
                    sess_trial_type_wise_cca_cor_test[sess_idx,i] = cca_cor_test


                    '''
                    Compute r2.
                    '''
                    sess_trial_type_wise_cca_r2_train[sess_idx,i] = r2_score(target_proj_train, pred_proj_train, multioutput='raw_values') #(n_comp)
                    sess_trial_type_wise_cca_r2_test[sess_idx,i] = r2_score(target_proj_test, pred_proj_test, multioutput='raw_values') #(n_comp)

                    '''
                    r2_score gives 1 if y_true and y_pred are identical, and y_true has zero std. We set r2 to 0 in such cases.
                    '''
                    zero_r2_mask_train = np.isnan(cca_cor_train)
                    sess_trial_type_wise_cca_r2_train[sess_idx,i][zero_r2_mask_train] = 0

                    zero_r2_mask_test = np.isnan(cca_cor_test)
                    sess_trial_type_wise_cca_r2_test[sess_idx,i][zero_r2_mask_test] = 0



            np.save(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_weight_target.npy'), sess_trial_type_wise_cca_weight_target)
            np.save(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_weight_pred.npy'), sess_trial_type_wise_cca_weight_pred)
            np.save(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_cor_train.npy'), sess_trial_type_wise_cca_cor_train)
            np.save(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_cor_test.npy'), sess_trial_type_wise_cca_cor_test)
            np.save(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_r2_train.npy'), sess_trial_type_wise_cca_r2_train)
            np.save(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_r2_test.npy'), sess_trial_type_wise_cca_r2_test)

            np.save(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_target_proj_train.npy'), sess_trial_type_wise_cca_target_proj_train)
            np.save(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_pred_proj_train.npy'), sess_trial_type_wise_cca_pred_proj_train)

            np.save(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_target_proj_test.npy'), sess_trial_type_wise_cca_target_proj_test)
            np.save(os.path.join(cca_save_path, 'sess_trial_type_wise_cca_pred_proj_test.npy'), sess_trial_type_wise_cca_pred_proj_test)


        '''
        Some checks
        '''

        # for sess_idx, (f1, f2) in enumerate(sess_map):

        #     for i in self.n_trial_types_list:

        #         print('')
        #         print('[{}/{}]-th Session'.format(sess_idx+1, n_sess))
        #         print('f1: {}'.format(f1))
        #         print('f2: {}'.format(f2))
        #         print('i: {}'.format(i))
        #         print('')

        #         n_trials, T, n_neurons = cv_agg_sess_trial_type_wise_target[f1][f2][i].shape

        #         print('n_samples: {}'.format(n_trials*T))
        #         print('n_neurons: {}'.format(n_neurons))

        '''
        Plot
        '''
        print('')
        print('')
        print('-----------------')
        print('Start plotting...')
        print('-----------------')
        print('')
        print('')


        if False:
            '''
            Bar graph of cca_cor.
            '''
            fig = plt.figure(figsize=(30,30))
            n_row = int(np.floor(np.sqrt(n_sess))) + 1 if not (int(np.sqrt(n_sess)))**2 == n_sess else int(np.sqrt(n_sess))
            n_col = n_row

            gs = gridspec.GridSpec(n_row, n_col)

            from scipy.stats import t
            one_sided_p = 0.05

            for sess_idx, (f1, f2) in enumerate(sess_map):

                ax = fig.add_subplot(gs[sess_idx])
                ax.set_title('{}/{}'.format(f1, f2))
                ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])

                sub_gs = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=gs[sess_idx])

                color = ['c', 'm']

                for k, data_type in enumerate(['train', 'test']):
                    for i in self.n_trial_types_list:
                        # First col: train
                        ax = fig.add_subplot(sub_gs[i,k])
                        ax.set_title('{} / {}'.format(data_type, 'L' if i == 0 else 'R'))

                        if k == 0:
                            cur_cor = sess_trial_type_wise_cca_cor_train[sess_idx,i]
                        elif k == 1:
                            cur_cor = sess_trial_type_wise_cca_cor_test[sess_idx,i]

                        # print('')
                        # print('[{}/{}]-th Session'.format(sess_idx+1, n_sess))
                        # print('f1: {}'.format(f1))
                        # print('f2: {}'.format(f2))
                        # print('i: {}'.format(i))
                        # print('data_type: {}'.format(data_type))
                        # print('n_neurons: {}'.format(len(cur_cor)))
                        # print(np.nonzero(np.isnan(cur_cor))[0])
                        # print('')

                        n_comp = len(cur_cor)
                        ax.bar(np.arange(n_comp), cur_cor, width=0.5, color=color[k])
                        ax.set_ylim(0,1)

                        '''
                        Show the critical correlation corresponding to p value of 5%
                        '''
                        n_trials, T, _ = cv_agg_sess_trial_type_wise_target[f1][f2][i].shape
                        if k == 0:
                            n_samples = int(n_trials*T*train_frac)
                        else:
                            n_samples = n_trials*T - int(n_trials*T*train_frac)

                        t_val = t.isf(one_sided_p, df=n_samples-2)

                        critical_cor = t_val/np.sqrt(n_samples - 2 + t_val**2)

                        ax.axhline(critical_cor, c='k', ls='--')






            fig.savefig(os.path.join(cca_save_path, 'cca_cor_bar.png'))



        if False:
            '''
            Bar graph of cca_r2.
            '''
            fig = plt.figure(figsize=(30,30))
            n_row = int(np.floor(np.sqrt(n_sess))) + 1 if not (int(np.sqrt(n_sess)))**2 == n_sess else int(np.sqrt(n_sess))
            n_col = n_row

            gs = gridspec.GridSpec(n_row, n_col)

            from scipy.stats import t
            one_sided_p = 0.05

            for sess_idx, (f1, f2) in enumerate(sess_map):

                ax = fig.add_subplot(gs[sess_idx])
                ax.set_title('{}/{}'.format(f1, f2))
                ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])

                sub_gs = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=gs[sess_idx])

                color = ['c', 'm']

                for k, data_type in enumerate(['train', 'test']):
                    for i in self.n_trial_types_list:
                        # First col: train
                        ax = fig.add_subplot(sub_gs[i,k])
                        ax.set_title('{} / {}'.format(data_type, 'L' if i == 0 else 'R'))

                        if k == 0:
                            cur_r2 = sess_trial_type_wise_cca_r2_train[sess_idx,i]
                        elif k == 1:
                            cur_r2 = sess_trial_type_wise_cca_r2_test[sess_idx,i]

                        # print('')
                        # print('[{}/{}]-th Session'.format(sess_idx+1, n_sess))
                        # print('f1: {}'.format(f1))
                        # print('f2: {}'.format(f2))
                        # print('i: {}'.format(i))
                        # print('data_type: {}'.format(data_type))
                        # print('n_neurons: {}'.format(len(cur_r2)))
                        # print(np.nonzero(np.isnan(cur_r2))[0])
                        # print('')

                        n_comp = len(cur_r2)
                        ax.bar(np.arange(n_comp), cur_r2, width=0.5, color=color[k])
                        ax.set_ylim(0,1)




            fig.savefig(os.path.join(cca_save_path, 'cca_r2_bar.png'))



        if True:
            comp_idx_list = [0,9]

            '''
            Scatter plots of target_proj and pred_proj in test set. 
            '''
            fig = plt.figure(figsize=(30,30))
            n_row = int(np.floor(np.sqrt(n_sess))) + 1 if not (int(np.sqrt(n_sess)))**2 == n_sess else int(np.sqrt(n_sess))
            n_col = n_row

            gs = gridspec.GridSpec(n_row, n_col)


            for sess_idx, (f1, f2) in enumerate(sess_map):

                ax = fig.add_subplot(gs[sess_idx])
                ax.set_title('{}/{}'.format(f1, f2))
                ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])

                sub_gs = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=gs[sess_idx])

                color = ['r', 'b']

                for i in self.n_trial_types_list:

                    for k, comp_idx in enumerate(comp_idx_list):

                        ax = fig.add_subplot(sub_gs[i,k])

                        cur_cor = sess_trial_type_wise_cca_cor_test[sess_idx,i][comp_idx]
                        cur_r2 = sess_trial_type_wise_cca_r2_test[sess_idx,i][comp_idx]

                        ax.set_title('{} / top {} / cor: {:.2f} / r2: {:.2f}'.format('L' if i == 0 else 'R', comp_idx+1, cur_cor, cur_r2))

                        cur_x = sess_trial_type_wise_cca_pred_proj_test[sess_idx,i][:,comp_idx]
                        cur_y = sess_trial_type_wise_cca_target_proj_test[sess_idx,i][:,comp_idx]
                        ax.scatter(cur_x, cur_y, color=color[i])

                        min_val = cat([cur_x, cur_y], 0).min()
                        max_val = cat([cur_x, cur_y], 0).max()
                        ax.plot([min_val, max_val], [min_val, max_val], c='k', ls='--')

            fig.savefig(os.path.join(cca_save_path, 'cca_scatter_plot_top_{}_top_{}.png'.format(comp_idx_list[0], comp_idx_list[1])))




    def compute_pred(self):
        n_trial_types_list = list(range(2))


        # Detect devices
        use_cuda = torch.cuda.is_available()                   # check if GPU exists
        device = torch.device("cuda:{}".format(self.configs['gpu_ids'][0]) if use_cuda else "cpu")   # use CPU or GPU
        print("USE_CUDA: " + str(use_cuda))

        # Data loading parameters
        params = {'batch_size': self.configs['bs'], 'shuffle': False, 'num_workers': self.configs['num_workers'], \
        'pin_memory': bool(self.configs['pin_memory'])} if use_cuda else {}

        # Collect video filenames.
        video_root_path = os.path.join(self.configs['video_data_dir'], 'png_frames')

        target_path = os.path.join(self.configs['neural_data_dir'], self.configs['data_prefix'] + 'NeuronalData')

        # Some basic checks
        if 'all_views' in self.configs['model_name']:
            assert len(self.configs['view_type']) == 2
            assert self.configs['view_type'][0] == 'side'
            assert self.configs['view_type'][1] == 'bottom'

        if 'static' not in self.configs['model_name']:
            assert self.configs['_input_channel'] == 1

        if 'downsample' not in self.configs['model_name']:
            assert self.configs['image_shape'][0] == 86
            assert self.configs['image_shape'][1] == 130
            assert len(self.configs['_maxPoolLayers']) == 2



        def mouse_sort(x):
            # e.g. x = 'BAYLORGC[#]'
            return int(x[8:])

        def video_trial_sort(x):
            # e.g. x = 'E3-BAYLORGC25-2018_09_17-13'
            return int(x.split('-')[3])




        '''
        Need to modify from here.
        Use no_stim_correct_i_good_trials.
        video_path_dict[f1][f2][i] will contain paths to videos for trials in no_stim_correct_i_good_trials[i].
        And we will separately split trials of each trial type into train and test set.
        '''



        video_path_dict = nestdict()
        for f1 in self.f1_list:
            # f1 = 'BAYLORGC[#]'
            if os.path.isfile(os.path.join(video_root_path, f1)):
                continue
            

            for f2 in os.listdir(os.path.join(video_root_path, f1)):
                # f2 = '[yyyy]_[mm]_[dd]'
                if os.path.isfile(os.path.join(video_root_path, f1, f2)):
                    print("F2 is file")
                    continue
                
                # We want to only select no stim trials.
                # CW: select all trials including incorrect

                no_stim_correct_i_good_trials = np.load(os.path.join(target_path, f1, f2, 'no_stim_i_good_trials.npy'), allow_pickle= True)

                for i in n_trial_types_list:
                    # Just a sanity check to make sure i_good_trials are sorted.
                    #assert (no_stim_correct_i_good_trials[i] == sorted(no_stim_correct_i_good_trials[i])).all()

                    video_path_dict[f1][f2][i] = []


                for f3 in os.listdir(os.path.join(video_root_path, f1, f2)):
                    # f3 = 'E3-BAYLORGC25-2018_09_17-13'
                    if os.path.isfile(os.path.join(video_root_path, f1, f2, f3)):
                        continue


                    for i in n_trial_types_list:
                        if int(f3.split('-')[3]) in no_stim_correct_i_good_trials[i]:
                            video_path_dict[f1][f2][i].append(f3)



                for i in n_trial_types_list:
                    # Sort cur_path_dict[f2] according to the trial index at the end.
                    video_path_dict[f1][f2][i].sort(key=video_trial_sort)




        # Figure out n_neurons.

        n_neurons_dict = nestdict()
        self.configs['max_n_neurons'] = 0

        for f1 in os.listdir(target_path):
            # f1 = 'BAYLORGC[#]'
            if os.path.isfile(os.path.join(target_path, f1)):
                continue

            # We do not want to include labels for mice where video data is not available.        
            if f1 not in video_path_dict.keys():
                continue

            for f2 in os.listdir(os.path.join(target_path, f1)):
                # f2 = '[yyyy]_[mm]_[dd]'
                if os.path.isfile(os.path.join(target_path, f1, f2)):
                    continue

                # We do not want to include labels for sessions where video data is not available.        
                if f2 not in video_path_dict[f1].keys():
                    continue


                with open(os.path.join(target_path, f1, f2, 'all_neurons_{}.obj'.format(self.configs['fr_percentile'])), 'rb') as read_file:
                    all_neurons = pickle.load(read_file)


                n_neurons_dict[f1][f2] = all_neurons[0].shape[2]

                if n_neurons_dict[f1][f2] > self.configs['max_n_neurons']:
                    self.configs['max_n_neurons'] = n_neurons_dict[f1][f2]

        n_neurons_dict = nestdict_to_dict(n_neurons_dict)


        print('')
        print('max_n_neurons: {}'.format(self.configs['max_n_neurons']))
        print('')


        # Collect target.

        # Needed below to figure out begin and end_bin.
        f1 = list(video_path_dict.keys())[0]
        f2 = list(video_path_dict[f1].keys())[0]
        bin_centers = np.load(os.path.join(target_path, f1, f2, 'bin_centers.npy'))

        # These are used for cur_all_neurons.
        begin_bin = np.argmin(np.abs(bin_centers - self.configs['neural_begin_time']))
        end_bin = np.argmin(np.abs(bin_centers - self.configs['neural_end_time']))

        bin_stride = 0.1
        neural_skip_bin = int(self.configs['neural_skip_time']//bin_stride)

        pred_times = bin_centers[begin_bin:end_bin+1:neural_skip_bin]


        target_dict = nestdict()

        for f1 in os.listdir(target_path):
            # f1 = 'BAYLORGC[#]'
            if os.path.isfile(os.path.join(target_path, f1)):
                continue

            # We do not want to include labels for mice where video data is not available.        
            if f1 not in video_path_dict.keys():
                continue

            for f2 in os.listdir(os.path.join(target_path, f1)):
                # f2 = '[yyyy]_[mm]_[dd]'
                if os.path.isfile(os.path.join(target_path, f1, f2)):
                    continue

                # We do not want to include labels for sessions where video data is not available.        
                if f2 not in video_path_dict[f1].keys():
                    continue


                with open(os.path.join(target_path, f1, f2, 'all_neurons_{}.obj'.format(int(self.configs['fr_percentile']))), 'rb') as read_file:
                    all_neurons = pickle.load(read_file)

                # Need to check if the trial numbers are equal between video and label data. If they are unequal, we use only trials present in the video data.
                no_stim_correct_i_good_trials = np.load(os.path.join(target_path, f1, f2, 'no_stim_i_good_trials.npy'), allow_pickle = True)

                for i in n_trial_types_list:

                    cur_all_neurons = all_neurons[i]

                    # We transpose cur_all_neurons so that its shape is (n_trials, T, n_comp).
                    cur_all_neurons = np.transpose(cur_all_neurons, (1,0,2))


                    cur_all_neurons = cur_all_neurons[:,begin_bin:end_bin+1:neural_skip_bin]


                    # Pad the neuron dimension with nan.
                    n_trials, T, n_neurons = cur_all_neurons.shape

                    assert n_neurons <= self.configs['max_n_neurons']
                    new_all_neurons = np.empty((n_trials, T, self.configs['max_n_neurons']))
                    new_all_neurons.fill(np.nan)
                    new_all_neurons[...,:n_neurons] = cur_all_neurons

                    cur_all_neurons = new_all_neurons

                    video_f3_list = video_path_dict[f1][f2][i]
                    if len(video_f3_list) != len(no_stim_correct_i_good_trials[i]):
                        '''
                        Usually happens when debugging.
                        '''
                        print('')
                        print('The number of trials in the video data is not equal to that in the neural data.')
                        print('We select only those trials present in the video data.')
                        print('')

                        targets_for_only_video_trials = []
                        for f3 in video_f3_list:
                            video_trial_idx = int(f3.split('-')[3])
                            video_trial_mask = (no_stim_correct_i_good_trials[i] == video_trial_idx)
                            targets_for_only_video_trials.append(cur_all_neurons[video_trial_mask])
                        target_dict[f1][f2][i] = cat(targets_for_only_video_trials, 0)

                    else:
                        target_dict[f1][f2][i] = cur_all_neurons




        video_path_dict = nestdict_to_dict(video_path_dict)
        target_dict = nestdict_to_dict(target_dict)



        # Determine n_sess and sess_inds for sess_cond model.
        self.configs['n_sess'] = 0
        sess_inds_dict = nestdict()
        for f1 in sorted(target_dict.keys(), key=mouse_sort):

            for f2 in sorted(target_dict[f1].keys()):
                self.configs['n_sess'] += 1
                for i in n_trial_types_list:
                    # n_sess -1 because sess_inds ranges from 0 to n_sess-1.
                    sess_inds_dict[f1][f2][i] = [self.configs['n_sess']-1]*len(target_dict[f1][f2][i])


        sess_inds_dict = nestdict_to_dict(sess_inds_dict)


        # Determine self.configs["n_neurons"]
        self.configs["n_neurons"] = np.zeros(self.configs['n_sess']).astype(int)
        for f1 in sorted(target_dict.keys(), key=mouse_sort):
            for f2 in sorted(target_dict[f1].keys()):
                cur_sess_ind = sess_inds_dict[f1][f2][0][0]
                self.configs["n_neurons"][cur_sess_ind] = n_neurons_dict[f1][f2]



        # We also want to evaluate r2 separately for each session.
        sess_wise_test_video_path_dict = nestdict()
        sess_wise_test_target_dict = nestdict()
        sess_wise_test_sess_inds_dict = nestdict()
        sess_wise_test_trial_type_dict = nestdict()

        for f1 in self.f1_list:
            # f1 = 'BAYLORGC[#]'
            for f2 in os.listdir(os.path.join(video_root_path, f1)):
                # f2 = '[yyyy]_[mm]_[dd]'

                for i in n_trial_types_list:

                    cur_video_path = np.array([os.path.join(f1,f2,f3) for f3 in video_path_dict[f1][f2][i]])
                    cur_target = target_dict[f1][f2][i]

                    cur_sess_inds = np.array(sess_inds_dict[f1][f2][i])
                    cur_trial_type = np.array([i]*len(cur_sess_inds))

                    kf = KFold(n_splits=self.n_cv, shuffle=True, random_state=1)

                    for cv_ind, (train_inds, test_inds) in enumerate(kf.split(cur_target)):


                        cur_train_video_path, cur_test_video_path = cur_video_path[train_inds], cur_video_path[test_inds]
                        cur_train_target, cur_test_target = cur_target[train_inds], cur_target[test_inds]
                        cur_train_sess_inds, cur_test_sess_inds = cur_sess_inds[train_inds], cur_sess_inds[test_inds]
                        cur_train_trial_type, cur_test_trial_type = cur_trial_type[train_inds], cur_trial_type[test_inds]

                        sess_wise_test_video_path_dict[cv_ind][f1][f2][i] = cur_test_video_path
                        sess_wise_test_target_dict[cv_ind][f1][f2][i] = cur_test_target
                        sess_wise_test_sess_inds_dict[cv_ind][f1][f2][i] = cur_test_sess_inds
                        sess_wise_test_trial_type_dict[cv_ind][f1][f2][i] = cur_test_trial_type



            
        '''
        Notes on transform:
        1. self.configs['image_shape'] is set to (height, width) = [86, 130], so that it has roughly the same aspect ratio as the cropped image, which has
        (height, width) = (266, 400).
        
        2. ToTensor normalizes the range 0~255 to 0~1.
        3. I am not normalizing the input by mean and std, just like Aiden, I believe.
        '''


        if len(self.configs['view_type']) == 1:
            transform_list = [transforms.Compose([transforms.Grayscale(),
                                            CropOutPoles(),
                                            transforms.Resize(self.configs['image_shape']),
                                            transforms.ToTensor()])]

        else:
            transform_side = transforms.Compose([transforms.Grayscale(),
                                            CropOutPoles(),
                                            transforms.Resize(self.configs['image_shape_side']),
                                            transforms.ToTensor()])

            transform_bottom = transforms.Compose([transforms.Grayscale(),
                                            transforms.Resize(self.configs['image_shape_bottom']),
                                            transforms.ToTensor()])

            # Assume that side always comes before bottom in view_type.
            transform_list = [transform_side, transform_bottom]


        # selected_frames include both begin and end_frame.
        n_frames = int(np.rint((self.configs['end_frame'] - self.configs['begin_frame'])/self.configs['skip_frame'])) + 1
        selected_frames = (self.configs['begin_frame'] + self.configs['skip_frame']*np.arange(n_frames))


        # Determine self.configs['pred_timesteps']
        selected_frame_times = selected_frames/1000*5 - 3.6

        pred_timesteps = []
        for pred_time in pred_times:
            pred_timesteps.append(np.argmin(np.abs(selected_frame_times - pred_time)))

        self.configs['pred_timesteps']  = np.array(pred_timesteps)



        # To deal with lrcn_neureg_bn_static2 etc.
        if '_input_channel' in self.configs.keys() and self.configs['_input_channel'] != 1:
            n_input_frames = self.configs['_input_channel']

            n_pre_frames = (n_input_frames-1)//2
            n_post_frames = (n_input_frames-1)//2

            if (n_input_frames-1)%2 != 0:
                n_pre_frames += 1

            # out of range for early times.
            if (self.configs['pred_timesteps'][0] - n_pre_frames) < 0:
                new_begin_frame = self.configs['begin_frame'] - self.configs['skip_frame']*n_pre_frames
            else:
                new_begin_frame = self.configs['begin_frame']

            # out of range for late times.
            if (self.configs['pred_timesteps'][-1] + n_post_frames) >= n_frames:
                new_end_frame = self.configs['end_frame'] + self.configs['skip_frame']*n_post_frames
            else:
                new_end_frame = self.configs['end_frame']


            n_frames = int(np.rint((new_end_frame - new_begin_frame)/self.configs['skip_frame'])) + 1
            selected_frames = (new_begin_frame + self.configs['skip_frame']*np.arange(n_frames))

            # Determine self.configs['pred_timesteps']
            selected_frame_times = selected_frames/1000*5 - 3.6

            pred_timesteps = []
            for pred_time in pred_times:
                pred_timesteps.append(np.argmin(np.abs(selected_frame_times - pred_time)))

            self.configs['pred_timesteps']  = np.array(pred_timesteps)




        # Iterate over cv_ind
        for cv_ind in range(self.n_cv):
            print('')
            print('')
            print('')
            print('------------------------------')
            print('------------------------------')
            print('{}-th Cross-Validation Begins!'.format(cv_ind+1))
            print('------------------------------')
            print('------------------------------')
            print('')
            print('')
            print('')

            self.compute_pred_cv_helper(cv_ind, video_root_path, n_trial_types_list,\
            sess_wise_test_video_path_dict[cv_ind], sess_wise_test_target_dict[cv_ind],\
            sess_wise_test_sess_inds_dict[cv_ind], sess_wise_test_trial_type_dict[cv_ind],\
            selected_frames, transform_list, params, mouse_sort, video_path_dict, target_dict, sess_inds_dict, device)


        # Also save pred that are aggregated over cross-validation folds.

        cv_agg_sess_trial_type_wise_target = nestdict()
        cv_agg_sess_trial_type_wise_pred = nestdict()
        cv_agg_sess_trial_type_wise_trial_idx = nestdict()


        for cv_ind in range(self.n_cv):
            recon_save_path = os.path.join(self.recon_save_path, 'compute_pred', 'cv_ind_{}'.format(cv_ind))

            with open(os.path.join(recon_save_path, 'sess_trial_type_wise_target.obj'), 'rb') as f:
                cur_cv_sess_trial_type_wise_target = pickle.load(f)

            with open(os.path.join(recon_save_path, 'sess_trial_type_wise_pred.obj'), 'rb') as f:
                cur_cv_sess_trial_type_wise_pred = pickle.load(f)

            with open(os.path.join(recon_save_path, 'sess_trial_type_wise_trial_idx.obj'), 'rb') as f:
                cur_cv_sess_trial_type_wise_trial_idx = pickle.load(f)


            for f1 in self.f1_list:
                # f1 = 'BAYLORGC[#]'
                for f2 in os.listdir(os.path.join(video_root_path, f1)):
                    # f2 = '[yyyy]_[mm]_[dd]'

                    for i in n_trial_types_list:

                        if cv_ind == 0:
                            cv_agg_sess_trial_type_wise_target[f1][f2][i] = cur_cv_sess_trial_type_wise_target[f1][f2][i]
                            cv_agg_sess_trial_type_wise_pred[f1][f2][i] = cur_cv_sess_trial_type_wise_pred[f1][f2][i]
                            cv_agg_sess_trial_type_wise_trial_idx[f1][f2][i] = cur_cv_sess_trial_type_wise_trial_idx[f1][f2][i]

                        else:
                            cv_agg_sess_trial_type_wise_target[f1][f2][i] \
                            = cat([cv_agg_sess_trial_type_wise_target[f1][f2][i], cur_cv_sess_trial_type_wise_target[f1][f2][i]], 0)

                            cv_agg_sess_trial_type_wise_pred[f1][f2][i] \
                            = cat([cv_agg_sess_trial_type_wise_pred[f1][f2][i], cur_cv_sess_trial_type_wise_pred[f1][f2][i]], 0)

                            cv_agg_sess_trial_type_wise_trial_idx[f1][f2][i] \
                            = cat([cv_agg_sess_trial_type_wise_trial_idx[f1][f2][i], cur_cv_sess_trial_type_wise_trial_idx[f1][f2][i]], 0)



        cv_agg_sess_trial_type_wise_target = nestdict_to_dict(cv_agg_sess_trial_type_wise_target)
        cv_agg_sess_trial_type_wise_pred = nestdict_to_dict(cv_agg_sess_trial_type_wise_pred)
        cv_agg_sess_trial_type_wise_trial_idx = nestdict_to_dict(cv_agg_sess_trial_type_wise_trial_idx)


        os.makedirs(os.path.join(self.recon_save_path, 'compute_pred', 'cv_agg'), exist_ok=True)

        with open(os.path.join(self.recon_save_path, 'compute_pred', 'cv_agg', 'cv_agg_sess_trial_type_wise_target.obj'), 'wb') as f:
            pickle.dump(cv_agg_sess_trial_type_wise_target, f)

        with open(os.path.join(self.recon_save_path, 'compute_pred', 'cv_agg', 'cv_agg_sess_trial_type_wise_pred.obj'), 'wb') as f:
            pickle.dump(cv_agg_sess_trial_type_wise_pred, f)

        with open(os.path.join(self.recon_save_path, 'compute_pred', 'cv_agg', 'cv_agg_sess_trial_type_wise_trial_idx.obj'), 'wb') as f:
            pickle.dump(cv_agg_sess_trial_type_wise_trial_idx, f)



        # Sanity check: check if there is any duplicate trial idx.
        for f1 in self.f1_list:
            # f1 = 'BAYLORGC[#]'
            for f2 in os.listdir(os.path.join(video_root_path, f1)):
                # f2 = '[yyyy]_[mm]_[dd]'

                for i in n_trial_types_list:

                    assert len(cv_agg_sess_trial_type_wise_trial_idx[f1][f2][i]) == len(np.unique(cv_agg_sess_trial_type_wise_trial_idx[f1][f2][i]))




    def compute_pred_cv_helper(self, cv_ind, video_root_path, n_trial_types_list,\
        sess_wise_test_video_path_dict, sess_wise_test_target_dict, \
        sess_wise_test_sess_inds_dict, sess_wise_test_trial_type_dict,\
        selected_frames, transform_list, params, mouse_sort, video_path_dict, target_dict, sess_inds_dict, device):

        model_save_path = os.path.join(self.model_save_path, 'cv_ind_{}'.format(cv_ind))
        recon_save_path = os.path.join(self.recon_save_path, 'compute_pred', 'cv_ind_{}'.format(cv_ind))

        # We also want to evaluate r2 separately for each session.
        sess_wise_valid_loader_dict = nestdict()

        for f1 in self.f1_list:
            # f1 = 'BAYLORGC[#]'

            for f2 in os.listdir(os.path.join(video_root_path, f1)):
                # f2 = '[yyyy]_[mm]_[dd]'
                for i in n_trial_types_list:

                    cur_test_video_path = sess_wise_test_video_path_dict[f1][f2][i]
                    cur_test_target = sess_wise_test_target_dict[f1][f2][i]
                    cur_test_sess_inds = sess_wise_test_sess_inds_dict[f1][f2][i]
                    cur_test_trial_type = sess_wise_test_trial_type_dict[f1][f2][i]

                    cur_valid_set = MyDatasetNeuRegRecon(video_root_path, cur_test_video_path, cur_test_target, \
                        cur_test_sess_inds, cur_test_trial_type, selected_frames, self.configs['view_type'], transform_list=transform_list,\
                         img_type=self.configs['img_type'])

                    sess_wise_valid_loader_dict[f1][f2][i] = data.DataLoader(cur_valid_set, **params)



        for key in sorted(self.configs.keys()):
            print('{}: {}'.format(key, self.configs[key]))
        print('')
        print('')


        import sys
        model = getattr(sys.modules[__name__], self.configs['model_name'])(self.configs).to(device)


        # Parallelize model to multiple GPUs
        if torch.cuda.device_count() > 1:
            if self.configs['gpu_ids'] is None:
                print("Using", torch.cuda.device_count(), "GPUs!")
                print('')
                model = nn.DataParallel(model)
            else:
                print("Using", len(self.configs['gpu_ids']), "GPUs!")
                print('')
                model = nn.DataParallel(model, device_ids=self.configs['gpu_ids'])



        # Load the saved model.
        
        model.load_state_dict(torch.load(os.path.join(model_save_path,'best_model.pth')))

        loss_fct = nn.MSELoss()

        # We also want to evaluate r2 separately for each session.
        sess_trial_type_wise_target = nestdict()
        sess_trial_type_wise_pred = nestdict()
        # trial_idx is given by int(f3.split('-')[3]).
        sess_trial_type_wise_trial_idx = nestdict()

        sess_trial_type_wise_best_test_scores = nestdict()


        for f1 in self.f1_list:
            # f1 = 'BAYLORGC[#]'

            for f2 in os.listdir(os.path.join(video_root_path, f1)):
                # f2 = '[yyyy]_[mm]_[dd]'

                for i in n_trial_types_list:

                    cur_valid_loader = sess_wise_valid_loader_dict[f1][f2][i]

                    # best_test_score is set to float('inf') so that we don't save the model.
                    print('Sess-specific validation results')
                    print('f1: ', f1)
                    print('f2: ', f2)
                    print('i: ', i)
                    sess_trial_type_wise_target[f1][f2][i], sess_trial_type_wise_pred[f1][f2][i], sess_trial_type_wise_trial_idx[f1][f2][i], \
                    sess_trial_type_wise_best_test_scores[f1][f2][i] = \
                    neureg_recon(self.configs, model, device, cur_valid_loader, loss_fct)



        sess_trial_type_wise_target = nestdict_to_dict(sess_trial_type_wise_target) # sess_trial_tye_wise_target[f1][f2][i] has shape (n_trials, T, n_comp)
        sess_trial_type_wise_pred = nestdict_to_dict(sess_trial_type_wise_pred) # sess_trial_tye_wise_target[f1][f2][i] has shape (n_trials, T, n_comp)
        sess_trial_type_wise_trial_idx = nestdict_to_dict(sess_trial_type_wise_trial_idx)
        
        sess_trial_type_wise_best_test_scores = nestdict_to_dict(sess_trial_type_wise_best_test_scores)



        # Save data.

        os.makedirs(recon_save_path, exist_ok=True)

        with open(os.path.join(recon_save_path, 'sess_trial_type_wise_target.obj'), 'wb') as f:
            pickle.dump(sess_trial_type_wise_target, f)

        with open(os.path.join(recon_save_path, 'sess_trial_type_wise_pred.obj'), 'wb') as f:
            pickle.dump(sess_trial_type_wise_pred, f)

        with open(os.path.join(recon_save_path, 'sess_trial_type_wise_trial_idx.obj'), 'wb') as f:
            pickle.dump(sess_trial_type_wise_trial_idx, f)





    def get_sess_map(self, return_n_sess=True):

        save_path = os.path.join(self.recon_save_path, 'get_sess_map')
        os.makedirs(save_path, exist_ok=True)

        if os.path.isfile(os.path.join(save_path, 'sess_map.npy')):

            sess_map = np.load(os.path.join(save_path, 'sess_map.npy'), allow_pickle=True)

        else:

            with open(os.path.join(self.recon_save_path, 'compute_pred', 'cv_agg', 'cv_agg_sess_trial_type_wise_target.obj'), 'rb') as f:
                cv_agg_sess_trial_type_wise_target = pickle.load(f)

            n_sess = 0
            sess_map = []
            for f1 in sorted(cv_agg_sess_trial_type_wise_target.keys(), key=mouse_sort):

                for f2 in sorted(cv_agg_sess_trial_type_wise_target[f1].keys()):

                    n_sess += 1
                    sess_map.append((f1,f2))

            sess_map = np.array(sess_map, dtype=object)


            np.save(os.path.join(save_path, 'sess_map.npy'), sess_map)


        if return_n_sess:
        
            return sess_map, len(sess_map)

        else:

            return sess_map

    def compute_r2(self):

        #save_path = os.path.join(self.recon_save_path, 'compute_r2')
        #os.makedirs(save_path, exist_ok=True)

        #if os.path.isfile(os.path.join(save_path, 'r2.npy')):
        #    r2_array = np.load(os.path.join(save_path, 'r2.npy'), allow_pickle = True)

        #else:
        print("HERE")

        with open(os.path.join(self.recon_save_path, 'compute_pred', 'cv_agg', 'cv_agg_sess_trial_type_wise_target.obj'), 'rb') as f:
            cv_agg_sess_trial_type_wise_target = pickle.load(f)

        with open(os.path.join(self.recon_save_path, 'compute_pred', 'cv_agg', 'cv_agg_sess_trial_type_wise_pred.obj'), 'rb') as f:
            cv_agg_sess_trial_type_wise_pred = pickle.load(f)

        #activity = np.concatenate((pred['BAYLORAT40']['201023'][0])) # , pred['BAYLORAT40']['201023'][1]))
        #predicted = activity.T[8].T

        #targetact = np.concatenate((target['BAYLORAT40']['201023'][0]).T[8].T # , target['BAYLORAT40']['201023'][1]))
        #targetact

        sess_map, n_sess = self.get_sess_map()

        #r2_array = np.zeros((n_sess, self.n_trial_types), dtype=object)
        r2_array = []
        #r2_array = []

        for sess_idx, (f1,f2) in enumerate(sess_map):
            cur_target = cat((cv_agg_sess_trial_type_wise_target[f1][f2][0], cv_agg_sess_trial_type_wise_target[f1][f2][1])).T
            cur_pred = cat((cv_agg_sess_trial_type_wise_pred[f1][f2][0], cv_agg_sess_trial_type_wise_pred[f1][f2][1])).T # n trials, T, neurons
            n_arr = []
            for n in range(cur_target.shape[0]): # iterate through each neuron in each session
                    n_arr += [r2_score(cat(cur_target[n].T), cat(cur_pred[n].T))]

                
            r2_array += [n_arr] # save for each session a list of r2 scores for each neuron
                

           
        #    for i in self.n_trial_types_list:

        #        cur_target = cv_agg_sess_trial_type_wise_target[f1][f2][i] # (n_trials, T, n_comp)
        #        cur_pred = cv_agg_sess_trial_type_wise_pred[f1][f2][i] # (n_trials, T, n_comp)

        #        n_trials, T, n_comp = cur_target.shape

        #        cur_r2 = r2_score(cur_target.reshape(n_trials, T*n_comp), cur_pred.reshape(n_trials, T*n_comp), multioutput='raw_values').reshape(T, n_comp)

        #        r2_array[sess_idx,i] = cur_r2.mean(0) # (n_comp)

        np.save(os.path.join('D:\\', 'r2.npy'), r2_array)



        return r2_array



    def r2_hist(self):


        with open(os.path.join(self.recon_save_path, 'compute_pred', 'cv_agg', 'cv_agg_sess_trial_type_wise_target.obj'), 'rb') as f:
            cv_agg_sess_trial_type_wise_target = pickle.load(f)

        with open(os.path.join(self.recon_save_path, 'compute_pred', 'cv_agg', 'cv_agg_sess_trial_type_wise_pred.obj'), 'rb') as f:
            cv_agg_sess_trial_type_wise_pred = pickle.load(f)

        sess_map, n_sess = self.get_sess_map()

        r2_array = self.compute_r2()


        #fig = plt.figure(figsize=(30, 20))
        #n_row = int(np.floor(np.sqrt(n_sess))) + 1 if not (int(np.sqrt(n_sess)))**2 == n_sess else int(np.sqrt(n_sess))
        #n_col = n_row

        #gs = gridspec.GridSpec(n_row, n_col, wspace=0.4, hspace=0.5)



        #for sess_idx, (f1,f2) in enumerate(sess_map):

        #    cur_scores = np.array(r2_array[sess_idx]).mean(0) # (n_trial_types, n_neurons) -> (n_neurons)

        #    ax = fig.add_subplot(gs[sess_idx])
        #    ax.hist(cur_scores)
        #    ax.axvline(0.0, ls='--', c='k')
            
        #    n_neurons = len(cur_scores)
        #    r2_pos_n_neurons = len(np.nonzero(cur_scores>0)[0])

        #    ax.set_title('{}/{}\n(max:{:.2f}, median:{:.2f}, min:{:.2f})\n(total n neurons: {}, r2-positive n neurons: {})'.format(f1, f2, \
        #        np.max(cur_scores), np.median(cur_scores), np.min(cur_scores), n_neurons, r2_pos_n_neurons), fontsize=14)
        #    ax.set_xlabel('r2_tc', fontsize=14)
        #    ax.set_ylabel('n_neurons', fontsize=14)


        #os.makedirs(os.path.join(self.recon_save_path, 'plots'), exist_ok=True)

        #fig.savefig(os.path.join(self.recon_save_path, 'plots', 'r2_hist.png'))






    def r2_hist_sep_trial_type_and_hemi(self):


        sess_map, n_sess = self.get_sess_map()

        r2_array = self.compute_r2()


        fig = plt.figure(figsize=(30, 30))
        n_row = int(np.floor(np.sqrt(n_sess))) + 1 if not (int(np.sqrt(n_sess)))**2 == n_sess else int(np.sqrt(n_sess))
        n_col = n_row

        gs = gridspec.GridSpec(n_row, n_col, wspace=0.4, hspace=0.5)

        color = ['m', 'c']

        for sess_idx, (f1,f2) in enumerate(sess_map):

            cur_filename = os.path.join('NeuronalData', '_'.join([f1, f2]) + '.mat')

            cur_neural_unit_location = self.get_neural_unit_location_from_filename(self.configs['data_prefix'], cur_filename)

            ax = fig.add_subplot(gs[sess_idx])
            ax.set_title('{}/{}'.format(f1, f2, fontsize=14))
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])

            sub_gs = gridspec.GridSpecFromSubplotSpec(2,1, wspace=0.4, hspace=0.5, subplot_spec=gs[sess_idx])

            for i in self.n_trial_types_list:

                ax = fig.add_subplot(sub_gs[i])

                cur_scores = r2_array[sess_idx,i] # (n_neurons)

                for j, loc_name in enumerate(self.n_loc_names_list):
                    ax.hist(cur_scores[cur_neural_unit_location==loc_name], color=color[j], label=loc_name, alpha=0.3)
                
                ax.axvline(0.0, ls='--', c='k')
                
                ax.set_xlabel('r2_tc ({})'.format('lick left' if i == 0 else 'lick right'), fontsize=14)
                ax.set_ylabel('n_neurons', fontsize=14)
                ax.legend()

        os.makedirs(os.path.join(self.recon_save_path, 'plots'), exist_ok=True)

        fig.savefig(os.path.join(self.recon_save_path, 'plots', 'r2_hist_sep_trial_type_and_hemi.png'))






    def r2_vs_neural_depth(self):
        r2_array = self.compute_r2() # (n_sess, n_trial_types)[n_comp]
        sess_map, n_sess = self.get_sess_map(return_n_sess=True)



        fig = plt.figure(figsize=(30, 20))
        n_row = int(np.floor(np.sqrt(n_sess))) + 1 if not (int(np.sqrt(n_sess)))**2 == n_sess else int(np.sqrt(n_sess))
        n_col = n_row

        gs = gridspec.GridSpec(n_row, n_col, wspace=0.4, hspace=0.5)


        for sess_idx, (f1,f2) in enumerate(sess_map):

            cur_r2 = np.stack(r2_array[sess_idx]).mean(0) # (n_neurons)

            '''
            Load neural_depth
            '''
            cur_filename = os.path.join('NeuronalData', '_'.join([f1, f2]) + '.mat')
            cur_neural_unit_depth = self.get_neural_unit_depth_from_filename(self.configs['data_prefix'], cur_filename)

            ax = fig.add_subplot(gs[sess_idx])
            ax.scatter(cur_neural_unit_depth, cur_r2, s=10)

            ax.set_xlabel('neural_depth', fontsize=14)
            ax.set_ylabel('r2', fontsize=14)

            ax.axhline(0, ls='--', c='k')

        os.makedirs(os.path.join(self.recon_save_path, 'plots'), exist_ok=True)

        fig.savefig(os.path.join(self.recon_save_path, 'plots', 'r2_vs_neural_depth.png'))




    def r2_vs_neuron_type(self):
        r2_array = self.compute_r2() # (n_sess, n_trial_types)[n_comp]
        sess_map, n_sess = self.get_sess_map(return_n_sess=True)



        fig = plt.figure(figsize=(30, 20))
        n_row = int(np.floor(np.sqrt(n_sess))) + 1 if not (int(np.sqrt(n_sess)))**2 == n_sess else int(np.sqrt(n_sess))
        n_col = n_row

        gs = gridspec.GridSpec(n_row, n_col, wspace=0.4, hspace=0.5)

        neuron_type_list = ['putative_pyramidal', 'putative_interneuron',  'unclassified']

        for sess_idx, (f1,f2) in enumerate(sess_map):

            cur_r2 = np.stack(r2_array[sess_idx]).mean(0) # (n_neurons)

            '''
            Load neural_depth
            '''
            cur_filename = os.path.join('NeuronalData', '_'.join([f1, f2]) + '.mat')
            cur_neural_unit_type = self.get_neural_unit_type_from_filename(self.configs['data_prefix'], cur_filename)

            ax = fig.add_subplot(gs[sess_idx])

            for k, neuron_type in enumerate(neuron_type_list):
                
                ax.scatter([k]*len(cur_r2[cur_neural_unit_type==neuron_type]), cur_r2[cur_neural_unit_type==neuron_type], s=10)

            ax.set_xticks(np.arange(len(neuron_type_list)))
            ax.set_xticklabels(neuron_type_list, fontsize=14)
            ax.set_ylabel('r2', fontsize=14)

            ax.axhline(0, ls='--', c='k')

        os.makedirs(os.path.join(self.recon_save_path, 'plots'), exist_ok=True)

        fig.savefig(os.path.join(self.recon_save_path, 'plots', 'r2_vs_neuron_type.png'))




    def get_neural_data_from_filename(self, data_prefix, filename, start_time, end_time):

        '''
        neural_data_list[i] = (rates, trial_type_labels) for i's stim type. stim_type = ['no_stim', 'left', 'right', 'bi']
        '''

        n_trial_types_list = range(2)
        loc_name_list = ['left_ALM', 'right_ALM']


        # Get bin_max and neural_unit_location.
        split_filename = filename.split('\\')
        prep_save_path = os.path.join(data_prefix + split_filename[0], split_filename[1][:-4])

        assert os.path.isfile(os.path.join(prep_save_path, 'culled_session.obj'))

        filehandler = open(os.path.join(prep_save_path, 'culled_session.obj'), 'rb')
        sess = pickle.load(filehandler)

        cw_path = os.path.join(prep_save_path, 'bin_centers.npy')

        bin_centers = npyio.load(cw_path)
        if 'Bin0.1' in data_prefix:
            bin_width = 0.1
        else:
            bin_width = 0.4



        samp_bin_min = np.abs((bin_centers-bin_width/2)-sess.task_pole_on_time.min()).argmin()
        if not np.isclose(((bin_centers-bin_width/2)-sess.task_pole_on_time.min())[samp_bin_min], 0, atol=1e-2) and ((bin_centers-bin_width/2)-sess.task_pole_on_time.min())[samp_bin_min] < 0:
            print('warning: np.isclose(((bin_centers-bin_width/2)-sess.task_pole_off_time.min())[bin_min], 0, atol=1e-2) is False.')
            samp_bin_min = samp_bin_min+1 

        samp_bin_max = np.abs((bin_centers+bin_width/2)-sess.task_pole_off_time.min()).argmin()
        if not np.isclose(((bin_centers+bin_width/2)-sess.task_pole_off_time.min())[samp_bin_max], 0, atol=1e-2) and ((bin_centers+bin_width/2)-sess.task_pole_off_time.min())[samp_bin_max] > 0:
            print('warning: np.isclose(((bin_centers+bin_width/2)-sess.task_cue_on_time.min())[bin_max], 0, atol=1e-2) is False.')
            samp_bin_max = samp_bin_max-1


        bin_min = np.abs((bin_centers-bin_width/2)-sess.task_pole_off_time.min()).argmin()
        if not np.isclose(((bin_centers-bin_width/2)-sess.task_pole_off_time.min())[bin_min], 0, atol=1e-2) and ((bin_centers-bin_width/2)-sess.task_pole_off_time.min())[bin_min] < 0:
            print('warning: np.isclose(((bin_centers-bin_width/2)-sess.task_pole_off_time.min())[bin_min], 0, atol=1e-2) is False.')
            bin_min = bin_min+1 


        bin_max = np.abs((bin_centers+bin_width/2)-sess.task_cue_on_time.min()).argmin()
        if not np.isclose(((bin_centers+bin_width/2)-sess.task_cue_on_time.min())[bin_max], 0, atol=1e-2) and ((bin_centers+bin_width/2)-sess.task_cue_on_time.min())[bin_max] > 0:
            print('warning: np.isclose(((bin_centers+bin_width/2)-sess.task_cue_on_time.min())[bin_max], 0, atol=1e-2) is False.')
            bin_max = bin_max-1


        neural_unit_location = sess.neural_unit_location.copy()

        stim_type_list = ['no_stim', 'left_stim', 'right_stim', 'bi_stim']

        neural_data_list = []
        
        # Compute start and end bin.
        start_bin = np.argmin(np.abs(bin_centers-start_time))
        end_bin = np.argmin(np.abs(bin_centers-end_time))

        for stim_type in stim_type_list:

            train_set = ALMDataset(filename, data_prefix=data_prefix, stim_type=stim_type, data_type='train', neural_unit_location=neural_unit_location, loc_name='both', bin_min=0, bin_max=len(bin_centers)-1)
            test_set = ALMDataset(filename, data_prefix=data_prefix, stim_type=stim_type, data_type='test', neural_unit_location=neural_unit_location, loc_name='both', bin_min=0, bin_max=len(bin_centers)-1)
            
            train_rates = train_set.rates
            test_rates = test_set.rates
            rates = np.concatenate([train_rates, test_rates], 1)


            train_trial_type_labels = train_set.trial_type_labels
            test_trial_type_labels = test_set.trial_type_labels
            trial_type_labels = np.concatenate([train_trial_type_labels, test_trial_type_labels], 0)

            train_labels = train_set.labels
            test_labels = test_set.labels
            labels = np.concatenate([train_labels, test_labels], 0)

            suc_labels = (trial_type_labels==labels).astype(int)


            '''
            Select relevant timesteps.
            '''
            rates = rates[start_bin:end_bin+1]

            neural_data_list.append((rates, trial_type_labels))

        return neural_data_list, bin_centers[start_bin:end_bin+1]




    def get_neural_unit_location_from_filename(self, data_prefix, filename):

        n_trial_types_list = range(2)
        loc_name_list = ['left_ALM', 'right_ALM']


        # Get bin_max and neural_unit_location.
        split_filename = filename.split('\\')
        prep_save_path = os.path.join(data_prefix + split_filename[0], split_filename[1][:-4])

        assert os.path.isfile(os.path.join(prep_save_path, 'culled_session.obj'))

        filehandler = open(os.path.join(prep_save_path, 'culled_session.obj'), 'rb')
        sess = pickle.load(filehandler)

        bin_centers = np.load(os.path.join(prep_save_path, 'bin_centers.npy'))
        if 'Bin0.1' in data_prefix:
            bin_width = 0.1
        else:
            bin_width = 0.4



        samp_bin_min = np.abs((bin_centers-bin_width/2)-sess.task_pole_on_time.min()).argmin()
        if not np.isclose(((bin_centers-bin_width/2)-sess.task_pole_on_time.min())[samp_bin_min], 0, atol=1e-2) and ((bin_centers-bin_width/2)-sess.task_pole_on_time.min())[samp_bin_min] < 0:
            print('warning: np.isclose(((bin_centers-bin_width/2)-sess.task_pole_off_time.min())[bin_min], 0, atol=1e-2) is False.')
            samp_bin_min = samp_bin_min+1 

        samp_bin_max = np.abs((bin_centers+bin_width/2)-sess.task_pole_off_time.min()).argmin()
        if not np.isclose(((bin_centers+bin_width/2)-sess.task_pole_off_time.min())[samp_bin_max], 0, atol=1e-2) and ((bin_centers+bin_width/2)-sess.task_pole_off_time.min())[samp_bin_max] > 0:
            print('warning: np.isclose(((bin_centers+bin_width/2)-sess.task_cue_on_time.min())[bin_max], 0, atol=1e-2) is False.')
            samp_bin_max = samp_bin_max-1


        bin_min = np.abs((bin_centers-bin_width/2)-sess.task_pole_off_time.min()).argmin()
        if not np.isclose(((bin_centers-bin_width/2)-sess.task_pole_off_time.min())[bin_min], 0, atol=1e-2) and ((bin_centers-bin_width/2)-sess.task_pole_off_time.min())[bin_min] < 0:
            print('warning: np.isclose(((bin_centers-bin_width/2)-sess.task_pole_off_time.min())[bin_min], 0, atol=1e-2) is False.')
            bin_min = bin_min+1 


        bin_max = np.abs((bin_centers+bin_width/2)-sess.task_cue_on_time.min()).argmin()
        if not np.isclose(((bin_centers+bin_width/2)-sess.task_cue_on_time.min())[bin_max], 0, atol=1e-2) and ((bin_centers+bin_width/2)-sess.task_cue_on_time.min())[bin_max] > 0:
            print('warning: np.isclose(((bin_centers+bin_width/2)-sess.task_cue_on_time.min())[bin_max], 0, atol=1e-2) is False.')
            bin_max = bin_max-1

        neural_unit_location = sess.neural_unit_location.copy()

        return neural_unit_location




    def get_neural_unit_depth_from_filename(self, data_prefix, filename):

        n_trial_types_list = range(2)
        loc_name_list = ['left_ALM', 'right_ALM']


        # Get bin_max and neural_unit_location.
        split_filename = filename.split('\\')
        prep_save_path = os.path.join(data_prefix + split_filename[0], split_filename[1][:-4])

        assert os.path.isfile(os.path.join(prep_save_path, 'culled_session.obj'))

        filehandler = open(os.path.join(prep_save_path, 'culled_session.obj'), 'rb')
        sess = pickle.load(filehandler)

        bin_centers = np.load(os.path.join(prep_save_path, 'bin_centers.npy'), allow_picke=True)
        if 'Bin0.1' in data_prefix:
            bin_width = 0.1
        else:
            bin_width = 0.4



        samp_bin_min = np.abs((bin_centers-bin_width/2)-sess.task_pole_on_time.min()).argmin()
        if not np.isclose(((bin_centers-bin_width/2)-sess.task_pole_on_time.min())[samp_bin_min], 0, atol=1e-2) and ((bin_centers-bin_width/2)-sess.task_pole_on_time.min())[samp_bin_min] < 0:
            print('warning: np.isclose(((bin_centers-bin_width/2)-sess.task_pole_off_time.min())[bin_min], 0, atol=1e-2) is False.')
            samp_bin_min = samp_bin_min+1 

        samp_bin_max = np.abs((bin_centers+bin_width/2)-sess.task_pole_off_time.min()).argmin()
        if not np.isclose(((bin_centers+bin_width/2)-sess.task_pole_off_time.min())[samp_bin_max], 0, atol=1e-2) and ((bin_centers+bin_width/2)-sess.task_pole_off_time.min())[samp_bin_max] > 0:
            print('warning: np.isclose(((bin_centers+bin_width/2)-sess.task_cue_on_time.min())[bin_max], 0, atol=1e-2) is False.')
            samp_bin_max = samp_bin_max-1


        bin_min = np.abs((bin_centers-bin_width/2)-sess.task_pole_off_time.min()).argmin()
        if not np.isclose(((bin_centers-bin_width/2)-sess.task_pole_off_time.min())[bin_min], 0, atol=1e-2) and ((bin_centers-bin_width/2)-sess.task_pole_off_time.min())[bin_min] < 0:
            print('warning: np.isclose(((bin_centers-bin_width/2)-sess.task_pole_off_time.min())[bin_min], 0, atol=1e-2) is False.')
            bin_min = bin_min+1 


        bin_max = np.abs((bin_centers+bin_width/2)-sess.task_cue_on_time.min()).argmin()
        if not np.isclose(((bin_centers+bin_width/2)-sess.task_cue_on_time.min())[bin_max], 0, atol=1e-2) and ((bin_centers+bin_width/2)-sess.task_cue_on_time.min())[bin_max] > 0:
            print('warning: np.isclose(((bin_centers+bin_width/2)-sess.task_cue_on_time.min())[bin_max], 0, atol=1e-2) is False.')
            bin_max = bin_max-1

        neural_unit_depth = sess.neural_unit_depth.copy()

        return neural_unit_depth


    def get_neural_unit_type_from_filename(self, data_prefix, filename):

        n_trial_types_list = range(2)
        loc_name_list = ['left_ALM', 'right_ALM']


        # Get bin_max and neural_unit_location.
        split_filename = filename.split('\\')
        prep_save_path = os.path.join(data_prefix + split_filename[0], split_filename[1][:-4])

        assert os.path.isfile(os.path.join(prep_save_path, 'culled_session.obj'))

        filehandler = open(os.path.join(prep_save_path, 'culled_session.obj'), 'rb')
        sess = pickle.load(filehandler)

        bin_centers = np.load(os.path.join(prep_save_path, 'bin_centers.npy'), allow_picke=True)
        if 'Bin0.1' in data_prefix:
            bin_width = 0.1
        else:
            bin_width = 0.4



        samp_bin_min = np.abs((bin_centers-bin_width/2)-sess.task_pole_on_time.min()).argmin()
        if not np.isclose(((bin_centers-bin_width/2)-sess.task_pole_on_time.min())[samp_bin_min], 0, atol=1e-2) and ((bin_centers-bin_width/2)-sess.task_pole_on_time.min())[samp_bin_min] < 0:
            print('warning: np.isclose(((bin_centers-bin_width/2)-sess.task_pole_off_time.min())[bin_min], 0, atol=1e-2) is False.')
            samp_bin_min = samp_bin_min+1 

        samp_bin_max = np.abs((bin_centers+bin_width/2)-sess.task_pole_off_time.min()).argmin()
        if not np.isclose(((bin_centers+bin_width/2)-sess.task_pole_off_time.min())[samp_bin_max], 0, atol=1e-2) and ((bin_centers+bin_width/2)-sess.task_pole_off_time.min())[samp_bin_max] > 0:
            print('warning: np.isclose(((bin_centers+bin_width/2)-sess.task_cue_on_time.min())[bin_max], 0, atol=1e-2) is False.')
            samp_bin_max = samp_bin_max-1


        bin_min = np.abs((bin_centers-bin_width/2)-sess.task_pole_off_time.min()).argmin()
        if not np.isclose(((bin_centers-bin_width/2)-sess.task_pole_off_time.min())[bin_min], 0, atol=1e-2) and ((bin_centers-bin_width/2)-sess.task_pole_off_time.min())[bin_min] < 0:
            print('warning: np.isclose(((bin_centers-bin_width/2)-sess.task_pole_off_time.min())[bin_min], 0, atol=1e-2) is False.')
            bin_min = bin_min+1 


        bin_max = np.abs((bin_centers+bin_width/2)-sess.task_cue_on_time.min()).argmin()
        if not np.isclose(((bin_centers+bin_width/2)-sess.task_cue_on_time.min())[bin_max], 0, atol=1e-2) and ((bin_centers+bin_width/2)-sess.task_cue_on_time.min())[bin_max] > 0:
            print('warning: np.isclose(((bin_centers+bin_width/2)-sess.task_cue_on_time.min())[bin_max], 0, atol=1e-2) is False.')
            bin_max = bin_max-1

        neural_unit_type = sess.neural_unit_type.copy()

        return neural_unit_type




    def r2_with_std_across_cv(self):

        sess_map, n_sess = self.get_sess_map(return_n_sess=True)

        r2_array = np.zeros((n_sess, self.n_cv, self.n_trial_types), dtype=object)


        for cv_ind in range(self.n_cv):
            recon_save_path = os.path.join(self.recon_save_path, 'compute_pred', 'cv_ind_{}'.format(cv_ind))

            with open(os.path.join(recon_save_path, 'sess_trial_type_wise_target.obj'), 'rb') as f:
                cur_cv_sess_trial_type_wise_target = pickle.load(f)

            with open(os.path.join(recon_save_path, 'sess_trial_type_wise_pred.obj'), 'rb') as f:
                cur_cv_sess_trial_type_wise_pred = pickle.load(f)



            for sess_idx, (f1,f2) in enumerate(sess_map):

                for i in self.n_trial_types_list:

                    cur_target = cur_cv_sess_trial_type_wise_target[f1][f2][i] # (n_trials, T, n_comp)
                    cur_pred = cur_cv_sess_trial_type_wise_pred[f1][f2][i] # (n_trials, T, n_comp)

                    n_trials, T, n_comp = cur_target.shape

                    cur_r2 = r2_score(cur_target.reshape(n_trials, T*n_comp), cur_pred.reshape(n_trials, T*n_comp), multioutput='raw_values').reshape(T, n_comp)

                    r2_array[sess_idx,cv_ind,i] = cur_r2.mean(0) # (n_comp)


        '''
        Plot
        '''

        fig = plt.figure(figsize=(30, 20))
        n_row = int(np.floor(np.sqrt(n_sess))) + 1 if not (int(np.sqrt(n_sess)))**2 == n_sess else int(np.sqrt(n_sess))
        n_col = n_row

        gs = gridspec.GridSpec(n_row, n_col, wspace=0.4, hspace=0.5)


        for sess_idx, (f1,f2) in enumerate(sess_map):

            cur_scores = r2_array[sess_idx].mean(1) # (n_cv, n_trial_types) -> (n_cv)

            print('')
            print(cur_scores.shape)
            print(cur_scores[0].shape)
            print('')

            sorted_inds = np.argsort(cur_scores.mean(0))[::-1] # (n_neurons)

            n_neurons = len(sorted_inds)
            r2_pos_n_neurons = len(np.nonzero(cur_scores.mean(0)>0)[0])


            ax = fig.add_subplot(gs[sess_idx])

            ax.errorbar(np.arange(n_neurons), cur_scores.mean(0)[sorted_inds], yerr=cur_scores.std(0, ddof=1)[sorted_inds], capsize=2)
            

            ax.set_title('{}/{}\n(max:{:.2f}, median:{:.2f}, min:{:.2f})\n(total n neurons: {}, r2-positive n neurons: {})'.format(f1, f2, \
                np.max(cur_scores.mean(0)), np.median(cur_scores.mean(0)), np.min(cur_scores.mean(0)), n_neurons, r2_pos_n_neurons), fontsize=14)

            ax.set_xlabel('neurons sorted', fontsize=14)
            ax.set_ylabel('r2_tc', fontsize=14)
            ax.set_ylim(None, 1.2)
            ax.axhline(0, ls='--', c='k')

        os.makedirs(os.path.join(self.recon_save_path, 'plots'), exist_ok=True)

        fig.savefig(os.path.join(self.recon_save_path, 'plots', 'r2_with_std_across_cv.png'))