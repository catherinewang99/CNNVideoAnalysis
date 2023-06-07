import os, sys, time, argparse, pickle, math, shutil
sys.path.append('NeuralPopulationAnalysisUtils/Code/Python')
from neupop import Session as Session
from neupop import PopulationAnalysis as PA
from neupop import MultiSessionAnalysis as MS
from neupop import decoders as ld
from neupop import session_parsers as parsers

from alm_datasets import ALMDataset


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable


import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt
from matplotlib import gridspec
from video_pred_functions import *
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
import json
from collections import defaultdict

from sklearn.metrics import r2_score

from scipy import stats

import pert_pred_utils

cat = np.concatenate

import matplotlib as mpl


'''
As of 10/08/20, I don't have sudo access,
and couldn't do "sudo apt install msttcorefonts -qq" and "rm ~/.cache/matplotlib -rf"
as described in my Evernote note (search for "sans-serif").

Without it, I don't have 'sans-serif' in font.family, at least in node 09.
'''

# mpl.rcParams['text.usetex'] = False
# mpl.rcParams['svg.fonttype'] = 'none'

# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = 'Arial'


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





# Modified from https://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph
def barplot_annotate_brackets(ax, num1, num2, data, center, height, yerr=None, y=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:

        # # * is p < 0.05
        # # ** is p < 0.005
        # # *** is p < 0.0005
        # # etc.
        # text = ''
        # p = .05

        # while data < p:
        #     text += '*'
        #     p /= 10.

        #     if maxasterix and len(text) == maxasterix:
        #         break

        '''
        The standard convention for asterisks is as follows:
        '''
        # * is p < 0.05
        # ** is p < 0.01
        # *** is p < 0.001
        # etc.
        if data < 0.001:
            text = '***'
        
        elif data < 0.01:
            text = '**'

        elif data < 0.05:
            text = '*'

        else:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = ax.get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    if y is None:
        y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    ax.text(*mid, text, **kwargs)



class PertPredFromVideosAnalysisExp(object):
    def __init__(self):
        # Load pred_self.configs
        with open('state_pred_from_videos_analysis_configs.json','r') as read_file:
            self.configs = json.load(read_file)

        self.sess_type = self.configs['sess_type']

        self.n_cv = self.configs['n_cv']

        self.n_trial_types_list = range(2)
        self.n_loc_names_list = ['left_ALM', 'right_ALM']

        self.n_trial_types = 2
        self.n_loc_names = 2

        self.init_model_and_analysis_save_paths()



    def init_model_and_analysis_save_paths(self):



        cnn_channel_str = '_'.join([str(i) for i in self.configs['_cnn_channel_list']])

        if self.configs['img_type'] == 'jpg':
            img_type_str = ''
        elif self.configs['img_type'] == 'png':
            img_type_str = 'png_frames'

        if self.configs['neural_begin_time'] == -1.4 and self.configs['neural_end_time'] == -0.2 and self.configs['neural_skip_time'] == 0.2:
            neural_time_str = ''
        else:
            neural_time_str = 'begin_{:.1f}_end_{:.1f}_skip_{:.1f}'.format(self.configs['neural_begin_time'], self.configs['neural_end_time'], self.configs['neural_skip_time'])


        # Note that we are not including the last 'cv_ind_{}'.format(cv_ind) here. It will be included in individual methods.
        self.model_save_path = os.path.join(self.configs['models_cv_dir'], self.configs['pert_type'], \
            'random_seed_{}'.format(self.configs['random_seed']), 'inst_trial_type_{}'.format(self.configs['inst_trial_type']), \
            self.configs['sess_type'], img_type_str, neural_time_str, 'view_type_{}'.format('_'.join(self.configs['view_type'])), \
            'model_name_{}'.format(self.configs['model_name']), 'cnn_channel_list_{}'.format(cnn_channel_str), \
            'bs_{}_epochs_{}'.format(self.configs['bs'], self.configs['epochs']), 'n_cv_{}'.format(self.configs['n_cv'])) 

        os.makedirs(self.model_save_path, exist_ok=True)


        # analysis_save_path.
        # model_cv_dir: /data5/bkang/bk_video_ALM/NeuronalData5_codes/cd_reg_from_videos_all_no_stim_trials_cv_frame_conversion_corrected_full_cv_models
        if len(self.configs['models_cv_dir'].split('\\')) != 1:
            analysis_dir = self.configs['models_cv_dir'].split('\\')
            analysis_dir[-1] = 'state_pred_from_videos_analysis_results'
            analysis_dir = '\\'.join(analysis_dir)
        else:
            analysis_dir = 'state_pred_from_videos_analysis_results'


        # Note that we are not including the last 'cv_ind_{}'.format(cv_ind) here. It will be included in individual methods.
        self.analysis_save_path = os.path.join(analysis_dir, self.configs['pert_type'], \
            'random_seed_{}'.format(self.configs['random_seed']), 'inst_trial_type_{}'.format(self.configs['inst_trial_type']), \
            self.configs['sess_type'], img_type_str, neural_time_str, 'view_type_{}'.format('_'.join(self.configs['view_type'])), \
            'model_name_{}'.format(self.configs['model_name']), 'cnn_channel_list_{}'.format(cnn_channel_str), \
            'bs_{}_epochs_{}'.format(self.configs['bs'], self.configs['epochs']), 'n_cv_{}'.format(self.configs['n_cv']))

        os.makedirs(self.analysis_save_path, exist_ok=True)


    def get_sess_specific_analysis_save_path(self, sess_type):

        cnn_channel_str = '_'.join([str(i) for i in self.configs['_cnn_channel_list']])

        if self.configs['img_type'] == 'jpg':
            img_type_str = ''
        elif self.configs['img_type'] == 'png':
            img_type_str = 'png_frames'

        if self.configs['neural_begin_time'] == -1.4 and self.configs['neural_end_time'] == -0.2 and self.configs['neural_skip_time'] == 0.2:
            neural_time_str = ''
        else:
            neural_time_str = 'begin_{:.1f}_end_{:.1f}_skip_{:.1f}'.format(self.configs['neural_begin_time'], self.configs['neural_end_time'], self.configs['neural_skip_time'])


        # analysis_save_path.
        # model_cv_dir: /data5/bkang/bk_video_ALM/NeuronalData5_codes/cd_reg_from_videos_all_no_stim_trials_cv_frame_conversion_corrected_full_cv_models
        if len(self.configs['models_cv_dir'].split('\\')) != 1:
            analysis_dir = self.configs['models_cv_dir'].split('\\')
            analysis_dir[-1] = 'state_pred_from_videos_analysis_results'
            analysis_dir = '\\'.join(analysis_dir)
        else:
            analysis_dir = 'state_pred_from_videos_analysis_results'

        data_load_path = os.path.join(analysis_dir, self.configs['pert_type'], \
            'random_seed_{}'.format(self.configs['random_seed']), 'inst_trial_type_{}'.format(self.configs['inst_trial_type']), \
            sess_type, img_type_str, neural_time_str, 'view_type_{}'.format('_'.join(self.configs['view_type'])), \
            'model_name_{}'.format(self.configs['model_name']), 'cnn_channel_list_{}'.format(cnn_channel_str), \
            'bs_{}_epochs_{}'.format(self.configs['bs'], self.configs['epochs']), 'n_cv_{}'.format(self.configs['n_cv']))


        return data_load_path



    def compute_pred(self, sess_type_list):

        for sess_type in sess_type_list:
            
            self.configs['sess_type'] = sess_type

            #for i in self.n_trial_types_list: #cw: commented out bc don't need 1

            self.configs['inst_trial_type'] = 0

            self.init_model_and_analysis_save_paths()

            self.compute_pred_for_each_inst_trial_type()



    def compute_pred_for_each_inst_trial_type(self):
        n_trial_types_list = self.n_trial_types_list



        '''
        ###
        To avoid having duplicate processes on gpu 0,
        I need to use CUDA_VISIBLE_DEVICES.
        Once I set visible devices, the gpu ids need to start from 0.
        ###
        '''
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(', '.join([str(x) for x in self.configs['gpu_ids_before_masking']]))
        print('')
        print('os.environ["CUDA_VISIBLE_DEVICES"]')
        print(os.environ["CUDA_VISIBLE_DEVICES"])
        print('')
        self.configs['gpu_ids'] = [x for x in range(len(self.configs['gpu_ids_before_masking']))]




        '''
        ###
        Set random seeds for determinism.
        ###
        '''
        np.random.seed(self.configs['random_seed'])
        torch.manual_seed(self.configs['random_seed'])






        self.configs['use_cuda'] = bool(self.configs['use_cuda'])
        self.configs['do_pos_weight'] = bool(self.configs['do_pos_weight'])

        if 'rnn' in self.configs['model_name']:
            self.configs['_input_channel'] = 1

        assert self.configs['inst_trial_type'] in [0, 1]
        
        # Create directories to save results.
        os.makedirs(self.configs['logs_cv_dir'], exist_ok=True)
        os.makedirs(self.configs['models_cv_dir'], exist_ok=True)

        # Detect devices
        use_cuda = torch.cuda.is_available() and self.configs['use_cuda']                  # check if GPU exists
        device = torch.device("cuda:{}".format(self.configs['gpu_ids'][0]) if use_cuda else "cpu")   # use CPU or GPU

        # Data loading parameters

        '''
        To use WeightedRandomSampler, we must set shuffle to False.
        '''
        params = {'batch_size': self.configs['test_bs'], 'shuffle': False, 'num_workers': self.configs['num_workers'], \
        'pin_memory': bool(self.configs['pin_memory']), 'drop_last': False}\
        if use_cuda else {'batch_size': self.configs['test_bs']}

        # Collect video filenames.
        if self.configs['img_type'] == 'jpg':
            video_root_path = os.path.join(self.configs['video_data_dir'], 'frames')
        elif self.configs['img_type'] == 'png':
            video_root_path = os.path.join(self.configs['video_data_dir'], 'png_frames')


        '''
        Session selection.
        Here, we define f1_f2_list.
        '''
        sess_type = self.configs['sess_type']

        #assert sess_type in ['{}_{}'.format(task_type, sort_type) for sort_type in ['guang_one_laser'] for task_type in ['original', 'full_reverse']]

        filehandler = open('sorted_filenames.obj', 'rb')
        filenames = pickle.load(filehandler)
        
        #sess_by_task_dict = pert_pred_utils.load_sess_by_task_dict()

        #print('')
        #print('<Before restricting to sessions that have videos>')
        #print('sess_type: ', sess_type)
        #print('n_sess: ', len(sess_by_task_dict[sess_type]))

        #pre_f1_f2_list = []
        #for sess_idx in sess_by_task_dict[sess_type]:
        #    sess_name = filenames[sess_idx-1].split('\\')[1][:-4].replace('kilosort', '') #BAYLORGC100_2020_01_21
        #    # f1 = sess_name.split('_')[0]
        #    # start = sess_name.find('_') + 1
        #    # f2 = sess_name[start:]
        #    pre_f1_f2_list.append(sess_name)

        #pre_f1_f2_list = np.array(pre_f1_f2_list)


        '''
        Load f1_f2_list, which contains tuples of form (f1,f2)
        '''
        if bool(self.configs['debug']):
            f1_f2_list = []
            for f1 in os.listdir(video_root_path):
                if os.path.isfile(os.path.join(video_root_path, f1)):
                    continue
                for f2 in os.listdir(os.path.join(video_root_path, f1)):

                    if os.path.isfile(os.path.join(video_root_path, f1, f2)):
                        continue

                    f1_f2_list.append((f1,f2))


        else:
            # if not os.path.isfile(os.path.join('make_f1_f2_list_results', sess_type, 'f1_f2_list.pkl')):
            #     import make_f1_f2_list
            #     make_f1_f2_list.main(sess_type)

            # Since I modify make_f1_f2_list from time to time, I want to generate f1_f2_list each time I run this code.
            import make_f1_f2_list
            #task_type = 'standard' if 'original' in sess_type else 'full_reverse'
            make_f1_f2_list.main(sess_type)

            with open(os.path.join('make_f1_f2_list_results', sess_type, 'f1_f2_list.pkl'), 'rb') as f:
                f1_f2_list = pickle.load(f)


        video_f1_f2_list = np.array(['_'.join([f1,f2]) for (f1,f2) in f1_f2_list])


        '''
        We take those sessions in pre_f1_f2_list that are in video_f1_f2_list.
        '''
        #f1_f2_list = pre_f1_f2_list[np.isin(pre_f1_f2_list, video_f1_f2_list)]

        f1_f2_list = video_f1_f2_list

        # sanity check
        # pre_f1_f2_set = set(list(pre_f1_f2_list))
        # video_f1_f2_set = set(list(video_f1_f2_list))
        # print('')
        # print('sanity check')
        # print('pre_f1_f2_set.difference(video_f1_f2_set)')
        # for x in list(pre_f1_f2_set.difference(video_f1_f2_set)):
        #     print(x)

        #print('')
        #print('sanity check')
        #print('<pre_f1_f2_list>')
        #print('n_sess: ', len(pre_f1_f2_list))
        #for x in pre_f1_f2_list:
        #    print(x)
        #print('')
        #print('<video_f1_f2_list>')
        #print('n_sess: ', len(video_f1_f2_list))
        #for x in video_f1_f2_list:
        #    print(x)
        #print('')
        #print('<Sessions in pre_f1_f2_list that are not in video_f1_f2_list>')
        #print('n_sess: ', len(pre_f1_f2_list[~np.isin(pre_f1_f2_list, video_f1_f2_list)]))
        #for x in pre_f1_f2_list[~np.isin(pre_f1_f2_list, video_f1_f2_list)]:
        #    print(x)



        print('')
        print('<After restricting to sessions that have videos>')
        print('sess_type: ', sess_type)
        print('n_sess: ', len(f1_f2_list))
        print('')


        f1_f2_list_old = f1_f2_list.copy()

        f1_f2_list = []
        for sess_name in f1_f2_list_old:
            # sess_name: BAYLORGC100_2020_01_21
            f1 = sess_name.split('_')[0]
            start = sess_name.find('_') + 1
            f2 = sess_name[start:]
            f1_f2_list.append((f1,f2))

        f1_f2_list = np.array(f1_f2_list)



        # if not self.configs['debug']:
        #     '''
        #     I found that certain sessions such as GC90_2019_11_18 don't have side view videos.
        #     I want to remove such sessions from f1_f2_list.
        #     '''
        #     if 'full_reverse' in sess_type:
        #         avi_root_path = '/mnt/fs4/bkang/alm_video_data/E3/avi_files'
        #     elif 'standard' in sess_type:
        #         # We know that sessions in /mnt/fs6/zwang71/MiceBehavior/data/E3/Session all have both view videos.
        #         avi_root_path = '/mnt/fs2/bkang/alm_video_data/E3/avi_files'


        #     print('')
        #     print('len(f1_f2_list) before checking for bad sess: {}'.format(len(f1_f2_list)))
        #     good_sess_idx = []

        #     for sess_idx, (f1, f2) in enumerate(f1_f2_list):
        #         print('')
        #         print('Checking f1: {} / f2: {}'.format(f1,f2))
                
        #         if f1 in ['BAYLORGC25', 'BAYLORGC26', 'BAYLORGC50']:
        #             good_sess_idx.append(sess_idx)
        #             continue

        #         # avi_file: GC50_20181211_bottom_trial_100-0000.avi
        #         if not any('side' in avi_file for avi_file in os.listdir(os.path.join(avi_root_path, f1, f2, 'Camera'))):
        #             print('')
        #             print('No side view in:')
        #             print('f1: {} / f2: {}'.format(f1, f2))
        #             continue

        #         if not any('bottom' in avi_file for avi_file in os.listdir(os.path.join(avi_root_path, f1, f2, 'Camera'))):
        #             print('')
        #             print('No bottom view in:')
        #             print('f1: {} / f2: {}'.format(f1, f2))
        #             continue

        #         good_sess_idx.append(sess_idx)


        #     f1_f2_list = f1_f2_list[good_sess_idx]

        #     print('')
        #     print('len(f1_f2_list) after checking for bad sess: {}'.format(len(f1_f2_list)))




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
        Collect video path.

        Use no_stim_i_good_trials.
        video_path_dict[f1][f2][i] will contain paths to videos for trials in no_stim_i_good_trials[i].
        And we will separately split trials of each trial type into train and test set.
        '''

        video_path_dict = nestdict()
        for f1, f2 in f1_f2_list:

            state_1_good_trials = pert_pred_utils.load_stim_i_good_trials_for_state_pred("state_1", self.configs['inst_trial_type'], self.configs['prep_root_dir'], f1, f2)
            state_2_good_trials = pert_pred_utils.load_stim_i_good_trials_for_state_pred("state_2", self.configs['inst_trial_type'], self.configs['prep_root_dir'], f1, f2)


            total_i_good_trials = np.array([state_1_good_trials, state_2_good_trials])


            for i in n_trial_types_list:
                # Just a sanity check to make sure i_good_trials are sorted.
                assert (total_i_good_trials[i] == sorted(total_i_good_trials[i])).all()

                video_path_dict[f1][f2][i] = []


            for f3 in os.listdir(os.path.join(video_root_path, f1, f2)):
                # f3 = 'E3-BAYLORGC25-2018_09_17-13'
                if os.path.isfile(os.path.join(video_root_path, f1, f2, f3)):
                    continue


                for i in n_trial_types_list:
                    if pert_pred_utils.map_f3_to_trial_idx(f1,f2,f3) in total_i_good_trials[i]:
                        video_path_dict[f1][f2][i].append(f3)


            for i in n_trial_types_list:
                # Sort cur_path_dict[f2] according to the trial index at the end.
                # Not necessary to sort it really, but let's do it anyways.
                video_path_dict[f1][f2][i].sort(key=video_trial_sort)




        '''
        Frame selection
        '''


        selected_frames_dict = nestdict()
        self.configs['pred_timesteps'] = None


        # Needed below to figure out begin and end_bin.
        f1 = list(video_path_dict.keys())[0]
        f2 = list(video_path_dict[f1].keys())[0]

        bin_centers = pert_pred_utils.load_bin_centers(self.configs['prep_root_dir'],f1,f2)


        # These are used for cur_cd_non_cd_pca.
        begin_bin = np.argmin(np.abs(bin_centers - self.configs['neural_begin_time']))
        end_bin = np.argmin(np.abs(bin_centers - self.configs['neural_end_time']))

        bin_stride = 0.1
        skip_bin = int(self.configs['neural_skip_time']//bin_stride)

        pred_times = bin_centers[begin_bin:end_bin+1:skip_bin]



        print('')
        print('Selecting frames for each session...')


        # Load n_frames_dict
        if not self.configs['debug']:
            save_path = 'get_n_frames_for_each_sess_results'
            if not os.path.isfile(os.path.join(save_path, 'n_frames_dict.pkl')):
                import get_n_frames_for_each_sess
                get_n_frames_for_each_sess.main()

            with open(os.path.join(save_path, 'n_frames_dict.pkl'), 'rb') as f:
                n_frames_dict = pickle.load(f)

        for f1, f2 in f1_f2_list:
            if not self.configs['debug']:
                cur_n_frames = n_frames_dict[(f1,f2)]
                cur_n_frames = 1000

            else:
                cur_n_frames = 1000

            if cur_n_frames == 1000:

                go_cue_time = 3.57
                begin_frame = int(np.rint((pred_times[0] + go_cue_time)*200)) # 200 frames per sec.
                end_frame = int(np.rint((pred_times[-1] + go_cue_time)*200))
                skip_frame = self.configs['skip_frame']

            elif cur_n_frames == 1200:
                go_cue_time = 4.57
                begin_frame = int(np.rint((pred_times[0] + go_cue_time)*200)) # 200 frames per sec.
                end_frame = int(np.rint((pred_times[-1] + go_cue_time)*200))
                skip_frame = self.configs['skip_frame']

            else:
                raise ValueError('Invalid cur_n_frames: {}. Needs to be either 1000 or 1200'.format(cur_n_frames))

            # Both of the returned variables are 1d np array.
            selected_frames, pred_timesteps = \
            pert_pred_utils.compute_selected_frames_and_pred_timesteps(self.configs, pred_times, begin_frame, end_frame, skip_frame, go_cue_time)

            # print(pred_times)
            # print(selected_frames)
            # print(pred_timesteps)

            if self.configs['pred_timesteps'] is None:
                self.configs['pred_timesteps'] = pred_timesteps
            else:
                # Even when n_frames is different across sessions, pred_timesteps should be the same.
                if (self.configs['pred_timesteps'] != pred_timesteps).any():
                    raise ValueError('self.configs pred_timesteps (={}) is \
                        NOT the same as current (f1:{}/f2:{}) pred_timesteps (={})'.format(\
                        ', '.join([str(x) for x in self.configs['pred_timesteps']]), f1, f2, ', '.join([str(x) for x in pred_timesteps])))

            for i in n_trial_types_list:
                selected_frames_dict[f1][f2][i] = np.broadcast_to(selected_frames, (len(video_path_dict[f1][f2][i]), len(selected_frames)))

            # print('')
            # print('f1: {} / f2: {}'.format(f1, f2))
            # print('n_frames: {}'.format(cur_n_frames))
            # print('go_cue_time: {}'.format(go_cue_time))
            # print('begin_frame: {}'.format(begin_frame))
            # print('end_frame: {}'.format(end_frame))
            # print('skip_frame: {}'.format(skip_frame))





        # Determine n_sess and sess_inds for sess_cond model.
        self.configs['n_sess'] = 0
        sess_inds_dict = nestdict()

        for f1, f2 in f1_f2_list:

            self.configs['n_sess'] += 1

            for i in n_trial_types_list:
                # n_sess -1 because sess_inds ranges from 0 to n_sess-1.
                sess_inds_dict[f1][f2][i] = [self.configs['n_sess']-1]*len(video_path_dict[f1][f2][i])










        '''
        Cross-validation train-test split.
        We determine the split using video trials, and then
        use it when generating neural data in which CD is computed only using the train set's correct trials.
        '''

        n_cv = self.configs['n_cv']

        sess_wise_test_video_path_dict = nestdict()
        sess_wise_test_sess_inds_dict = nestdict()
        sess_wise_test_trial_type_dict = nestdict()
        sess_wise_test_selected_frames_dict = nestdict()

        sess_wise_test_trial_idxs_dict = nestdict()
        '''
        I realized that for some sessions and inst_trial_types, there are very few 
        error trials, less than n_cv, which makes it impossible to perform a separate k-fold
        split for each inst_trial_type. To circumvent this issue, we first identify
        sessions in which at least one of inst_trial_types have error trials fewer
        than n_cv. For those sessions, we perform k-fold split for the entire set of 
        trials in a given inst_trial_type.
        '''
        few_error_trial_f1_f2_list = []


        for f1, f2 in f1_f2_list:
            # f1 = 'BAYLORGC[#]'
            # f2 = '[yyyy]_[mm]_[dd]'

            print('')
            print(f1, f2)

            for i in n_trial_types_list:
                cur_video_path = np.array([os.path.join(f1,f2,f3) for f3 in video_path_dict[f1][f2][i]]) # an array of each trial's video path
                
                print('{}'.format('no_stim' if i==0 else self.configs['pert_type']))
                print(len(cur_video_path))

                if len(cur_video_path) < n_cv:
                    few_error_trial_f1_f2_list.append((f1, f2))
                    break
        
        print('')
        print('few_error_trial_f1_f2_list')
        print('')
        for f1, f2 in few_error_trial_f1_f2_list:
            print(f1,f2)




        '''
        Here, only iterate through sessions that are not in few_error_trial_f1_f2_list.
        '''
        for f1, f2 in f1_f2_list:
            # f1 = 'BAYLORGC[#]'
            # f2 = '[yyyy]_[mm]_[dd]'
            if (f1,f2) in few_error_trial_f1_f2_list:
                continue

            for i in n_trial_types_list:

                # We wrap lists into arrays so that they can be indexed with advanced indexing.
                cur_video_path = np.array([os.path.join(f1,f2,f3) for f3 in video_path_dict[f1][f2][i]]) # an array of each trial's video path

                cur_sess_inds = np.array(sess_inds_dict[f1][f2][i]) # an array [sess_ind]*n_trials.
                cur_trial_type = np.array([i]*len(cur_sess_inds)) # an array [i]*n_trials.

                cur_selected_frames = selected_frames_dict[f1][f2][i] # np array of shape (n_trials, n_selected_frames)

                '''
                In most cases, map_f3_to_trial_idx is int(f3.split('-')[3]).But,
                there are a few exceptional sessions where this is not true.
                '''
                cur_trial_idxs = np.array([pert_pred_utils.map_f3_to_trial_idx(f1,f2,f3) for f3 in video_path_dict[f1][f2][i]]) # an array of each trial's video path


                kf = KFold(n_splits=n_cv, shuffle=True, random_state=1)

                for cv_ind, (train_inds, test_inds) in enumerate(kf.split(cur_video_path)):


                    cur_train_video_path, cur_test_video_path = cur_video_path[train_inds], cur_video_path[test_inds]
                    cur_train_sess_inds, cur_test_sess_inds = cur_sess_inds[train_inds], cur_sess_inds[test_inds]
                    cur_train_trial_type, cur_test_trial_type = cur_trial_type[train_inds], cur_trial_type[test_inds]
                    cur_train_selected_frames, cur_test_selected_frames = cur_selected_frames[train_inds], cur_selected_frames[test_inds]


                    sess_wise_test_video_path_dict[cv_ind][f1][f2][i] = cur_test_video_path
                    sess_wise_test_sess_inds_dict[cv_ind][f1][f2][i] = cur_test_sess_inds
                    sess_wise_test_trial_type_dict[cv_ind][f1][f2][i] = cur_test_trial_type
                    sess_wise_test_selected_frames_dict[cv_ind][f1][f2][i] = cur_test_selected_frames

                    sess_wise_test_trial_idxs_dict[cv_ind][f1][f2][i] = cur_trial_idxs[test_inds]




        '''
        ###
        Note added on 10/01/20 (while revisiting this code for Cell-revision):
        
        Note that in the for loop below for few_error_trial_f1_f2_list, the index i
        is fixed at 1 after the for loop for i, and thus various dict will have an empty
        entry for dict[cv_ind][f1][f2][0].

        This didn't cause any error afterwards because list's extend simply does not add
        any element from dict[cv_ind][f1][f2][0] (I confirmed this behavior), and I initially
        thought this would have a bad effect on learning. But, I realize that in fact, it
        doesn't affect learning at all, since we are going to concatenate dict[cv_ind][f1][f2][i]
        anyways for training and testing! The same holds for sess-wise validation set evaluation.
        ###
        '''


        for f1, f2 in few_error_trial_f1_f2_list:

            cur_video_path = []
            cur_sess_inds = []
            cur_trial_type = []
            cur_selected_frames = []
            cur_trial_idxs = []

            for i in n_trial_types_list:

                # We wrap lists into arrays so that they can be indexed with advanced indexing.
                cur_video_path.extend([os.path.join(f1,f2,f3) for f3 in video_path_dict[f1][f2][i]]) # an array of each trial's video path

                cur_sess_inds.extend(sess_inds_dict[f1][f2][i]) # an array [sess_ind]*n_trials.
                cur_trial_type.extend([i]*len(cur_sess_inds)) # an array [i]*n_trials.

                cur_selected_frames.extend(selected_frames_dict[f1][f2][i]) # np array of shape (n_trials, n_selected_frames)

                '''
                In most cases, map_f3_to_trial_idx is int(f3.split('-')[3]).But,
                there are a few exceptional sessions where this is not true.
                '''

                cur_trial_idxs.extend([pert_pred_utils.map_f3_to_trial_idx(f1,f2,f3) for f3 in video_path_dict[f1][f2][i]]) # an array of each trial's video path


            cur_video_path = np.array(cur_video_path)
            cur_sess_inds = np.array(cur_sess_inds)
            cur_trial_type = np.array(cur_trial_type)
            cur_selected_frames = np.array(cur_selected_frames)
            cur_trial_idxs = np.array(cur_trial_idxs)


            kf = KFold(n_splits=n_cv, shuffle=True, random_state=1)

            for cv_ind, (train_inds, test_inds) in enumerate(kf.split(cur_video_path)):


                cur_train_video_path, cur_test_video_path = cur_video_path[train_inds], cur_video_path[test_inds]
                cur_train_sess_inds, cur_test_sess_inds = cur_sess_inds[train_inds], cur_sess_inds[test_inds]
                cur_train_trial_type, cur_test_trial_type = cur_trial_type[train_inds], cur_trial_type[test_inds]
                cur_train_selected_frames, cur_test_selected_frames = cur_selected_frames[train_inds], cur_selected_frames[test_inds]


                sess_wise_test_video_path_dict[cv_ind][f1][f2][i] = cur_test_video_path
                sess_wise_test_sess_inds_dict[cv_ind][f1][f2][i] = cur_test_sess_inds
                sess_wise_test_trial_type_dict[cv_ind][f1][f2][i] = cur_test_trial_type
                sess_wise_test_selected_frames_dict[cv_ind][f1][f2][i] = cur_test_selected_frames

                sess_wise_test_trial_idxs_dict[cv_ind][f1][f2][i] = cur_trial_idxs[test_inds]






        '''
        ###
        NOTE:

        I realized that I can save some time by first combining all sessions data into a single data loader
        and then later dividing up the outputs of the network into different sessions.
        ###
        '''



        test_video_path = np.zeros(n_cv, dtype=object)
        
        test_sess_inds = np.zeros(n_cv, dtype=object)

        test_trial_type = np.zeros(n_cv, dtype=object)

        test_selected_frames = np.zeros(n_cv, dtype=object)    

        test_trial_idxs = np.zeros(n_cv, dtype=object)    



        for cv_ind in range(n_cv):

            test_video_path[cv_ind] = []

            test_sess_inds[cv_ind] = []

            test_trial_type[cv_ind] = []

            test_selected_frames[cv_ind] = []

            test_trial_idxs[cv_ind] = []


        for cv_ind in range(n_cv):

            for f1, f2 in f1_f2_list:
                # f1 = 'BAYLORGC[#]'
                # f2 = '[yyyy]_[mm]_[dd]'

                for i in n_trial_types_list:

                    # Test set
                    test_video_path[cv_ind].extend(sess_wise_test_video_path_dict[cv_ind][f1][f2][i])
                    test_sess_inds[cv_ind].extend(sess_wise_test_sess_inds_dict[cv_ind][f1][f2][i])
                    test_trial_type[cv_ind].extend(sess_wise_test_trial_type_dict[cv_ind][f1][f2][i])
                    test_selected_frames[cv_ind].extend(sess_wise_test_selected_frames_dict[cv_ind][f1][f2][i])
                    test_trial_idxs[cv_ind].extend(sess_wise_test_trial_idxs_dict[cv_ind][f1][f2][i])

            # Turn the lists into arrays.

            test_video_path[cv_ind] = np.array(test_video_path[cv_ind])
            test_sess_inds[cv_ind] = np.array(test_sess_inds[cv_ind])
            test_trial_type[cv_ind] = np.array(test_trial_type[cv_ind])
            test_selected_frames[cv_ind] = np.array(test_selected_frames[cv_ind])
            test_trial_idxs[cv_ind] = np.array(test_trial_idxs[cv_ind])












            
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





        # Iterate over cv_ind
        n_sess = self.configs['n_sess']

        '''
        How do we measure an accuracy in a highly imbalanced dataset?
        I use all the minority class samples, and a random subset of the majority class samples with the equal size, and then average
        over different samplings of the latter. This is also how Finkelstein et al. did it.
        '''




        cv_agg_sess_wise_trial_type = np.zeros((n_cv, n_sess), dtype=object) #  X[cv_ind,sess_ind] has shape (n_trials)
        cv_agg_sess_wise_pred = np.zeros((n_cv, n_sess), dtype=object) # X[cv_ind,sess_ind] has shape (n_trials, T)
        cv_agg_sess_wise_pre_sigmoid_score = np.zeros((n_cv, n_sess), dtype=object) # X[cv_ind,sess_ind] has shape (n_trials, T)
        cv_agg_sess_wise_trial_idx = np.zeros((n_cv, n_sess), dtype=object) # X[cv_ind,sess_ind] has shape (n_trials)
        cv_agg_sess_wise_acc = np.zeros((n_cv, n_sess), dtype=object) # X[cv_ind,sess_ind] has shape (n_samplings, T)
        cv_agg_sess_wise_roc_auc = np.zeros((n_cv, n_sess), dtype=object) # X[cv_ind,sess_ind] has shape (n_samplings, T)


        # Save sess_ind to (f1,f2) mapping.
        sess_ind_to_f1_f2_map = np.zeros((n_sess,), dtype=object)

        for sess_ind, (f1, f2) in enumerate(f1_f2_list):

            sess_ind_to_f1_f2_map[sess_ind] = '_'.join([f1,f2])

        for cv_ind in range(n_cv):
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

            cv_agg_sess_wise_trial_type[cv_ind], cv_agg_sess_wise_pred[cv_ind], \
            cv_agg_sess_wise_pre_sigmoid_score[cv_ind], cv_agg_sess_wise_trial_idx[cv_ind],\
            cv_agg_sess_wise_acc[cv_ind], cv_agg_sess_wise_roc_auc[cv_ind] = \
            self.compute_pred_helper_cv(f1_f2_list, cv_ind, self.configs, video_root_path, n_trial_types_list,\
            test_video_path[cv_ind],test_sess_inds[cv_ind], \
            test_trial_type[cv_ind], test_selected_frames[cv_ind],\
            test_trial_idxs[cv_ind],\
            transform_list, params, device)



        os.makedirs(os.path.join(self.analysis_save_path, 'compute_pred'), exist_ok=True)

        np.save(os.path.join(self.analysis_save_path, 'compute_pred', 'cv_agg_sess_wise_trial_type.npy'), cv_agg_sess_wise_trial_type)
        np.save(os.path.join(self.analysis_save_path, 'compute_pred', 'cv_agg_sess_wise_pred.npy'), cv_agg_sess_wise_pred)
        np.save(os.path.join(self.analysis_save_path, 'compute_pred', 'cv_agg_sess_wise_pre_sigmoid_score.npy'), \
            cv_agg_sess_wise_pre_sigmoid_score)
        np.save(os.path.join(self.analysis_save_path, 'compute_pred', 'cv_agg_sess_wise_trial_idx.npy'), cv_agg_sess_wise_trial_idx)
        np.save(os.path.join(self.analysis_save_path, 'compute_pred', \
            'cv_agg_sess_wise_acc_n_samplings_{}.npy'.format(self.configs['n_samplings'])), cv_agg_sess_wise_acc)
        np.save(os.path.join(self.analysis_save_path, 'compute_pred', \
            'cv_agg_sess_wise_roc_auc_n_samplings_{}.npy'.format(self.configs['n_samplings'])), cv_agg_sess_wise_roc_auc)
        np.save(os.path.join(self.analysis_save_path, 'compute_pred', 'sess_ind_to_f1_f2_map.npy'), sess_ind_to_f1_f2_map)




    def compute_pred_helper_cv(self, f1_f2_list, cv_ind, configs, video_root_path, n_trial_types_list,\
        test_video_path, test_sess_inds,\
        test_trial_type, test_selected_frames,\
        test_trial_idxs,\
        transform_list, params, device):
        '''
        Return:
        sess_wise_trial_type = np.zeros((n_sess,), dtype=object) #  X[sess_ind] has shape (n_trials)
        sess_wise_pred = np.zeros((n_sess,), dtype=object) # X[sess_ind] has shape (n_trials, T)
        sess_wise_pre_sigmoid_score = np.zeros((n_sess,), dtype=object) # X[sess_ind] has shape (n_trials, T)
        sess_wise_trial_idx = np.zeros((n_sess,), dtype=object) # X[sess_ind] has shape (n_trials)
        sess_wise_acc = np.zeros((n_sess,), dtype=object) # X[sess_ind] has shape (n_samplings, T)
        sess_wise_roc_auc = np.zeros((n_sess,), dtype=object) # X[sess_ind] has shape (n_samplings, T)
        '''




        '''
        ###
        Important:
        We will not define a dataloader separately for each
        random subsampling of the majority class.

        Instead we will first test the network on all samples
        in one go, and then subsample the majority class samples' predictions afterward.
        ###
        '''

        combined_valid_set = MyDatasetChoicePredVariableFramesRecon(video_root_path, test_video_path,\
            test_sess_inds, test_trial_type, test_selected_frames, \
            test_trial_idxs, configs['view_type'], \
            transform_list=transform_list, img_type=configs['img_type'])

        combined_valid_loader = data.DataLoader(combined_valid_set, **params)


        for key in sorted(self.configs.keys()):
            print('{}: {}'.format(key, self.configs[key]))
        print('')
        print('')


        '''
        Test the best saved model. Copied from cd_reg_from_videos_test.py.
        '''
        print('')
        print('Test begins!')
        print('')



        import sys
        model = getattr(sys.modules[__name__], self.configs['model_name'])(self.configs).to(device)


        # Parallelize model to multiple GPUs
        if self.configs['use_cuda'] and torch.cuda.device_count() > 1:
            if self.configs['gpu_ids'] is None:
                print("Using", torch.cuda.device_count(), "GPUs!")
                print('')
                model = nn.DataParallel(model)
            else:
                print("Using", len(self.configs['gpu_ids']), "GPUs!")
                print('')
                model = nn.DataParallel(model, device_ids=self.configs['gpu_ids'])

        '''
        If we don't specify map_location, by default, the saved model is loaded onto the gpus it was trained on.
        This could be problematic if the gpus it was trained on are currently occupied by other processes.

        If I set map_location to device as below, the model will still be run on multiple gpus as specified in DataParallel.
        '''
        model.load_state_dict(torch.load(os.path.join(self.model_save_path, 'cv_ind_{}'.format(cv_ind), 'best_model.pth'), map_location=device))
        




        '''
        All numpy arrays:
        combined_trial_type: (n_trials)
        combined_pred: (n_trials, T)
        combined_pre_sigmoid_score: (n_trials, T)
        combined_trial_idx: (n_trials)
        combined_sess_inds: (n_trials)
        '''
        combined_trial_type, combined_pred, combined_pre_sigmoid_score, combined_trial_idx, combined_sess_inds \
         = pert_pred_analysis(self.configs, model, device, combined_valid_loader)





        n_samplings = self.configs['n_samplings']

        cls_names = ['no_stim', self.configs['pert_type']]

        n_sess = self.configs['n_sess']


        sess_wise_trial_type = np.zeros((n_sess,), dtype=object) #  X[sess_ind] has shape (n_trials)
        sess_wise_pred = np.zeros((n_sess,), dtype=object) # X[sess_ind] has shape (n_trials, T)
        sess_wise_pre_sigmoid_score = np.zeros((n_sess,), dtype=object) # X[sess_ind] has shape (n_trials, T)
        sess_wise_trial_idx = np.zeros((n_sess,), dtype=object) # X[sess_ind] has shape (n_trials)
        sess_wise_acc = np.zeros((n_sess,), dtype=object) # X[sess_ind] has shape (n_samplings, T)
        sess_wise_roc_auc = np.zeros((n_sess,), dtype=object) # X[sess_ind] has shape (n_samplings, T)



        '''
        Now divide up results into different sessions.
        '''

        for sess_ind, (f1, f2) in enumerate(f1_f2_list):

            cur_sess_mask = (combined_sess_inds == sess_ind)

            cur_test_trial_type = combined_trial_type[cur_sess_mask]
            cur_test_pred = combined_pred[cur_sess_mask]
            cur_test_pre_sigmoid_score = combined_pre_sigmoid_score[cur_sess_mask]
            cur_test_trial_idx = combined_trial_idx[cur_sess_mask]

            sess_wise_trial_type[sess_ind] = cur_test_trial_type
            sess_wise_pred[sess_ind] = cur_test_pred
            sess_wise_pre_sigmoid_score[sess_ind] = cur_test_pre_sigmoid_score
            sess_wise_trial_idx[sess_ind] = cur_test_trial_idx

            T = cur_test_pred.shape[1]

            sess_wise_acc[sess_ind] = np.zeros((n_samplings, T))
            sess_wise_roc_auc[sess_ind] = np.zeros((n_samplings, T))


            '''
            ###
            We have to balance the test set by random undersampling of the majority class.
            ###
            '''
            cls_types = np.unique(cur_test_trial_type)
            #if len(cls_types) != 2:
            #    continue
            #assert len(cls_types) == 2
            #assert (cls_types == np.array([0,1])).all()


            '''
            Determine majority and minority classes.
            '''
            n_samples_each_cls = np.array([len(np.nonzero(cur_test_trial_type==k)[0]) for k in cls_types])

            major_cls_type = cls_types[n_samples_each_cls.argmax()]
            minor_cls_type = cls_types[n_samples_each_cls.argmin()]

            if minor_cls_type == major_cls_type:
                # This could happen if the two classes have the same number of samples.
                # If so, we want them to be different.
                minor_cls_type = 1 - major_cls_type 

            print('')
            print('f1: {}'.format(f1))
            print('f2: {}'.format(f2))
            print('major_cls: {}'.format(cls_names[major_cls_type]))
            print('minor_cls: {}'.format(cls_names[minor_cls_type]))


            '''
            Take all minority class samples, compute its number, and 
            sample majority class samples of the same number.
            '''
            n_minor_cls_samples = len(np.nonzero(cur_test_trial_type==minor_cls_type)[0])
            minor_mask = (cur_test_trial_type==minor_cls_type)

            cur_minor_test_trial_type = cur_test_trial_type[minor_mask]
            cur_minor_test_pred = cur_test_pred[minor_mask]
            cur_minor_test_pre_sigmoid_score = cur_test_pre_sigmoid_score[minor_mask]


            major_inds = np.nonzero(cur_test_trial_type==major_cls_type)[0]
            n_major_cls_samples = len(major_inds)

            for s in range(n_samplings):
                temp = np.random.permutation(n_major_cls_samples)[:n_minor_cls_samples]
                sampled_major_inds = major_inds[temp]

                cur_major_test_trial_type = cur_test_trial_type[sampled_major_inds]
                cur_major_test_pred = cur_test_pred[sampled_major_inds]
                cur_major_test_pre_sigmoid_score = cur_test_pre_sigmoid_score[sampled_major_inds]


                cur_balanced_test_trial_type = cat([cur_minor_test_trial_type, cur_major_test_trial_type], 0)
                cur_balanced_test_pred = cat([cur_minor_test_pred, cur_major_test_pred], 0)
                cur_balanced_test_pre_sigmoid_score = cat([cur_minor_test_pre_sigmoid_score, cur_major_test_pre_sigmoid_score], 0)

                '''
                Compute the accuracy.
                '''
                sess_wise_acc[sess_ind][s] = (cur_balanced_test_trial_type[:,None] == cur_balanced_test_pred).astype(float).mean(0) # (T)


                '''
                Compute the roc_auc.
                '''
                for t in range(T):
                    sess_wise_roc_auc[sess_ind][s,t] = \
                    roc_auc_score(cur_balanced_test_trial_type, cur_balanced_test_pre_sigmoid_score[:,t])



        return sess_wise_trial_type, sess_wise_pred, \
        sess_wise_pre_sigmoid_score, sess_wise_trial_idx,\
        sess_wise_acc, sess_wise_roc_auc





    def plot_test_acc_time_course(self, sess_type_list):



        cnn_channel_str = '_'.join([str(i) for i in self.configs['_cnn_channel_list']])

        if self.configs['img_type'] == 'jpg':
            img_type_str = ''
        elif self.configs['img_type'] == 'png':
            img_type_str = 'png_frames'

        if self.configs['neural_begin_time'] == -1.4 and self.configs['neural_end_time'] == -0.2 and self.configs['neural_skip_time'] == 0.2:
            neural_time_str = ''
        else:
            neural_time_str = 'begin_{:.1f}_end_{:.1f}_skip_{:.1f}'.format(self.configs['neural_begin_time'], self.configs['neural_end_time'], self.configs['neural_skip_time'])



        # analysis_save_path.
        # model_cv_dir: /data5/bkang/bk_video_ALM/NeuronalData5_codes/cd_reg_from_videos_all_no_stim_trials_cv_frame_conversion_corrected_full_cv_models
        if len(self.configs['models_cv_dir'].split('\\')) != 1:
            analysis_dir = self.configs['models_cv_dir'].split('\\')
            analysis_dir[-1] = 'pert_pred_from_videos_analysis_results'
            analysis_dir = '\\'.join(analysis_dir)
        else:
            analysis_dir = 'pert_pred_from_videos_analysis_results'


        # Note that we are not including the last 'cv_ind_{}'.format(cv_ind) here. It will be included in individual methods.
        fig_save_path = os.path.join(analysis_dir, 'plot_test_acc_time_course', self.configs['pert_type'], \
            'random_seed_{}'.format(self.configs['random_seed']), \
            img_type_str, neural_time_str, 'view_type_{}'.format('_'.join(self.configs['view_type'])), \
            'model_name_{}'.format(self.configs['model_name']), 'cnn_channel_list_{}'.format(cnn_channel_str), \
            'bs_{}_epochs_{}'.format(self.configs['bs'], self.configs['epochs']), 'n_cv_{}'.format(self.configs['n_cv']))

        os.makedirs(fig_save_path, exist_ok=True)


        combined_acc_across_everything = []

        if len(sess_type_list) == 1:
            sess_type_str = '({})'.format(sess_type_list[0])
        else:
            sess_type_str = ''


        for sess_type in sess_type_list:

            self.configs['sess_type'] = sess_type

            combined_acc_across_inst_trial_types = []

            for i in self.n_trial_types_list:

                self.configs['inst_trial_type'] = i

                self.init_model_and_analysis_save_paths()

                combined_acc_across_inst_trial_types.append(self.plot_test_acc_time_course_helper())

            # (n_trial_types, n_cv, n_sess, n_samplings, T)
            combined_acc_across_inst_trial_types = np.stack(combined_acc_across_inst_trial_types, 0)

            combined_acc_across_everything.append(combined_acc_across_inst_trial_types)

        # (n_trial_types, n_cv, n_sess, n_samplings, T)
        combined_acc_across_everything = cat(combined_acc_across_everything, 2)


        combined_acc_to_plot = combined_acc_across_everything.mean(0).mean(0).mean(1) # (n_sess, T)


        n_sess = len(combined_acc_to_plot)

        n_steps = int(np.rint((self.configs['neural_end_time'] - self.configs['neural_begin_time'])/self.configs['neural_skip_time'])) + 1
        timesteps = self.configs['neural_begin_time'] + self.configs['neural_skip_time']*np.arange(n_steps)


        fig = plt.figure()

        ax = fig.add_subplot(111)

        ax.set_title(self.configs['pert_type'])

        ax.errorbar(timesteps, combined_acc_to_plot.mean(0), yerr=combined_acc_to_plot.std(0,ddof=1)/np.sqrt(n_sess))

        fig.savefig(os.path.join(fig_save_path, 'plot_test_acc_time_course{}.png'.format(sess_type_str)))



    def plot_test_acc_time_course_helper(self):

        cv_agg_sess_wise_acc = np.load(os.path.join(self.analysis_save_path, 'compute_pred', \
            'cv_agg_sess_wise_acc_n_samplings_{}.npy'.format(self.configs['n_samplings'])))





        n_cv, n_sess = cv_agg_sess_wise_acc.shape

        # (n_cv, n_sess, n_samplings, T)
        combined_acc = np.array([[cv_agg_sess_wise_acc[cv_ind,sess_ind] for sess_ind in range(n_sess)] for cv_ind in range(n_cv)])

        return combined_acc






    def plot_score_distribution(self, sess_type_list):



        cnn_channel_str = '_'.join([str(i) for i in self.configs['_cnn_channel_list']])

        if self.configs['img_type'] == 'jpg':
            img_type_str = ''
        elif self.configs['img_type'] == 'png':
            img_type_str = 'png_frames'

        if self.configs['neural_begin_time'] == -1.4 and self.configs['neural_end_time'] == -0.2 and self.configs['neural_skip_time'] == 0.2:
            neural_time_str = ''
        else:
            neural_time_str = 'begin_{:.1f}_end_{:.1f}_skip_{:.1f}'.format(self.configs['neural_begin_time'], self.configs['neural_end_time'], self.configs['neural_skip_time'])



        # analysis_save_path.
        # model_cv_dir: /data5/bkang/bk_video_ALM/NeuronalData5_codes/cd_reg_from_videos_all_no_stim_trials_cv_frame_conversion_corrected_full_cv_models
        if len(self.configs['models_cv_dir'].split('\\')) != 1:
            analysis_dir = self.configs['models_cv_dir'].split('\\')
            analysis_dir[-1] = 'state_pred_from_videos_analysis_results'
            analysis_dir = '/'.join(analysis_dir)
        else:
            analysis_dir = 'state_pred_from_videos_analysis_results'


        # Note that we are not including the last 'cv_ind_{}'.format(cv_ind) here. It will be included in individual methods.
        fig_save_path = os.path.join(analysis_dir, 'plot_score_distribution', self.configs['pert_type'], \
            'random_seed_{}'.format(self.configs['random_seed']), \
            img_type_str, neural_time_str, 'view_type_{}'.format('_'.join(self.configs['view_type'])), \
            'model_name_{}'.format(self.configs['model_name']), 'cnn_channel_list_{}'.format(cnn_channel_str), \
            'bs_{}_epochs_{}'.format(self.configs['bs'], self.configs['epochs']), 'n_cv_{}'.format(self.configs['n_cv']))

        os.makedirs(fig_save_path, exist_ok=True)


        combined_trial_type_across_everything = []
        combined_score_across_everything = []


        if len(sess_type_list) == 1:
            sess_type_str = '({})'.format(sess_type_list[0])
        else:
            sess_type_str = ''


        for sess_type in sess_type_list:

            self.configs['sess_type'] = sess_type


            for inst_i in self.n_trial_types_list:

                self.configs['inst_trial_type'] = inst_i

                self.init_model_and_analysis_save_paths()

                combined_trial_type, combined_score = self.plot_score_distribution_helper()

                combined_trial_type_across_everything.append(combined_trial_type)
                combined_score_across_everything.append(combined_score)



        # (n_trials)
        combined_trial_type_across_everything = cat(combined_trial_type_across_everything, 0)

        # (n_trials, T)
        combined_score_across_everything = cat(combined_score_across_everything, 0)

        n_steps = int(np.rint((self.configs['neural_end_time'] - self.configs['neural_begin_time'])/self.configs['neural_skip_time'])) + 1
        timesteps = self.configs['neural_begin_time'] + self.configs['neural_skip_time']*np.arange(n_steps)

        cur_dict = {}
        cur_dict['Time'] = np.broadcast_to(timesteps[None], combined_score_across_everything.shape).reshape(-1)
        cur_dict['Score'] = combined_score_across_everything.reshape(-1)
        cur_dict['Pert type'] = \
        np.broadcast_to(combined_trial_type_across_everything[:,None], combined_score_across_everything.shape).reshape(-1)


        import seaborn as sns
        import pandas as pd

        cur_df = pd.DataFrame(cur_dict)
        my_pal = {}
        my_pal[0] = 'gray'
        my_pal[1] = 'orange'


        fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.set_title(self.configs['pert_type'], fontsize=10)

        ax = sns.violinplot(x='Time', y='Score', hue='Pert type', data=cur_df, palette=my_pal, split=True, \
            ax=ax, inner=None, linewidth=0.1)

        ax.set_xticks(np.arange(len(timesteps)))
        ax.set_xticklabels(['{:.1f}'.format(x) for x in timesteps])

        fig.savefig(os.path.join(fig_save_path, 'plot_score_distribution{}.png'.format(sess_type_str)))



    def plot_score_distribution_helper(self):

        cv_agg_sess_wise_trial_type = np.load(os.path.join(self.analysis_save_path, 'compute_pred', \
        'cv_agg_sess_wise_trial_type.npy'))

        cv_agg_sess_wise_pre_sigmoid_score = np.load(os.path.join(self.analysis_save_path, 'compute_pred', \
        'cv_agg_sess_wise_pre_sigmoid_score.npy'))



        n_cv, n_sess = cv_agg_sess_wise_pre_sigmoid_score.shape


        # (n_trials)
        combined_trial_type = cat(cv_agg_sess_wise_trial_type.reshape(-1), 0)


        # (n_trials, T)
        combined_score = cat(cv_agg_sess_wise_pre_sigmoid_score.reshape(-1), 0)


        return combined_trial_type, combined_score





    def correlate_roc_auc_vs_bi_stim_cross_hemi_cor_across_sessions(self, sess_type_list, cross_hemi_cor_type,\
        analyzed_time):



        '''
        ###
        This plot is only concerned with bi_stim trials.
        ###
        '''
        self.configs['pert_type'] = 'bi_stim'


        cnn_channel_str = '_'.join([str(i) for i in self.configs['_cnn_channel_list']])

        if self.configs['img_type'] == 'jpg':
            img_type_str = ''
        elif self.configs['img_type'] == 'png':
            img_type_str = 'png_frames'

        if self.configs['neural_begin_time'] == -1.4 and self.configs['neural_end_time'] == -0.2 and self.configs['neural_skip_time'] == 0.2:
            neural_time_str = ''
        else:
            neural_time_str = 'begin_{:.1f}_end_{:.1f}_skip_{:.1f}'.format(self.configs['neural_begin_time'], self.configs['neural_end_time'], self.configs['neural_skip_time'])



        # analysis_save_path.
        # model_cv_dir: /data5/bkang/bk_video_ALM/NeuronalData5_codes/cd_reg_from_videos_all_no_stim_trials_cv_frame_conversion_corrected_full_cv_models
        if len(self.configs['models_cv_dir'].split('\\')) != 1:
            analysis_dir = self.configs['models_cv_dir'].split('\\')
            analysis_dir[-1] = 'pert_pred_from_videos_analysis_results'
            analysis_dir = '/'.join(analysis_dir)
        else:
            analysis_dir = 'pert_pred_from_videos_analysis_results'


        # Note that we are not including the last 'cv_ind_{}'.format(cv_ind) here. It will be included in individual methods.
        fig_save_path = os.path.join(analysis_dir, 'correlate_roc_auc_vs_bi_stim_cross_hemi_cor_across_sessions', self.configs['pert_type'], \
            'random_seed_{}'.format(self.configs['random_seed']), \
            img_type_str, neural_time_str, 'view_type_{}'.format('_'.join(self.configs['view_type'])), \
            'model_name_{}'.format(self.configs['model_name']), 'cnn_channel_list_{}'.format(cnn_channel_str), \
            'bs_{}_epochs_{}'.format(self.configs['bs'], self.configs['epochs']), 'n_cv_{}'.format(self.configs['n_cv']),\
            'n_samplings_{}'.format(self.configs['n_samplings']))

        os.makedirs(fig_save_path, exist_ok=True)

        # We will save data for plots to save time.
        data_save_path = os.path.join(fig_save_path, 'data_for_plots', 'analyzed_time_{:.1f}'.format(analyzed_time))
        os.makedirs(data_save_path, exist_ok=True)


        '''
        ###
        There are two different ways to define roc_auc.
        The first is to compute it separately for each train-test split,
        and then average it.

        The second is to aggregate test samples across train-test slpits, and then compute
        (balanced) auc score on them.
        ###
        '''

        # combined_roc_auc1 will have shape (n_sess, n_trial_types, n_cv, n_samplings)
        combined_roc_auc1 = np.zeros((self.n_trial_types,), dtype=object)

        for i in self.n_trial_types_list:
            combined_roc_auc1[i] = []
 
        # combined_roc_auc2 will have shape (n_sess, n_trial_types, n_samplings)
        combined_roc_auc2 = np.zeros((self.n_trial_types,), dtype=object)

        for i in self.n_trial_types_list:
            combined_roc_auc2[i] = []

        '''
        ###
        Since CD will be derived from control trials and we are using bi-stim CD projs, 
        we don't need to worry about train-test splits. So, we will use all correct control
        trials to compute CD.

        Also, I will aggregate bi-stim trials across train-test splits to compute the cross-hemi
        cor, because I don't have enough bi-stim trials. For example, if we have 50 bi-stim trials,
        for a given instructed trial type, we have 25, and in the test set, we have 5.
        ###
        '''

        # combined_bi_stim_cross_hemi_cor will have shape (n_sess, n_trial_types)
        combined_bi_stim_cross_hemi_cor = np.zeros((self.n_trial_types,), dtype=object)

        for i in self.n_trial_types_list:
            combined_bi_stim_cross_hemi_cor[i] = []



        if len(sess_type_list) == 1:
            sess_type_str = '({})'.format(sess_type_list[0])
        else:
            sess_type_str = ''


        for sess_type in sess_type_list:

            self.configs['sess_type'] = sess_type


            for inst_i in self.n_trial_types_list:

                self.configs['inst_trial_type'] = inst_i

                self.init_model_and_analysis_save_paths()

                '''
                roc_auc1: (n_sess, n_cv, n_samplings) for a given sess_type and inst_i.
                roc_auc2: (n_sess, n_samplings)
                bi_stim_cross_hemi_cor: (n_sess,)
                '''
                roc_auc1, roc_auc2, bi_stim_cross_hemi_cor = \
                self.correlate_roc_auc_vs_bi_stim_cross_hemi_cor_across_sessions_helper(cross_hemi_cor_type, analyzed_time, \
                    data_save_path)

                print('')
                print('sess_type: ', sess_type)
                print('inst_i: ', inst_i)
                print('roc_auc1.shape: ', roc_auc1.shape)
                print('roc_auc2.shape: ', roc_auc2.shape)
                print('bi_stim_cross_hemi_cor.shape: ', bi_stim_cross_hemi_cor.shape)


                combined_roc_auc1[inst_i].extend(roc_auc1)
                combined_roc_auc2[inst_i].extend(roc_auc2)
                combined_bi_stim_cross_hemi_cor[inst_i].extend(bi_stim_cross_hemi_cor)


        '''
        Aggregate across different sess types.
        '''
        for inst_i in self.n_trial_types_list:

            # (n_sess, n_cv, n_samplings)
            combined_roc_auc1[inst_i] = np.array(combined_roc_auc1[inst_i]) 

            # (n_sess, n_samplings)
            combined_roc_auc2[inst_i] = np.array(combined_roc_auc2[inst_i]) 

            # (n_sess,)
            combined_bi_stim_cross_hemi_cor[inst_i] = np.array(combined_bi_stim_cross_hemi_cor[inst_i]) 


        print('')
        print('')
        print('combined_roc_auc1.shape: ', combined_roc_auc1.shape)
        print('combined_roc_auc2.shape: ', combined_roc_auc2.shape)
        print('combined_bi_stim_cross_hemi_cor.shape: ', combined_bi_stim_cross_hemi_cor.shape)
        print('')
        print('')
            
        '''
        Finally turn them into the desired shapes.
        (n_sess, n_trial_types, n_cv, n_samplings)
        (n_sess, n_trial_types,n_samplings)
        (n_sess, n_trial_types)
        '''
        combined_roc_auc1 = np.stack(combined_roc_auc1, 1)
        combined_roc_auc2 = np.stack(combined_roc_auc2, 1)
        combined_bi_stim_cross_hemi_cor = np.stack(combined_bi_stim_cross_hemi_cor, 1)


        fig = plt.figure(figsize=(20, 10))

        gs = gridspec.GridSpec(1,2, wspace=0.4, hspace=0.4)

        for k, cur_combined_roc_auc in enumerate([combined_roc_auc1, combined_roc_auc2]):

            ax = fig.add_subplot(gs[k])
            ax.tick_params(length=5, width=2, labelsize=20)


            '''
            xs: (n_sess)
            ys: (n_sess)
            '''
            if len(cur_combined_roc_auc.shape) == 4:
                xs = cur_combined_roc_auc.mean(1).mean(1).mean(1)
            else:
                xs = cur_combined_roc_auc.mean(1).mean(1)

            ys = combined_bi_stim_cross_hemi_cor.mean(1)


            ax.scatter(xs, ys, s=600, marker='o', edgecolor='k', facecolor='None')


            ax.plot(np.unique(xs), np.poly1d(np.polyfit(xs, ys, 1))(np.unique(xs)), c='k', lw=2)
            coeffs, residuals, _, _, _ = np.polyfit(xs, ys, 1, full=True)
            r2 = 1 - residuals/((ys-ys.mean())**2).sum()
            #p-value for linear regression: https://stattrek.com/regression/slope-test.aspx
            se = np.sqrt(residuals/(len(ys)-2))/np.sqrt(((xs-xs.mean())**2).sum())
            slope = coeffs[0]
            t_stat = slope/se
            if slope < 0:
                t_stat = -t_stat
            df = len(ys)-2
            from scipy.stats import t
            one_sided_p_value = t.sf(t_stat, df)


            '''
            I am showing two-sided p value.
            '''

            if k == 0:
                ax.set_title('AUC separately for\neach train-test split\n(r={:.2f}, p={:.1e})'.format(\
                    np.sqrt(r2[0]), 2*one_sided_p_value[0]), fontsize=20)

            else:
                ax.set_title('AUC for test samples\nagg over train-test-splits\n(r={:.2f}, p={:.1e})'.format(\
                    np.sqrt(r2[0]), 2*one_sided_p_value[0]), fontsize=20)


            ax.set_xlabel('ROC AUC', fontsize=20)
            ax.set_ylabel('Bi-stim cross-hemi CD cor', fontsize=20)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)

        fig.savefig(os.path.join(fig_save_path, 'correlate_roc_auc_vs_bi_stim_cross_hemi_cor_across_sessions{}.png'.format(sess_type_str)))



    def correlate_roc_auc_vs_bi_stim_cross_hemi_cor_across_sessions_helper(self, cross_hemi_cor_type, \
        analyzed_time, data_save_path):
        '''
        Return:
        roc_auc1: (n_sess, n_cv, n_samplings) for a given sess_type and inst_i.
        roc_auc2: (n_sess, n_samplings)
        bi_stim_cross_hemi_cor: (n_sess,)
        '''

        data_save_path = os.path.join(data_save_path, 'sess_type_{}'.format(self.configs['sess_type']), \
            'inst_trial_type_{}'.format(self.configs['inst_trial_type']))
        os.makedirs(data_save_path, exist_ok=True)



        roc_auc1_save_name = 'roc_auc1.npy'
        roc_auc2_save_name = 'roc_auc2.npy'
        bi_stim_cross_hemi_cor_save_name = 'bi_stim_cross_hemi_cor_type_{}.npy'.format(cross_hemi_cor_type)

        if all([os.path.isfile(os.path.join(data_save_path, save_name)) for save_name in \
            [roc_auc1_save_name, roc_auc2_save_name, bi_stim_cross_hemi_cor_save_name]]):

            roc_auc1 = np.load(os.path.join(data_save_path, roc_auc1_save_name))
            roc_auc2 = np.load(os.path.join(data_save_path, roc_auc2_save_name))
            bi_stim_cross_hemi_cor = np.load(os.path.join(data_save_path, bi_stim_cross_hemi_cor_save_name))


        else:


            '''
            cv_agg_sess_wise_trial_type = np.zeros((n_cv, n_sess), dtype=object) #  X[cv_ind,sess_ind] has shape (n_trials)
            cv_agg_sess_wise_pre_sigmoid_score = np.zeros((n_cv, n_sess), dtype=object) # X[cv_ind,sess_ind] has shape (n_trials, T)
            cv_agg_sess_wise_roc_auc = np.zeros((n_cv, n_sess), dtype=object) # X[cv_ind,sess_ind] has shape (n_samplings, T)
            '''
            n_steps = int(np.rint((self.configs['neural_end_time'] - self.configs['neural_begin_time'])/self.configs['neural_skip_time'])) + 1
            pred_timesteps = self.configs['neural_begin_time'] + self.configs['neural_skip_time']*np.arange(n_steps)

            timestep_ind = np.abs(pred_timesteps - analyzed_time).argmin()



            cv_agg_sess_wise_trial_type = np.load(os.path.join(self.analysis_save_path, 'compute_pred', \
            'cv_agg_sess_wise_trial_type.npy'))


            cv_agg_sess_wise_pre_sigmoid_score = np.load(os.path.join(self.analysis_save_path, 'compute_pred', \
            'cv_agg_sess_wise_pre_sigmoid_score.npy'))


            cv_agg_sess_wise_roc_auc = np.load(os.path.join(self.analysis_save_path, 'compute_pred', \
            'cv_agg_sess_wise_roc_auc_n_samplings_{}.npy'.format(self.configs['n_samplings'])))



            n_cv, n_sess = cv_agg_sess_wise_pre_sigmoid_score.shape

            n_samplings = self.configs['n_samplings']

            
            roc_auc1 = np.zeros((n_sess, n_cv, n_samplings))
            roc_auc2 = np.zeros((n_sess, n_samplings))
            bi_stim_cross_hemi_cor = np.zeros((n_sess,))


            '''
            ###
            roc_auc1.
            ###
            '''

            # (n_sess, n_cv, n_samplings, T)
            roc_auc1 = np.array([[cv_agg_sess_wise_roc_auc[cv_ind,sess_ind] for cv_ind in range(n_cv)] for sess_ind in range(n_sess)])
            # (n_sess, n_cv, n_samplings)
            roc_auc1 = roc_auc1[...,timestep_ind]


            '''
            ###
            roc_auc2.
            ###
            '''
            cls_names = ['no_stim', self.configs['pert_type']]


            for sess_ind in range(n_sess):
                # For each session, we collect trial_type and pre_sigmoid_score across train-test splits.

                cur_test_trial_type = []
                cur_test_pre_sigmoid_score = []

                for cv_ind in range(n_cv):

                    # (n_trials)
                    cur_test_trial_type.append(cv_agg_sess_wise_trial_type[cv_ind,sess_ind])

                    # (n_trials)
                    cur_test_pre_sigmoid_score.append(cv_agg_sess_wise_pre_sigmoid_score[cv_ind,sess_ind][:,timestep_ind])

                cur_test_trial_type = cat(cur_test_trial_type, 0)
                cur_test_pre_sigmoid_score = cat(cur_test_pre_sigmoid_score, 0)



                '''
                ###
                We have to balance the test set by random undersampling of the majority class.
                ###
                '''
                cls_types = np.unique(cur_test_trial_type)

                assert len(cls_types) == 2
                assert (cls_types == np.array([0,1])).all()


                '''
                Determine majority and minority classes.
                '''
                n_samples_each_cls = np.array([len(np.nonzero(cur_test_trial_type==k)[0]) for k in cls_types])

                major_cls_type = cls_types[n_samples_each_cls.argmax()]
                minor_cls_type = cls_types[n_samples_each_cls.argmin()]

                if minor_cls_type == major_cls_type:
                    # This could happen if the two classes have the same number of samples.
                    # If so, we want them to be different.
                    minor_cls_type = 1 - major_cls_type 

                print('')
                print('sess_ind: {}'.format(sess_ind))
                print('major_cls: {}'.format(cls_names[major_cls_type]))
                print('minor_cls: {}'.format(cls_names[minor_cls_type]))


                '''
                Take all minority class samples, compute its number, and 
                sample majority class samples of the same number.
                '''
                n_minor_cls_samples = len(np.nonzero(cur_test_trial_type==minor_cls_type)[0])
                minor_mask = (cur_test_trial_type==minor_cls_type)

                cur_minor_test_trial_type = cur_test_trial_type[minor_mask]
                cur_minor_test_pre_sigmoid_score = cur_test_pre_sigmoid_score[minor_mask]


                major_inds = np.nonzero(cur_test_trial_type==major_cls_type)[0]
                n_major_cls_samples = len(major_inds)


                for s in range(n_samplings):

                    temp = np.random.permutation(n_major_cls_samples)[:n_minor_cls_samples]
                    sampled_major_inds = major_inds[temp]

                    cur_major_test_trial_type = cur_test_trial_type[sampled_major_inds]
                    cur_major_test_pre_sigmoid_score = cur_test_pre_sigmoid_score[sampled_major_inds]

                    cur_balanced_test_trial_type = cat([cur_minor_test_trial_type, cur_major_test_trial_type], 0)
                    cur_balanced_test_pre_sigmoid_score = cat([cur_minor_test_pre_sigmoid_score, cur_major_test_pre_sigmoid_score], 0)


                    roc_auc2[sess_ind,s] = roc_auc_score(cur_balanced_test_trial_type, cur_balanced_test_pre_sigmoid_score)


            '''
            ###
            bi_stim_cross_hemi_cor
            ###
            '''



            '''
            We need to select only those bi-stim trials (of the current instructed trial type) that have videos.
            '''

            # Maps sess_ind to f1_f2 string.
            sess_ind_to_f1_f2_map = np.load(os.path.join(self.analysis_save_path, 'compute_pred', 'sess_ind_to_f1_f2_map.npy'))


            # Collect video filenames.
            if self.configs['img_type'] == 'jpg':
                video_root_path = os.path.join(self.configs['video_data_dir'], 'frames')
            elif self.configs['img_type'] == 'png':
                video_root_path = os.path.join(self.configs['video_data_dir'], 'png_frames')


            prep_root_path = self.configs['prep_root_dir']
            data_prefix = self.configs['data_prefix']

            pert_type = self.configs['pert_type']

            for sess_ind in range(n_sess):

                bi_stim_i_good_trial_numbers_with_videos = []


                # f1 = 'BAYLORGC[#]'
                # f2 = '[yyyy]_[mm]_[dd]'
                f1_f2 = sess_ind_to_f1_f2_map[sess_ind]
                temp = f1_f2.find('_')

                f1 = f1_f2[:temp]
                f2 = f1_f2[temp+1:]


                bi_stim_i_good_trials = pert_pred_utils.load_stim_i_good_trials_for_pert_pred(pert_type, \
                    self.configs['inst_trial_type'], self.configs['prep_root_dir'], f1, f2)

                print('')
                print('<bi_stim_i_good_trials (before selecting those with videos)>')
                print('inst_trial_type: ', self.configs['inst_trial_type'])
                print(bi_stim_i_good_trials)
                print('')

                for f3 in os.listdir(os.path.join(video_root_path, f1, f2)):
                    # f3 = 'E3-BAYLORGC25-2018_09_17-13'
                    if os.path.isfile(os.path.join(video_root_path, f1, f2, f3)):
                        continue

                    trial_idx_for_f3 = pert_pred_utils.map_f3_to_trial_idx(f1,f2,f3)

                    if trial_idx_for_f3 in bi_stim_i_good_trials:
                        bi_stim_i_good_trial_numbers_with_videos.append(trial_idx_for_f3)

                bi_stim_i_good_trial_numbers_with_videos.sort()

                '''
                Now compute bi_stim_cross_hemi_cor.
                '''

                # cd[j]
                cd = self.get_instant_cd_from_all_correct_control_trials(prep_root_path, data_prefix, f1, f2, analyzed_time)


                bi_stim_rates, cur_bin_centers, cur_neural_unit_location = \
                self.get_rates_for_given_i_good_trial_numbers(pert_type, prep_root_path, data_prefix, f1, f2, \
                    bi_stim_i_good_trial_numbers_with_videos)

                cur_time_ind = np.abs(cur_bin_centers - analyzed_time).argmin()

                # (n_trials, n_neurons)
                cur_bi_stim_rates = bi_stim_rates[cur_time_ind]

                cur_n_trials = len(cur_bi_stim_rates)
                
                cur_bi_stim_cd_projs = np.zeros((self.n_loc_names, cur_n_trials))

                for j, loc_name in enumerate(self.n_loc_names_list):
                    cur_bi_stim_cd_projs[j] = cur_bi_stim_rates[...,cur_neural_unit_location==loc_name].dot(cd[j])


                if cross_hemi_cor_type == 'rank':
                    bi_stim_cross_hemi_cor[sess_ind], _ = stats.spearmanr(cur_bi_stim_cd_projs[0], cur_bi_stim_cd_projs[1])
                else:
                    raise ValueError('cross_hemi_cor_type {} is not supported yet.'.format(cross_hemi_cor_type))



            '''
            Save
            '''

            np.save(os.path.join(data_save_path, roc_auc1_save_name), roc_auc1)
            np.save(os.path.join(data_save_path, roc_auc2_save_name), roc_auc2)
            np.save(os.path.join(data_save_path, bi_stim_cross_hemi_cor_save_name), bi_stim_cross_hemi_cor)



        return roc_auc1, roc_auc2, bi_stim_cross_hemi_cor







    def correlate_cls_acc_vs_bi_stim_cross_hemi_cor_across_sessions(self, sess_type_list, cross_hemi_cor_type,\
        analyzed_time):



        '''
        ###
        This plot is only concerned with bi_stim trials.
        ###
        '''
        self.configs['pert_type'] = 'bi_stim'


        cnn_channel_str = '_'.join([str(i) for i in self.configs['_cnn_channel_list']])

        if self.configs['img_type'] == 'jpg':
            img_type_str = ''
        elif self.configs['img_type'] == 'png':
            img_type_str = 'png_frames'

        if self.configs['neural_begin_time'] == -1.4 and self.configs['neural_end_time'] == -0.2 and self.configs['neural_skip_time'] == 0.2:
            neural_time_str = ''
        else:
            neural_time_str = 'begin_{:.1f}_end_{:.1f}_skip_{:.1f}'.format(self.configs['neural_begin_time'], self.configs['neural_end_time'], self.configs['neural_skip_time'])



        # analysis_save_path.
        # model_cv_dir: /data5/bkang/bk_video_ALM/NeuronalData5_codes/cd_reg_from_videos_all_no_stim_trials_cv_frame_conversion_corrected_full_cv_models
        if len(self.configs['models_cv_dir'].split('\\')) != 1:
            analysis_dir = self.configs['models_cv_dir'].split('\\')
            analysis_dir[-1] = 'pert_pred_from_videos_analysis_results'
            analysis_dir = '/'.join(analysis_dir)
        else:
            analysis_dir = 'pert_pred_from_videos_analysis_results'


        # Note that we are not including the last 'cv_ind_{}'.format(cv_ind) here. It will be included in individual methods.
        fig_save_path = os.path.join(analysis_dir, 'correlate_cls_acc_vs_bi_stim_cross_hemi_cor_across_sessions', self.configs['pert_type'], \
            'random_seed_{}'.format(self.configs['random_seed']), \
            img_type_str, neural_time_str, 'view_type_{}'.format('_'.join(self.configs['view_type'])), \
            'model_name_{}'.format(self.configs['model_name']), 'cnn_channel_list_{}'.format(cnn_channel_str), \
            'bs_{}_epochs_{}'.format(self.configs['bs'], self.configs['epochs']), 'n_cv_{}'.format(self.configs['n_cv']),\
            'n_samplings_{}'.format(self.configs['n_samplings']))

        os.makedirs(fig_save_path, exist_ok=True)

        # We will save data for plots to save time.
        data_save_path = os.path.join(fig_save_path, 'data_for_plots', 'analyzed_time_{:.1f}'.format(analyzed_time))
        os.makedirs(data_save_path, exist_ok=True)


        '''
        ###
        We compute cls_acc separately for each train-test split,
        and then average it.

        ###
        '''

        # combined_cls_acc will have shape (n_sess, n_trial_types, n_cv, n_samplings)
        combined_cls_acc = np.zeros((self.n_trial_types,), dtype=object)

        '''
        IMPORTANT:
        I previously used combined_cls_acc.fill([]), and I found that this led to a bug, 
        because (I believe) it initializes combined_cls_acc[0] and combined_cls_acc[1] 
        to the same list (i.e. they point to the same underlying list).
        '''

        for i in self.n_trial_types_list:
            combined_cls_acc[i] = []

        '''
        ###
        Since CD will be derived from control trials and we are using bi-stim CD projs, 
        we don't need to worry about train-test splits. So, we will use all correct control
        trials to compute CD.

        Also, I will aggregate bi-stim trials across train-test splits to compute the cross-hemi
        cor, because I don't have enough bi-stim trials. For example, if we have 50 bi-stim trials,
        for a given instructed trial type, we have 25, and in the test set, we have 5.
        ###
        '''

        # combined_bi_stim_cross_hemi_cor will have shape (n_sess, n_trial_types)
        combined_bi_stim_cross_hemi_cor = np.zeros((self.n_trial_types,), dtype=object)

        for i in self.n_trial_types_list:
            combined_bi_stim_cross_hemi_cor[i] = []


        # combined_n_trials_each_sess will have shape (n_sess, n_trial_types)
        combined_n_trials_each_sess = np.zeros((self.n_trial_types,), dtype=object)

        for i in self.n_trial_types_list:
            combined_n_trials_each_sess[i] = []






        if len(sess_type_list) == 1:
            sess_type_str = '({})'.format(sess_type_list[0])
        else:
            sess_type_str = ''


        for sess_type in sess_type_list:

            self.configs['sess_type'] = sess_type


            for inst_i in self.n_trial_types_list:

                self.configs['inst_trial_type'] = inst_i

                self.init_model_and_analysis_save_paths()

                '''
                cls_acc: (n_sess, n_cv, n_samplings) for a given sess_type and inst_i.
                bi_stim_cross_hemi_cor: (n_sess,)
                '''
                cls_acc, bi_stim_cross_hemi_cor, n_trials_each_sess = \
                self.correlate_cls_acc_vs_bi_stim_cross_hemi_cor_across_sessions_helper(cross_hemi_cor_type, analyzed_time, \
                    data_save_path)

                print('')
                print('sess_type: ', sess_type)
                print('inst_i: ', inst_i)
                print('cls_acc.shape: ', cls_acc.shape)
                print('bi_stim_cross_hemi_cor.shape: ', bi_stim_cross_hemi_cor.shape)

                combined_cls_acc[inst_i].extend(cls_acc)
                combined_bi_stim_cross_hemi_cor[inst_i].extend(bi_stim_cross_hemi_cor)
                combined_n_trials_each_sess[inst_i].extend(n_trials_each_sess)


        '''
        Aggregate across different sess types.
        '''
        for inst_i in self.n_trial_types_list:

            # (n_sess, n_cv, n_samplings)
            combined_cls_acc[inst_i] = np.array(combined_cls_acc[inst_i]) 

            # (n_sess,)
            combined_bi_stim_cross_hemi_cor[inst_i] = np.array(combined_bi_stim_cross_hemi_cor[inst_i]) 

            
        '''
        Finally turn them into the desired shapes.
        (n_sess, n_trial_types, n_cv, n_samplings)
        (n_sess, n_trial_types)
        '''
        combined_cls_acc = np.stack(combined_cls_acc, 1)
        combined_bi_stim_cross_hemi_cor = np.stack(combined_bi_stim_cross_hemi_cor, 1)

        print('')
        print('')
        print('combined_cls_acc.shape: ', combined_cls_acc.shape)
        print('combined_bi_stim_cross_hemi_cor.shape: ', combined_bi_stim_cross_hemi_cor.shape)
        print('')
        print('')


        # (n_sess, n_trial_types) -> (n_sess)
        combined_n_trials_each_sess = np.stack([combined_n_trials_each_sess[inst_i] for inst_i in self.n_trial_types_list], 1)
        combined_n_trials_each_sess = combined_n_trials_each_sess.mean(1)

        print(combined_n_trials_each_sess)

        normalized_n_trials = combined_n_trials_each_sess - combined_n_trials_each_sess.min()
        normalized_n_trials /= normalized_n_trials.max()

        fig = plt.figure(figsize=(10, 10))

        gs = gridspec.GridSpec(1,1, wspace=0.4, hspace=0.4)


        ax = fig.add_subplot(gs[0])
        ax.tick_params(length=5, width=2, labelsize=20)


        '''
        xs: (n_sess)
        ys: (n_sess)
        '''
        xs = combined_cls_acc.mean(1).mean(1).mean(1)

        ys = combined_bi_stim_cross_hemi_cor.mean(1)

        print('len(xs)')
        print(len(xs))


        print('len(ys)')
        print(len(ys))

        print('len(normalized_n_trials)')
        print(len(normalized_n_trials))


        # ax.scatter(xs, ys, s=600, marker='o', edgecolor='k', facecolor='None')
        for k in range(len(xs)):
            ax.scatter(xs[k], ys[k], s=600, marker='o', edgecolor='k', facecolor=plt.get_cmap('Reds')(normalized_n_trials[k]))


        ax.plot(np.unique(xs), np.poly1d(np.polyfit(xs, ys, 1))(np.unique(xs)), c='k', lw=2)
        coeffs, residuals, _, _, _ = np.polyfit(xs, ys, 1, full=True)
        r2 = 1 - residuals/((ys-ys.mean())**2).sum()
        #p-value for linear regression: https://stattrek.com/regression/slope-test.aspx
        se = np.sqrt(residuals/(len(ys)-2))/np.sqrt(((xs-xs.mean())**2).sum())
        slope = coeffs[0]
        t_stat = slope/se
        if slope < 0:
            t_stat = -t_stat
        df = len(ys)-2
        from scipy.stats import t
        one_sided_p_value = t.sf(t_stat, df)


        '''
        Add a color bar for n_trials.
        '''
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        norm = mpl.colors.Normalize(vmin=combined_n_trials_each_sess.min(), vmax=combined_n_trials_each_sess.max())

        cb = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap('Reds'),
                                        norm=norm,
                                        orientation='vertical')
        cb.set_label('n_trials')


        '''
        I am showing two-sided p value.
        '''

        ax.set_title('r={:.2f}, p={:.1e}'.format(\
            np.sqrt(r2[0]), 2*one_sided_p_value[0]), fontsize=20)


        ax.set_xlabel('Classification acc.', fontsize=20)
        ax.set_ylabel('Bi-stim cross-hemi CD cor', fontsize=20)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)

        fig.savefig(os.path.join(fig_save_path, 'correlate_cls_acc_vs_bi_stim_cross_hemi_cor_across_sessions{}.png'.format(sess_type_str)))



    def correlate_cls_acc_vs_bi_stim_cross_hemi_cor_across_sessions_helper(self, cross_hemi_cor_type, \
        analyzed_time, data_save_path):
        '''
        Return:
        cls_acc1: (n_sess, n_cv, n_samplings) for a given sess_type and inst_i.
        bi_stim_cross_hemi_cor: (n_sess,)
        '''

        data_save_path = os.path.join(data_save_path, 'sess_type_{}'.format(self.configs['sess_type']), \
            'inst_trial_type_{}'.format(self.configs['inst_trial_type']))
        os.makedirs(data_save_path, exist_ok=True)



        cls_acc_save_name = 'cls_acc.npy'
        bi_stim_cross_hemi_cor_save_name = 'bi_stim_cross_hemi_cor_type_{}.npy'.format(cross_hemi_cor_type)
        n_trials_each_sess_save_name = 'n_trials_each_sess.npy'

        if all([os.path.isfile(os.path.join(data_save_path, save_name)) for save_name in \
            [cls_acc_save_name, bi_stim_cross_hemi_cor_save_name]]):

            cls_acc = np.load(os.path.join(data_save_path, cls_acc_save_name))
            bi_stim_cross_hemi_cor = np.load(os.path.join(data_save_path, bi_stim_cross_hemi_cor_save_name))
            n_trials_each_sess = np.load(os.path.join(data_save_path, n_trials_each_sess_save_name))


        else:


            '''
            cv_agg_sess_wise_cls_acc = np.zeros((n_cv, n_sess), dtype=object) # X[cv_ind,sess_ind] has shape (n_samplings, T)
            '''
            n_steps = int(np.rint((self.configs['neural_end_time'] - self.configs['neural_begin_time'])/self.configs['neural_skip_time'])) + 1
            pred_timesteps = self.configs['neural_begin_time'] + self.configs['neural_skip_time']*np.arange(n_steps)

            timestep_ind = np.abs(pred_timesteps - analyzed_time).argmin()


            cv_agg_sess_wise_cls_acc = np.load(os.path.join(self.analysis_save_path, 'compute_pred', \
            'cv_agg_sess_wise_acc_n_samplings_{}.npy'.format(self.configs['n_samplings'])))



            n_cv, n_sess = cv_agg_sess_wise_cls_acc.shape

            n_samplings = self.configs['n_samplings']

            
            cls_acc = np.zeros((n_sess, n_cv, n_samplings))
            bi_stim_cross_hemi_cor = np.zeros((n_sess,))
            n_trials_each_sess = np.zeros((n_sess,))

            '''
            ###
            cls_acc.
            ###
            '''

            # (n_sess, n_cv, n_samplings, T)
            cls_acc = np.array([[cv_agg_sess_wise_cls_acc[cv_ind,sess_ind] for cv_ind in range(n_cv)] for sess_ind in range(n_sess)])
            # (n_sess, n_cv, n_samplings)
            cls_acc = cls_acc[...,timestep_ind]




            '''
            ###
            bi_stim_cross_hemi_cor
            ###
            '''

            '''
            We need to select only those bi-stim trials (of the current instructed trial type) that have videos.
            '''

            # Maps sess_ind to f1_f2 string.
            sess_ind_to_f1_f2_map = np.load(os.path.join(self.analysis_save_path, 'compute_pred', 'sess_ind_to_f1_f2_map.npy'))


            # Collect video filenames.
            if self.configs['img_type'] == 'jpg':
                video_root_path = os.path.join(self.configs['video_data_dir'], 'frames')
            elif self.configs['img_type'] == 'png':
                video_root_path = os.path.join(self.configs['video_data_dir'], 'png_frames')


            prep_root_path = self.configs['prep_root_dir']
            data_prefix = self.configs['data_prefix']

            pert_type = self.configs['pert_type']

            for sess_ind in range(n_sess):

                bi_stim_i_good_trial_numbers_with_videos = []


                # f1 = 'BAYLORGC[#]'
                # f2 = '[yyyy]_[mm]_[dd]'
                f1_f2 = sess_ind_to_f1_f2_map[sess_ind]
                temp = f1_f2.find('_')

                f1 = f1_f2[:temp]
                f2 = f1_f2[temp+1:]


                bi_stim_i_good_trials = pert_pred_utils.load_stim_i_good_trials_for_pert_pred(pert_type, \
                    self.configs['inst_trial_type'], self.configs['prep_root_dir'], f1, f2)

                print('')
                print('<bi_stim_i_good_trials (before selecting those with videos)>')
                print('inst_trial_type: ', self.configs['inst_trial_type'])
                print(bi_stim_i_good_trials)
                print('')

                for f3 in os.listdir(os.path.join(video_root_path, f1, f2)):
                    # f3 = 'E3-BAYLORGC25-2018_09_17-13'
                    if os.path.isfile(os.path.join(video_root_path, f1, f2, f3)):
                        continue

                    trial_idx_for_f3 = pert_pred_utils.map_f3_to_trial_idx(f1,f2,f3)

                    if trial_idx_for_f3 in bi_stim_i_good_trials:
                        bi_stim_i_good_trial_numbers_with_videos.append(trial_idx_for_f3)

                bi_stim_i_good_trial_numbers_with_videos.sort()

                '''
                Now compute bi_stim_cross_hemi_cor.
                '''

                # cd[j]
                cd = self.get_instant_cd_from_all_correct_control_trials(prep_root_path, data_prefix, f1, f2, analyzed_time)


                bi_stim_rates, cur_bin_centers, cur_neural_unit_location = \
                self.get_rates_for_given_i_good_trial_numbers(pert_type, prep_root_path, data_prefix, f1, f2, \
                    bi_stim_i_good_trial_numbers_with_videos)

                cur_time_ind = np.abs(cur_bin_centers - analyzed_time).argmin()

                # (n_trials, n_neurons)
                cur_bi_stim_rates = bi_stim_rates[cur_time_ind]

                cur_n_trials = len(cur_bi_stim_rates)
                
                cur_bi_stim_cd_projs = np.zeros((self.n_loc_names, cur_n_trials))

                for j, loc_name in enumerate(self.n_loc_names_list):
                    cur_bi_stim_cd_projs[j] = cur_bi_stim_rates[...,cur_neural_unit_location==loc_name].dot(cd[j])


                if cross_hemi_cor_type == 'rank':
                    bi_stim_cross_hemi_cor[sess_ind], _ = stats.spearmanr(cur_bi_stim_cd_projs[0], cur_bi_stim_cd_projs[1])
                else:
                    raise ValueError('cross_hemi_cor_type {} is not supported yet.'.format(cross_hemi_cor_type))

                n_trials_each_sess[sess_ind] = cur_n_trials


            '''
            Save
            '''

            np.save(os.path.join(data_save_path, cls_acc_save_name), cls_acc)
            np.save(os.path.join(data_save_path, bi_stim_cross_hemi_cor_save_name), bi_stim_cross_hemi_cor)
            np.save(os.path.join(data_save_path, n_trials_each_sess_save_name), n_trials_each_sess)


        return cls_acc, bi_stim_cross_hemi_cor, n_trials_each_sess










    def correlate_cls_acc_vs_bi_stim_cross_hemi_cor_not_cond_on_trial_type_across_sessions(self, sess_type_list, cross_hemi_cor_type,\
        analyzed_time):



        '''
        ###
        This plot is only concerned with bi_stim trials.
        ###
        '''
        self.configs['pert_type'] = 'bi_stim'


        cnn_channel_str = '_'.join([str(i) for i in self.configs['_cnn_channel_list']])

        if self.configs['img_type'] == 'jpg':
            img_type_str = ''
        elif self.configs['img_type'] == 'png':
            img_type_str = 'png_frames'

        if self.configs['neural_begin_time'] == -1.4 and self.configs['neural_end_time'] == -0.2 and self.configs['neural_skip_time'] == 0.2:
            neural_time_str = ''
        else:
            neural_time_str = 'begin_{:.1f}_end_{:.1f}_skip_{:.1f}'.format(self.configs['neural_begin_time'], self.configs['neural_end_time'], self.configs['neural_skip_time'])



        # analysis_save_path.
        # model_cv_dir: /data5/bkang/bk_video_ALM/NeuronalData5_codes/cd_reg_from_videos_all_no_stim_trials_cv_frame_conversion_corrected_full_cv_models
        if len(self.configs['models_cv_dir'].split('\\')) != 1:
            analysis_dir = self.configs['models_cv_dir'].split('\\')
            analysis_dir[-1] = 'pert_pred_from_videos_analysis_results'
            analysis_dir = '/'.join(analysis_dir)
        else:
            analysis_dir = 'pert_pred_from_videos_analysis_results'


        # Note that we are not including the last 'cv_ind_{}'.format(cv_ind) here. It will be included in individual methods.
        fig_save_path = os.path.join(analysis_dir, 'correlate_cls_acc_vs_bi_stim_cross_hemi_cor_not_cond_on_trial_type_across_sessions', self.configs['pert_type'], \
            'random_seed_{}'.format(self.configs['random_seed']), \
            img_type_str, neural_time_str, 'view_type_{}'.format('_'.join(self.configs['view_type'])), \
            'model_name_{}'.format(self.configs['model_name']), 'cnn_channel_list_{}'.format(cnn_channel_str), \
            'bs_{}_epochs_{}'.format(self.configs['bs'], self.configs['epochs']), 'n_cv_{}'.format(self.configs['n_cv']),\
            'n_samplings_{}'.format(self.configs['n_samplings']))

        os.makedirs(fig_save_path, exist_ok=True)

        # We will save data for plots to save time.
        data_save_path = os.path.join(fig_save_path, 'data_for_plots', 'analyzed_time_{:.1f}'.format(analyzed_time))
        os.makedirs(data_save_path, exist_ok=True)


        '''
        ###
        We compute cls_acc separately for each train-test split,
        and then average it.

        ###
        '''

        # combined_cls_acc will have shape (n_sess, n_trial_types, n_cv, n_samplings)
        combined_cls_acc = np.zeros((self.n_trial_types,), dtype=object)

        '''
        IMPORTANT:
        I previously used combined_cls_acc.fill([]), and I found that this led to a bug, 
        because (I believe) it initializes combined_cls_acc[0] and combined_cls_acc[1] 
        to the same list (i.e. they point to the same underlying list).
        '''

        for i in self.n_trial_types_list:
            combined_cls_acc[i] = []

        '''
        ###
        Since CD will be derived from control trials and we are using bi-stim CD projs, 
        we don't need to worry about train-test splits. So, we will use all correct control
        trials to compute CD.

        Also, I will aggregate bi-stim trials across train-test splits to compute the cross-hemi
        cor, because I don't have enough bi-stim trials. For example, if we have 50 bi-stim trials,
        for a given instructed trial type, we have 25, and in the test set, we have 5.
        ###
        '''

        # combined_bi_stim_cross_hemi_cor_not_cond_on_trial_type will have shape (n_sess, n_trial_types)
        combined_bi_stim_cd_proj = np.zeros((self.n_trial_types,), dtype=object)

        for i in self.n_trial_types_list:
            combined_bi_stim_cd_proj[i] = []


        # combined_n_trials_each_sess will have shape (n_sess, n_trial_types)
        combined_n_trials_each_sess = np.zeros((self.n_trial_types,), dtype=object)

        for i in self.n_trial_types_list:
            combined_n_trials_each_sess[i] = []






        if len(sess_type_list) == 1:
            sess_type_str = '({})'.format(sess_type_list[0])
        else:
            sess_type_str = ''


        for sess_type in sess_type_list:

            self.configs['sess_type'] = sess_type


            for inst_i in self.n_trial_types_list:

                self.configs['inst_trial_type'] = inst_i

                self.init_model_and_analysis_save_paths()

                '''
                cls_acc: (n_sess, n_cv, n_samplings) for a given sess_type and inst_i.
                bi_stim_cross_hemi_cor_not_cond_on_trial_type: (n_sess, n_loc_names) and X[sess_ind,j] has shape (n_trials)
                '''
                cls_acc, bi_stim_cd_proj, n_trials_each_sess = \
                self.correlate_cls_acc_vs_bi_stim_cross_hemi_cor_not_cond_on_trial_type_across_sessions_helper(cross_hemi_cor_type, analyzed_time, \
                    data_save_path)

                print('')
                print('sess_type: ', sess_type)
                print('inst_i: ', inst_i)
                print('cls_acc.shape: ', cls_acc.shape)

                combined_cls_acc[inst_i].extend(cls_acc)
                combined_bi_stim_cd_proj[inst_i].extend(bi_stim_cd_proj)
                combined_n_trials_each_sess[inst_i].extend(n_trials_each_sess)


        '''
        Aggregate across different sess types.
        '''
        for inst_i in self.n_trial_types_list:

            # (n_sess, n_cv, n_samplings)
            combined_cls_acc[inst_i] = np.array(combined_cls_acc[inst_i]) 

            # (n_sess, n_loc_names) and X[sess_ind,j] has shape (n_trials)
            combined_bi_stim_cd_proj[inst_i] = np.array(combined_bi_stim_cd_proj[inst_i]) 

            
        '''
        Finally turn them into the desired shapes.
        (n_sess, n_trial_types, n_cv, n_samplings)
        (n_sess, n_trial_types)
        '''
        combined_cls_acc = np.stack(combined_cls_acc, 1)

        n_sess = len(combined_bi_stim_cd_proj[0])

        print('')
        print('n_sess')
        print(n_sess)
        
        combined_bi_stim_cross_hemi_cor_not_cond_on_trial_type = np.zeros(n_sess)

        for sess_ind in range(n_sess):

            left_cd_proj = []
            right_cd_proj = []
            for j in range(self.n_loc_names):
                for inst_i in self.n_trial_types_list:
                    if j == 0:
                        left_cd_proj.append(combined_bi_stim_cd_proj[inst_i][sess_ind,j])
                    else:
                        right_cd_proj.append(combined_bi_stim_cd_proj[inst_i][sess_ind,j])

            # (n_trials)
            left_cd_proj = cat(left_cd_proj, 0)
            right_cd_proj = cat(right_cd_proj, 0)


            if cross_hemi_cor_type == 'rank':
                combined_bi_stim_cross_hemi_cor_not_cond_on_trial_type[sess_ind], _ = stats.spearmanr(\
                    left_cd_proj, right_cd_proj)
            else:
                raise ValueError('cross_hemi_cor_type {} is not supported yet.'.format(cross_hemi_cor_type))



        # (n_sess, n_trial_types) -> (n_sess)
        combined_n_trials_each_sess = np.stack([combined_n_trials_each_sess[inst_i] for inst_i in self.n_trial_types_list], 1)
        combined_n_trials_each_sess = combined_n_trials_each_sess.sum(1)

        print(combined_n_trials_each_sess)

        normalized_n_trials = combined_n_trials_each_sess - combined_n_trials_each_sess.min()
        normalized_n_trials /= normalized_n_trials.max()

        fig = plt.figure(figsize=(10, 10))

        gs = gridspec.GridSpec(1,1, wspace=0.4, hspace=0.4)


        ax = fig.add_subplot(gs[0])
        ax.tick_params(length=5, width=2, labelsize=20)


        '''
        xs: (n_sess)
        ys: (n_sess)
        '''
        xs = combined_cls_acc.mean(1).mean(1).mean(1)

        ys = combined_bi_stim_cross_hemi_cor_not_cond_on_trial_type

        print('len(xs)')
        print(len(xs))


        print('len(ys)')
        print(len(ys))

        print('len(normalized_n_trials)')
        print(len(normalized_n_trials))


        # ax.scatter(xs, ys, s=600, marker='o', edgecolor='k', facecolor='None')
        for k in range(len(xs)):
            ax.scatter(xs[k], ys[k], s=600, marker='o', edgecolor='k', facecolor=plt.get_cmap('Reds')(normalized_n_trials[k]))


        ax.plot(np.unique(xs), np.poly1d(np.polyfit(xs, ys, 1))(np.unique(xs)), c='k', lw=2)
        coeffs, residuals, _, _, _ = np.polyfit(xs, ys, 1, full=True)
        r2 = 1 - residuals/((ys-ys.mean())**2).sum()
        #p-value for linear regression: https://stattrek.com/regression/slope-test.aspx
        se = np.sqrt(residuals/(len(ys)-2))/np.sqrt(((xs-xs.mean())**2).sum())
        slope = coeffs[0]
        t_stat = slope/se
        if slope < 0:
            t_stat = -t_stat
        df = len(ys)-2
        from scipy.stats import t
        one_sided_p_value = t.sf(t_stat, df)


        '''
        Add a color bar for n_trials.
        '''
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        norm = mpl.colors.Normalize(vmin=combined_n_trials_each_sess.min(), vmax=combined_n_trials_each_sess.max())

        cb = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap('Reds'),
                                        norm=norm,
                                        orientation='vertical')
        cb.set_label('n_trials')


        '''
        I am showing two-sided p value.
        '''

        ax.set_title('r={:.2f}, p={:.1e}'.format(\
            np.sqrt(r2[0]), 2*one_sided_p_value[0]), fontsize=20)


        ax.set_xlabel('Classification acc.', fontsize=20)
        ax.set_ylabel('Bi-stim cross-hemi CD cor (not cond on trial-type)', fontsize=20)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)

        fig.savefig(os.path.join(fig_save_path, 'correlate_cls_acc_vs_bi_stim_cross_hemi_cor_not_cond_on_trial_type_across_sessions{}.png'.format(sess_type_str)))



    def correlate_cls_acc_vs_bi_stim_cross_hemi_cor_not_cond_on_trial_type_across_sessions_helper(self, cross_hemi_cor_type, \
        analyzed_time, data_save_path):
        '''
        Return:
        cls_acc1: (n_sess, n_cv, n_samplings) for a given sess_type and inst_i.
        bi_stim_cross_hemi_cor_not_cond_on_trial_type: (n_sess,)
        '''

        data_save_path = os.path.join(data_save_path, 'sess_type_{}'.format(self.configs['sess_type']), \
            'inst_trial_type_{}'.format(self.configs['inst_trial_type']))
        os.makedirs(data_save_path, exist_ok=True)



        cls_acc_save_name = 'cls_acc.npy'
        bi_stim_cd_proj_save_name = 'bi_stim_cd_proj.npy'
        n_trials_each_sess_save_name = 'n_trials_each_sess.npy'

        if all([os.path.isfile(os.path.join(data_save_path, save_name)) for save_name in \
            [cls_acc_save_name, bi_stim_cd_proj_save_name]]):

            cls_acc = np.load(os.path.join(data_save_path, cls_acc_save_name))
            bi_stim_cd_proj = np.load(os.path.join(data_save_path, bi_stim_cd_proj_save_name))
            n_trials_each_sess = np.load(os.path.join(data_save_path, n_trials_each_sess_save_name))


        else:


            '''
            cv_agg_sess_wise_cls_acc = np.zeros((n_cv, n_sess), dtype=object) # X[cv_ind,sess_ind] has shape (n_samplings, T)
            '''
            n_steps = int(np.rint((self.configs['neural_end_time'] - self.configs['neural_begin_time'])/self.configs['neural_skip_time'])) + 1
            pred_timesteps = self.configs['neural_begin_time'] + self.configs['neural_skip_time']*np.arange(n_steps)

            timestep_ind = np.abs(pred_timesteps - analyzed_time).argmin()


            cv_agg_sess_wise_cls_acc = np.load(os.path.join(self.analysis_save_path, 'compute_pred', \
            'cv_agg_sess_wise_acc_n_samplings_{}.npy'.format(self.configs['n_samplings'])))



            n_cv, n_sess = cv_agg_sess_wise_cls_acc.shape

            n_samplings = self.configs['n_samplings']

            
            cls_acc = np.zeros((n_sess, n_cv, n_samplings))
            bi_stim_cd_proj = np.zeros((n_sess, self.n_loc_names), dtype=object)
            n_trials_each_sess = np.zeros((n_sess,))

            '''
            ###
            cls_acc.
            ###
            '''

            # (n_sess, n_cv, n_samplings, T)
            cls_acc = np.array([[cv_agg_sess_wise_cls_acc[cv_ind,sess_ind] for cv_ind in range(n_cv)] for sess_ind in range(n_sess)])
            # (n_sess, n_cv, n_samplings)
            cls_acc = cls_acc[...,timestep_ind]




            '''
            ###
            bi_stim_cross_hemi_cor_not_cond_on_trial_type
            ###
            '''

            '''
            We need to select only those bi-stim trials (of the current instructed trial type) that have videos.
            '''

            # Maps sess_ind to f1_f2 string.
            sess_ind_to_f1_f2_map = np.load(os.path.join(self.analysis_save_path, 'compute_pred', 'sess_ind_to_f1_f2_map.npy'))


            # Collect video filenames.
            if self.configs['img_type'] == 'jpg':
                video_root_path = os.path.join(self.configs['video_data_dir'], 'frames')
            elif self.configs['img_type'] == 'png':
                video_root_path = os.path.join(self.configs['video_data_dir'], 'png_frames')


            prep_root_path = self.configs['prep_root_dir']
            data_prefix = self.configs['data_prefix']

            pert_type = self.configs['pert_type']

            for sess_ind in range(n_sess):

                bi_stim_i_good_trial_numbers_with_videos = []


                # f1 = 'BAYLORGC[#]'
                # f2 = '[yyyy]_[mm]_[dd]'
                f1_f2 = sess_ind_to_f1_f2_map[sess_ind]
                temp = f1_f2.find('_')

                f1 = f1_f2[:temp]
                f2 = f1_f2[temp+1:]


                bi_stim_i_good_trials = pert_pred_utils.load_stim_i_good_trials_for_pert_pred(pert_type, \
                    self.configs['inst_trial_type'], self.configs['prep_root_dir'], f1, f2)

                print('')
                print('<bi_stim_i_good_trials (before selecting those with videos)>')
                print('inst_trial_type: ', self.configs['inst_trial_type'])
                print(bi_stim_i_good_trials)
                print('')

                for f3 in os.listdir(os.path.join(video_root_path, f1, f2)):
                    # f3 = 'E3-BAYLORGC25-2018_09_17-13'
                    if os.path.isfile(os.path.join(video_root_path, f1, f2, f3)):
                        continue

                    trial_idx_for_f3 = pert_pred_utils.map_f3_to_trial_idx(f1,f2,f3)

                    if trial_idx_for_f3 in bi_stim_i_good_trials:
                        bi_stim_i_good_trial_numbers_with_videos.append(trial_idx_for_f3)

                bi_stim_i_good_trial_numbers_with_videos.sort()

                '''
                Now compute bi_stim_cross_hemi_cor_not_cond_on_trial_type.
                '''

                # cd[j]
                cd = self.get_instant_cd_from_all_correct_control_trials(prep_root_path, data_prefix, f1, f2, analyzed_time)


                bi_stim_rates, cur_bin_centers, cur_neural_unit_location = \
                self.get_rates_for_given_i_good_trial_numbers(pert_type, prep_root_path, data_prefix, f1, f2, \
                    bi_stim_i_good_trial_numbers_with_videos)

                cur_time_ind = np.abs(cur_bin_centers - analyzed_time).argmin()

                # (n_trials, n_neurons)
                cur_bi_stim_rates = bi_stim_rates[cur_time_ind]

                cur_n_trials = len(cur_bi_stim_rates)
                
                for j, loc_name in enumerate(self.n_loc_names_list):
                    bi_stim_cd_proj[sess_ind,j] = cur_bi_stim_rates[...,cur_neural_unit_location==loc_name].dot(cd[j])


                n_trials_each_sess[sess_ind] = cur_n_trials


            '''
            Save
            '''

            np.save(os.path.join(data_save_path, cls_acc_save_name), cls_acc)
            np.save(os.path.join(data_save_path, bi_stim_cd_proj_save_name), bi_stim_cd_proj)
            np.save(os.path.join(data_save_path, n_trials_each_sess_save_name), n_trials_each_sess)


        return cls_acc, bi_stim_cd_proj, n_trials_each_sess








    def correlate_cls_acc_vs_bi_stim_cross_hemi_cor_not_cond_on_trial_type_across_sessions_using_guang_bi_stim_cor(self, sess_type_list, cross_hemi_cor_type,\
        analyzed_time):



        '''
        ###
        This plot is only concerned with bi_stim trials.
        ###
        '''
        self.configs['pert_type'] = 'bi_stim'


        cnn_channel_str = '_'.join([str(i) for i in self.configs['_cnn_channel_list']])

        if self.configs['img_type'] == 'jpg':
            img_type_str = ''
        elif self.configs['img_type'] == 'png':
            img_type_str = 'png_frames'

        if self.configs['neural_begin_time'] == -1.4 and self.configs['neural_end_time'] == -0.2 and self.configs['neural_skip_time'] == 0.2:
            neural_time_str = ''
        else:
            neural_time_str = 'begin_{:.1f}_end_{:.1f}_skip_{:.1f}'.format(self.configs['neural_begin_time'], self.configs['neural_end_time'], self.configs['neural_skip_time'])



        # analysis_save_path.
        # model_cv_dir: /data5/bkang/bk_video_ALM/NeuronalData5_codes/cd_reg_from_videos_all_no_stim_trials_cv_frame_conversion_corrected_full_cv_models
        if len(self.configs['models_cv_dir'].split('\\')) != 1:
            analysis_dir = self.configs['models_cv_dir'].split('\\')
            analysis_dir[-1] = 'pert_pred_from_videos_analysis_results'
            analysis_dir = '/'.join(analysis_dir)
        else:
            analysis_dir = 'pert_pred_from_videos_analysis_results'


        # Note that we are not including the last 'cv_ind_{}'.format(cv_ind) here. It will be included in individual methods.
        fig_save_path = os.path.join(analysis_dir, 'correlate_cls_acc_vs_bi_stim_cross_hemi_cor_not_cond_on_trial_type_across_sessions_using_guang_bi_stim_cor', self.configs['pert_type'], \
            'random_seed_{}'.format(self.configs['random_seed']), \
            img_type_str, neural_time_str, 'view_type_{}'.format('_'.join(self.configs['view_type'])), \
            'model_name_{}'.format(self.configs['model_name']), 'cnn_channel_list_{}'.format(cnn_channel_str), \
            'bs_{}_epochs_{}'.format(self.configs['bs'], self.configs['epochs']), 'n_cv_{}'.format(self.configs['n_cv']),\
            'n_samplings_{}'.format(self.configs['n_samplings']))

        os.makedirs(fig_save_path, exist_ok=True)

        # We will save data for plots to save time.
        data_save_path = os.path.join(fig_save_path, 'data_for_plots', 'analyzed_time_{:.1f}'.format(analyzed_time))
        os.makedirs(data_save_path, exist_ok=True)


        '''
        ###
        We compute cls_acc separately for each train-test split,
        and then average it.

        ###
        '''

        # combined_cls_acc will have shape (n_sess, n_trial_types, n_cv, n_samplings)
        combined_cls_acc = np.zeros((self.n_trial_types,), dtype=object)

        '''
        IMPORTANT:
        I previously used combined_cls_acc.fill([]), and I found that this led to a bug, 
        because (I believe) it initializes combined_cls_acc[0] and combined_cls_acc[1] 
        to the same list (i.e. they point to the same underlying list).
        '''

        for i in self.n_trial_types_list:
            combined_cls_acc[i] = []

        '''
        ###
        Since CD will be derived from control trials and we are using bi-stim CD projs, 
        we don't need to worry about train-test splits. So, we will use all correct control
        trials to compute CD.

        Also, I will aggregate bi-stim trials across train-test splits to compute the cross-hemi
        cor, because I don't have enough bi-stim trials. For example, if we have 50 bi-stim trials,
        for a given instructed trial type, we have 25, and in the test set, we have 5.
        ###
        '''




        if len(sess_type_list) == 1:
            sess_type_str = '({})'.format(sess_type_list[0])
        else:
            sess_type_str = ''


        for sess_type in sess_type_list:

            self.configs['sess_type'] = sess_type


            for inst_i in self.n_trial_types_list:

                self.configs['inst_trial_type'] = inst_i

                self.init_model_and_analysis_save_paths()

                '''
                cls_acc: (n_sess, n_cv, n_samplings) for a given sess_type and inst_i.
                '''
                cls_acc = \
                self.correlate_cls_acc_vs_bi_stim_cross_hemi_cor_not_cond_on_trial_type_across_sessions_using_guang_bi_stim_cor_helper(cross_hemi_cor_type, analyzed_time, \
                    data_save_path)

                print('')
                print('sess_type: ', sess_type)
                print('inst_i: ', inst_i)
                print('cls_acc.shape: ', cls_acc.shape)

                combined_cls_acc[inst_i].extend(cls_acc)


        '''
        Aggregate across different sess types.
        '''
        for inst_i in self.n_trial_types_list:

            # (n_sess, n_cv, n_samplings)
            combined_cls_acc[inst_i] = np.array(combined_cls_acc[inst_i]) 


            
        '''
        Finally turn them into the desired shapes.
        (n_sess, n_trial_types, n_cv, n_samplings)
        (n_sess)
        '''
        combined_cls_acc = np.stack(combined_cls_acc, 1)

        
        combined_bi_stim_cross_hemi_cor_not_cond_on_trial_type = np.zeros(n_sess)

        for sess_ind in range(n_sess):
            
            '''
            TBD below.
            '''



        fig = plt.figure(figsize=(10, 10))

        gs = gridspec.GridSpec(1,1, wspace=0.4, hspace=0.4)


        ax = fig.add_subplot(gs[0])
        ax.tick_params(length=5, width=2, labelsize=20)


        '''
        xs: (n_sess)
        ys: (n_sess)
        '''
        xs = combined_cls_acc.mean(1).mean(1).mean(1)

        ys = combined_bi_stim_cross_hemi_cor_not_cond_on_trial_type

        print('len(xs)')
        print(len(xs))


        print('len(ys)')
        print(len(ys))

        print('len(normalized_n_trials)')
        print(len(normalized_n_trials))


        # ax.scatter(xs, ys, s=600, marker='o', edgecolor='k', facecolor='None')
        for k in range(len(xs)):
            ax.scatter(xs[k], ys[k], s=600, marker='o', edgecolor='k', facecolor=plt.get_cmap('Reds')(normalized_n_trials[k]))


        ax.plot(np.unique(xs), np.poly1d(np.polyfit(xs, ys, 1))(np.unique(xs)), c='k', lw=2)
        coeffs, residuals, _, _, _ = np.polyfit(xs, ys, 1, full=True)
        r2 = 1 - residuals/((ys-ys.mean())**2).sum()
        #p-value for linear regression: https://stattrek.com/regression/slope-test.aspx
        se = np.sqrt(residuals/(len(ys)-2))/np.sqrt(((xs-xs.mean())**2).sum())
        slope = coeffs[0]
        t_stat = slope/se
        if slope < 0:
            t_stat = -t_stat
        df = len(ys)-2
        from scipy.stats import t
        one_sided_p_value = t.sf(t_stat, df)


        '''
        Add a color bar for n_trials.
        '''
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        norm = mpl.colors.Normalize(vmin=combined_n_trials_each_sess.min(), vmax=combined_n_trials_each_sess.max())

        cb = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap('Reds'),
                                        norm=norm,
                                        orientation='vertical')
        cb.set_label('n_trials')


        '''
        I am showing two-sided p value.
        '''

        ax.set_title('r={:.2f}, p={:.1e}'.format(\
            np.sqrt(r2[0]), 2*one_sided_p_value[0]), fontsize=20)


        ax.set_xlabel('Classification acc.', fontsize=20)
        ax.set_ylabel('Bi-stim cross-hemi CD cor (not cond on trial-type)', fontsize=20)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)

        fig.savefig(os.path.join(fig_save_path, 'correlate_cls_acc_vs_bi_stim_cross_hemi_cor_not_cond_on_trial_type_across_sessions_using_guang_bi_stim_cor{}.png'.format(sess_type_str)))



    def correlate_cls_acc_vs_bi_stim_cross_hemi_cor_not_cond_on_trial_type_across_sessions_using_guang_bi_stim_cor_helper(self, cross_hemi_cor_type, \
        analyzed_time, data_save_path):



        '''
        TBD below.
        '''










        '''
        Return:
        cls_acc1: (n_sess, n_cv, n_samplings) for a given sess_type and inst_i.
        bi_stim_cross_hemi_cor_not_cond_on_trial_type: (n_sess,)
        '''

        data_save_path = os.path.join(data_save_path, 'sess_type_{}'.format(self.configs['sess_type']), \
            'inst_trial_type_{}'.format(self.configs['inst_trial_type']))
        os.makedirs(data_save_path, exist_ok=True)



        cls_acc_save_name = 'cls_acc.npy'
        bi_stim_cd_proj_save_name = 'bi_stim_cd_proj.npy'
        n_trials_each_sess_save_name = 'n_trials_each_sess.npy'

        if all([os.path.isfile(os.path.join(data_save_path, save_name)) for save_name in \
            [cls_acc_save_name, bi_stim_cd_proj_save_name]]):

            cls_acc = np.load(os.path.join(data_save_path, cls_acc_save_name))
            bi_stim_cd_proj = np.load(os.path.join(data_save_path, bi_stim_cd_proj_save_name))
            n_trials_each_sess = np.load(os.path.join(data_save_path, n_trials_each_sess_save_name))


        else:


            '''
            cv_agg_sess_wise_cls_acc = np.zeros((n_cv, n_sess), dtype=object) # X[cv_ind,sess_ind] has shape (n_samplings, T)
            '''
            n_steps = int(np.rint((self.configs['neural_end_time'] - self.configs['neural_begin_time'])/self.configs['neural_skip_time'])) + 1
            pred_timesteps = self.configs['neural_begin_time'] + self.configs['neural_skip_time']*np.arange(n_steps)

            timestep_ind = np.abs(pred_timesteps - analyzed_time).argmin()


            cv_agg_sess_wise_cls_acc = np.load(os.path.join(self.analysis_save_path, 'compute_pred', \
            'cv_agg_sess_wise_acc_n_samplings_{}.npy'.format(self.configs['n_samplings'])))



            n_cv, n_sess = cv_agg_sess_wise_cls_acc.shape

            n_samplings = self.configs['n_samplings']

            
            cls_acc = np.zeros((n_sess, n_cv, n_samplings))
            bi_stim_cd_proj = np.zeros((n_sess, self.n_loc_names), dtype=object)
            n_trials_each_sess = np.zeros((n_sess,))

            '''
            ###
            cls_acc.
            ###
            '''

            # (n_sess, n_cv, n_samplings, T)
            cls_acc = np.array([[cv_agg_sess_wise_cls_acc[cv_ind,sess_ind] for cv_ind in range(n_cv)] for sess_ind in range(n_sess)])
            # (n_sess, n_cv, n_samplings)
            cls_acc = cls_acc[...,timestep_ind]




            '''
            ###
            bi_stim_cross_hemi_cor_not_cond_on_trial_type
            ###
            '''

            '''
            We need to select only those bi-stim trials (of the current instructed trial type) that have videos.
            '''

            # Maps sess_ind to f1_f2 string.
            sess_ind_to_f1_f2_map = np.load(os.path.join(self.analysis_save_path, 'compute_pred', 'sess_ind_to_f1_f2_map.npy'))


            # Collect video filenames.
            if self.configs['img_type'] == 'jpg':
                video_root_path = os.path.join(self.configs['video_data_dir'], 'frames')
            elif self.configs['img_type'] == 'png':
                video_root_path = os.path.join(self.configs['video_data_dir'], 'png_frames')


            prep_root_path = self.configs['prep_root_dir']
            data_prefix = self.configs['data_prefix']

            pert_type = self.configs['pert_type']

            for sess_ind in range(n_sess):

                bi_stim_i_good_trial_numbers_with_videos = []


                # f1 = 'BAYLORGC[#]'
                # f2 = '[yyyy]_[mm]_[dd]'
                f1_f2 = sess_ind_to_f1_f2_map[sess_ind]
                temp = f1_f2.find('_')

                f1 = f1_f2[:temp]
                f2 = f1_f2[temp+1:]


                bi_stim_i_good_trials = pert_pred_utils.load_stim_i_good_trials_for_pert_pred(pert_type, \
                    self.configs['inst_trial_type'], self.configs['prep_root_dir'], f1, f2)

                print('')
                print('<bi_stim_i_good_trials (before selecting those with videos)>')
                print('inst_trial_type: ', self.configs['inst_trial_type'])
                print(bi_stim_i_good_trials)
                print('')

                for f3 in os.listdir(os.path.join(video_root_path, f1, f2)):
                    # f3 = 'E3-BAYLORGC25-2018_09_17-13'
                    if os.path.isfile(os.path.join(video_root_path, f1, f2, f3)):
                        continue

                    trial_idx_for_f3 = pert_pred_utils.map_f3_to_trial_idx(f1,f2,f3)

                    if trial_idx_for_f3 in bi_stim_i_good_trials:
                        bi_stim_i_good_trial_numbers_with_videos.append(trial_idx_for_f3)

                bi_stim_i_good_trial_numbers_with_videos.sort()

                '''
                Now compute bi_stim_cross_hemi_cor_not_cond_on_trial_type.
                '''

                # cd[j]
                cd = self.get_instant_cd_from_all_correct_control_trials(prep_root_path, data_prefix, f1, f2, analyzed_time)


                bi_stim_rates, cur_bin_centers, cur_neural_unit_location = \
                self.get_rates_for_given_i_good_trial_numbers(pert_type, prep_root_path, data_prefix, f1, f2, \
                    bi_stim_i_good_trial_numbers_with_videos)

                cur_time_ind = np.abs(cur_bin_centers - analyzed_time).argmin()

                # (n_trials, n_neurons)
                cur_bi_stim_rates = bi_stim_rates[cur_time_ind]

                cur_n_trials = len(cur_bi_stim_rates)
                
                for j, loc_name in enumerate(self.n_loc_names_list):
                    bi_stim_cd_proj[sess_ind,j] = cur_bi_stim_rates[...,cur_neural_unit_location==loc_name].dot(cd[j])


                n_trials_each_sess[sess_ind] = cur_n_trials


            '''
            Save
            '''

            np.save(os.path.join(data_save_path, cls_acc_save_name), cls_acc)
            np.save(os.path.join(data_save_path, bi_stim_cd_proj_save_name), bi_stim_cd_proj)
            np.save(os.path.join(data_save_path, n_trials_each_sess_save_name), n_trials_each_sess)


        return cls_acc, bi_stim_cd_proj, n_trials_each_sess




















































    def compare_bi_stim_cross_hemi_cor_between_high_and_low_score_trials(self, sess_type_list, cross_hemi_cor_type,\
        analyzed_time):



        '''
        ###
        This plot is only concerned with bi_stim trials.
        ###
        '''
        self.configs['pert_type'] = 'bi_stim'


        cnn_channel_str = '_'.join([str(i) for i in self.configs['_cnn_channel_list']])

        if self.configs['img_type'] == 'jpg':
            img_type_str = ''
        elif self.configs['img_type'] == 'png':
            img_type_str = 'png_frames'

        if self.configs['neural_begin_time'] == -1.4 and self.configs['neural_end_time'] == -0.2 and self.configs['neural_skip_time'] == 0.2:
            neural_time_str = ''
        else:
            neural_time_str = 'begin_{:.1f}_end_{:.1f}_skip_{:.1f}'.format(self.configs['neural_begin_time'], self.configs['neural_end_time'], self.configs['neural_skip_time'])



        # analysis_save_path.
        # model_cv_dir: /data5/bkang/bk_video_ALM/NeuronalData5_codes/cd_reg_from_videos_all_no_stim_trials_cv_frame_conversion_corrected_full_cv_models
        if len(self.configs['models_cv_dir'].split('\\')) != 1:
            analysis_dir = self.configs['models_cv_dir'].split('\\')
            analysis_dir[-1] = 'pert_pred_from_videos_analysis_results'
            analysis_dir = '/'.join(analysis_dir)
        else:
            analysis_dir = 'pert_pred_from_videos_analysis_results'


        # Note that we are not including the last 'cv_ind_{}'.format(cv_ind) here. It will be included in individual methods.
        fig_save_path = os.path.join(analysis_dir, 'compare_bi_stim_cross_hemi_cor_between_high_and_low_score_trials', self.configs['pert_type'], \
            'random_seed_{}'.format(self.configs['random_seed']), \
            img_type_str, neural_time_str, 'view_type_{}'.format('_'.join(self.configs['view_type'])), \
            'model_name_{}'.format(self.configs['model_name']), 'cnn_channel_list_{}'.format(cnn_channel_str), \
            'bs_{}_epochs_{}'.format(self.configs['bs'], self.configs['epochs']), 'n_cv_{}'.format(self.configs['n_cv']),\
            'n_samplings_{}'.format(self.configs['n_samplings']))

        os.makedirs(fig_save_path, exist_ok=True)

        # We will save data for plots to save time.
        data_save_path = os.path.join(fig_save_path, 'data_for_plots', 'analyzed_time_{:.1f}'.format(analyzed_time))
        os.makedirs(data_save_path, exist_ok=True)


        '''
        ###
        We will first aggregate test scores across train-test splits. And then,
        divide them into high and low score groups, and with each group, we calculate
        cross-hemi correlations. We will do this separately for each instructed trial type.

        ###
        '''

        # combined_bi_stim_cross_hemi_cor will have shape (n_sess, n_trial_types, n_groups)
        combined_bi_stim_cross_hemi_cor = np.zeros((self.n_trial_types,), dtype=object)

        for i in self.n_trial_types_list:
            combined_bi_stim_cross_hemi_cor[i] = []


        if len(sess_type_list) == 1:
            sess_type_str = '({})'.format(sess_type_list[0])
        else:
            sess_type_str = ''


        for sess_type in sess_type_list:

            self.configs['sess_type'] = sess_type


            for inst_i in self.n_trial_types_list:

                self.configs['inst_trial_type'] = inst_i

                self.init_model_and_analysis_save_paths()

                '''
                bi_stim_cross_hemi_cor: (n_sess, n_groups)
                '''
                bi_stim_cross_hemi_cor = \
                self.compare_bi_stim_cross_hemi_cor_between_high_and_low_score_trials_helper(cross_hemi_cor_type, analyzed_time, \
                    data_save_path)

                print('')
                print('sess_type: ', sess_type)
                print('inst_i: ', inst_i)
                print('bi_stim_cross_hemi_cor.shape: ', bi_stim_cross_hemi_cor.shape)


                combined_bi_stim_cross_hemi_cor[inst_i].extend(bi_stim_cross_hemi_cor)


        '''
        Aggregate across different sess types.
        '''
        for inst_i in self.n_trial_types_list:


            # (n_sess,)
            combined_bi_stim_cross_hemi_cor[inst_i] = np.array(combined_bi_stim_cross_hemi_cor[inst_i]) 


        print('')
        print('combined_bi_stim_cross_hemi_cor.shape: ', combined_bi_stim_cross_hemi_cor.shape)
        print('')
            
        '''
        Finally turn them into the desired shapes.
        (n_sess, n_trial_types, n_groups)
        '''
        combined_bi_stim_cross_hemi_cor = np.stack(combined_bi_stim_cross_hemi_cor, 1)

        # Average over trial types.
        combined_bi_stim_cross_hemi_cor = combined_bi_stim_cross_hemi_cor.mean(1) # (n_sess, n_groups)



        fig = plt.figure(figsize=(7,10))

        gs = gridspec.GridSpec(1,1)

        # For t-test
        from scipy import stats


        ax = fig.add_subplot(gs[0])




        ax.bar(np.arange(2), combined_bi_stim_cross_hemi_cor.mean(0), width=0.4, edgecolor='k', facecolor=plt.get_cmap('Greys')(0.5), zorder=-1)

        t_stat, pval = stats.ttest_rel(combined_bi_stim_cross_hemi_cor[:,0], combined_bi_stim_cross_hemi_cor[:,1])

        print('')
        print('t_stat: ', t_stat)
        print('two-sided pval: {:.2e}'.format(pval))

        pval_str = 'p = {:.1e}'.format(pval)

        y_max = combined_bi_stim_cross_hemi_cor.max()

        barplot_annotate_brackets(ax, 0, 1, pval_str, np.arange(2), combined_bi_stim_cross_hemi_cor.mean(0), barh=0.1, fs=25, y=y_max*1.1)


        ax.scatter([0]*len(combined_bi_stim_cross_hemi_cor[:,0]), combined_bi_stim_cross_hemi_cor[:,0], edgecolor='k', facecolor='w', s=200)
        ax.scatter([1]*len(combined_bi_stim_cross_hemi_cor[:,1]), combined_bi_stim_cross_hemi_cor[:,1], edgecolor='k', facecolor='w', s=200)


        ax.axhline(0, ls='--', c='k')

        for idx in range(len(combined_bi_stim_cross_hemi_cor[:,0])):
            ax.plot(np.arange(2), [combined_bi_stim_cross_hemi_cor[idx,0], combined_bi_stim_cross_hemi_cor[idx,1]], c='k', ls='-')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)


        ax.tick_params(length=5, width=2, labelsize=25)


        ax.set_xticklabels(['More similar\nto control', 'Less similar\nto control'], fontsize=25)
        ax.set_xticks(np.arange(2))
        ax.set_ylabel('Bi-stim cross-hemi CD cor', fontsize=25)

        # ax.set_yticks(np.arange(0, 0.16+0.04, 0.04))
        # ax.set_yticklabels(['{:.2f}'.format(x) for x in np.arange(0, 0.16+0.04, 0.04)], fontsize=20)

        # ytick_arr = np.array([0, 0.10, 0.2])
        # ax.set_yticks(ytick_arr)
        # ax.set_yticklabels(['{:.2f}'.format(x) for x in ytick_arr], fontsize=30)


        plt.gcf().subplots_adjust(left=0.25)



        fig.savefig(os.path.join(fig_save_path, 'compare_bi_stim_cross_hemi_cor_between_high_and_low_score_trials{}.png'.format(sess_type_str)))






    def compare_bi_stim_cross_hemi_cor_between_high_and_low_score_trials_helper(self, cross_hemi_cor_type, \
        analyzed_time, data_save_path):
        '''
        Return:
        bi_stim_cross_hemi_cor: (n_sess, n_groups)
        '''

        data_save_path = os.path.join(data_save_path, 'sess_type_{}'.format(self.configs['sess_type']), \
            'inst_trial_type_{}'.format(self.configs['inst_trial_type']))
        os.makedirs(data_save_path, exist_ok=True)



        bi_stim_cross_hemi_cor_save_name = 'bi_stim_cross_hemi_in_high_and_low_score_groups_cor_type_{}.npy'.format(cross_hemi_cor_type)

        if all([os.path.isfile(os.path.join(data_save_path, save_name)) for save_name in \
            [bi_stim_cross_hemi_cor_save_name]]):

            bi_stim_cross_hemi_cor = np.load(os.path.join(data_save_path, bi_stim_cross_hemi_cor_save_name))


        else:


            '''
            cv_agg_sess_wise_trial_type = np.zeros((n_cv, n_sess), dtype=object) #  X[cv_ind,sess_ind] has shape (n_trials)
            cv_agg_sess_wise_pre_sigmoid_score = np.zeros((n_cv, n_sess), dtype=object) # X[cv_ind,sess_ind] has shape (n_trials, T)
            cv_agg_sess_wise_trial_idx = np.zeros((n_cv, n_sess), dtype=object) # X[cv_ind,sess_ind] has shape (n_trials)
            '''
            n_steps = int(np.rint((self.configs['neural_end_time'] - self.configs['neural_begin_time'])/self.configs['neural_skip_time'])) + 1
            pred_timesteps = self.configs['neural_begin_time'] + self.configs['neural_skip_time']*np.arange(n_steps)

            timestep_ind = np.abs(pred_timesteps - analyzed_time).argmin()



            cv_agg_sess_wise_trial_type = np.load(os.path.join(self.analysis_save_path, 'compute_pred', \
            'cv_agg_sess_wise_trial_type.npy'))


            cv_agg_sess_wise_pre_sigmoid_score = np.load(os.path.join(self.analysis_save_path, 'compute_pred', \
            'cv_agg_sess_wise_pre_sigmoid_score.npy'))

            cv_agg_sess_wise_trial_idx = np.load(os.path.join(self.analysis_save_path, 'compute_pred', \
            'cv_agg_sess_wise_trial_idx.npy'))


            n_cv, n_sess = cv_agg_sess_wise_pre_sigmoid_score.shape

            n_groups = 2



            '''
            ###
            First, need to get trial_idx_in_each_group.
            ###
            '''
            trial_idx_in_each_group = np.zeros((n_sess, n_groups), dtype=object)


            '''
            1. Sort bi-stim trials according to pre_sigmoid_score at the analyzed time step aggregated over train-test splits.
            '''

            n_steps = int(np.rint((self.configs['neural_end_time'] - self.configs['neural_begin_time'])/self.configs['neural_skip_time'])) + 1
            pred_timesteps = self.configs['neural_begin_time'] + self.configs['neural_skip_time']*np.arange(n_steps)

            timestep_ind = np.abs(pred_timesteps - analyzed_time).argmin()

            for sess_ind in range(n_sess):

                cur_trial_type = []
                cur_score = []
                cur_trial_idx = []

                for cv_ind in range(n_cv):
                    cur_trial_type.extend(cv_agg_sess_wise_trial_type[cv_ind,sess_ind])
                    cur_score.extend(cv_agg_sess_wise_pre_sigmoid_score[cv_ind,sess_ind][:,timestep_ind])
                    cur_trial_idx.extend(cv_agg_sess_wise_trial_idx[cv_ind,sess_ind])

                cur_trial_type = np.array(cur_trial_type)
                cur_score = np.array(cur_score)
                cur_trial_idx = np.array(cur_trial_idx)

                assert len(np.unique(cur_trial_idx)) == len(cur_trial_idx)

                '''
                Need to select bi-stim trials.
                '''
                cur_bi_stim_mask = (cur_trial_type==1)

                cur_bi_stim_score = cur_score[cur_bi_stim_mask]
                cur_bi_stim_trial_idx = cur_trial_idx[cur_bi_stim_mask]

                score_sorted_idxs = np.argsort(cur_bi_stim_score)
                cur_n_trials = len(score_sorted_idxs)

                # print('')
                # print('score_sorted_idxs')
                # print(score_sorted_idxs)

                trial_idx_in_each_group[sess_ind,0] = cur_bi_stim_trial_idx[score_sorted_idxs[:cur_n_trials//2]]
                trial_idx_in_each_group[sess_ind,1] = cur_bi_stim_trial_idx[score_sorted_idxs[cur_n_trials//2:]]

                # print('')
                # print('trial_idx_in_each_group[sess_ind,0]')
                # print(trial_idx_in_each_group[sess_ind,0])
                # print('')
                # print('trial_idx_in_each_group[sess_ind,1]')
                # print(trial_idx_in_each_group[sess_ind,1])


            '''
            ###
            bi_stim_cross_hemi_cor
            ###
            '''

            bi_stim_cross_hemi_cor = np.zeros((n_sess, n_groups))


            '''
            We need to select only those bi-stim trials (of the current instructed trial type) that have videos.
            '''

            # Maps sess_ind to f1_f2 string.
            sess_ind_to_f1_f2_map = np.load(os.path.join(self.analysis_save_path, 'compute_pred', 'sess_ind_to_f1_f2_map.npy'))


            # Collect video filenames.
            if self.configs['img_type'] == 'jpg':
                video_root_path = os.path.join(self.configs['video_data_dir'], 'frames')
            elif self.configs['img_type'] == 'png':
                video_root_path = os.path.join(self.configs['video_data_dir'], 'png_frames')


            prep_root_path = self.configs['prep_root_dir']
            data_prefix = self.configs['data_prefix']

            pert_type = self.configs['pert_type']

            for sess_ind in range(n_sess):

                # f1 = 'BAYLORGC[#]'
                # f2 = '[yyyy]_[mm]_[dd]'
                f1_f2 = sess_ind_to_f1_f2_map[sess_ind]
                temp = f1_f2.find('_')

                f1 = f1_f2[:temp]
                f2 = f1_f2[temp+1:]


                # cd[j]
                cd = self.get_instant_cd_from_all_correct_control_trials(prep_root_path, data_prefix, f1, f2, analyzed_time)


                for g in range(n_groups):

                    cur_i_good_trial_numbers = trial_idx_in_each_group[sess_ind,g]

                    '''
                    Now compute bi_stim_cross_hemi_cor.
                    '''

                    bi_stim_rates, cur_bin_centers, cur_neural_unit_location = \
                    self.get_rates_for_given_i_good_trial_numbers(pert_type, prep_root_path, data_prefix, f1, f2, \
                        cur_i_good_trial_numbers)

                    cur_time_ind = np.abs(cur_bin_centers - analyzed_time).argmin()

                    # (n_trials, n_neurons)
                    cur_bi_stim_rates = bi_stim_rates[cur_time_ind]

                    cur_n_trials = len(cur_bi_stim_rates)
                    
                    cur_bi_stim_cd_projs = np.zeros((self.n_loc_names, cur_n_trials))

                    for j, loc_name in enumerate(self.n_loc_names_list):
                        cur_bi_stim_cd_projs[j] = cur_bi_stim_rates[...,cur_neural_unit_location==loc_name].dot(cd[j])


                    if cross_hemi_cor_type == 'rank':
                        bi_stim_cross_hemi_cor[sess_ind,g], _ = stats.spearmanr(cur_bi_stim_cd_projs[0], cur_bi_stim_cd_projs[1])
                    else:
                        raise ValueError('cross_hemi_cor_type {} is not supported yet.'.format(cross_hemi_cor_type))



            '''
            Save
            '''

            np.save(os.path.join(data_save_path, bi_stim_cross_hemi_cor_save_name), bi_stim_cross_hemi_cor)


        return bi_stim_cross_hemi_cor





    def get_instant_cd_from_all_correct_control_trials(self, prep_root_path, data_prefix, f1, f2, analyzed_time):

        filename = os.path.join('NeuronalData', '_'.join([f1,f2])+'.mat')
        if not os.path.exists(filename):
            filename = os.path.join('NeuronalData', '_'.join([f1+'kilosort',f2])+'.mat')


        prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1,f2]))
        if not os.path.exists(prep_save_path):
            prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1+'kilosort',f2]))



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

        i_good_trials = np.array(sess.i_good_trials.copy())


        assert np.allclose(sess.task_pole_off_time.min(), np.nanmin(sess.stim_on_time))

        # print('')
        # print('f1: {} / f2: {}'.format(f1, f2))
        # print('task_pole_on_time: min {:.1f} max {:.1f}'.format(sess.task_pole_on_time.min(), sess.task_pole_on_time.max()))
        # print('task_pole_off_time: min {:.1f} max {:.1f}'.format(sess.task_pole_off_time.min(), sess.task_pole_off_time.max()))
        # print('stim_on_time: min {:.1f} max {:.1f}'.format(np.nanmin(sess.stim_on_time), np.nanmax(sess.stim_on_time)))
        # print('stim_off_time: min {:.1f} max {:.1f}'.format(np.nanmin(sess.stim_off_time), np.nanmax(sess.stim_off_time)))


        no_stim_train_set = ALMDataset(filename, root_path=prep_root_path, data_prefix=data_prefix, stim_type='no_stim', \
            data_type='train', neural_unit_location=neural_unit_location, loc_name='both', bin_min=0, bin_max=len(bin_centers)-1)
        no_stim_test_set = ALMDataset(filename, root_path=prep_root_path, data_prefix=data_prefix, stim_type='no_stim', \
            data_type='test', neural_unit_location=neural_unit_location, loc_name='both', bin_min=0, bin_max=len(bin_centers)-1)
        
        no_stim_train_rates = no_stim_train_set.rates
        no_stim_test_rates = no_stim_test_set.rates
        no_stim_rates = np.concatenate([no_stim_train_rates, no_stim_test_rates], 1)


        no_stim_train_trial_type_labels = no_stim_train_set.trial_type_labels
        no_stim_test_trial_type_labels = no_stim_test_set.trial_type_labels
        no_stim_trial_type_labels = np.concatenate([no_stim_train_trial_type_labels, no_stim_test_trial_type_labels], 0)

        no_stim_train_labels = no_stim_train_set.labels
        no_stim_test_labels = no_stim_test_set.labels
        no_stim_labels = np.concatenate([no_stim_train_labels, no_stim_test_labels], 0)

        no_stim_suc_labels = (no_stim_trial_type_labels==no_stim_labels).astype(int)


        analyzed_bin = np.abs(bin_centers - analyzed_time).argmin()

        cur_rates = no_stim_rates[analyzed_bin]

        trial_avg_fr_diff = cur_rates[(no_stim_trial_type_labels==1)*(no_stim_suc_labels==1)].mean(0) - \
        cur_rates[(no_stim_trial_type_labels==0)*(no_stim_suc_labels==1)].mean(0)

        cd = np.zeros((self.n_loc_names,), dtype=object)

        for j, loc_name in enumerate(self.n_loc_names_list):
            temp = trial_avg_fr_diff[neural_unit_location==loc_name]
            cd[j] = temp/np.linalg.norm(temp)

        return cd







    def get_rates_for_given_i_good_trial_numbers(self, pert_type, prep_root_path, data_prefix, f1, f2, i_good_trial_numbers):

        filename = os.path.join('NeuronalData', '_'.join([f1,f2])+'.mat')
        if not os.path.exists(filename):
            filename = os.path.join('NeuronalData', '_'.join([f1+'kilosort',f2])+'.mat')

        prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1,f2]))
        if not os.path.exists(prep_save_path):
            prep_save_path = os.path.join(prep_root_path, data_prefix + 'NeuronalData', '_'.join([f1+'kilosort',f2]))



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

        i_good_trials = np.array(sess.i_good_trials.copy())


        assert np.allclose(sess.task_pole_off_time.min(), np.nanmin(sess.stim_on_time))

        print('')
        print('f1: {} / f2: {}'.format(f1, f2))
        # print('task_pole_on_time: min {:.1f} max {:.1f}'.format(sess.task_pole_on_time.min(), sess.task_pole_on_time.max()))
        # print('task_pole_off_time: min {:.1f} max {:.1f}'.format(sess.task_pole_off_time.min(), sess.task_pole_off_time.max()))
        # print('stim_on_time: min {:.1f} max {:.1f}'.format(np.nanmin(sess.stim_on_time), np.nanmax(sess.stim_on_time)))
        # print('stim_off_time: min {:.1f} max {:.1f}'.format(np.nanmin(sess.stim_off_time), np.nanmax(sess.stim_off_time)))


        stim_train_set = ALMDataset(filename, root_path=prep_root_path, data_prefix=data_prefix, stim_type=pert_type, \
            data_type='train', neural_unit_location=neural_unit_location, loc_name='both', bin_min=0, bin_max=len(bin_centers)-1)
        stim_test_set = ALMDataset(filename, root_path=prep_root_path, data_prefix=data_prefix, stim_type=pert_type, \
            data_type='test', neural_unit_location=neural_unit_location, loc_name='both', bin_min=0, bin_max=len(bin_centers)-1)
        
        stim_train_rates = stim_train_set.rates
        stim_test_rates = stim_test_set.rates
        stim_rates = np.concatenate([stim_train_rates, stim_test_rates], 1)


        stim_train_trial_type_labels = stim_train_set.trial_type_labels
        stim_test_trial_type_labels = stim_test_set.trial_type_labels
        stim_trial_type_labels = np.concatenate([stim_train_trial_type_labels, stim_test_trial_type_labels], 0)

        stim_train_labels = stim_train_set.labels
        stim_test_labels = stim_test_set.labels
        stim_labels = np.concatenate([stim_train_labels, stim_test_labels], 0)

        stim_suc_labels = (stim_trial_type_labels==stim_labels).astype(int)

        '''
        Re-define stim_train and test data according to train and test trial_idx.
        '''
        # Load stim_mask.
        stim_mask_path = os.path.join(prep_save_path, pert_type, 'stim_mask.npy')
        stim_mask = np.load(stim_mask_path)

        cur_mask = np.isin(i_good_trials, i_good_trial_numbers)[stim_mask]

        # # temp
        # stim_i_good_trials_old = i_good_trials[stim_mask]
        # stim_i_good_trials = stim_i_good_trials_old[(stim_trial_type_labels==self.configs['inst_trial_type'])]

        # print('')
        # print('pert_type: ', pert_type)
        # print('')
        # print('<i_good_trials>')
        # print(i_good_trials)
        # print('')
        # print('<i_good_trial_numbers>')
        # print(i_good_trial_numbers)
        # print('<i_good_trial_numbers (recalculated)>')
        # print('cur_inst_trial_type: ', self.configs['inst_trial_type'])
        # print(stim_i_good_trials)        
        # print('')
        # print('<i_good_trials[stim_mask]>')
        # print(i_good_trials[stim_mask])
        # # temp

        # sanity check

        print(len(np.nonzero(cur_mask)[0]))
        print(len(i_good_trial_numbers))
        assert len(np.nonzero(cur_mask)[0]) == len(i_good_trial_numbers)
        print('assert len(np.nonzero(cur_mask)[0]) == len(i_good_trial_numbers)')
        cur_rates = stim_rates[:,cur_mask]


        return cur_rates, bin_centers, neural_unit_location 






    def compute_score_grad_in_pixel_space(self, sess_type_list):

        '''
        A note about test_bs:
        I found that using test_bs = 24 already takes 5~6 GB of the gpu memory.
        So, if the gpu is already running other processes, I need to choose test_bs appropriately.
        Otherwise, I think I can choose a maximum test_bs of 48.
        '''



        for sess_type in sess_type_list:
            
            self.configs['sess_type'] = sess_type

            for i in self.n_trial_types_list:

                self.configs['inst_trial_type'] = i

                self.init_model_and_analysis_save_paths()

                self.compute_score_grad_in_pixel_space_for_each_inst_trial_type()



    def compute_score_grad_in_pixel_space_for_each_inst_trial_type(self):
        n_trial_types_list = self.n_trial_types_list



        '''
        ###
        To avoid having duplicate processes on gpu 0,
        I need to use CUDA_VISIBLE_DEVICES.
        Once I set visible devices, the gpu ids need to start from 0.
        ###
        '''
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(', '.join([str(x) for x in self.configs['gpu_ids_before_masking']]))
        print('')
        print('os.environ["CUDA_VISIBLE_DEVICES"]')
        print(os.environ["CUDA_VISIBLE_DEVICES"])
        print('')
        self.configs['gpu_ids'] = [x for x in range(len(self.configs['gpu_ids_before_masking']))]




        '''
        ###
        Set random seeds for determinism.
        ###
        '''
        np.random.seed(self.configs['random_seed'])
        torch.manual_seed(self.configs['random_seed'])






        self.configs['use_cuda'] = bool(self.configs['use_cuda'])
        self.configs['do_pos_weight'] = bool(self.configs['do_pos_weight'])

        if 'rnn' in self.configs['model_name']:
            self.configs['_input_channel'] = 1

        assert self.configs['inst_trial_type'] in [0, 1]
        
        # Create directories to save results.
        os.makedirs(self.configs['logs_cv_dir'], exist_ok=True)
        os.makedirs(self.configs['models_cv_dir'], exist_ok=True)

        # Detect devices
        use_cuda = torch.cuda.is_available() and self.configs['use_cuda']                  # check if GPU exists
        device = torch.device("cuda:{}".format(self.configs['gpu_ids'][0]) if use_cuda else "cpu")   # use CPU or GPU

        # Data loading parameters

        '''
        To use WeightedRandomSampler, we must set shuffle to False.
        '''
        params = {'batch_size': self.configs['test_bs'], 'shuffle': False, 'num_workers': self.configs['num_workers'], \
        'pin_memory': bool(self.configs['pin_memory']), 'drop_last': False}\
        if use_cuda else {'batch_size': self.configs['test_bs']}

        # Collect video filenames.
        if self.configs['img_type'] == 'jpg':
            video_root_path = os.path.join(self.configs['video_data_dir'], 'frames')
        elif self.configs['img_type'] == 'png':
            video_root_path = os.path.join(self.configs['video_data_dir'], 'png_frames')


        '''
        Session selection.
        Here, we define f1_f2_list.
        '''
        sess_type = self.configs['sess_type']

        assert sess_type in ['{}_{}'.format(task_type, sort_type) for sort_type in ['guang_one_laser'] for task_type in ['original', 'full_reverse']]

        filehandler = open('sorted_filenames.obj', 'rb')
        filenames = pickle.load(filehandler)
        
        sess_by_task_dict = pert_pred_utils.load_sess_by_task_dict()

        print('')
        print('<Before restricting to sessions that have videos>')
        print('sess_type: ', sess_type)
        print('n_sess: ', len(sess_by_task_dict[sess_type]))

        pre_f1_f2_list = []
        for sess_idx in sess_by_task_dict[sess_type]:
            sess_name = filenames[sess_idx-1].split('\\')[1][:-4].replace('kilosort', '') #BAYLORGC100_2020_01_21
            # f1 = sess_name.split('_')[0]
            # start = sess_name.find('_') + 1
            # f2 = sess_name[start:]
            pre_f1_f2_list.append(sess_name)

        pre_f1_f2_list = np.array(pre_f1_f2_list)


        '''
        Load f1_f2_list, which contains tuples of form (f1,f2)
        '''
        if bool(self.configs['debug']):
            f1_f2_list = []
            for f1 in os.listdir(video_root_path):
                if os.path.isfile(os.path.join(video_root_path, f1)):
                    continue
                for f2 in os.listdir(os.path.join(video_root_path, f1)):

                    if os.path.isfile(os.path.join(video_root_path, f1, f2)):
                        continue

                    f1_f2_list.append((f1,f2))


        else:
            # if not os.path.isfile(os.path.join('make_f1_f2_list_results', sess_type, 'f1_f2_list.pkl')):
            #     import make_f1_f2_list
            #     make_f1_f2_list.main(sess_type)

            # Since I modify make_f1_f2_list from time to time, I want to generate f1_f2_list each time I run this code.
            import make_f1_f2_list
            task_type = 'standard' if 'original' in sess_type else 'full_reverse'
            make_f1_f2_list.main('{}_all'.format(task_type))

            with open(os.path.join('make_f1_f2_list_results', '{}_all'.format(task_type), 'f1_f2_list.pkl'), 'rb') as f:
                f1_f2_list = pickle.load(f)


        video_f1_f2_list = np.array(['_'.join([f1,f2]) for (f1,f2) in f1_f2_list])


        '''
        We take those sessions in pre_f1_f2_list that are in video_f1_f2_list.
        '''
        f1_f2_list = pre_f1_f2_list[np.isin(pre_f1_f2_list, video_f1_f2_list)]

        # sanity check
        # pre_f1_f2_set = set(list(pre_f1_f2_list))
        # video_f1_f2_set = set(list(video_f1_f2_list))
        # print('')
        # print('sanity check')
        # print('pre_f1_f2_set.difference(video_f1_f2_set)')
        # for x in list(pre_f1_f2_set.difference(video_f1_f2_set)):
        #     print(x)

        print('')
        print('sanity check')
        print('<pre_f1_f2_list>')
        print('n_sess: ', len(pre_f1_f2_list))
        for x in pre_f1_f2_list:
            print(x)
        print('')
        print('<video_f1_f2_list>')
        print('n_sess: ', len(video_f1_f2_list))
        for x in video_f1_f2_list:
            print(x)
        print('')
        print('<Sessions in pre_f1_f2_list that are not in video_f1_f2_list>')
        print('n_sess: ', len(pre_f1_f2_list[~np.isin(pre_f1_f2_list, video_f1_f2_list)]))
        for x in pre_f1_f2_list[~np.isin(pre_f1_f2_list, video_f1_f2_list)]:
            print(x)



        print('')
        print('<After restricting to sessions that have videos>')
        print('sess_type: ', sess_type)
        print('n_sess: ', len(f1_f2_list))
        print('')


        f1_f2_list_old = f1_f2_list.copy()

        f1_f2_list = []
        for sess_name in f1_f2_list_old:
            # sess_name: BAYLORGC100_2020_01_21
            f1 = sess_name.split('_')[0]
            start = sess_name.find('_') + 1
            f2 = sess_name[start:]
            f1_f2_list.append((f1,f2))

        f1_f2_list = np.array(f1_f2_list)



        # if not self.configs['debug']:
        #     '''
        #     I found that certain sessions such as GC90_2019_11_18 don't have side view videos.
        #     I want to remove such sessions from f1_f2_list.
        #     '''
        #     if 'full_reverse' in sess_type:
        #         avi_root_path = '/mnt/fs4/bkang/alm_video_data/E3/avi_files'
        #     elif 'standard' in sess_type:
        #         # We know that sessions in /mnt/fs6/zwang71/MiceBehavior/data/E3/Session all have both view videos.
        #         avi_root_path = '/mnt/fs2/bkang/alm_video_data/E3/avi_files'


        #     print('')
        #     print('len(f1_f2_list) before checking for bad sess: {}'.format(len(f1_f2_list)))
        #     good_sess_idx = []

        #     for sess_idx, (f1, f2) in enumerate(f1_f2_list):
        #         print('')
        #         print('Checking f1: {} / f2: {}'.format(f1,f2))
                
        #         if f1 in ['BAYLORGC25', 'BAYLORGC26', 'BAYLORGC50']:
        #             good_sess_idx.append(sess_idx)
        #             continue

        #         # avi_file: GC50_20181211_bottom_trial_100-0000.avi
        #         if not any('side' in avi_file for avi_file in os.listdir(os.path.join(avi_root_path, f1, f2, 'Camera'))):
        #             print('')
        #             print('No side view in:')
        #             print('f1: {} / f2: {}'.format(f1, f2))
        #             continue

        #         if not any('bottom' in avi_file for avi_file in os.listdir(os.path.join(avi_root_path, f1, f2, 'Camera'))):
        #             print('')
        #             print('No bottom view in:')
        #             print('f1: {} / f2: {}'.format(f1, f2))
        #             continue

        #         good_sess_idx.append(sess_idx)


        #     f1_f2_list = f1_f2_list[good_sess_idx]

        #     print('')
        #     print('len(f1_f2_list) after checking for bad sess: {}'.format(len(f1_f2_list)))




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
        Collect video path.

        Use no_stim_i_good_trials.
        video_path_dict[f1][f2][i] will contain paths to videos for trials in no_stim_i_good_trials[i].
        And we will separately split trials of each trial type into train and test set.
        '''

        video_path_dict = nestdict()
        for f1, f2 in f1_f2_list:

            no_stim_i_good_trials = pert_pred_utils.load_no_stim_i_good_trials_for_pert_pred(self.configs['inst_trial_type'], self.configs['prep_root_dir'], f1, f2)
            stim_i_good_trials = pert_pred_utils.load_stim_i_good_trials_for_pert_pred(self.configs['pert_type'], self.configs['inst_trial_type'], self.configs['prep_root_dir'], f1, f2)

            total_i_good_trials = np.array([no_stim_i_good_trials, stim_i_good_trials])


            for i in n_trial_types_list:
                # Just a sanity check to make sure i_good_trials are sorted.
                assert (total_i_good_trials[i] == sorted(total_i_good_trials[i])).all()

                video_path_dict[f1][f2][i] = []


            for f3 in os.listdir(os.path.join(video_root_path, f1, f2)):
                # f3 = 'E3-BAYLORGC25-2018_09_17-13'
                if os.path.isfile(os.path.join(video_root_path, f1, f2, f3)):
                    continue


                for i in n_trial_types_list:
                    if pert_pred_utils.map_f3_to_trial_idx(f1,f2,f3) in total_i_good_trials[i]:
                        video_path_dict[f1][f2][i].append(f3)


            for i in n_trial_types_list:
                # Sort cur_path_dict[f2] according to the trial index at the end.
                # Not necessary to sort it really, but let's do it anyways.
                video_path_dict[f1][f2][i].sort(key=video_trial_sort)




        '''
        Frame selection
        '''


        selected_frames_dict = nestdict()
        self.configs['pred_timesteps'] = None


        # Needed below to figure out begin and end_bin.
        f1 = list(video_path_dict.keys())[0]
        f2 = list(video_path_dict[f1].keys())[0]

        bin_centers = pert_pred_utils.load_bin_centers(self.configs['prep_root_dir'],f1,f2)


        # These are used for cur_cd_non_cd_pca.
        begin_bin = np.argmin(np.abs(bin_centers - self.configs['neural_begin_time']))
        end_bin = np.argmin(np.abs(bin_centers - self.configs['neural_end_time']))

        bin_stride = 0.1
        skip_bin = int(self.configs['neural_skip_time']//bin_stride)

        pred_times = bin_centers[begin_bin:end_bin+1:skip_bin]



        print('')
        print('Selecting frames for each session...')


        # Load n_frames_dict
        if not self.configs['debug']:
            save_path = 'get_n_frames_for_each_sess_results'
            if not os.path.isfile(os.path.join(save_path, 'n_frames_dict.pkl')):
                import get_n_frames_for_each_sess
                get_n_frames_for_each_sess.main()

            with open(os.path.join(save_path, 'n_frames_dict.pkl'), 'rb') as f:
                n_frames_dict = pickle.load(f)

        for f1, f2 in f1_f2_list:
            if not self.configs['debug']:
                cur_n_frames = n_frames_dict[(f1,f2)]
            else:
                cur_n_frames = 1000

            if cur_n_frames == 1000:

                go_cue_time = 3.57
                begin_frame = int(np.rint((pred_times[0] + go_cue_time)*200)) # 200 frames per sec.
                end_frame = int(np.rint((pred_times[-1] + go_cue_time)*200))
                skip_frame = self.configs['skip_frame']

            elif cur_n_frames == 1200:
                go_cue_time = 4.57
                begin_frame = int(np.rint((pred_times[0] + go_cue_time)*200)) # 200 frames per sec.
                end_frame = int(np.rint((pred_times[-1] + go_cue_time)*200))
                skip_frame = self.configs['skip_frame']

            else:
                raise ValueError('Invalid cur_n_frames: {}. Needs to be either 1000 or 1200'.format(cur_n_frames))

            # Both of the returned variables are 1d np array.
            selected_frames, pred_timesteps = \
            pert_pred_utils.compute_selected_frames_and_pred_timesteps(self.configs, pred_times, begin_frame, end_frame, skip_frame, go_cue_time)

            # print(pred_times)
            # print(selected_frames)
            # print(pred_timesteps)

            if self.configs['pred_timesteps'] is None:
                self.configs['pred_timesteps'] = pred_timesteps
            else:
                # Even when n_frames is different across sessions, pred_timesteps should be the same.
                if (self.configs['pred_timesteps'] != pred_timesteps).any():
                    raise ValueError('self.configs pred_timesteps (={}) is \
                        NOT the same as current (f1:{}/f2:{}) pred_timesteps (={})'.format(\
                        ', '.join([str(x) for x in self.configs['pred_timesteps']]), f1, f2, ', '.join([str(x) for x in pred_timesteps])))

            for i in n_trial_types_list:
                selected_frames_dict[f1][f2][i] = np.broadcast_to(selected_frames, (len(video_path_dict[f1][f2][i]), len(selected_frames)))

            # print('')
            # print('f1: {} / f2: {}'.format(f1, f2))
            # print('n_frames: {}'.format(cur_n_frames))
            # print('go_cue_time: {}'.format(go_cue_time))
            # print('begin_frame: {}'.format(begin_frame))
            # print('end_frame: {}'.format(end_frame))
            # print('skip_frame: {}'.format(skip_frame))





        # Determine n_sess and sess_inds for sess_cond model.
        self.configs['n_sess'] = 0
        sess_inds_dict = nestdict()

        for f1, f2 in f1_f2_list:

            self.configs['n_sess'] += 1

            for i in n_trial_types_list:
                # n_sess -1 because sess_inds ranges from 0 to n_sess-1.
                sess_inds_dict[f1][f2][i] = [self.configs['n_sess']-1]*len(video_path_dict[f1][f2][i])










        '''
        Cross-validation train-test split.
        We determine the split using video trials, and then
        use it when generating neural data in which CD is computed only using the train set's correct trials.
        '''

        n_cv = self.configs['n_cv']

        sess_wise_test_video_path_dict = nestdict()
        sess_wise_test_sess_inds_dict = nestdict()
        sess_wise_test_trial_type_dict = nestdict()
        sess_wise_test_selected_frames_dict = nestdict()

        sess_wise_test_trial_idxs_dict = nestdict()
        '''
        I realized that for some sessions and inst_trial_types, there are very few 
        error trials, less than n_cv, which makes it impossible to perform a separate k-fold
        split for each inst_trial_type. To circumvent this issue, we first identify
        sessions in which at least one of inst_trial_types have error trials fewer
        than n_cv. For those sessions, we perform k-fold split for the entire set of 
        trials in a given inst_trial_type.
        '''
        few_error_trial_f1_f2_list = []


        for f1, f2 in f1_f2_list:
            # f1 = 'BAYLORGC[#]'
            # f2 = '[yyyy]_[mm]_[dd]'

            print('')
            print(f1, f2)

            for i in n_trial_types_list:
                cur_video_path = np.array([os.path.join(f1,f2,f3) for f3 in video_path_dict[f1][f2][i]]) # an array of each trial's video path
                
                print('{}'.format('no_stim' if i==0 else self.configs['pert_type']))
                print(len(cur_video_path))

                if len(cur_video_path) < n_cv:
                    few_error_trial_f1_f2_list.append((f1, f2))
                    break
        
        print('')
        print('few_error_trial_f1_f2_list')
        print('')
        for f1, f2 in few_error_trial_f1_f2_list:
            print(f1,f2)




        '''
        Here, only iterate through sessions that are not in few_error_trial_f1_f2_list.
        '''
        for f1, f2 in f1_f2_list:
            # f1 = 'BAYLORGC[#]'
            # f2 = '[yyyy]_[mm]_[dd]'
            if (f1,f2) in few_error_trial_f1_f2_list:
                continue

            for i in n_trial_types_list:

                # We wrap lists into arrays so that they can be indexed with advanced indexing.
                cur_video_path = np.array([os.path.join(f1,f2,f3) for f3 in video_path_dict[f1][f2][i]]) # an array of each trial's video path

                cur_sess_inds = np.array(sess_inds_dict[f1][f2][i]) # an array [sess_ind]*n_trials.
                cur_trial_type = np.array([i]*len(cur_sess_inds)) # an array [i]*n_trials.

                cur_selected_frames = selected_frames_dict[f1][f2][i] # np array of shape (n_trials, n_selected_frames)

                '''
                In most cases, map_f3_to_trial_idx is int(f3.split('-')[3]).But,
                there are a few exceptional sessions where this is not true.
                '''
                cur_trial_idxs = np.array([pert_pred_utils.map_f3_to_trial_idx(f1,f2,f3) for f3 in video_path_dict[f1][f2][i]]) # an array of each trial's video path


                kf = KFold(n_splits=n_cv, shuffle=True, random_state=1)

                for cv_ind, (train_inds, test_inds) in enumerate(kf.split(cur_video_path)):


                    cur_train_video_path, cur_test_video_path = cur_video_path[train_inds], cur_video_path[test_inds]
                    cur_train_sess_inds, cur_test_sess_inds = cur_sess_inds[train_inds], cur_sess_inds[test_inds]
                    cur_train_trial_type, cur_test_trial_type = cur_trial_type[train_inds], cur_trial_type[test_inds]
                    cur_train_selected_frames, cur_test_selected_frames = cur_selected_frames[train_inds], cur_selected_frames[test_inds]


                    sess_wise_test_video_path_dict[cv_ind][f1][f2][i] = cur_test_video_path
                    sess_wise_test_sess_inds_dict[cv_ind][f1][f2][i] = cur_test_sess_inds
                    sess_wise_test_trial_type_dict[cv_ind][f1][f2][i] = cur_test_trial_type
                    sess_wise_test_selected_frames_dict[cv_ind][f1][f2][i] = cur_test_selected_frames

                    sess_wise_test_trial_idxs_dict[cv_ind][f1][f2][i] = cur_trial_idxs[test_inds]




        '''
        ###
        Note added on 10/01/20 (while revisiting this code for Cell-revision):
        
        Note that in the for loop below for few_error_trial_f1_f2_list, the index i
        is fixed at 1 after the for loop for i, and thus various dict will have an empty
        entry for dict[cv_ind][f1][f2][0].

        This didn't cause any error afterwards because list's extend simply does not add
        any element from dict[cv_ind][f1][f2][0] (I confirmed this behavior), and I initially
        thought this would have a bad effect on learning. But, I realize that in fact, it
        doesn't affect learning at all, since we are going to concatenate dict[cv_ind][f1][f2][i]
        anyways for training and testing! The same holds for sess-wise validation set evaluation.
        ###
        '''


        for f1, f2 in few_error_trial_f1_f2_list:

            cur_video_path = []
            cur_sess_inds = []
            cur_trial_type = []
            cur_selected_frames = []
            cur_trial_idxs = []

            for i in n_trial_types_list:

                # We wrap lists into arrays so that they can be indexed with advanced indexing.
                cur_video_path.extend([os.path.join(f1,f2,f3) for f3 in video_path_dict[f1][f2][i]]) # an array of each trial's video path

                cur_sess_inds.extend(sess_inds_dict[f1][f2][i]) # an array [sess_ind]*n_trials.
                cur_trial_type.extend([i]*len(cur_sess_inds)) # an array [i]*n_trials.

                cur_selected_frames.extend(selected_frames_dict[f1][f2][i]) # np array of shape (n_trials, n_selected_frames)

                '''
                In most cases, map_f3_to_trial_idx is int(f3.split('-')[3]).But,
                there are a few exceptional sessions where this is not true.
                '''

                cur_trial_idxs.extend([pert_pred_utils.map_f3_to_trial_idx(f1,f2,f3) for f3 in video_path_dict[f1][f2][i]]) # an array of each trial's video path


            cur_video_path = np.array(cur_video_path)
            cur_sess_inds = np.array(cur_sess_inds)
            cur_trial_type = np.array(cur_trial_type)
            cur_selected_frames = np.array(cur_selected_frames)
            cur_trial_idxs = np.array(cur_trial_idxs)


            kf = KFold(n_splits=n_cv, shuffle=True, random_state=1)

            for cv_ind, (train_inds, test_inds) in enumerate(kf.split(cur_video_path)):


                cur_train_video_path, cur_test_video_path = cur_video_path[train_inds], cur_video_path[test_inds]
                cur_train_sess_inds, cur_test_sess_inds = cur_sess_inds[train_inds], cur_sess_inds[test_inds]
                cur_train_trial_type, cur_test_trial_type = cur_trial_type[train_inds], cur_trial_type[test_inds]
                cur_train_selected_frames, cur_test_selected_frames = cur_selected_frames[train_inds], cur_selected_frames[test_inds]


                sess_wise_test_video_path_dict[cv_ind][f1][f2][i] = cur_test_video_path
                sess_wise_test_sess_inds_dict[cv_ind][f1][f2][i] = cur_test_sess_inds
                sess_wise_test_trial_type_dict[cv_ind][f1][f2][i] = cur_test_trial_type
                sess_wise_test_selected_frames_dict[cv_ind][f1][f2][i] = cur_test_selected_frames

                sess_wise_test_trial_idxs_dict[cv_ind][f1][f2][i] = cur_trial_idxs[test_inds]






        '''
        ###
        NOTE:

        I realized that I can save some time by first combining all sessions data into a single data loader
        and then later dividing up the outputs of the network into different sessions.
        ###
        '''



        test_video_path = np.zeros(n_cv, dtype=object)
        
        test_sess_inds = np.zeros(n_cv, dtype=object)

        test_trial_type = np.zeros(n_cv, dtype=object)

        test_selected_frames = np.zeros(n_cv, dtype=object)    

        test_trial_idxs = np.zeros(n_cv, dtype=object)    



        for cv_ind in range(n_cv):

            test_video_path[cv_ind] = []

            test_sess_inds[cv_ind] = []

            test_trial_type[cv_ind] = []

            test_selected_frames[cv_ind] = []

            test_trial_idxs[cv_ind] = []


        for cv_ind in range(n_cv):

            for f1, f2 in f1_f2_list:
                # f1 = 'BAYLORGC[#]'
                # f2 = '[yyyy]_[mm]_[dd]'

                for i in n_trial_types_list:

                    '''
                    For now, to save disk space, we are only computing grad for bi stim trials.
                    So, we exclude control trials which have i==0.
                    '''
                    if i == 0:
                        continue

                    # Test set
                    test_video_path[cv_ind].extend(sess_wise_test_video_path_dict[cv_ind][f1][f2][i])
                    test_sess_inds[cv_ind].extend(sess_wise_test_sess_inds_dict[cv_ind][f1][f2][i])
                    test_trial_type[cv_ind].extend(sess_wise_test_trial_type_dict[cv_ind][f1][f2][i])
                    test_selected_frames[cv_ind].extend(sess_wise_test_selected_frames_dict[cv_ind][f1][f2][i])
                    test_trial_idxs[cv_ind].extend(sess_wise_test_trial_idxs_dict[cv_ind][f1][f2][i])

            # Turn the lists into arrays.

            test_video_path[cv_ind] = np.array(test_video_path[cv_ind])
            test_sess_inds[cv_ind] = np.array(test_sess_inds[cv_ind])
            test_trial_type[cv_ind] = np.array(test_trial_type[cv_ind])
            test_selected_frames[cv_ind] = np.array(test_selected_frames[cv_ind])
            test_trial_idxs[cv_ind] = np.array(test_trial_idxs[cv_ind])












            
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





        # Iterate over cv_ind
        n_sess = self.configs['n_sess']



        '''
        ###
        IMPORTANT:
        
        For now, to save disk space, we are only computing grad for bi stim trials.
        In fact, when I collect test data above, I only included those with i == 1.

        ###
        '''

        # X[cv_ind,sess_ind] has shape (n_trials, T, n_input_channels, H, W)
        cv_agg_sess_wise_score_grad_for_pert_trials_side_view = np.zeros((n_cv, n_sess), dtype=object)
        cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view = np.zeros((n_cv, n_sess), dtype=object)


        # Save sess_ind to (f1,f2) mapping.
        sess_ind_to_f1_f2_map = np.zeros((n_sess,), dtype=object)

        for sess_ind, (f1, f2) in enumerate(f1_f2_list):

            sess_ind_to_f1_f2_map[sess_ind] = '_'.join([f1,f2])

        for cv_ind in range(n_cv):
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

            cv_agg_sess_wise_score_grad_for_pert_trials_side_view[cv_ind], \
            cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view[cv_ind] = \
            self.compute_score_grad_in_pixel_space_helper_cv(f1_f2_list, cv_ind, self.configs, video_root_path, n_trial_types_list,\
            test_video_path[cv_ind],test_sess_inds[cv_ind], \
            test_trial_type[cv_ind], test_selected_frames[cv_ind],\
            test_trial_idxs[cv_ind],\
            transform_list, params, device)



        # '''
        # I suspect I can reduce time it takes to save and load numpy arrays by avoiding object arrays.
        # '''
        # cv_agg_sess_wise_score_grad_for_pert_trials_side_view_old = cv_agg_sess_wise_score_grad_for_pert_trials_side_view
        # cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view_old = cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view

        # n_total_trials = len(cat(cv_agg_sess_wise_score_grad_for_pert_trials_side_view_old.reshape(-1), 0))

        # _, T, n_input_channels, H_side, W_side = cv_agg_sess_wise_score_grad_for_pert_trials_side_view_old[0,0].shape
        # _, T, n_input_channels, H_bottom, W_bottom = cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view_old[0,0].shape

        # cv_and_sess_mask = np.zeros((n_cv, n_sess, n_total_trials), dtype=bool)
        # cv_agg_sess_wise_score_grad_for_pert_trials_side_view = np.zeros((n_total_trials, T, n_input_channels, H_side, W_side))
        # cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view = np.zeros((n_total_trials, T, n_input_channels, H_bottom, W_bottom))

        # n_trials_count = 0
        # for cv_ind in range(n_cv):
        #     for sess_ind in range(n_sess):
                
        #         cur_n_trials_side = len(cv_agg_sess_wise_score_grad_for_pert_trials_side_view_old[cv_ind,sess_ind])
        #         cur_n_trials_bottom = len(cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view_old[cv_ind,sess_ind])

        #         assert cur_n_trials_side == cur_n_trials_bottom

        #         cv_and_sess_mask[cv_ind,sess_ind,n_trials_count:n_trials_count+cur_n_trials_side] = True

        #         cv_agg_sess_wise_score_grad_for_pert_trials_side_view[n_trials_count:n_trials_count+cur_n_trials_side] = \
        #         cv_agg_sess_wise_score_grad_for_pert_trials_side_view_old[cv_ind,sess_ind]

        #         cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view[n_trials_count:n_trials_count+cur_n_trials_bottom] = \
        #         cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view_old[cv_ind,sess_ind]


        #         n_trials_count += cur_n_trials_side





        os.makedirs(os.path.join(self.analysis_save_path, 'compute_score_grad_in_pixel_space'), exist_ok=True)


        # np.save(os.path.join(self.analysis_save_path, 'compute_score_grad_in_pixel_space',\
        #  'cv_and_sess_mask.npy'), cv_and_sess_mask)

        np.save(os.path.join(self.analysis_save_path, 'compute_score_grad_in_pixel_space',\
         'cv_agg_sess_wise_score_grad_for_pert_trials_side_view.npy'), cv_agg_sess_wise_score_grad_for_pert_trials_side_view)

        np.save(os.path.join(self.analysis_save_path, 'compute_score_grad_in_pixel_space',\
         'cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view.npy'), cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view)


    def compute_score_grad_in_pixel_space_helper_cv(self, f1_f2_list, cv_ind, configs, video_root_path, n_trial_types_list,\
        test_video_path, test_sess_inds,\
        test_trial_type, test_selected_frames,\
        test_trial_idxs,\
        transform_list, params, device):
        '''
        Return:
        cv_agg_sess_wise_score_grad_for_pert_trials_side_view = np.zeros((n_sess,), dtype=object)
        X[sess_ind] has shape (n_trials, T, n_input_channels, H, W)

        cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view = np.zeros((n_sess,), dtype=object)
        X[sess_ind] has shape (n_trials, T, n_input_channels, H, W)

        '''




        '''
        ###
        Important:
        We will not define a dataloader separately for each
        random subsampling of the majority class.

        Instead we will first test the network on all samples
        in one go, and then subsample the majority class samples' predictions afterward.
        ###
        '''

        combined_valid_set = MyDatasetChoicePredVariableFramesRecon(video_root_path, test_video_path,\
            test_sess_inds, test_trial_type, test_selected_frames, \
            test_trial_idxs, configs['view_type'], \
            transform_list=transform_list, img_type=configs['img_type'])

        combined_valid_loader = data.DataLoader(combined_valid_set, **params)






        for key in sorted(self.configs.keys()):
            print('{}: {}'.format(key, self.configs[key]))
        print('')
        print('')


        '''
        Test the best saved model. Copied from cd_reg_from_videos_test.py.
        '''
        print('')
        print('Test begins!')
        print('')



        import sys
        model = getattr(sys.modules[__name__], self.configs['model_name'])(self.configs).to(device)


        # Parallelize model to multiple GPUs
        if self.configs['use_cuda'] and torch.cuda.device_count() > 1:
            if self.configs['gpu_ids'] is None:
                print("Using", torch.cuda.device_count(), "GPUs!")
                print('')
                model = nn.DataParallel(model)
            else:
                print("Using", len(self.configs['gpu_ids']), "GPUs!")
                print('')
                model = nn.DataParallel(model, device_ids=self.configs['gpu_ids'])

        '''
        If we don't specify map_location, by default, the saved model is loaded onto the gpus it was trained on.
        This could be problematic if the gpus it was trained on are currently occupied by other processes.

        If I set map_location to device as below, the model will still be run on multiple gpus as specified in DataParallel.
        '''
        model.load_state_dict(torch.load(os.path.join(self.model_save_path, 'cv_ind_{}'.format(cv_ind), 'best_model.pth'), map_location=device))
        


        '''
        All numpy arrays:
        combined_score_grad_side_view: (n_trials, T, n_input_channels, H, W)
        combined_score_grad_bottom_view: (n_trials, T, n_input_channels, H, W)
        combined_trial_idx: (n_trials)
        combined_sess_inds: (n_trials)
        '''
        combined_score_grad_side_view, combined_score_grad_bottom_view, combined_trial_idx, combined_sess_inds \
         = self.compute_score_grad_in_pixel_space_core(self.configs, model, device, combined_valid_loader)


        n_sess = len(f1_f2_list)

        # X[sess_ind] has shape (n_trials, T, n_input_channels, H, W)
        cv_agg_sess_wise_score_grad_for_pert_trials_side_view = np.zeros((n_sess,), dtype=object)
        cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view = np.zeros((n_sess,), dtype=object)


        for sess_ind, (f1, f2) in enumerate(f1_f2_list):

            cur_sess_mask = (combined_sess_inds == sess_ind)

            cv_agg_sess_wise_score_grad_for_pert_trials_side_view[sess_ind] = combined_score_grad_side_view[cur_sess_mask]
            cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view[sess_ind] = combined_score_grad_bottom_view[cur_sess_mask]


        return cv_agg_sess_wise_score_grad_for_pert_trials_side_view, cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view




    def compute_score_grad_in_pixel_space_core(self, configs, model, device, test_loader):
        '''
        Return:

        All numpy arrays:
        combined_score_grad_side_view: (n_trials, T, n_input_channels, H, W)
        combined_score_grad_bottom_view: (n_trials, T, n_input_channels, H, W)
        combined_trial_idx: (n_trials)
        combined_sess_inds: (n_trials)
        '''


        # set model as testing mode
        begin_time = time.time()
        model.eval()




        '''
        We don't need gradients wrt model params.
        '''
        for param in model.parameters():
            param.requires_grad = False




        all_score_grad_side_view = []
        all_score_grad_bottom_view = []
        all_trial_idx = []
        all_sess_inds = []



        for batch_idx, data in enumerate(test_loader):
            '''
            y is trial type label.
            z is sess ind.
            '''

            model.zero_grad()


            if 'all_views' in configs['model_name']:
                X_side, X_bottom, y, z, trial_idx = data
                # Note that trial_idx is not needed for the forward pass.
                X_side, X_bottom, y, z = X_side.to(device), X_bottom.to(device), y.to(device), z.to(device)

                '''
                For gradient calculations.
                '''
                X_side.requires_grad = True
                X_bottom.requires_grad = True


            else:
                X, y, z, trial_idx = data
                # Note that trial_idx is not needed for the forward pass.
                X, y, z = X.to(device), y.to(device), z.to(device)

                '''
                For gradient calculations.
                '''
                X.requires_grad = True



            if 'sess_cond' in configs['model_name']:
                
                if 'all_views' in configs['model_name']:
                    output = model(X_side, X_bottom, z) # (batch, T)
                else:
                    output = model(X, z)

            else:

                if 'all_views' in configs['model_name']:
                    output = model(X_side, X_bottom)
                else:
                    output = model(X)



            cur_n_trials, T = output.size()

            n_frames = X_side.size()[1]

            frame_idxs_for_each_pred_timestep = model.module.input_timesteps.reshape(T,-1) # (T, n_input_channels)


            # For convenience, I am not going to implement the below part of the code for non all_views models.
            assert 'all_views' in configs['model_name']


            cur_score_grad_side_view = []
            cur_score_grad_bottom_view = []


            for t in range(T):

                    '''
                    For the side view.
                    '''

                    # (n_trials, n_frames, c, h, w)
                    # The output of autograd.grad is a tuple of length equal to the length of the outputs of 
                    # the functions that are differentiated. In our case, the latter is 1.
                    grad_side_view = torch.autograd.grad(output[:,t].sum(), X_side, retain_graph=True)[0].data.cpu().numpy()


                    '''
                    Sanity check:
                    grad_side_view should be zero in frames not used for current t.
                    '''
                    assert grad_side_view.shape == \
                    (cur_n_trials, n_frames, 1, self.configs['image_shape_side'][0], self.configs['image_shape_side'][1])

                    # (cur_n_trials, n_frames, H, W)
                    grad_side_view = grad_side_view[:,:,0]

                    cur_frame_mask = np.zeros(n_frames, dtype=bool)
                    cur_frame_mask[frame_idxs_for_each_pred_timestep[t]] = True


                    assert np.allclose(grad_side_view[:,~cur_frame_mask], 0)


                    grad_side_view[:,cur_frame_mask]

                    cur_score_grad_side_view.append(grad_side_view)



                    '''
                    For the bottom view.
                    '''

                    # (n_trials, n_frames, c, h, w)
                    grad_bottom_view = torch.autograd.grad(output[:,t].sum(), X_bottom, retain_graph=True)[0].data.cpu().numpy()


                    '''
                    Sanity check:
                    grad_bottom_view should be zero in frames not used for current t.
                    '''
                    assert grad_bottom_view.shape == \
                    (cur_n_trials, n_frames, 1, self.configs['image_shape_bottom'][0], self.configs['image_shape_bottom'][1])

                    # (cur_n_trials, n_frames, H, W)
                    grad_bottom_view = grad_bottom_view[:,:,0]

                    cur_frame_mask = np.zeros(n_frames, dtype=bool)
                    cur_frame_mask[frame_idxs_for_each_pred_timestep[t]] = True


                    assert np.allclose(grad_bottom_view[:,~cur_frame_mask], 0)


                    grad_bottom_view[:,cur_frame_mask]

                    cur_score_grad_bottom_view.append(grad_bottom_view)




            # (cur_n_trials, T, n_input_channels, H, W)
            cur_score_grad_side_view = np.stack(cur_score_grad_side_view, 1)
            all_score_grad_side_view.append(cur_score_grad_side_view)


            # (cur_n_trials, T, n_input_channels, H, W)
            cur_score_grad_bottom_view = np.stack(cur_score_grad_bottom_view, 1)
            all_score_grad_bottom_view.append(cur_score_grad_bottom_view)


            all_trial_idx.append(trial_idx.cpu().data.numpy())
            all_sess_inds.append(z.cpu().data.numpy())



        all_score_grad_side_view = cat(all_score_grad_side_view, 0)
        all_score_grad_bottom_view = cat(all_score_grad_bottom_view, 0)
        all_trial_idx = cat(all_trial_idx, 0)
        all_sess_inds = cat(all_sess_inds, 0)



        return all_score_grad_side_view, all_score_grad_bottom_view, all_trial_idx, all_sess_inds



    def visualize_score_grad_in_pixel_space(self, sess_type_list):

        '''
        A note about test_bs:
        I found that using test_bs = 24 already takes 5~6 GB of the gpu memory.
        So, if the gpu is already running other processes, I need to choose test_bs appropriately.
        Otherwise, I think I can choose a maximum test_bs of 48.
        '''



        for sess_type in sess_type_list:
            
            self.configs['sess_type'] = sess_type

            for i in self.n_trial_types_list:

                self.configs['inst_trial_type'] = i

                self.init_model_and_analysis_save_paths()

                self.visualize_score_grad_in_pixel_space_for_each_inst_trial_type()


    def visualize_score_grad_in_pixel_space_for_each_inst_trial_type(self):


        cnn_channel_str = '_'.join([str(i) for i in self.configs['_cnn_channel_list']])

        if self.configs['img_type'] == 'jpg':
            img_type_str = ''
        elif self.configs['img_type'] == 'png':
            img_type_str = 'png_frames'

        if self.configs['neural_begin_time'] == -1.4 and self.configs['neural_end_time'] == -0.2 and self.configs['neural_skip_time'] == 0.2:
            neural_time_str = ''
        else:
            neural_time_str = 'begin_{:.1f}_end_{:.1f}_skip_{:.1f}'.format(self.configs['neural_begin_time'], self.configs['neural_end_time'], self.configs['neural_skip_time'])



        # analysis_save_path.
        # model_cv_dir: /data5/bkang/bk_video_ALM/NeuronalData5_codes/cd_reg_from_videos_all_no_stim_trials_cv_frame_conversion_corrected_full_cv_models
        if len(self.configs['models_cv_dir'].split('\\')) != 1:
            analysis_dir = self.configs['models_cv_dir'].split('\\')
            analysis_dir[-1] = 'pert_pred_from_videos_analysis_results'
            analysis_dir = '/'.join(analysis_dir)
        else:
            analysis_dir = 'pert_pred_from_videos_analysis_results'


        # Note that we are not including the last 'cv_ind_{}'.format(cv_ind) here. It will be included in individual methods.
        fig_save_path = os.path.join(analysis_dir, 'visualize_score_grad_in_pixel_space', self.configs['pert_type'], \
            'random_seed_{}'.format(self.configs['random_seed']), \
            img_type_str, neural_time_str, 'view_type_{}'.format('_'.join(self.configs['view_type'])), \
            'model_name_{}'.format(self.configs['model_name']), 'cnn_channel_list_{}'.format(cnn_channel_str), \
            'bs_{}_epochs_{}'.format(self.configs['bs'], self.configs['epochs']), 'n_cv_{}'.format(self.configs['n_cv']),\
            self.configs['sess_type'], 'inst_trial_type_{}'.format(self.configs['inst_trial_type']))

        os.makedirs(fig_save_path, exist_ok=True)




        # (n_sess, T, H, W)
        trial_avg_inputs_for_pert_trials_side_view, trial_avg_inputs_for_pert_trials_bottom_view = \
        self.get_trial_avg_inputs_for_pert_trials()



        '''
        X has shape (n_cv, n_sess)
        X[cv_ind,sess_ind] has shape (n_trials, T, n_input_channels, H, W)
        '''

        # cv_and_sess_mask = \
        # np.load(os.path.join(self.analysis_save_path, 'compute_score_grad_in_pixel_space',\
        #  'cv_and_sess_mask.npy'))

        # cv_agg_sess_wise_score_grad_for_pert_trials_side_view_old = \
        # np.load(os.path.join(self.analysis_save_path, 'compute_score_grad_in_pixel_space',\
        #  'cv_agg_sess_wise_score_grad_for_pert_trials_side_view.npy'))

        # cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view_old = \
        # np.load(os.path.join(self.analysis_save_path, 'compute_score_grad_in_pixel_space',\
        #  'cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view.npy'))



        # n_cv = self.configs['n_cv']
        # n_sess = len(trial_avg_inputs_for_pert_trials_side_view)

        # cv_agg_sess_wise_score_grad_for_pert_trials_side_view = np.zeros((n_cv, n_sess), dtype=object)
        # cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view = np.zeros((n_cv, n_sess), dtype=object)

        # for cv_ind in range(n_cv):
        #     for sess_ind in range(n_sess):

        #         cur_mask = cv_and_sess_mask[cv_ind,sess_ind]
                
        #         cv_agg_sess_wise_score_grad_for_pert_trials_side_view[cv_ind,sess_ind] = \
        #         cv_agg_sess_wise_score_grad_for_pert_trials_side_view_old[cur_mask]

        #         cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view[cv_ind,sess_ind] = \
        #         cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view_old[cur_mask]



        cv_agg_sess_wise_score_grad_for_pert_trials_side_view = \
        np.load(os.path.join(self.analysis_save_path, 'compute_score_grad_in_pixel_space',\
         'cv_agg_sess_wise_score_grad_for_pert_trials_side_view.npy'))

        cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view = \
        np.load(os.path.join(self.analysis_save_path, 'compute_score_grad_in_pixel_space',\
         'cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view.npy'))


        n_cv, n_sess = cv_agg_sess_wise_score_grad_for_pert_trials_side_view.shape

        _, T, n_input_channels, H_side, W_side = cv_agg_sess_wise_score_grad_for_pert_trials_side_view[0,0].shape

        _, _, _, H_bottom, W_bottom = cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view[0,0].shape


        '''
        Trial-average the l1 norm across channels of gradient across train-test splits.
        '''
        # (n_sess, T, H, W)
        trial_avg_grad_side_view_l1_norm_in_channel = np.zeros((n_sess, T, H_side, W_side))
        trial_std_grad_side_view_l1_norm_in_channel = np.zeros((n_sess, T, H_side, W_side))

        trial_avg_grad_bottom_view_l1_norm_in_channel = np.zeros((n_sess, T, H_bottom, W_bottom))
        trial_std_grad_bottom_view_l1_norm_in_channel = np.zeros((n_sess, T, H_bottom, W_bottom))

        for sess_ind in range(n_sess):

            '''
            side view
            '''
            # (n_trials, T, n_input_channels, H, W)
            cur_grad_side_view = cat(cv_agg_sess_wise_score_grad_for_pert_trials_side_view[:,sess_ind], 0)

            trial_avg_grad_side_view_l1_norm_in_channel[sess_ind] = \
            np.linalg.norm(cur_grad_side_view, axis=2, ord=1).mean(0) # (n_trials, T, H, W) -> (T, H, W)

            trial_std_grad_side_view_l1_norm_in_channel[sess_ind] = \
            np.linalg.norm(cur_grad_side_view, axis=2, ord=1).std(0, ddof=1) # (n_trials, T, H, W) -> (T, H, W)


            '''
            bottom view
            '''
            # (n_trials, T, n_input_channels, H, W)
            cur_grad_bottom_view = cat(cv_agg_sess_wise_score_grad_for_pert_trials_bottom_view[:,sess_ind], 0)

            trial_avg_grad_bottom_view_l1_norm_in_channel[sess_ind] = \
            np.linalg.norm(cur_grad_bottom_view, axis=2, ord=1).mean(0) # (n_trials, T, H, W) -> (T, H, W)

            trial_std_grad_bottom_view_l1_norm_in_channel[sess_ind] = \
            np.linalg.norm(cur_grad_bottom_view, axis=2, ord=1).std(0, ddof=1) # (n_trials, T, H, W) -> (T, H, W)


        print('')
        print('<side view>')
        print('trial_avg of grad l1 norm: {:.2e}'.format(trial_avg_grad_side_view_l1_norm_in_channel.mean()))
        print('trial_std of grad l1 norm: {:.2e}'.format(trial_std_grad_side_view_l1_norm_in_channel.mean()))

        print('')
        print('<bottom view>')
        print('trial_avg of grad l1 norm: {:.2e}'.format(trial_avg_grad_bottom_view_l1_norm_in_channel.mean()))
        print('trial_std of grad l1 norm: {:.2e}'.format(trial_std_grad_bottom_view_l1_norm_in_channel.mean()))


        '''
        ###
        We want to visualize the trial-avg score_grad (l1 norm across channel dimensions)
        on top of the trial-avg inputs, for each pred time step separately.
        We are just using test trials aggregated across train-test splits as for the score grad.
        ###
        '''



        n_rows = 20
        n_cols = n_sess//n_rows
        if n_sess%n_rows != 0:
            n_cols += 1


        sess_ind_to_f1_f2_map = np.load(os.path.join(self.analysis_save_path, 'compute_pred', 'sess_ind_to_f1_f2_map.npy'))



        # # temp
        # fig = plt.figure()
        # ax = fig.add_subplot(111)

        # sess_ind = 0
        # t = 0

        # ax.imshow(trial_avg_grad_side_view_l1_norm_in_channel[sess_ind,t], cmap=plt.get_cmap('Reds'))
        # ax.imshow(trial_avg_inputs_for_pert_trials_side_view[sess_ind,t], cmap='gray', vmin=0, vmax=1, alpha=0.7)

        # fig.savefig(os.path.join(fig_save_path, 'grad_vis_side_view.png'))

        # return
        # # temp

        '''
        side view
        '''


        fig = plt.figure(figsize=(n_cols*T*7, n_rows*7))

        gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.4, wspace=0.4)

        for sess_ind in range(n_sess):

            ax = fig.add_subplot(gs[sess_ind])

            f1_f2 = sess_ind_to_f1_f2_map[sess_ind]

            ax.set_title(f1_f2)
            ax.axis('off')


            sub_gs = gridspec.GridSpecFromSubplotSpec(1,T, subplot_spec=gs[sess_ind])

            for t in range(T):
                ax = fig.add_subplot(sub_gs[t])
                ax.axis('off')

                ax.imshow(trial_avg_grad_side_view_l1_norm_in_channel[sess_ind,t], cmap=plt.get_cmap('Reds'))
                ax.imshow(trial_avg_inputs_for_pert_trials_side_view[sess_ind,t], cmap='gray', vmin=0, vmax=1, alpha=0.9)


        fig.savefig(os.path.join(fig_save_path, 'grad_vis_side_view.png'))





        '''
        bottom view
        '''


        fig = plt.figure(figsize=(n_cols*T*7, n_rows*7))

        gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.4, wspace=0.4)

        for sess_ind in range(n_sess):

            ax = fig.add_subplot(gs[sess_ind])

            f1_f2 = sess_ind_to_f1_f2_map[sess_ind]

            ax.set_title(f1_f2)
            ax.axis('off')


            sub_gs = gridspec.GridSpecFromSubplotSpec(1,T, subplot_spec=gs[sess_ind])

            for t in range(T):
                ax = fig.add_subplot(sub_gs[t])
                ax.axis('off')

                ax.imshow(trial_avg_grad_bottom_view_l1_norm_in_channel[sess_ind,t], cmap=plt.get_cmap('Reds'))
                ax.imshow(trial_avg_inputs_for_pert_trials_bottom_view[sess_ind,t], cmap='gray', vmin=0, vmax=1, alpha=0.9)


        fig.savefig(os.path.join(fig_save_path, 'grad_vis_bottom_view.png'))

    

    def get_trial_avg_inputs_for_pert_trials(self):

        n_trial_types_list = self.n_trial_types_list
        configs = self.configs



        if 'rnn' in self.configs['model_name']:
            self.configs['_input_channel'] = 1

        assert self.configs['inst_trial_type'] in [0, 1]
        

        # Data loading parameters

        '''
        To use WeightedRandomSampler, we must set shuffle to False.
        '''
        params = {'batch_size': self.configs['test_bs'], 'shuffle': False, 'num_workers': self.configs['num_workers'], \
        'pin_memory': bool(self.configs['pin_memory']), 'drop_last': False}\

        # Collect video filenames.
        if self.configs['img_type'] == 'jpg':
            video_root_path = os.path.join(self.configs['video_data_dir'], 'frames')
        elif self.configs['img_type'] == 'png':
            video_root_path = os.path.join(self.configs['video_data_dir'], 'png_frames')


        '''
        Session selection.
        Here, we define f1_f2_list.
        '''
        sess_type = self.configs['sess_type']

        assert sess_type in ['{}_{}'.format(task_type, sort_type) for sort_type in ['guang_one_laser'] for task_type in ['original', 'full_reverse']]

        filehandler = open('sorted_filenames.obj', 'rb')
        filenames = pickle.load(filehandler)
        
        sess_by_task_dict = pert_pred_utils.load_sess_by_task_dict()

        print('')
        print('<Before restricting to sessions that have videos>')
        print('sess_type: ', sess_type)
        print('n_sess: ', len(sess_by_task_dict[sess_type]))

        pre_f1_f2_list = []
        for sess_idx in sess_by_task_dict[sess_type]:
            sess_name = filenames[sess_idx-1].split('\\')[1][:-4].replace('kilosort', '') #BAYLORGC100_2020_01_21
            # f1 = sess_name.split('_')[0]
            # start = sess_name.find('_') + 1
            # f2 = sess_name[start:]
            pre_f1_f2_list.append(sess_name)

        pre_f1_f2_list = np.array(pre_f1_f2_list)


        '''
        Load f1_f2_list, which contains tuples of form (f1,f2)
        '''
        if bool(self.configs['debug']):
            f1_f2_list = []
            for f1 in os.listdir(video_root_path):
                if os.path.isfile(os.path.join(video_root_path, f1)):
                    continue
                for f2 in os.listdir(os.path.join(video_root_path, f1)):

                    if os.path.isfile(os.path.join(video_root_path, f1, f2)):
                        continue

                    f1_f2_list.append((f1,f2))


        else:
            # if not os.path.isfile(os.path.join('make_f1_f2_list_results', sess_type, 'f1_f2_list.pkl')):
            #     import make_f1_f2_list
            #     make_f1_f2_list.main(sess_type)

            # Since I modify make_f1_f2_list from time to time, I want to generate f1_f2_list each time I run this code.
            import make_f1_f2_list
            task_type = 'standard' if 'original' in sess_type else 'full_reverse'
            make_f1_f2_list.main('{}_all'.format(task_type))

            with open(os.path.join('make_f1_f2_list_results', '{}_all'.format(task_type), 'f1_f2_list.pkl'), 'rb') as f:
                f1_f2_list = pickle.load(f)


        video_f1_f2_list = np.array(['_'.join([f1,f2]) for (f1,f2) in f1_f2_list])


        '''
        We take those sessions in pre_f1_f2_list that are in video_f1_f2_list.
        '''
        f1_f2_list = pre_f1_f2_list[np.isin(pre_f1_f2_list, video_f1_f2_list)]

        # sanity check
        # pre_f1_f2_set = set(list(pre_f1_f2_list))
        # video_f1_f2_set = set(list(video_f1_f2_list))
        # print('')
        # print('sanity check')
        # print('pre_f1_f2_set.difference(video_f1_f2_set)')
        # for x in list(pre_f1_f2_set.difference(video_f1_f2_set)):
        #     print(x)

        print('')
        print('sanity check')
        print('<pre_f1_f2_list>')
        print('n_sess: ', len(pre_f1_f2_list))
        for x in pre_f1_f2_list:
            print(x)
        print('')
        print('<video_f1_f2_list>')
        print('n_sess: ', len(video_f1_f2_list))
        for x in video_f1_f2_list:
            print(x)
        print('')
        print('<Sessions in pre_f1_f2_list that are not in video_f1_f2_list>')
        print('n_sess: ', len(pre_f1_f2_list[~np.isin(pre_f1_f2_list, video_f1_f2_list)]))
        for x in pre_f1_f2_list[~np.isin(pre_f1_f2_list, video_f1_f2_list)]:
            print(x)



        print('')
        print('<After restricting to sessions that have videos>')
        print('sess_type: ', sess_type)
        print('n_sess: ', len(f1_f2_list))
        print('')


        f1_f2_list_old = f1_f2_list.copy()

        f1_f2_list = []
        for sess_name in f1_f2_list_old:
            # sess_name: BAYLORGC100_2020_01_21
            f1 = sess_name.split('_')[0]
            start = sess_name.find('_') + 1
            f2 = sess_name[start:]
            f1_f2_list.append((f1,f2))

        f1_f2_list = np.array(f1_f2_list)



        # if not self.configs['debug']:
        #     '''
        #     I found that certain sessions such as GC90_2019_11_18 don't have side view videos.
        #     I want to remove such sessions from f1_f2_list.
        #     '''
        #     if 'full_reverse' in sess_type:
        #         avi_root_path = '/mnt/fs4/bkang/alm_video_data/E3/avi_files'
        #     elif 'standard' in sess_type:
        #         # We know that sessions in /mnt/fs6/zwang71/MiceBehavior/data/E3/Session all have both view videos.
        #         avi_root_path = '/mnt/fs2/bkang/alm_video_data/E3/avi_files'


        #     print('')
        #     print('len(f1_f2_list) before checking for bad sess: {}'.format(len(f1_f2_list)))
        #     good_sess_idx = []

        #     for sess_idx, (f1, f2) in enumerate(f1_f2_list):
        #         print('')
        #         print('Checking f1: {} / f2: {}'.format(f1,f2))
                
        #         if f1 in ['BAYLORGC25', 'BAYLORGC26', 'BAYLORGC50']:
        #             good_sess_idx.append(sess_idx)
        #             continue

        #         # avi_file: GC50_20181211_bottom_trial_100-0000.avi
        #         if not any('side' in avi_file for avi_file in os.listdir(os.path.join(avi_root_path, f1, f2, 'Camera'))):
        #             print('')
        #             print('No side view in:')
        #             print('f1: {} / f2: {}'.format(f1, f2))
        #             continue

        #         if not any('bottom' in avi_file for avi_file in os.listdir(os.path.join(avi_root_path, f1, f2, 'Camera'))):
        #             print('')
        #             print('No bottom view in:')
        #             print('f1: {} / f2: {}'.format(f1, f2))
        #             continue

        #         good_sess_idx.append(sess_idx)


        #     f1_f2_list = f1_f2_list[good_sess_idx]

        #     print('')
        #     print('len(f1_f2_list) after checking for bad sess: {}'.format(len(f1_f2_list)))




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
        Collect video path.

        Use no_stim_i_good_trials.
        video_path_dict[f1][f2][i] will contain paths to videos for trials in no_stim_i_good_trials[i].
        And we will separately split trials of each trial type into train and test set.
        '''

        video_path_dict = nestdict()
        for f1, f2 in f1_f2_list:

            no_stim_i_good_trials = pert_pred_utils.load_no_stim_i_good_trials_for_pert_pred(self.configs['inst_trial_type'], self.configs['prep_root_dir'], f1, f2)
            stim_i_good_trials = pert_pred_utils.load_stim_i_good_trials_for_pert_pred(self.configs['pert_type'], self.configs['inst_trial_type'], self.configs['prep_root_dir'], f1, f2)

            total_i_good_trials = np.array([no_stim_i_good_trials, stim_i_good_trials])


            for i in n_trial_types_list:
                # Just a sanity check to make sure i_good_trials are sorted.
                assert (total_i_good_trials[i] == sorted(total_i_good_trials[i])).all()

                video_path_dict[f1][f2][i] = []


            for f3 in os.listdir(os.path.join(video_root_path, f1, f2)):
                # f3 = 'E3-BAYLORGC25-2018_09_17-13'
                if os.path.isfile(os.path.join(video_root_path, f1, f2, f3)):
                    continue


                for i in n_trial_types_list:
                    if pert_pred_utils.map_f3_to_trial_idx(f1,f2,f3) in total_i_good_trials[i]:
                        video_path_dict[f1][f2][i].append(f3)


            for i in n_trial_types_list:
                # Sort cur_path_dict[f2] according to the trial index at the end.
                # Not necessary to sort it really, but let's do it anyways.
                video_path_dict[f1][f2][i].sort(key=video_trial_sort)




        '''
        Frame selection
        '''


        selected_frames_dict = nestdict()
        self.configs['pred_timesteps'] = None


        # Needed below to figure out begin and end_bin.
        f1 = list(video_path_dict.keys())[0]
        f2 = list(video_path_dict[f1].keys())[0]

        bin_centers = pert_pred_utils.load_bin_centers(self.configs['prep_root_dir'],f1,f2)


        # These are used for cur_cd_non_cd_pca.
        begin_bin = np.argmin(np.abs(bin_centers - self.configs['neural_begin_time']))
        end_bin = np.argmin(np.abs(bin_centers - self.configs['neural_end_time']))

        bin_stride = 0.1
        skip_bin = int(self.configs['neural_skip_time']//bin_stride)

        pred_times = bin_centers[begin_bin:end_bin+1:skip_bin]



        print('')
        print('Selecting frames for each session...')


        # Load n_frames_dict
        if not self.configs['debug']:
            save_path = 'get_n_frames_for_each_sess_results'
            if not os.path.isfile(os.path.join(save_path, 'n_frames_dict.pkl')):
                import get_n_frames_for_each_sess
                get_n_frames_for_each_sess.main()

            with open(os.path.join(save_path, 'n_frames_dict.pkl'), 'rb') as f:
                n_frames_dict = pickle.load(f)

        for f1, f2 in f1_f2_list:
            if not self.configs['debug']:
                cur_n_frames = n_frames_dict[(f1,f2)]
            else:
                cur_n_frames = 1000

            if cur_n_frames == 1000:

                go_cue_time = 3.57
                begin_frame = int(np.rint((pred_times[0] + go_cue_time)*200)) # 200 frames per sec.
                end_frame = int(np.rint((pred_times[-1] + go_cue_time)*200))
                skip_frame = self.configs['skip_frame']

            elif cur_n_frames == 1200:
                go_cue_time = 4.57
                begin_frame = int(np.rint((pred_times[0] + go_cue_time)*200)) # 200 frames per sec.
                end_frame = int(np.rint((pred_times[-1] + go_cue_time)*200))
                skip_frame = self.configs['skip_frame']

            else:
                raise ValueError('Invalid cur_n_frames: {}. Needs to be either 1000 or 1200'.format(cur_n_frames))

            # Both of the returned variables are 1d np array.
            selected_frames, pred_timesteps = \
            pert_pred_utils.compute_selected_frames_and_pred_timesteps(self.configs, pred_times, begin_frame, end_frame, skip_frame, go_cue_time)

            # print(pred_times)
            # print(selected_frames)
            # print(pred_timesteps)

            if self.configs['pred_timesteps'] is None:
                self.configs['pred_timesteps'] = pred_timesteps
            else:
                # Even when n_frames is different across sessions, pred_timesteps should be the same.
                if (self.configs['pred_timesteps'] != pred_timesteps).any():
                    raise ValueError('self.configs pred_timesteps (={}) is \
                        NOT the same as current (f1:{}/f2:{}) pred_timesteps (={})'.format(\
                        ', '.join([str(x) for x in self.configs['pred_timesteps']]), f1, f2, ', '.join([str(x) for x in pred_timesteps])))

            for i in n_trial_types_list:
                selected_frames_dict[f1][f2][i] = np.broadcast_to(selected_frames, (len(video_path_dict[f1][f2][i]), len(selected_frames)))

            # print('')
            # print('f1: {} / f2: {}'.format(f1, f2))
            # print('n_frames: {}'.format(cur_n_frames))
            # print('go_cue_time: {}'.format(go_cue_time))
            # print('begin_frame: {}'.format(begin_frame))
            # print('end_frame: {}'.format(end_frame))
            # print('skip_frame: {}'.format(skip_frame))





        # Determine n_sess and sess_inds for sess_cond model.
        self.configs['n_sess'] = 0
        sess_inds_dict = nestdict()

        for f1, f2 in f1_f2_list:

            self.configs['n_sess'] += 1

            for i in n_trial_types_list:
                # n_sess -1 because sess_inds ranges from 0 to n_sess-1.
                sess_inds_dict[f1][f2][i] = [self.configs['n_sess']-1]*len(video_path_dict[f1][f2][i])










        '''
        Cross-validation train-test split.
        We determine the split using video trials, and then
        use it when generating neural data in which CD is computed only using the train set's correct trials.
        '''

        n_cv = self.configs['n_cv']

        sess_wise_test_video_path_dict = nestdict()
        sess_wise_test_sess_inds_dict = nestdict()
        sess_wise_test_trial_type_dict = nestdict()
        sess_wise_test_selected_frames_dict = nestdict()

        sess_wise_test_trial_idxs_dict = nestdict()
        '''
        I realized that for some sessions and inst_trial_types, there are very few 
        error trials, less than n_cv, which makes it impossible to perform a separate k-fold
        split for each inst_trial_type. To circumvent this issue, we first identify
        sessions in which at least one of inst_trial_types have error trials fewer
        than n_cv. For those sessions, we perform k-fold split for the entire set of 
        trials in a given inst_trial_type.
        '''
        few_error_trial_f1_f2_list = []


        for f1, f2 in f1_f2_list:
            # f1 = 'BAYLORGC[#]'
            # f2 = '[yyyy]_[mm]_[dd]'

            print('')
            print(f1, f2)

            for i in n_trial_types_list:
                cur_video_path = np.array([os.path.join(f1,f2,f3) for f3 in video_path_dict[f1][f2][i]]) # an array of each trial's video path
                
                print('{}'.format('no_stim' if i==0 else self.configs['pert_type']))
                print(len(cur_video_path))

                if len(cur_video_path) < n_cv:
                    few_error_trial_f1_f2_list.append((f1, f2))
                    break
        
        print('')
        print('few_error_trial_f1_f2_list')
        print('')
        for f1, f2 in few_error_trial_f1_f2_list:
            print(f1,f2)




        '''
        Here, only iterate through sessions that are not in few_error_trial_f1_f2_list.
        '''
        for f1, f2 in f1_f2_list:
            # f1 = 'BAYLORGC[#]'
            # f2 = '[yyyy]_[mm]_[dd]'
            if (f1,f2) in few_error_trial_f1_f2_list:
                continue

            for i in n_trial_types_list:

                # We wrap lists into arrays so that they can be indexed with advanced indexing.
                cur_video_path = np.array([os.path.join(f1,f2,f3) for f3 in video_path_dict[f1][f2][i]]) # an array of each trial's video path

                cur_sess_inds = np.array(sess_inds_dict[f1][f2][i]) # an array [sess_ind]*n_trials.
                cur_trial_type = np.array([i]*len(cur_sess_inds)) # an array [i]*n_trials.

                cur_selected_frames = selected_frames_dict[f1][f2][i] # np array of shape (n_trials, n_selected_frames)

                '''
                In most cases, map_f3_to_trial_idx is int(f3.split('-')[3]).But,
                there are a few exceptional sessions where this is not true.
                '''
                cur_trial_idxs = np.array([pert_pred_utils.map_f3_to_trial_idx(f1,f2,f3) for f3 in video_path_dict[f1][f2][i]]) # an array of each trial's video path


                kf = KFold(n_splits=n_cv, shuffle=True, random_state=1)

                for cv_ind, (train_inds, test_inds) in enumerate(kf.split(cur_video_path)):


                    cur_train_video_path, cur_test_video_path = cur_video_path[train_inds], cur_video_path[test_inds]
                    cur_train_sess_inds, cur_test_sess_inds = cur_sess_inds[train_inds], cur_sess_inds[test_inds]
                    cur_train_trial_type, cur_test_trial_type = cur_trial_type[train_inds], cur_trial_type[test_inds]
                    cur_train_selected_frames, cur_test_selected_frames = cur_selected_frames[train_inds], cur_selected_frames[test_inds]


                    sess_wise_test_video_path_dict[cv_ind][f1][f2][i] = cur_test_video_path
                    sess_wise_test_sess_inds_dict[cv_ind][f1][f2][i] = cur_test_sess_inds
                    sess_wise_test_trial_type_dict[cv_ind][f1][f2][i] = cur_test_trial_type
                    sess_wise_test_selected_frames_dict[cv_ind][f1][f2][i] = cur_test_selected_frames

                    sess_wise_test_trial_idxs_dict[cv_ind][f1][f2][i] = cur_trial_idxs[test_inds]




        '''
        ###
        Note added on 10/01/20 (while revisiting this code for Cell-revision):
        
        Note that in the for loop below for few_error_trial_f1_f2_list, the index i
        is fixed at 1 after the for loop for i, and thus various dict will have an empty
        entry for dict[cv_ind][f1][f2][0].

        This didn't cause any error afterwards because list's extend simply does not add
        any element from dict[cv_ind][f1][f2][0] (I confirmed this behavior), and I initially
        thought this would have a bad effect on learning. But, I realize that in fact, it
        doesn't affect learning at all, since we are going to concatenate dict[cv_ind][f1][f2][i]
        anyways for training and testing! The same holds for sess-wise validation set evaluation.
        ###
        '''


        for f1, f2 in few_error_trial_f1_f2_list:

            cur_video_path = []
            cur_sess_inds = []
            cur_trial_type = []
            cur_selected_frames = []
            cur_trial_idxs = []

            for i in n_trial_types_list:

                # We wrap lists into arrays so that they can be indexed with advanced indexing.
                cur_video_path.extend([os.path.join(f1,f2,f3) for f3 in video_path_dict[f1][f2][i]]) # an array of each trial's video path

                cur_sess_inds.extend(sess_inds_dict[f1][f2][i]) # an array [sess_ind]*n_trials.
                cur_trial_type.extend([i]*len(cur_sess_inds)) # an array [i]*n_trials.

                cur_selected_frames.extend(selected_frames_dict[f1][f2][i]) # np array of shape (n_trials, n_selected_frames)

                '''
                In most cases, map_f3_to_trial_idx is int(f3.split('-')[3]).But,
                there are a few exceptional sessions where this is not true.
                '''

                cur_trial_idxs.extend([pert_pred_utils.map_f3_to_trial_idx(f1,f2,f3) for f3 in video_path_dict[f1][f2][i]]) # an array of each trial's video path


            cur_video_path = np.array(cur_video_path)
            cur_sess_inds = np.array(cur_sess_inds)
            cur_trial_type = np.array(cur_trial_type)
            cur_selected_frames = np.array(cur_selected_frames)
            cur_trial_idxs = np.array(cur_trial_idxs)


            kf = KFold(n_splits=n_cv, shuffle=True, random_state=1)

            for cv_ind, (train_inds, test_inds) in enumerate(kf.split(cur_video_path)):


                cur_train_video_path, cur_test_video_path = cur_video_path[train_inds], cur_video_path[test_inds]
                cur_train_sess_inds, cur_test_sess_inds = cur_sess_inds[train_inds], cur_sess_inds[test_inds]
                cur_train_trial_type, cur_test_trial_type = cur_trial_type[train_inds], cur_trial_type[test_inds]
                cur_train_selected_frames, cur_test_selected_frames = cur_selected_frames[train_inds], cur_selected_frames[test_inds]


                sess_wise_test_video_path_dict[cv_ind][f1][f2][i] = cur_test_video_path
                sess_wise_test_sess_inds_dict[cv_ind][f1][f2][i] = cur_test_sess_inds
                sess_wise_test_trial_type_dict[cv_ind][f1][f2][i] = cur_test_trial_type
                sess_wise_test_selected_frames_dict[cv_ind][f1][f2][i] = cur_test_selected_frames

                sess_wise_test_trial_idxs_dict[cv_ind][f1][f2][i] = cur_trial_idxs[test_inds]






        '''
        ###
        NOTE:

        I realized that I can save some time by first combining all sessions data into a single data loader
        and then later dividing up the outputs of the network into different sessions.
        ###
        '''


        test_video_path = []

        test_sess_inds = []

        test_trial_type = []

        test_selected_frames = []

        test_trial_idxs = []


        for cv_ind in range(n_cv):

            for f1, f2 in f1_f2_list:
                # f1 = 'BAYLORGC[#]'
                # f2 = '[yyyy]_[mm]_[dd]'

                for i in n_trial_types_list:

                    '''
                    For now, to save disk space, we are only computing grad for bi stim trials.
                    So, we exclude control trials which have i==0.
                    '''
                    if i == 0:
                        continue

                    # Test set
                    test_video_path.extend(sess_wise_test_video_path_dict[cv_ind][f1][f2][i])
                    test_sess_inds.extend(sess_wise_test_sess_inds_dict[cv_ind][f1][f2][i])
                    test_trial_type.extend(sess_wise_test_trial_type_dict[cv_ind][f1][f2][i])
                    test_selected_frames.extend(sess_wise_test_selected_frames_dict[cv_ind][f1][f2][i])
                    test_trial_idxs.extend(sess_wise_test_trial_idxs_dict[cv_ind][f1][f2][i])



        # Turn the lists into arrays.

        test_video_path = np.array(test_video_path)
        test_sess_inds = np.array(test_sess_inds)
        test_trial_type = np.array(test_trial_type)
        test_selected_frames = np.array(test_selected_frames)
        test_trial_idxs = np.array(test_trial_idxs)

 
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





        combined_valid_set = MyDatasetChoicePredVariableFramesRecon(video_root_path, test_video_path,\
            test_sess_inds, test_trial_type, test_selected_frames, \
            test_trial_idxs, configs['view_type'], \
            transform_list=transform_list, img_type=configs['img_type'])


        # Because I am using a variable named "data" in the for loop for combined_valid_loader below,
        # I got an UnboundedLocalError before. To avoid it, I replaced "data" with "torch.utils.data".
        combined_valid_loader = torch.utils.data.DataLoader(combined_valid_set, **params)


        # (n_trials, T, H, W)
        # We need to keep individual trials at this point to trial-average separately for each sesssion.
        combined_X_side = []
        combined_X_bottom = []
        combined_sess_inds = []


        '''
        Copied the definition of input_timesteps from lrcn_choice_pred_bn_static2_all_views_time_indp_dec_sess_cond_v2.
        '''

        n_input_frames = self.configs['_input_channel']


        # n_pre and n_post_frames = 4 covers +-0.2 s around the bin center (assuming that skip_frame = 10).
        n_pre_frames = (n_input_frames-1)//2
        n_post_frames = (n_input_frames-1)//2

        if (n_input_frames-1)%2 != 0:
            n_pre_frames += 1


        T = len(self.configs['pred_timesteps'])

        frame_idxs_for_each_pred_timestep = np.zeros((T, n_input_frames), dtype=int)
        for t, k in enumerate(self.configs['pred_timesteps']):
            frame_idxs_for_each_pred_timestep[t] = np.arange(k-n_pre_frames, k+n_post_frames+1, 1)


        for batch_idx, data in enumerate(combined_valid_loader):

            # X: (n_trials, n_frames, c, h, w)
            X_side, X_bottom, y, z, trial_idx = data

            # X: (n_trials, n_frames, h, w)
            X_side, X_bottom, z = X_side[:,:,0].data.cpu().numpy(), X_bottom[:,:,0].data.cpu().numpy(), z.data.cpu().numpy()



            '''
            side view
            '''

            # (n_trials, T, H, W)
            cur_X_side = []
            for t in range(T):
                cur_X_side.append(X_side[:,frame_idxs_for_each_pred_timestep[t]].mean(1)) # (n_trials, H, W)

            # (n_trials, T, H, W)
            cur_X_side = np.stack(cur_X_side, 1)

            combined_X_side.append(cur_X_side)


            '''
            bottom view
            '''

            # (n_trials, T, H, W)
            cur_X_bottom = []
            for t in range(T):
                cur_X_bottom.append(X_bottom[:,frame_idxs_for_each_pred_timestep[t]].mean(1)) # (n_trials, H, W)

            # (n_trials, T, H, W)
            cur_X_bottom = np.stack(cur_X_bottom, 1)

            combined_X_bottom.append(cur_X_bottom)



            combined_sess_inds.append(z)


        combined_X_side = cat(combined_X_side, 0)
        combined_X_bottom = cat(combined_X_bottom, 0)
        combined_sess_inds = cat(combined_sess_inds, 0)

        _, T, H_side, W_side = combined_X_side.shape
        _, T, H_bottom, W_bottom = combined_X_bottom.shape


        n_sess = len(f1_f2_list)

        trial_avg_inputs_for_pert_trials_side_view = np.zeros((n_sess, T, H_side, W_side))
        trial_avg_inputs_for_pert_trials_bottom_view = np.zeros((n_sess, T, H_bottom, W_bottom))

        for sess_ind in range(n_sess):
            cur_sess_mask = (combined_sess_inds==sess_ind)

            trial_avg_inputs_for_pert_trials_side_view[sess_ind] = combined_X_side[cur_sess_mask].mean(0)
            trial_avg_inputs_for_pert_trials_bottom_view[sess_ind] = combined_X_bottom[cur_sess_mask].mean(0)

        return trial_avg_inputs_for_pert_trials_side_view, trial_avg_inputs_for_pert_trials_bottom_view