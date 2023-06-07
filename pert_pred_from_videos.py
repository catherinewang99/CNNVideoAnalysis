'''
Adapted from choice_pred_from_videos_cv_frame_conversion_corrected_full_cv_v4_alt.

Some things that are useful to know:
1. i=0 means control trials, i=1 means pert trials.


'''

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from video_pred_functions import *
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import json
from collections import defaultdict

import pert_pred_utils

import random

cat = np.concatenate


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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(configs=None):

    n_trial_types_list = list(range(2))

    # Load pred_configs
    if configs is None:
        with open('pert_pred_configs.json','r') as read_file:
            configs = json.load(read_file)


    '''
    ###
    To avoid having duplicate processes on gpu 0,
    I need to use CUDA_VISIBLE_DEVICES.
    Once I set visible devices, the gpu ids need to start from 0.
    ###
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(', '.join([str(x) for x in configs['gpu_ids_before_masking']]))
    print('')
    print('os.environ["CUDA_VISIBLE_DEVICES"]')
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    print('')
    configs['gpu_ids'] = [x for x in range(len(configs['gpu_ids_before_masking']))]



    '''
    ###
    Set random seeds for determinism.
    ###
    '''
    set_seed(configs['random_seed'])
    #set_seed(torch.random.initial_seed())






    configs['use_cuda'] = bool(configs['use_cuda'])
    configs['do_pos_weight'] = bool(configs['do_pos_weight'])

    if 'rnn' in configs['model_name']:
        configs['_input_channel'] = 1

    assert configs['inst_trial_type'] in [0, 1]
    
    # Create directories to save results.
    os.makedirs(configs['logs_cv_dir'], exist_ok=True)
    os.makedirs(configs['models_cv_dir'], exist_ok=True)

    # Detect devices
    use_cuda = torch.cuda.is_available() and configs['use_cuda']                  # check if GPU exists
    device = torch.device("cuda:{}".format(configs['gpu_ids'][0]) if use_cuda else "cpu")   # use CPU or GPU

    # Data loading parameters

    '''
    To use WeightedRandomSampler, we must set shuffle to False.
    '''
    params = {'batch_size': configs['bs'], 'shuffle': False, 'num_workers': configs['num_workers'], 'pin_memory': bool(configs['pin_memory'])}\
     if use_cuda else {'batch_size': configs['bs']}

    # Collect video filenames.
    if configs['img_type'] == 'jpg':
        video_root_path = os.path.join(configs['video_data_dir'], 'frames')
    elif configs['img_type'] == 'png':
        video_root_path = os.path.join(configs['video_data_dir'], 'png_frames')


    '''
    Session selection.
    Here, we define f1_f2_list.
    '''
    sess_type = configs['sess_type']

    #assert sess_type in ['{}_{}'.format(task_type, sort_type) for sort_type in ['guang_one_laser'] for task_type in ['original', 'full_reverse']]

    filehandler = open('sorted_filenames.obj', 'rb')
    filenames = pickle.load(filehandler)
    
    #sess_by_task_dict = pert_pred_utils.load_sess_by_task_dict()

    print('')
    print('<Before restricting to sessions that have videos>')
    print('sess_type: ', sess_type)
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
    if bool(configs['debug']):
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

    # CW: add this to replace above line

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



    # if not configs['debug']:
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
    if 'all_views' in configs['model_name']:
        assert len(configs['view_type']) == 2
        assert configs['view_type'][0] == 'side'
        assert configs['view_type'][1] == 'bottom'

    if 'static' not in configs['model_name']:
        assert configs['_input_channel'] == 1

    if 'downsample' not in configs['model_name']:
        assert configs['image_shape'][0] == 86
        assert configs['image_shape'][1] == 130
        assert len(configs['_maxPoolLayers']) == 2



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

        no_stim_i_good_trials = pert_pred_utils.load_no_stim_i_good_trials_for_pert_pred(configs['inst_trial_type'], configs['prep_root_dir'], f1, f2)
        stim_i_good_trials = pert_pred_utils.load_stim_i_good_trials_for_pert_pred(configs['pert_type'], configs['inst_trial_type'], configs['prep_root_dir'], f1, f2)

        total_i_good_trials = np.array([no_stim_i_good_trials, stim_i_good_trials])


        for i in n_trial_types_list:
            # Just a sanity check to make sure i_good_trials are sorted.
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
    configs['pred_timesteps'] = None


    # Needed below to figure out begin and end_bin.
    f1 = list(video_path_dict.keys())[0]
    f2 = list(video_path_dict[f1].keys())[0]

    bin_centers = pert_pred_utils.load_bin_centers(configs['prep_root_dir'],f1,f2)


    # These are used for cur_cd_non_cd_pca.
    begin_bin = np.argmin(np.abs(bin_centers - configs['neural_begin_time']))
    end_bin = np.argmin(np.abs(bin_centers - configs['neural_end_time']))

    bin_stride = 0.1
    skip_bin = int(configs['neural_skip_time']//bin_stride)

    pred_times = bin_centers[begin_bin:end_bin+1:skip_bin]



    print('')
    print('Selecting frames for each session...')


    # Load n_frames_dict
    if not configs['debug']:
        save_path = 'get_n_frames_for_each_sess_results'
        if not os.path.isfile(os.path.join(save_path, 'n_frames_dict.pkl')):
            import get_n_frames_for_each_sess
            get_n_frames_for_each_sess.main()

        with open(os.path.join(save_path, 'n_frames_dict.pkl'), 'rb') as f:
            n_frames_dict = pickle.load(f)

    for f1, f2 in f1_f2_list:
        if not configs['debug']:
            cur_n_frames = n_frames_dict[(f1,f2)]
        else:
            cur_n_frames = 1000

        if cur_n_frames == 1000:

            go_cue_time = 3.57
            begin_frame = int(np.rint((pred_times[0] + go_cue_time)*200)) # 200 frames per sec.
            end_frame = int(np.rint((pred_times[-1] + go_cue_time)*200))
            skip_frame = configs['skip_frame']

        elif cur_n_frames == 1200:
            go_cue_time = 4.57
            begin_frame = int(np.rint((pred_times[0] + go_cue_time)*200)) # 200 frames per sec.
            end_frame = int(np.rint((pred_times[-1] + go_cue_time)*200))
            skip_frame = configs['skip_frame']

        else:
            raise ValueError('Invalid cur_n_frames: {}. Needs to be either 1000 or 1200'.format(cur_n_frames))

        # Both of the returned variables are 1d np array.
        selected_frames, pred_timesteps = \
        pert_pred_utils.compute_selected_frames_and_pred_timesteps(configs, pred_times, begin_frame, end_frame, skip_frame, go_cue_time)

        # print(pred_times)
        # print(selected_frames)
        # print(pred_timesteps)

        if configs['pred_timesteps'] is None:
            configs['pred_timesteps'] = pred_timesteps
        else:
            # Even when n_frames is different across sessions, pred_timesteps should be the same.
            if (configs['pred_timesteps'] != pred_timesteps).any():
                raise ValueError('configs pred_timesteps (={}) is \
                    NOT the same as current (f1:{}/f2:{}) pred_timesteps (={})'.format(\
                    ', '.join([str(x) for x in configs['pred_timesteps']]), f1, f2, ', '.join([str(x) for x in pred_timesteps])))

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
    configs['n_sess'] = 0
    sess_inds_dict = nestdict()

    for f1, f2 in f1_f2_list:

        configs['n_sess'] += 1 # 10

        for i in n_trial_types_list:
            # n_sess -1 because sess_inds ranges from 0 to n_sess-1.
            sess_inds_dict[f1][f2][i] = [configs['n_sess']-1]*len(video_path_dict[f1][f2][i])










    '''
    Cross-validation train-test split.
    We determine the split using video trials, and then
    use it when generating neural data in which CD is computed only using the train set's correct trials.
    '''

    n_cv = configs['n_cv']

    sess_wise_train_video_path_dict = nestdict()
    sess_wise_train_sess_inds_dict = nestdict()
    sess_wise_train_trial_type_dict = nestdict()
    sess_wise_train_selected_frames_dict = nestdict()

    sess_wise_test_video_path_dict = nestdict()
    sess_wise_test_sess_inds_dict = nestdict()
    sess_wise_test_trial_type_dict = nestdict()
    sess_wise_test_selected_frames_dict = nestdict()


    # trial_idx as extracted from f3. They are integer-valued. This corresponds to values in no_stim_i_good_trials.
    sess_wise_train_trial_idxs = nestdict()
    sess_wise_test_trial_idxs = nestdict()

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
            
            print('{}'.format('no_stim' if i==0 else configs['pert_type']))
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


                sess_wise_train_video_path_dict[cv_ind][f1][f2][i] = cur_train_video_path
                sess_wise_train_sess_inds_dict[cv_ind][f1][f2][i] = cur_train_sess_inds
                sess_wise_train_trial_type_dict[cv_ind][f1][f2][i] = cur_train_trial_type
                sess_wise_train_selected_frames_dict[cv_ind][f1][f2][i] = cur_train_selected_frames


                sess_wise_test_video_path_dict[cv_ind][f1][f2][i] = cur_test_video_path
                sess_wise_test_sess_inds_dict[cv_ind][f1][f2][i] = cur_test_sess_inds
                sess_wise_test_trial_type_dict[cv_ind][f1][f2][i] = cur_test_trial_type
                sess_wise_test_selected_frames_dict[cv_ind][f1][f2][i] = cur_test_selected_frames


                sess_wise_train_trial_idxs[cv_ind][f1][f2][i] = cur_trial_idxs[train_inds]
                sess_wise_test_trial_idxs[cv_ind][f1][f2][i] = cur_trial_idxs[test_inds]



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


            sess_wise_train_video_path_dict[cv_ind][f1][f2][i] = cur_train_video_path
            sess_wise_train_sess_inds_dict[cv_ind][f1][f2][i] = cur_train_sess_inds
            sess_wise_train_trial_type_dict[cv_ind][f1][f2][i] = cur_train_trial_type
            sess_wise_train_selected_frames_dict[cv_ind][f1][f2][i] = cur_train_selected_frames


            sess_wise_test_video_path_dict[cv_ind][f1][f2][i] = cur_test_video_path
            sess_wise_test_sess_inds_dict[cv_ind][f1][f2][i] = cur_test_sess_inds
            sess_wise_test_trial_type_dict[cv_ind][f1][f2][i] = cur_test_trial_type
            sess_wise_test_selected_frames_dict[cv_ind][f1][f2][i] = cur_test_selected_frames


            sess_wise_train_trial_idxs[cv_ind][f1][f2][i] = cur_trial_idxs[train_inds]
            sess_wise_test_trial_idxs[cv_ind][f1][f2][i] = cur_trial_idxs[test_inds]



    train_video_path = np.zeros(n_cv, dtype=object)
    test_video_path = np.zeros(n_cv, dtype=object)
    
    train_sess_inds = np.zeros(n_cv, dtype=object)
    test_sess_inds = np.zeros(n_cv, dtype=object)

    # We also keep track of trial type for certain models that use this information (e.g. used for temporal smoothing). 
    train_trial_type = np.zeros(n_cv, dtype=object)
    test_trial_type = np.zeros(n_cv, dtype=object)

    train_selected_frames = np.zeros(n_cv, dtype=object)
    test_selected_frames = np.zeros(n_cv, dtype=object)    



    for cv_ind in range(n_cv):

        train_video_path[cv_ind] = []
        test_video_path[cv_ind] = []

        train_sess_inds[cv_ind] = []
        test_sess_inds[cv_ind] = []

        train_trial_type[cv_ind] = []
        test_trial_type[cv_ind] = []

        train_selected_frames[cv_ind] = []
        test_selected_frames[cv_ind] = []



    for cv_ind in range(n_cv):

        for f1, f2 in f1_f2_list:
            # f1 = 'BAYLORGC[#]'
            # f2 = '[yyyy]_[mm]_[dd]'

            for i in n_trial_types_list:

                # Train set
                train_video_path[cv_ind].extend(sess_wise_train_video_path_dict[cv_ind][f1][f2][i])
                train_sess_inds[cv_ind].extend(sess_wise_train_sess_inds_dict[cv_ind][f1][f2][i])
                train_trial_type[cv_ind].extend(sess_wise_train_trial_type_dict[cv_ind][f1][f2][i])
                train_selected_frames[cv_ind].extend(sess_wise_train_selected_frames_dict[cv_ind][f1][f2][i])


                # Test set
                test_video_path[cv_ind].extend(sess_wise_test_video_path_dict[cv_ind][f1][f2][i])
                test_sess_inds[cv_ind].extend(sess_wise_test_sess_inds_dict[cv_ind][f1][f2][i])
                test_trial_type[cv_ind].extend(sess_wise_test_trial_type_dict[cv_ind][f1][f2][i])
                test_selected_frames[cv_ind].extend(sess_wise_test_selected_frames_dict[cv_ind][f1][f2][i])



    # video_path_dict = nestdict_to_dict(video_path_dict)
    # target_dict = nestdict_to_dict(target_dict)
    # selected_frames_dict = nestdict_to_dict(selected_frames_dict)
    # sess_inds_dict = nestdict_to_dict(sess_inds_dict)



        
    '''
    Notes on transform:
    1. configs['image_shape'] is set to (height, width) = [86, 130], so that it has roughly the same aspect ratio as the cropped image, which has
    (height, width) = (266, 400).
    
    2. ToTensor normalizes the range 0~255 to 0~1.
    3. I am not normalizing the input by mean and std, just like Aiden, I believe.
    '''

    if len(configs['view_type']) == 1:
        transform_list = [transforms.Compose([transforms.Grayscale(),
                                        CropOutPoles(),
                                        transforms.Resize(configs['image_shape']),
                                        transforms.ToTensor()])]

    else:
        transform_side = transforms.Compose([transforms.Grayscale(),
                                        CropOutPoles(),
                                        transforms.Resize(configs['image_shape_side']),
                                        transforms.ToTensor()])

        transform_bottom = transforms.Compose([transforms.Grayscale(),
                                        transforms.Resize(configs['image_shape_bottom']),
                                        transforms.ToTensor()])

        # Assume that side always comes before bottom in view_type.
        transform_list = [transform_side, transform_bottom]



    # Iterate over cv_ind
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

        train_test_helper_cv(f1_f2_list, cv_ind, configs, video_root_path, n_trial_types_list,\
        train_video_path[cv_ind], train_sess_inds[cv_ind], train_trial_type[cv_ind], train_selected_frames[cv_ind],\
        test_video_path[cv_ind], test_sess_inds[cv_ind], test_trial_type[cv_ind], test_selected_frames[cv_ind],\
        sess_wise_test_video_path_dict[cv_ind], \
        sess_wise_test_sess_inds_dict[cv_ind], sess_wise_test_trial_type_dict[cv_ind], sess_wise_test_selected_frames_dict[cv_ind],\
        transform_list, params, device)

        # '''
        # Check if all trials have both bottom and side view videos.
        # '''
        # train_test_helper_cv_dry_run(f1_f2_list, cv_ind, configs, video_root_path, n_trial_types_list,\
        # train_video_path[cv_ind], train_sess_inds[cv_ind], train_trial_type[cv_ind], train_selected_frames[cv_ind],\
        # test_video_path[cv_ind], test_sess_inds[cv_ind], test_trial_type[cv_ind], test_selected_frames[cv_ind],\
        # sess_wise_test_video_path_dict[cv_ind], \
        # sess_wise_test_sess_inds_dict[cv_ind], sess_wise_test_trial_type_dict[cv_ind], sess_wise_test_selected_frames_dict[cv_ind],\
        # transform_list, params, device)



def train_test_helper_cv_dry_run(f1_f2_list, cv_ind, configs, video_root_path, n_trial_types_list,\
    train_video_path, train_sess_inds, train_trial_type, train_selected_frames,\
    test_video_path, test_sess_inds, test_trial_type, test_selected_frames,\
    sess_wise_test_video_path_dict, \
    sess_wise_test_sess_inds_dict, sess_wise_test_trial_type_dict, sess_wise_test_selected_frames_dict,\
    transform_list, params, device):


    train_set = MyDatasetChoicePredVariableFrames(video_root_path, train_video_path, train_sess_inds, train_trial_type, train_selected_frames, \
        configs['view_type'], transform_list=transform_list, img_type=configs['img_type'])
    valid_set = MyDatasetChoicePredVariableFrames(video_root_path, test_video_path, test_sess_inds, test_trial_type, test_selected_frames, \
        configs['view_type'], transform_list=transform_list, img_type=configs['img_type'])


    train_loader = data.DataLoader(train_set, **params, drop_last=True)
    valid_loader = data.DataLoader(valid_set, **params)


    # We also want to evaluate r2 separately for each session.
    sess_wise_valid_loader_dict = nestdict()

    for f1, f2 in f1_f2_list:
        # f1 = 'BAYLORGC[#]'
        # f2 = '[yyyy]_[mm]_[dd]'
        cur_test_video_path = []
        cur_test_sess_inds = []
        cur_test_trial_type = []
        cur_test_selected_frames = []        
        
        for i in n_trial_types_list:

            cur_test_video_path.extend(sess_wise_test_video_path_dict[f1][f2][i])
            cur_test_sess_inds.extend(sess_wise_test_sess_inds_dict[f1][f2][i])
            cur_test_trial_type.extend(sess_wise_test_trial_type_dict[f1][f2][i])
            cur_test_selected_frames.extend(sess_wise_test_selected_frames_dict[f1][f2][i])

        cur_valid_set = MyDatasetChoicePredVariableFrames(video_root_path, cur_test_video_path,\
            cur_test_sess_inds, cur_test_trial_type, cur_test_selected_frames, configs['view_type'], \
            transform_list=transform_list, img_type=configs['img_type'])

        sess_wise_valid_loader_dict[f1][f2] = data.DataLoader(cur_valid_set, **params)



    for key in sorted(configs.keys()):
        print('{}: {}'.format(key, configs[key]))
    print('')
    print('')


    train_choice_pred_dry_run(configs, train_loader)
    validation_choice_pred_dry_run(configs, valid_loader)







def train_test_helper_cv(f1_f2_list, cv_ind, configs, video_root_path, n_trial_types_list,\
    train_video_path, train_sess_inds, train_trial_type, train_selected_frames,\
    test_video_path, test_sess_inds, test_trial_type, test_selected_frames,\
    sess_wise_test_video_path_dict, \
    sess_wise_test_sess_inds_dict, sess_wise_test_trial_type_dict, sess_wise_test_selected_frames_dict,\
    transform_list, params, device):
    
    train_set = MyDatasetChoicePredVariableFrames(video_root_path, train_video_path, train_sess_inds, train_trial_type, train_selected_frames, \
        configs['view_type'], transform_list=transform_list, img_type=configs['img_type'])
    valid_set = MyDatasetChoicePredVariableFrames(video_root_path, test_video_path, test_sess_inds, test_trial_type, test_selected_frames, \
        configs['view_type'], transform_list=transform_list, img_type=configs['img_type'])

    '''
    Use WeightedRandomSampler.
    '''
    # train set
    temp = np.array(train_set.trial_type_labels)
    n_neg_samples = len(np.nonzero(temp==0)[0])
    n_pos_samples = len(np.nonzero(temp==1)[0])
    n_total_samples = len(temp)
    train_class_prob = [n_neg_samples/n_total_samples, n_pos_samples/n_total_samples]
    
    train_weights = torch.zeros(len(train_set))
    for idx in range(len(train_set)):
        train_weights[idx] = 1/train_class_prob[temp[idx]]

    train_sampler = data.WeightedRandomSampler(train_weights, len(train_set))

    # validation set
    temp = np.array(valid_set.trial_type_labels)
    n_neg_samples = len(np.nonzero(temp==0)[0])
    n_pos_samples = len(np.nonzero(temp==1)[0])
    n_total_samples = len(temp)
    valid_class_prob = [n_neg_samples/n_total_samples, n_pos_samples/n_total_samples]
    
    valid_weights = torch.zeros(len(valid_set))
    for idx in range(len(valid_set)):
        valid_weights[idx] = 1/valid_class_prob[temp[idx]]

    valid_sampler = data.WeightedRandomSampler(valid_weights, len(valid_set))




    train_loader = data.DataLoader(train_set, **params, sampler=train_sampler, drop_last=True)
    valid_loader = data.DataLoader(valid_set, **params, sampler=valid_sampler)


    # We also want to evaluate r2 separately for each session.
    sess_wise_valid_loader_dict = nestdict()

    for f1, f2 in f1_f2_list:
        # f1 = 'BAYLORGC[#]'
        # f2 = '[yyyy]_[mm]_[dd]'
        cur_test_video_path = []
        cur_test_sess_inds = []
        cur_test_trial_type = []
        cur_test_selected_frames = []        
        
        for i in n_trial_types_list:

            cur_test_video_path.extend(sess_wise_test_video_path_dict[f1][f2][i])
            cur_test_sess_inds.extend(sess_wise_test_sess_inds_dict[f1][f2][i])
            cur_test_trial_type.extend(sess_wise_test_trial_type_dict[f1][f2][i])
            cur_test_selected_frames.extend(sess_wise_test_selected_frames_dict[f1][f2][i])

        cur_valid_set = MyDatasetChoicePredVariableFrames(video_root_path, cur_test_video_path,\
            cur_test_sess_inds, cur_test_trial_type, cur_test_selected_frames, configs['view_type'], \
            transform_list=transform_list, img_type=configs['img_type'])

        # validation set
        temp = np.array(cur_valid_set.trial_type_labels)
        n_neg_samples = len(np.nonzero(temp==0)[0])
        n_pos_samples = len(np.nonzero(temp==1)[0])
        n_total_samples = len(temp)
        valid_class_prob = [n_neg_samples/n_total_samples, n_pos_samples/n_total_samples]
        
        valid_weights = torch.zeros(len(cur_valid_set))
        for idx in range(len(cur_valid_set)):
            valid_weights[idx] = 1/valid_class_prob[temp[idx]]

        cur_valid_sampler = data.WeightedRandomSampler(valid_weights, len(cur_valid_set))

        sess_wise_valid_loader_dict[f1][f2] = data.DataLoader(cur_valid_set, **params, sampler=cur_valid_sampler)



    for key in sorted(configs.keys()):
        print('{}: {}'.format(key, configs[key]))
    print('')
    print('')


    import sys
    model = getattr(sys.modules[__name__], configs['model_name'])(configs).to(device)


    # Parallelize model to multiple GPUs
    if configs['use_cuda'] and torch.cuda.device_count() > 1:
        if configs['gpu_ids'] is None:
            print("Using", torch.cuda.device_count(), "GPUs!")
            print('')
            model = nn.DataParallel(model)
        else:
            print("Using", len(configs['gpu_ids']), "GPUs!")
            print('')
            model = nn.DataParallel(model, device_ids=configs['gpu_ids'])



    if 'l2_layers' in configs.keys() and len(configs['l2_layers']) != 0:
        # for e.g. configs['l2_layers'] = ['linear_list3'].
        skip_list = []
        for name, param in model.named_parameters():
            skip_list.append(name)
            for layer in configs['l2_layers']:
                if layer in name:
                    skip_list.pop()
                    break

        param_groups = add_weight_decay(model, configs['l2'], skip_list=skip_list)
        optimizer = torch.optim.Adam(param_groups, lr=configs['lr'])


    else:
        param_groups = add_weight_decay(model, configs['l2'], skip_list=[])
        optimizer = torch.optim.Adam(param_groups, lr=configs['lr'])


    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_train_aucs = []
    
    epoch_test_losses = []
    epoch_test_scores = []
    epoch_test_aucs = []

    epoch_test_scores_mean = []    

    cnn_channel_str = '_'.join([str(i) for i in configs['_cnn_channel_list']])

    if configs['img_type'] == 'jpg':
        img_type_str = ''
    elif configs['img_type'] == 'png':
        img_type_str = 'png_frames'

    if configs['neural_begin_time'] == -1.4 and configs['neural_end_time'] == -0.2 and configs['neural_skip_time'] == 0.2:
        neural_time_str = ''
    else:
        neural_time_str = 'begin_{:.1f}_end_{:.1f}_skip_{:.1f}'.format(configs['neural_begin_time'], configs['neural_end_time'], configs['neural_skip_time'])

    
    model_save_path = os.path.join(configs['models_cv_dir'], configs['pert_type'], 'random_seed_{}'.format(configs['random_seed']), 'inst_trial_type_{}'.format(configs['inst_trial_type']), configs['sess_type'], img_type_str, neural_time_str,
        'view_type_{}'.format('_'.join(configs['view_type'])), 'model_name_{}'.format(configs['model_name']), \
        'cnn_channel_list_{}'.format(cnn_channel_str), 'bs_{}_epochs_{}'.format(configs['bs'], configs['epochs']), 'n_cv_{}'.format(configs['n_cv']),  'cv_ind_{}'.format(cv_ind))

    logs_path = os.path.join(configs['logs_cv_dir'], configs['pert_type'], 'random_seed_{}'.format(configs['random_seed']), 'inst_trial_type_{}'.format(configs['inst_trial_type']), configs['sess_type'], img_type_str, neural_time_str, \
        'view_type_{}'.format('_'.join(configs['view_type'])), 'model_name_{}'.format(configs['model_name']),\
        'cnn_channel_list_{}'.format(cnn_channel_str), 'bs_{}_epochs_{}'.format(configs['bs'], configs['epochs']), 'n_cv_{}'.format(configs['n_cv']), 'cv_ind_{}'.format(cv_ind))


    os.makedirs(model_save_path, exist_ok=True)

    os.makedirs(logs_path, exist_ok=True)

    # sess_trial_type_wise_best_test_aucs = nestdict()

    # start training
    best_test_score_mean = float('-inf')
    best_thr = 0.0
    '''
    Use a weighted loss to account for the imblance between negative and positive examples.
    pos_weight = negative_train_examples/positive_train_examples.
    '''
    if configs['do_pos_weight']:
        temp = np.array(train_loader.dataset.trial_type_labels)
        n_neg_samples = len(np.nonzero(temp==0)[0])
        n_pos_samples = len(np.nonzero(temp==1)[0])
        pos_weight = n_neg_samples/n_pos_samples
        print('')
        print('n_neg_samples: ', n_neg_samples)
        print('n_pos_samples: ', n_pos_samples)
        print('pos_weight: {:.2f}'.format(pos_weight))

        loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    else:
        loss_fct = nn.BCEWithLogitsLoss()

    for epoch in range(configs['epochs']):
        # if epoch > 0:
        #     break
        # train, test model
        epoch_begin_time = time.time()

        train_losses, train_scores, train_aucs, cur_thr = train_choice_pred_manual_db(configs, model, device, train_loader, optimizer, epoch, loss_fct, return_auc=True)
        epoch_test_loss, epoch_test_score, epoch_test_auc = validation_choice_pred_manual_db(configs, model, device, valid_loader, best_test_score_mean, \
            loss_fct, model_save_path, cur_thr, return_auc=True)

        # If the model does not make a separate prediction for each time point, we don't want to take the mean.
        if isinstance(epoch_test_score, np.ndarray):
            epoch_test_score_mean = epoch_test_score.mean()
            epoch_test_auc_mean = epoch_test_auc.mean()
        else:
            epoch_test_score_mean = epoch_test_score
            epoch_test_auc_mean = epoch_test_auc

        if epoch_test_score_mean > best_test_score_mean:
            best_thr = cur_thr

        epoch_end_time = time.time()

        print('Epoch {} total time: {:.3f} s'.format(epoch+1, epoch_end_time - epoch_begin_time))
        print('')

        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_train_aucs.append(train_aucs)

        epoch_test_losses.append(epoch_test_loss)
        epoch_test_scores.append(epoch_test_score)
        epoch_test_aucs.append(epoch_test_auc)

        epoch_test_scores_mean.append(epoch_test_score_mean)        


        # determine best_test_auc
        best_test_score_mean = np.max(epoch_test_scores_mean)

        # save all train test results
        A = np.array(epoch_train_losses)
        B = np.array(epoch_train_scores)        
        C = np.array(epoch_train_aucs)
        D = np.array(epoch_test_losses)
        E = np.array(epoch_test_scores)
        F = np.array(epoch_test_aucs)
        np.save(os.path.join(logs_path, 'epoch_training_losses.npy'), A)
        np.save(os.path.join(logs_path, 'epoch_training_scores.npy'), B)
        np.save(os.path.join(logs_path, 'epoch_training_aucs.npy'), C)
        np.save(os.path.join(logs_path, 'epoch_test_losses.npy'), D)
        np.save(os.path.join(logs_path, 'epoch_test_scores.npy'), E)
        np.save(os.path.join(logs_path, 'epoch_test_aucs.npy'), F)



    for key in sorted(configs.keys()):
        print('{}: {}'.format(key, configs[key]))
    print('')
    print('')



    '''
    Test the best saved model. Copied from cd_reg_from_videos_test.py.
    '''
    print('')
    print('Test begins!')
    print('')



    import sys
    model = getattr(sys.modules[__name__], configs['model_name'])(configs).to(device)


    # Parallelize model to multiple GPUs
    if configs['use_cuda'] and torch.cuda.device_count() > 1:
        if configs['gpu_ids'] is None:
            print("Using", torch.cuda.device_count(), "GPUs!")
            print('')
            model = nn.DataParallel(model)
        else:
            print("Using", len(configs['gpu_ids']), "GPUs!")
            print('')
            model = nn.DataParallel(model, device_ids=configs['gpu_ids'])



    # Load the saved model.

    model.load_state_dict(torch.load(os.path.join(model_save_path,'best_model.pth')))


    if configs['do_pos_weight']:    
        temp = np.array(train_loader.dataset.trial_type_labels)
        n_neg_samples = len(np.nonzero(temp==0)[0])
        n_pos_samples = len(np.nonzero(temp==1)[0])
        pos_weight = n_neg_samples/n_pos_samples

        loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    else:
        loss_fct = nn.BCEWithLogitsLoss()



    # We also want to evaluate r2 separately for each session.
    sess_trial_type_wise_best_test_scores = nestdict()
    sess_trial_type_wise_best_test_aucs = nestdict()
    total_test_scores = []
    total_test_aucs = []

    for f1, f2 in f1_f2_list:
        # f1 = 'BAYLORGC[#]'
        # f2 = '[yyyy]_[mm]_[dd]'

        cur_valid_loader = sess_wise_valid_loader_dict[f1][f2]

        # best_test_score is set to float('inf') so that we don't save the model.
        print('Sess-specific validation results')
        print('f1: ', f1)
        print('f2: ', f2)

        # sess_trial_type_wise_best_test_scores[f1][f2] is an np array of shape (T)
        _, cur_sess_best_test_score, cur_sess_best_test_auc\
         = validation_choice_pred_manual_db(configs, model, device, cur_valid_loader, \
            float('inf'), loss_fct, model_save_path, best_thr, return_auc=True)

        if np.isnan(cur_sess_best_test_score).any():
            print('The current session does not have positive samples, so we skip it.')
            continue

        sess_trial_type_wise_best_test_scores[f1][f2] = cur_sess_best_test_score
        sess_trial_type_wise_best_test_aucs[f1][f2] = cur_sess_best_test_auc
         

        if len(cur_valid_loader.dataset) >= configs['n_test_samples_thr']:
            total_test_scores.append(sess_trial_type_wise_best_test_scores[f1][f2])
            total_test_aucs.append(sess_trial_type_wise_best_test_aucs[f1][f2])

    total_test_scores = np.array(total_test_scores) # (n_sess, T)
    total_test_aucs = np.array(total_test_aucs) # (n_sess, T)

    print('Session and time average accuracy (for sessions that has at least {} test samples): {:.2f}'.format(configs['n_test_samples_thr'], \
        total_test_scores.mean()))
    print('Session and time average roc_auc_auc (for sessions that has at least {} test samples): {:.2f}'.format(configs['n_test_samples_thr'], \
        total_test_aucs.mean()))


    if len(total_test_scores.shape) == 1:
        sess_avg_test_scores = total_test_scores.mean()
        sess_avg_test_aucs = total_test_aucs.mean()

        print('')
        print('')
        print('<Session averaged accuracy>')
        print('')
        print('{:.2f}'.format(sess_avg_test_scores))        
        print('')


        print('')
        print('')
        print('<Session averaged roc_auc_score>')
        print('')
        print('{:.2f}'.format(sess_avg_test_aucs))        
        print('')


    else:
        sess_avg_test_scores = total_test_scores.mean(0) # (T)
        sess_avg_test_aucs = total_test_aucs.mean(0) # (T)

        T = len(sess_avg_test_scores)

        print('')
        print('')
        print('<Session averaged accuracy for each time point>')
        print('')

        list_str = ', '.join(['t={}: {:.2f}'.format(t, sess_avg_test_scores[t]) for t in range(T)])
        print(list_str)
        
        print('')


        print('')
        print('')
        print('<Session averaged roc_auc_score for each time point>')
        print('')

        list_str = ', '.join(['t={}: {:.2f}'.format(t, sess_avg_test_aucs[t]) for t in range(T)])
        print(list_str)
        
        print('')

    sess_trial_type_wise_best_test_scores = nestdict_to_dict(sess_trial_type_wise_best_test_scores)
    sess_trial_type_wise_best_test_aucs = nestdict_to_dict(sess_trial_type_wise_best_test_aucs)

    # Save the test results.
    with open(os.path.join(logs_path, 'sess_trial_type_wise_best_test_scores.obj'), 'wb') as write_file:
        pickle.dump(sess_trial_type_wise_best_test_scores,write_file)


    with open(os.path.join(logs_path, 'sess_trial_type_wise_best_test_aucs.obj'), 'wb') as write_file:
        pickle.dump(sess_trial_type_wise_best_test_aucs,write_file)





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


# From https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/
def add_weight_decay(net, l2_value, skip_list=[]):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: 
            continue # frozen weights                
        
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list: 
            no_decay.append(param)
        
        else: 
            decay.append(param)
 
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]





if __name__ == '__main__':
    main()
