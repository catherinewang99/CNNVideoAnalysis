'''
Differences from all_neurons_reg_from_videos
1. We perform cross-validation.
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
from all_neurons_video_pred_functions import *
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import json
from collections import defaultdict

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

        return img.crop((0, int(h / 6) + 1, w, h))

    def __repr__(self):
        return self.__class__.__name__


# From https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/
def add_weight_decay(net, l2_value, skip_list=[]):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)

        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


def main():
    n_trial_types_list = list(range(2))

    # Load pred_configs
    with open('all_neurons_reg_configs.json', 'r') as read_file:
        configs = json.load(read_file)

    # Create directories to save results.
    os.makedirs(configs['logs_cv_dir'], exist_ok=True)
    os.makedirs(configs['models_cv_dir'], exist_ok=True)

    # Detect devices
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda:{}".format(configs['gpu_ids'][0]) if use_cuda else "cpu")  # use CPU or GPU

    # Data loading parameters
    params = {'batch_size': configs['bs'], 'shuffle': True, 'num_workers': configs['num_workers'],
              'pin_memory': bool(configs['pin_memory'])} if use_cuda else {}

    # Collect video filenames.
    if configs['img_type'] == 'jpg':
        video_root_path = os.path.join(configs['video_data_dir'], 'frames')
    elif configs['img_type'] == 'png':
        video_root_path = os.path.join(configs['video_data_dir'], 'png_frames')

    # sess_type: standard, full_reverse, rule_reverse
    # TODO: modify to fit alyse's data
    sess_type = configs['sess_type']
    if sess_type == 'standard':
        f1_list = list(filter(lambda x: int(x[8:]) <= 50, os.listdir(video_root_path)))
    elif sess_type == 'full_reverse':
        f1_list = list(filter(lambda x: int(x[8:]) >= 79, os.listdir(video_root_path)))
    elif sess_type == 'rule_reverse':
        f1_list = list(filter(lambda x: int(x[8:]) >= 61 and int(x[8:]) <= 71, os.listdir(video_root_path)))
    else:
        raise ValueError('invalid sess type.')

    target_path = os.path.join(configs['neural_data_dir'], configs['data_prefix'] + 'NeuronalData')

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
    Need to modify from here.
    Use no_stim_correct_i_good_trials.
    video_path_dict[f1][f2][i] will contain paths to videos for trials in no_stim_correct_i_good_trials[i].
    And we will separately split trials of each trial type into train and test set.
    '''

    video_path_dict = nestdict()
    for f1 in f1_list:
        # f1 = 'BAYLORGC[#]'
        if os.path.isfile(os.path.join(video_root_path, f1)):
            continue

        for f2 in os.listdir(os.path.join(video_root_path, f1)):
            # f2 = '[yyyy]_[mm]_[dd]'
            if os.path.isfile(os.path.join(video_root_path, f1, f2)):
                continue

            # We want to only select no stim trials.
            no_stim_correct_i_good_trials = np.load(
                os.path.join(target_path, f1, f2, 'no_stim_correct_i_good_trials.npy'), allow_pickle=True)

            for i in n_trial_types_list:
                # Just a sanity check to make sure i_good_trials are sorted.
                assert (no_stim_correct_i_good_trials[i] == sorted(no_stim_correct_i_good_trials[i])).all()

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
    configs['max_n_neurons'] = 0

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

            with open(os.path.join(target_path, f1, f2, 'all_neurons_{}.obj'.format(configs['fr_percentile'])),
                      'rb') as read_file:
                all_neurons = pickle.load(read_file)

            n_neurons_dict[f1][f2] = all_neurons[0].shape[2]

            if n_neurons_dict[f1][f2] > configs['max_n_neurons']:
                configs['max_n_neurons'] = n_neurons_dict[f1][f2]

    n_neurons_dict = nestdict_to_dict(n_neurons_dict)

    print('')
    print('max_n_neurons: {}'.format(configs['max_n_neurons']))
    print('')

    # Collect target.

    # Needed below to figure out begin and end_bin.
    f1 = list(video_path_dict.keys())[0]
    f2 = list(video_path_dict[f1].keys())[0]
    bin_centers = np.load(os.path.join(target_path, f1, f2, 'bin_centers.npy'))

    # These are used for cur_all_neurons.
    begin_bin = np.argmin(np.abs(bin_centers - configs['neural_begin_time']))
    end_bin = np.argmin(np.abs(bin_centers - configs['neural_end_time']))

    bin_stride = 0.1
    neural_skip_bin = int(configs['neural_skip_time'] // bin_stride)

    pred_times = bin_centers[begin_bin:end_bin + 1:neural_skip_bin]

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

            with open(os.path.join(target_path, f1, f2, 'all_neurons_{}.obj'.format(int(configs['fr_percentile']))),
                      'rb') as read_file:
                all_neurons = pickle.load(read_file)

            # Need to check if the trial numbers are equal between video and label data. If they are unequal, we use only trials present in the video data.
            no_stim_correct_i_good_trials = np.load(
                os.path.join(target_path, f1, f2, 'no_stim_correct_i_good_trials.npy'), allow_pickle=True)

            for i in n_trial_types_list:

                cur_all_neurons = all_neurons[i]

                # We transpose cur_all_neurons so that its shape is (n_trials, T, n_comp).
                cur_all_neurons = np.transpose(cur_all_neurons, (1, 0, 2))

                cur_all_neurons = cur_all_neurons[:, begin_bin:end_bin + 1:neural_skip_bin]

                # Pad the neuron dimension with nan.
                n_trials, T, n_neurons = cur_all_neurons.shape

                assert n_neurons <= configs['max_n_neurons']
                new_all_neurons = np.empty((n_trials, T, configs['max_n_neurons']))
                new_all_neurons.fill(np.nan)
                new_all_neurons[..., :n_neurons] = cur_all_neurons

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
    configs['n_sess'] = 0
    sess_inds_dict = nestdict()
    for f1 in sorted(target_dict.keys(), key=mouse_sort):

        for f2 in sorted(target_dict[f1].keys()):
            configs['n_sess'] += 1
            for i in n_trial_types_list:
                # n_sess -1 because sess_inds ranges from 0 to n_sess-1.
                sess_inds_dict[f1][f2][i] = [configs['n_sess'] - 1] * len(target_dict[f1][f2][i])

    sess_inds_dict = nestdict_to_dict(sess_inds_dict)

    # Determine configs["n_neurons"]
    configs["n_neurons"] = np.zeros(configs['n_sess']).astype(int)
    for f1 in sorted(target_dict.keys(), key=mouse_sort):
        for f2 in sorted(target_dict[f1].keys()):
            cur_sess_ind = sess_inds_dict[f1][f2][0][0]
            configs["n_neurons"][cur_sess_ind] = n_neurons_dict[f1][f2]

    # We also want to evaluate r2 separately for each session.
    n_cv = configs['n_cv']
    train_video_path = np.zeros(n_cv, dtype=object)
    test_video_path = np.zeros(n_cv, dtype=object)

    train_target = np.zeros(n_cv, dtype=object)
    test_target = np.zeros(n_cv, dtype=object)

    train_sess_inds = np.zeros(n_cv, dtype=object)
    test_sess_inds = np.zeros(n_cv, dtype=object)

    # We also keep track of trial type for certain models that use this information (e.g. used for temporal smoothing).
    train_trial_type = np.zeros(n_cv, dtype=object)
    test_trial_type = np.zeros(n_cv, dtype=object)

    for cv_ind in range(n_cv):
        train_video_path[cv_ind] = []
        test_video_path[cv_ind] = []

        train_target[cv_ind] = []
        test_target[cv_ind] = []

        train_sess_inds[cv_ind] = []
        test_sess_inds[cv_ind] = []

        train_trial_type[cv_ind] = []
        test_trial_type[cv_ind] = []

    sess_wise_test_video_path_dict = nestdict()
    sess_wise_test_target_dict = nestdict()
    sess_wise_test_sess_inds_dict = nestdict()
    sess_wise_test_trial_type_dict = nestdict()

    for f1 in f1_list:
        # f1 = 'BAYLORGC[#]'
        for f2 in os.listdir(os.path.join(video_root_path, f1)):
            # f2 = '[yyyy]_[mm]_[dd]'

            for i in n_trial_types_list:

                # We wrap lists into arrays so that they can be indexed with advanced indexing.
                cur_video_path = np.array([os.path.join(f1, f2, f3) for f3 in
                                           video_path_dict[f1][f2][i]])  # an array of each trial's video path
                cur_target = target_dict[f1][f2][
                    i]  # an array of shape (n_trials, T, max_n_neurons). Some entries filled with np.nan if n_neurons < max_n_neurons.

                cur_sess_inds = np.array(sess_inds_dict[f1][f2][i])  # an array [sess_ind]*n_trials.
                cur_trial_type = np.array([i] * len(cur_sess_inds))  # an array [i]*n_trials.

                kf = KFold(n_splits=n_cv, shuffle=True, random_state=1)

                for cv_ind, (train_inds, test_inds) in enumerate(kf.split(cur_target)):
                    cur_train_video_path, cur_test_video_path = cur_video_path[train_inds], cur_video_path[test_inds]
                    cur_train_target, cur_test_target = cur_target[train_inds], cur_target[test_inds]
                    cur_train_sess_inds, cur_test_sess_inds = cur_sess_inds[train_inds], cur_sess_inds[test_inds]
                    cur_train_trial_type, cur_test_trial_type = cur_trial_type[train_inds], cur_trial_type[test_inds]

                    train_video_path[cv_ind].extend(cur_train_video_path)
                    test_video_path[cv_ind].extend(cur_test_video_path)

                    train_target[cv_ind].extend(cur_train_target)
                    test_target[cv_ind].extend(cur_test_target)

                    train_sess_inds[cv_ind].extend(cur_train_sess_inds)
                    test_sess_inds[cv_ind].extend(cur_test_sess_inds)

                    train_trial_type[cv_ind].extend(cur_train_trial_type)
                    test_trial_type[cv_ind].extend(cur_test_trial_type)

                    sess_wise_test_video_path_dict[cv_ind][f1][f2][i] = cur_test_video_path
                    sess_wise_test_target_dict[cv_ind][f1][f2][i] = cur_test_target
                    sess_wise_test_sess_inds_dict[cv_ind][f1][f2][i] = cur_test_sess_inds
                    sess_wise_test_trial_type_dict[cv_ind][f1][f2][i] = cur_test_trial_type

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

    # selected_frames include both begin and end_frame.

    n_frames = int(np.rint((configs['end_frame'] - configs['begin_frame']) / configs['skip_frame'])) + 1
    selected_frames = (configs['begin_frame'] + configs['skip_frame'] * np.arange(n_frames))

    # Determine configs['pred_timesteps']
    selected_frame_times = selected_frames / 1000 * 5 - 3.6  # 1000 frames for 5s. The go cue is at 3.6s in video while at 0s in neural data.

    pred_timesteps = []
    for pred_time in pred_times:
        pred_timesteps.append(np.argmin(np.abs(selected_frame_times - pred_time)))

    configs['pred_timesteps'] = np.array(pred_timesteps)  # This is an array of T dim indices, not the real time in sec.

    # To deal with lrcn_neureg_bn_static2 etc.
    if '_input_channel' in configs.keys() and configs['_input_channel'] != 1:
        n_input_frames = configs['_input_channel']

        n_pre_frames = (n_input_frames - 1) // 2
        n_post_frames = (n_input_frames - 1) // 2

        if (n_input_frames - 1) % 2 != 0:
            n_pre_frames += 1

        # out of range for early times.
        if (configs['pred_timesteps'][0] - n_pre_frames) < 0:
            new_begin_frame = configs['begin_frame'] - configs['skip_frame'] * n_pre_frames
        else:
            new_begin_frame = configs['begin_frame']

        # out of range for late times.
        if (configs['pred_timesteps'][-1] + n_post_frames) >= n_frames:
            new_end_frame = configs['end_frame'] + configs['skip_frame'] * n_post_frames
        else:
            new_end_frame = configs['end_frame']

        n_frames = int(np.rint((new_end_frame - new_begin_frame) / configs['skip_frame'])) + 1
        selected_frames = (new_begin_frame + configs['skip_frame'] * np.arange(n_frames))

        # Determine configs['pred_timesteps']
        selected_frame_times = selected_frames / 1000 * 5 - 3.6

        pred_timesteps = []
        for pred_time in pred_times:
            pred_timesteps.append(np.argmin(np.abs(selected_frame_times - pred_time)))

        configs['pred_timesteps'] = np.array(pred_timesteps)

    # Iterate over cv_ind
    for cv_ind in range(n_cv):
        print('')
        print('')
        print('')
        print('------------------------------')
        print('------------------------------')
        print('{}-th Cross-Validation Begins!'.format(cv_ind + 1))
        print('------------------------------')
        print('------------------------------')
        print('')
        print('')
        print('')

        train_test_helper_cv(f1_list, cv_ind, configs, video_root_path, n_trial_types_list, \
                             train_video_path[cv_ind], train_target[cv_ind], train_sess_inds[cv_ind],
                             train_trial_type[cv_ind], \
                             test_video_path[cv_ind], test_target[cv_ind], test_sess_inds[cv_ind],
                             test_trial_type[cv_ind], \
                             sess_wise_test_video_path_dict[cv_ind], sess_wise_test_target_dict[cv_ind],
                             sess_wise_test_sess_inds_dict[cv_ind], sess_wise_test_trial_type_dict[cv_ind], \
                             selected_frames, transform_list, params, mouse_sort, video_path_dict, target_dict,
                             sess_inds_dict, device)


def train_test_helper_cv(f1_list, cv_ind, configs, video_root_path, n_trial_types_list, \
                         train_video_path, train_target, train_sess_inds, train_trial_type, \
                         test_video_path, test_target, test_sess_inds, test_trial_type, \
                         sess_wise_test_video_path_dict, sess_wise_test_target_dict, sess_wise_test_sess_inds_dict,
                         sess_wise_test_trial_type_dict, \
                         selected_frames, transform_list, params, mouse_sort, video_path_dict, target_dict,
                         sess_inds_dict, device):
    train_set = MyDatasetNeuReg(video_root_path, train_video_path, train_target, train_sess_inds, train_trial_type,
                                selected_frames, \
                                configs['view_type'], transform_list=transform_list, img_type=configs['img_type'])
    valid_set = MyDatasetNeuReg(video_root_path, test_video_path, test_target, test_sess_inds, test_trial_type,
                                selected_frames, \
                                configs['view_type'], transform_list=transform_list, img_type=configs['img_type'])

    train_loader = data.DataLoader(train_set, **params, drop_last=True)
    valid_loader = data.DataLoader(valid_set, **params)

    # We also want to evaluate r2 separately for each session.
    sess_wise_valid_loader_dict = nestdict()

    for f1 in f1_list:
        # f1 = 'BAYLORGC[#]'

        for f2 in os.listdir(os.path.join(video_root_path, f1)):
            # f2 = '[yyyy]_[mm]_[dd]'
            for i in n_trial_types_list:
                cur_test_video_path = sess_wise_test_video_path_dict[f1][f2][i]
                cur_test_target = sess_wise_test_target_dict[f1][f2][i]
                cur_test_sess_inds = sess_wise_test_sess_inds_dict[f1][f2][i]
                cur_test_trial_type = sess_wise_test_trial_type_dict[f1][f2][i]

                cur_valid_set = MyDatasetNeuReg(video_root_path, cur_test_video_path, cur_test_target, \
                                                cur_test_sess_inds, cur_test_trial_type, selected_frames,
                                                configs['view_type'], \
                                                transform_list=transform_list, img_type=configs['img_type'])

                sess_wise_valid_loader_dict[f1][f2][i] = data.DataLoader(cur_valid_set, **params)

    # Compute x_avg_dict and v_std_dict.
    if 't_smooth_loss' in configs.keys() and bool(configs['t_smooth_loss']):
        x_avg_dict = nestdict()
        v_std_dict = nestdict()

        sess_ind = 0
        for f1 in sorted(target_dict.keys(), key=mouse_sort):

            for f2 in sorted(target_dict[f1].keys()):

                for i in n_trial_types_list:
                    cur_target = target_dict[f1][f2][i]
                    x_avg_dict[sess_ind][i] = cur_target.mean(0)

                    cur_v = cur_target[:, 1:] - cur_target[:, :-1]

                    v_std_dict[sess_ind][i] = cur_v.std(0, ddof=1)

                sess_ind += 1

        x_avg_dict = nestdict_to_dict(x_avg_dict)
        v_std_dict = nestdict_to_dict(v_std_dict)
    else:
        x_avg_dict = None
        v_std_dict = None

    for key in sorted(configs.keys()):
        print('{}: {}'.format(key, configs[key]))
    print('')
    print('')

    import sys
    model = getattr(sys.modules[__name__], configs['model_name'])(configs).to(device)

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
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
    epoch_test_losses = []
    epoch_test_scores = []

    if configs['img_type'] == 'jpg':
        img_type_str = ''
    elif configs['img_type'] == 'png':
        img_type_str = 'png_frames'

    if configs['neural_begin_time'] == -1.4 and configs['neural_end_time'] == -0.2 and configs[
        'neural_skip_time'] == 0.2:
        neural_time_str = ''
    else:
        neural_time_str = 'begin_{:.1f}_end_{:.1f}_skip_{:.1f}'.format(configs['neural_begin_time'],
                                                                       configs['neural_end_time'],
                                                                       configs['neural_skip_time'])

    cnn_channel_str = '_'.join([str(i) for i in configs['_cnn_channel_list']])

    fc_layer_str = '_'.join([str(configs['_ln1_out']), str(configs['_ln2_out']), str(configs['_lstmHidden'])])

    model_save_path = os.path.join(configs['models_cv_dir'], configs['sess_type'],
                                   'fr_percentile_{}'.format(int(configs['fr_percentile'])), \
                                   img_type_str, neural_time_str, 'view_type_{}'.format(configs['view_type'][0]),
                                   'model_name_{}'.format(configs['model_name']), \
                                   'cnn_channel_list_{}'.format(cnn_channel_str), 'fc_layer_{}'.format(fc_layer_str),
                                   'cv_ind_{}'.format(cv_ind))

    logs_path = os.path.join(configs['logs_cv_dir'], configs['sess_type'],
                             'fr_percentile_{}'.format(int(configs['fr_percentile'])), \
                             img_type_str, neural_time_str, 'view_type_{}'.format(configs['view_type'][0]),
                             'model_name_{}'.format(configs['model_name']), \
                             'cnn_channel_list_{}'.format(cnn_channel_str), 'fc_layer_{}'.format(fc_layer_str),
                             'cv_ind_{}'.format(cv_ind))

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)

    # start training
    best_test_score = float('-inf')

    loss_fct = nn.MSELoss()

    for epoch in range(configs['epochs']):
        # train, test model
        epoch_begin_time = time.time()

        train_losses = train(configs, model, device, train_loader, optimizer, epoch, loss_fct)
        # best_model is saved within validation.
        epoch_test_loss, epoch_test_score = validation(configs, model, device, valid_loader, best_test_score, loss_fct,
                                                       model_save_path)

        # We also want to evaluate r2 separately for each session.
        if len(epoch_test_scores) != 0:
            if epoch_test_score > np.max(epoch_test_scores):
                f1 = f1_list[0]
                f2 = os.listdir(os.path.join(video_root_path, f1))[0]
                i = 0

                cur_valid_loader = sess_wise_valid_loader_dict[f1][f2][i]

                # best_test_score is set to float('inf') so that we don't save the model.
                print('Sess-specific validation results')
                print('f1: ', f1)
                print('f2: ', f2)
                print('i: ', i)
                _, _ = validation(configs, model, device, cur_valid_loader, \
                                  float('inf'), loss_fct)

        epoch_end_time = time.time()

        print('Epoch {} total time: {:.3f} s'.format(epoch + 1, epoch_end_time - epoch_begin_time))
        print('')

        # save results
        epoch_train_losses.append(train_losses)
        epoch_test_losses.append(epoch_test_loss)
        epoch_test_scores.append(epoch_test_score)

        # determine best_test_score
        best_test_score = np.max(epoch_test_scores)

        # save all train test results
        np.save(os.path.join(logs_path, 'epoch_train_losses.npy'), epoch_train_losses)
        np.save(os.path.join(logs_path, 'epoch_test_losses.npy'), epoch_test_losses)
        np.save(os.path.join(logs_path, 'epoch_test_scores.npy'), epoch_test_scores)

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
    if torch.cuda.device_count() > 1:
        if configs['gpu_ids'] is None:
            print("Using", torch.cuda.device_count(), "GPUs!")
            print('')
            model = nn.DataParallel(model)
        else:
            print("Using", len(configs['gpu_ids']), "GPUs!")
            print('')
            model = nn.DataParallel(model, device_ids=configs['gpu_ids'])

    # Load the saved model.

    model.load_state_dict(torch.load(os.path.join(model_save_path, 'best_model.pth')))

    loss_fct = nn.MSELoss()

    # We also want to evaluate r2 separately for each session.
    sess_trial_type_wise_best_test_scores = nestdict()
    total_test_scores = []

    for f1 in f1_list:
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
                _, sess_trial_type_wise_best_test_scores[f1][f2][i] = validation(configs, model, device,
                                                                                 cur_valid_loader, \
                                                                                 float('inf'), loss_fct)

                if len(cur_valid_loader.dataset) >= configs['n_test_samples_thr']:
                    total_test_scores.append(sess_trial_type_wise_best_test_scores[f1][f2][i])

    total_test_scores = np.array(total_test_scores)  # (n_sess*n_trial_types)

    print(
        'Total average trial_type_cond R2 (for sessions and trial types that has at least {} test samples): {:.2f}'.format(
            configs['n_test_samples_thr'], \
            np.array(total_test_scores).mean()))

    sess_trial_type_wise_best_test_scores = nestdict_to_dict(sess_trial_type_wise_best_test_scores)

    # Save the test results.
    with open(os.path.join(logs_path, 'sess_trial_type_wise_best_test_scores.obj'), 'wb') as write_file:
        pickle.dump(sess_trial_type_wise_best_test_scores, write_file)


if __name__ == '__main__':

    print('ehh')
    main()
