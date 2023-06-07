import os, time, math, sys
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.parameter import Parameter
import torch.nn.init as init

from sklearn.metrics import accuracy_score, r2_score

sys.path.append('NeuralPopulationAnalysisUtils/Code/Python')
from neupop import Session as Session
from neupop import PopulationAnalysis as PA
from neupop import MultiSessionAnalysis as MS
from neupop import decoders as ld
from neupop import session_parsers as parsers

from alm_datasets import ALMDataset

cat = np.concatenate


class MyDatasetNeuReg(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_path, folders, targets, sess_inds, trial_type_labels, frames, view_type_list,
                 transform_list=None, img_type='jpg'):
        "Initialization"
        self.data_path = data_path
        self.folders = folders
        self.targets = targets
        self.sess_inds = sess_inds
        self.trial_type_labels = trial_type_labels

        self.transform_list = transform_list
        self.frames = frames

        self.view_type_list = view_type_list

        self.img_type = img_type

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, view_type, use_transform):
        X = []
        frame_prefix = selected_folder.split('\\')[2] + '-{}-'.format(view_type)
        # print("FRAME: " + frame_prefix)
        for i in self.frames:
            # print("PATH " + '{:05d}.{}'.format(i, self.img_type))
            image = Image.open(os.path.join(path, selected_folder, frame_prefix + '{:05d}.{}'.format(i, self.img_type)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        y = torch.tensor(self.targets[index]).float()  # (T, n_comp)
        z = torch.tensor(self.sess_inds[index]).long()
        trial_type_label = torch.tensor(self.trial_type_labels[index]).long()

        # Load data
        if len(self.view_type_list) == 1:
            X = self.read_images(self.data_path, folder, self.view_type_list[0],
                                 self.transform_list[0])  # (seq, C, H, W)

            return X, y, z, trial_type_label


        else:
            X_side = self.read_images(self.data_path, folder, self.view_type_list[0],
                                      self.transform_list[0])  # (seq, C, H, W)
            X_bottom = self.read_images(self.data_path, folder, self.view_type_list[1],
                                        self.transform_list[1])  # (seq, C, H, W)

            # In general, X_side and X_bottom may have different h and w.

            return X_side, X_bottom, y, z, trial_type_label


class MyDatasetNeuRegRecon(data.Dataset):
    '''
    Difference from MyDatasetNeuReg:
    1. It returns the trial number indicated in the video path.
    '''

    def __init__(self, data_path, folders, targets, sess_inds, trial_type_labels, frames, view_type_list,
                 transform_list=None, img_type='jpg'):
        "Initialization"
        self.data_path = data_path
        self.folders = folders
        self.targets = targets
        self.sess_inds = sess_inds
        self.trial_type_labels = trial_type_labels

        self.transform_list = transform_list
        self.frames = frames

        self.view_type_list = view_type_list

        self.img_type = img_type

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, view_type, use_transform):
        X = []
        frame_prefix = selected_folder.split('\\')[2] + '-{}-'.format(view_type)
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, frame_prefix + '{:05d}.{}'.format(i, self.img_type)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # extract trial idx.
        # f3 = 'E3-BAYLORGC25-2018_09_17-13'
        f3 = folder.split('\\')[2]
        trial_idx = int(f3.split('-')[3])

        y = torch.tensor(self.targets[index]).float()  # (T, n_comp)
        z = torch.tensor(self.sess_inds[index]).long()
        trial_type_label = torch.tensor(self.trial_type_labels[index]).long()
        trial_idx = torch.tensor(trial_idx).long()

        # Load data
        if len(self.view_type_list) == 1:
            X = self.read_images(self.data_path, folder, self.view_type_list[0],
                                 self.transform_list[0])  # (seq, C, H, W)

            return X, y, z, trial_type_label, trial_idx


        else:
            X_side = self.read_images(self.data_path, folder, self.view_type_list[0],
                                      self.transform_list[0])  # (seq, C, H, W)
            X_bottom = self.read_images(self.data_path, folder, self.view_type_list[1],
                                        self.transform_list[1])  # (seq, C, H, W)

            # In general, X_side and X_bottom may have different h and w.

            return X_side, X_bottom, y, z, trial_type_label, trial_idx


## ---------------------- end of Dataset---------------------- ##


def train(configs, model, device, train_loader, optimizer, epoch, loss_fct):
    # set model as training mode
    model.train()

    losses = []
    N_count = 0  # counting total trained sample in one epoch
    begin_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        # distribute data to device
        if 'all_views' in configs['model_name']:
            X_side, X_bottom, y, sess_inds, trial_type = data
            X_side, X_bottom, y, sess_inds, trial_type = X_side.to(device), X_bottom.to(device), y.to(
                device), sess_inds.to(device), trial_type.to(device)

        else:
            X, y, sess_inds, trial_type = data
            X, y, sess_inds, trial_type = X.to(device), y.to(device), sess_inds.to(device), trial_type.to(device)

        N_count += y.size(0)

        optimizer.zero_grad()

        if 'all_views' in configs['model_name']:
            y_pred = model(X_side, X_bottom, sess_inds)

        else:
            y_pred = model(X, sess_inds)  # output has dim = (batch, number of classes)

        # We have to select elements that are not nan.
        nan_mask = torch.isnan(y)

        loss = loss_fct(y_pred[~nan_mask], y[~nan_mask])

        losses.append(loss.item())

        loss.backward()

        optimizer.step()

        # show information
        if (batch_idx + 1) % configs['log_interval'] == 0:

            # compute r2: for each session, compute r2 and average over sessions.
            r2_all_sess = []
            for sess_ind in torch.unique(sess_inds):
                sess_mask = (sess_inds == sess_ind)
                cur_y_pred = y_pred[sess_mask]
                cur_y = y[sess_mask]
                T = cur_y.size(1)

                if len(torch.nonzero(torch.isnan(cur_y[0, 0]))) != 0:
                    n_neurons = torch.nonzero(torch.isnan(cur_y[0, 0])).squeeze()[
                        0]  # The index along neuron dim where first nan occurs.
                else:
                    n_neurons = cur_y.shape[2]

                cur_y_pred = cur_y_pred[..., :n_neurons].contiguous().view(-1, T * n_neurons)
                cur_y = cur_y[..., :n_neurons].contiguous().view(-1, T * n_neurons)

                cur_r2 = r2_score(cur_y.cpu().data.numpy(), cur_y_pred.cpu().data.numpy(),
                                  multioutput='uniform_average')

                r2_all_sess.append(cur_r2)

            cur_time = time.time()
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, R2: {:.2f} ({:.3f} s)'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), \
                loss.item(), np.mean(r2_all_sess), cur_time - begin_time))
            begin_time = time.time()

    return losses


def validation(configs, model, device, test_loader, best_test_score, loss_fct, model_save_path=None):
    # set model as testing mode
    begin_time = time.time()
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    all_sess_inds = []
    with torch.no_grad():
        for data in test_loader:
            if 'all_views' in configs['model_name']:
                X_side, X_bottom, y, sess_inds, trial_type = data
                X_side, X_bottom, y, sess_inds, trial_type = X_side.to(device), X_bottom.to(device), y.to(
                    device), sess_inds.to(device), trial_type.to(device)

            else:
                X, y, sess_inds, trial_type = data
                X, y, sess_inds, trial_type = X.to(device), y.to(device), sess_inds.to(device), trial_type.to(device)

            if 'all_views' in configs['model_name']:
                y_pred = model(X_side, X_bottom, sess_inds)

            else:
                y_pred = model(X, sess_inds)  # y_pred has dim = (batch, number of classes)

            # We have to select elements that are not nan.
            nan_mask = torch.isnan(y)

            # Multiply by numel so that we are not dividing by numel.
            loss = loss_fct(y_pred[~nan_mask], y[~nan_mask])
            test_loss += loss.item()  # sum up batch loss

            # collect all y and y_pred in all batches
            all_y.append(y)
            all_y_pred.append(y_pred)
            all_sess_inds.append(sess_inds)

    test_loss /= len(test_loader)

    all_y = torch.cat(all_y, dim=0)
    all_y_pred = torch.cat(all_y_pred, dim=0)
    all_sess_inds = torch.cat(all_sess_inds, dim=0)

    # compute r2: for each session, compute r2 and average over sessions.
    r2_all_sess = []
    for sess_ind in torch.unique(all_sess_inds):
        sess_mask = (all_sess_inds == sess_ind)
        cur_y_pred = all_y_pred[sess_mask]
        cur_y = all_y[sess_mask]
        T = cur_y.size(1)

        if len(torch.nonzero(torch.isnan(cur_y[0, 0]))) != 0:
            n_neurons = torch.nonzero(torch.isnan(cur_y[0, 0])).squeeze()[
                0]  # The index along neuron dim where first nan occurs.
        else:
            n_neurons = cur_y.shape[2]

        cur_y_pred = cur_y_pred[..., :n_neurons].contiguous().view(-1, T * n_neurons)
        cur_y = cur_y[..., :n_neurons].contiguous().view(-1, T * n_neurons)

        cur_r2 = r2_score(cur_y.cpu().data.numpy(), cur_y_pred.cpu().data.numpy())
        r2_all_sess.append(cur_r2)

    test_score = np.mean(r2_all_sess)

    # show information
    cur_time = time.time()

    print('')
    print('Test set ({:d} samples): Average loss: {:.4f}, R2: {:.2f} ({:.3f} s)'.format(len(all_y), test_loss, \
                                                                                        test_score,
                                                                                        cur_time - begin_time))

    # save Pytorch models of best record
    if model_save_path is not None:
        if test_score > best_test_score:
            print('current test_score > best_test_score')
            os.makedirs(model_save_path, exist_ok=True)

            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))  # save model
            print("Current model saved!")
            print('')

    return test_loss, test_score


def neureg_recon(configs, model, device, test_loader, loss_fct):
    '''
    We assume that test_loader contains data from a single session.
    '''

    # set model as testing mode
    begin_time = time.time()
    model.eval()

    all_y = []
    all_y_pred = []
    all_sess_inds = []
    all_trial_idx = []
    with torch.no_grad():
        for data in test_loader:
            if 'all_views' in configs['model_name']:
                X_side, X_bottom, y, sess_inds, trial_type, trial_idx = data
                X_side, X_bottom, y, sess_inds, trial_type, trial_idx = \
                    X_side.to(device), X_bottom.to(device), y.to(device), sess_inds.to(device), trial_type.to(
                        device), trial_idx.to(device)

            else:
                X, y, sess_inds, trial_type, trial_idx = data
                X, y, sess_inds, trial_type, trial_idx = \
                    X.to(device), y.to(device), sess_inds.to(device), trial_type.to(device), trial_idx.to(device)

            if 'all_views' in configs['model_name']:
                y_pred = model(X_side, X_bottom, sess_inds)

            else:
                y_pred = model(X, sess_inds)  # y_pred has dim = (batch, number of classes)

            # collect all y and y_pred in all batches
            all_y.append(y)
            all_y_pred.append(y_pred)
            all_sess_inds.append(sess_inds)
            all_trial_idx.append(trial_idx)

    all_y = torch.cat(all_y, dim=0)
    all_y_pred = torch.cat(all_y_pred, dim=0)
    all_sess_inds = torch.cat(all_sess_inds, dim=0)
    all_trial_idx = torch.cat(all_trial_idx, dim=0)

    assert len(torch.unique(all_sess_inds)) == 1

    T = all_y.size(1)

    if len(torch.nonzero(torch.isnan(all_y[0, 0]))) != 0:
        n_neurons = torch.nonzero(torch.isnan(all_y[0, 0])).squeeze()[
            0]  # The index along neuron dim where first nan occurs.
    else:
        n_neurons = all_y.shape[2]

    all_y_pred = all_y_pred[..., :n_neurons]
    all_y = all_y[..., :n_neurons]

    all_r2 = r2_score(all_y.contiguous().view(-1, T * n_neurons).cpu().data.numpy(), \
                      all_y_pred.contiguous().view(-1, T * n_neurons).cpu().data.numpy(),
                      multioutput='raw_values').reshape(T, n_neurons)
    test_score = all_r2
    test_score_mean = all_r2.mean()

    # show information
    cur_time = time.time()

    print('')
    print('Test set ({:d} samples): R2: {:.2f} ({:.3f} s)'.format(len(all_y), \
                                                                  test_score_mean, cur_time - begin_time))

    return all_y.cpu().data.numpy(), all_y_pred.cpu().data.numpy(), all_trial_idx.cpu().data.numpy(), test_score


## ---------------------- end of train/val---------------------- ##


class cnn_bn(nn.Module):
    '''
    difference from cnn:
    1. Batch Norm.
    '''

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if not '_cnn_channel_list' in self.configs.keys():
            self.configs['_cnn_channel_list'] = [8, 8, 16, 16, 32, 32]
        if not '_maxPoolLayers' in self.configs.keys():
            self.configs["_maxPoolLayers"] = [2, 4]
        if not '_cnn_kernel' in self.configs.keys():
            self.configs["_cnn_kernel"] = 5
        if not '_input_channel' in self.configs.keys():
            self.configs["_input_channel"] = 1

        s = 1
        k = configs['_cnn_kernel']
        os = [self.configs['_input_channel']] + self.configs['_cnn_channel_list']
        self.conv_list = nn.ModuleList([
            nn.Conv2d(os[i], os[i + 1], k, stride=s, padding=(k - s) // 2) for i in
            range(len(self.configs['_cnn_channel_list']))
        ])
        self.bn_list = nn.ModuleList([
            nn.BatchNorm2d(os[i + 1]) for i in range(len(self.configs['_cnn_channel_list']))
        ])

    def forward(self, x):
        for i in range(len(self.configs['_cnn_channel_list'])):
            x = F.relu(self.conv_list[i](x))
            x = self.bn_list[i](x)
            if i in self.configs["_maxPoolLayers"]:
                x = F.max_pool2d(x, 2)
        return x


class lrcn_neureg_bn_static2_all_views_time_indp_dec(nn.Module):
    '''
    Differecne from lrcn_neureg_bn_static2_all_views:
    1. We don't have a separate linear decoder for each time point.
    '''

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 512

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 512
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 512

        self.n_input_frames = self.configs['_input_channel']

        # n_pre and n_post_frames = 4 covers +-0.2 s around the bin center (assuming that skip_frame = 10).
        n_pre_frames = (self.n_input_frames - 1) // 2
        n_post_frames = (self.n_input_frames - 1) // 2

        if (self.n_input_frames - 1) % 2 != 0:
            n_pre_frames += 1

        input_timesteps = []
        for k in self.configs['pred_timesteps']:
            input_timesteps.append(np.arange(k - n_pre_frames, k + n_post_frames + 1, 1))

        self.input_timesteps = cat(input_timesteps, 0)

        self.cnn_list = nn.ModuleList([cnn_bn(self.configs) for k in range(len(self.configs['view_type']))])

        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen_side = (self.configs["image_shape_side"][0] // pool) * (
                    self.configs["image_shape_side"][1] // pool) * self.configs["_cnn_channel_list"][-1]
        featureVlen_bottom = (self.configs["image_shape_bottom"][0] // pool) * (
                    self.configs["image_shape_bottom"][1] // pool) * self.configs["_cnn_channel_list"][-1]

        self.linear0 = nn.Linear(featureVlen_side + featureVlen_bottom, 2 * self.configs["_lstmHidden"])

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer

        self.linear_list2 = nn.ModuleList([
            nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer

        self.linear_list3 = nn.ModuleList([
            nn.Linear(self.configs["_ln2_out"], self.configs["n_neurons"][s]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer

    def forward(self, x_side, x_bottom, sess_inds):

        n_views = 2
        x_list = [x_side, x_bottom]

        seq_length = len(self.configs['pred_timesteps'])

        x_all_views_list = []
        for k in range(n_views):
            cur_x = x_list[k]
            bs, _, c, h, w = cur_x.size()

            x = cur_x[:, self.input_timesteps].view(-1, self.n_input_frames, h, w)
            x = self.cnn_list[k](x)
            x = x.view(bs, seq_length, -1)
            x = F.dropout(x, p=self.configs['dropout'], training=self.training)
            x_all_views_list.append(x)

        x = torch.cat(x_all_views_list, 2)  # (bs, seq_length, n_views*feaureVlen)

        x = F.relu(self.linear0(x))

        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['max_n_neurons']])

        outputs.fill_(float('nan'))

        for s in torch.unique(sess_inds):
            sess_mask = (sess_inds == s)
            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            x_sub2task = F.relu(self.linear_list2[s](x_sub1task))
            # We apply relu to the output layer, because the target is non-negative (we don't center firing rates).
            outputs[sess_mask, :, :self.configs["n_neurons"][s]] = F.relu(self.linear_list3[s](x_sub2task))

        return outputs


