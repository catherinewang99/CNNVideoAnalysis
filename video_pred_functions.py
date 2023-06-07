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

from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, roc_curve

sys.path.append('NeuralPopulationAnalysisUtils/Code/Python')
from neupop import Session as Session
from neupop import PopulationAnalysis as PA
from neupop import MultiSessionAnalysis as MS
from neupop import decoders as ld
from neupop import session_parsers as parsers

from alm_datasets import ALMDataset

import itertools

cat = np.concatenate


## ------------------- label conversion tools ------------------ ##
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)


def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()


def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()


def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()


# for CRNN
class MyDataset(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_path, folders, labels, frames, view_type, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

        # For now, I only use one view_type not both.
        assert len(view_type) == 1

        self.view_type = view_type[0]

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, view_type, use_transform):
        X = []
        frame_prefix = selected_folder.split('/')[2] + '-{}-'.format(view_type)
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, frame_prefix + '{:05d}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.view_type, self.transform)  # (input) spatial images

        if isinstance(self.labels[index], np.int64):
            y = torch.tensor(self.labels[index]).long()  # (labels) LongTensor are for int64 instead of FloatTensor
        else:
            y = torch.tensor(self.labels[index]).float()
        # print(X.shape)
        return X, y


class MyDatasetSessCond(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_path, folders, labels, sess_inds, frames, view_type, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.sess_inds = sess_inds

        self.transform = transform
        self.frames = frames

        # For now, I only use one view_type not both.
        assert len(view_type) == 1

        self.view_type = view_type[0]

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, view_type, use_transform):
        X = []
        frame_prefix = selected_folder.split('/')[2] + '-{}-'.format(view_type)
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, frame_prefix + '{:05d}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.view_type, self.transform)  # (input) spatial images

        if isinstance(self.labels[index], np.int64):
            y = torch.tensor(self.labels[index]).long()  # (labels) LongTensor are for int64 instead of FloatTensor
        else:
            y = torch.tensor(self.labels[index]).float()

        z = torch.tensor(self.sess_inds[index]).long()

        # print(X.shape)
        return X, y, z


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
        frame_prefix = selected_folder.split('/')[2] + '-{}-'.format(view_type)
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


class MyDatasetNeuRegVariableFrames(data.Dataset):
    '''
    Main differences from MyDatasetNeuReg:
    1. self.frames is not fixed for all samples, but depend on them. This is to allow for selecting different sets of frames across sessions
    (which I need to do because later sessions have 1200 frames as opposed to 1000 frames).
    '''

    def __init__(self, data_path, folders, targets, sess_inds, trial_type_labels, frames, view_type_list,
                 transform_list=None, \
                 img_type='jpg', mask_side=None):
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

        self.mask_side = mask_side  # (sess_ind, H, W)
        if self.mask_side is not None:
            self.mask_side = torch.tensor(self.mask_side).float()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, frames, path, selected_folder, view_type, use_transform):
        X = []
        # print("selected_folder: ")
        # print(path + selected_folder)
        #
        # print("view type")
        # print(view_type)
        #
        # print("franes")
        # print(frames)
        frame_prefix = selected_folder.split('\\')[2] + '-{}-'.format(view_type)
        for i in frames:
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
            X = self.read_images(self.frames[index], self.data_path, folder, self.view_type_list[0],
                                 self.transform_list[0])  # (seq, C, H, W)

            if self.mask_side is not None:
                X = X * self.mask_side[self.sess_inds[index]]

            return X, y, z, trial_type_label


        else:
            X_side = self.read_images(self.frames[index], self.data_path, folder, self.view_type_list[0],
                                      self.transform_list[0])  # (seq, C, H, W)
            X_bottom = self.read_images(self.frames[index], self.data_path, folder, self.view_type_list[1],
                                        self.transform_list[1])  # (seq, C, H, W)

            # In general, X_side and X_bottom may have different h and w.

            if self.mask_side is not None:
                X_side = X_side * self.mask_side[self.sess_inds[index]]

            return X_side, X_bottom, y, z, trial_type_label


class MyDatasetNeuRegVariableFramesRecon(data.Dataset):
    '''
    Main differences from MyDatasetNeuRegVariableFrames:
    1. We return trial idx in addition.
    '''

    def __init__(self, data_path, folders, targets, sess_inds, trial_type_labels, frames, trial_idxs, view_type_list, \
                 transform_list=None, img_type='jpg', mask_side=None, pert_type_labels=None):
        "Initialization"
        self.data_path = data_path
        self.folders = folders
        self.targets = targets
        self.sess_inds = sess_inds
        self.trial_type_labels = trial_type_labels

        self.transform_list = transform_list
        self.frames = frames

        self.trial_idxs = trial_idxs

        self.view_type_list = view_type_list

        self.img_type = img_type

        self.mask_side = mask_side  # (sess_ind, H, W)
        if self.mask_side is not None:
            self.mask_side = torch.tensor(self.mask_side).float()

        self.pert_type_labels = pert_type_labels

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, frames, path, selected_folder, view_type, use_transform):
        X = []
        frame_prefix = selected_folder.split('/')[2] + '-{}-'.format(view_type)
        for i in frames:
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
        trial_idx = self.trial_idxs[index]

        if self.pert_type_labels is not None:
            pert_type_label = torch.tensor(self.pert_type_labels[index]).long()

        # Load data
        if len(self.view_type_list) == 1:
            X = self.read_images(self.frames[index], self.data_path, folder, self.view_type_list[0],
                                 self.transform_list[0])  # (seq, C, H, W)

            if self.mask_side is not None:
                X = X * self.mask_side[self.sess_inds[index]]

            if self.pert_type_labels is not None:
                return X, y, z, trial_type_label, trial_idx, pert_type_label

            else:
                return X, y, z, trial_type_label, trial_idx


        else:
            X_side = self.read_images(self.frames[index], self.data_path, folder, self.view_type_list[0],
                                      self.transform_list[0])  # (seq, C, H, W)
            X_bottom = self.read_images(self.frames[index], self.data_path, folder, self.view_type_list[1],
                                        self.transform_list[1])  # (seq, C, H, W)

            # In general, X_side and X_bottom may have different h and w.

            if self.mask_side is not None:
                X_side = X_side * self.mask_side[self.sess_inds[index]]

            if self.pert_type_labels is not None:
                return X_side, X_bottom, y, z, trial_type_label, trial_idx, pert_type_label

            else:
                return X_side, X_bottom, y, z, trial_type_label, trial_idx


class MyDatasetChoicePredVariableFrames(data.Dataset):
    '''
    Main differences from MyDatasetNeuRegVariableFrames:
    1. Predict binary choice.
    '''

    def __init__(self, data_path, folders, sess_inds, trial_type_labels, frames, view_type_list, transform_list=None, \
                 img_type='jpg', mask_side=None):
        "Initialization"
        self.data_path = data_path
        self.folders = folders
        self.sess_inds = sess_inds
        self.trial_type_labels = trial_type_labels

        self.transform_list = transform_list
        self.frames = frames

        self.view_type_list = view_type_list

        self.img_type = img_type

        self.mask_side = mask_side  # (sess_ind, H, W)
        if self.mask_side is not None:
            self.mask_side = torch.tensor(self.mask_side).float()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, frames, path, selected_folder, view_type, use_transform):
        X = []
        frame_prefix = selected_folder.split('\\')[2] + '-{}-'.format(view_type)
        for i in frames:
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

        z = torch.tensor(self.sess_inds[index]).long()
        trial_type_label = torch.tensor(self.trial_type_labels[index]).long()

        # Load data
        if len(self.view_type_list) == 1:
            X = self.read_images(self.frames[index], self.data_path, folder, self.view_type_list[0],
                                 self.transform_list[0])  # (seq, C, H, W)

            if self.mask_side is not None:
                X = X * self.mask_side[self.sess_inds[index]]

            return X, trial_type_label, z


        else:
            X_side = self.read_images(self.frames[index], self.data_path, folder, self.view_type_list[0],
                                      self.transform_list[0])  # (seq, C, H, W)
            X_bottom = self.read_images(self.frames[index], self.data_path, folder, self.view_type_list[1],
                                        self.transform_list[1])  # (seq, C, H, W)

            # In general, X_side and X_bottom may have different h and w.

            if self.mask_side is not None:
                X_side = X_side * self.mask_side[self.sess_inds[index]]

            return X_side, X_bottom, trial_type_label, z


class MyDatasetChoicePredVariableFramesRecon(data.Dataset):
    '''
    Main differences from MyDatasetNeuRegVariableFrames:
    1. Predict binary choice.
    '''

    def __init__(self, data_path, folders, sess_inds, trial_type_labels, frames, trial_idxs, view_type_list,
                 transform_list=None, \
                 img_type='jpg', mask_side=None):
        "Initialization"
        self.data_path = data_path
        self.folders = folders
        self.sess_inds = sess_inds
        self.trial_type_labels = trial_type_labels

        self.transform_list = transform_list
        self.frames = frames

        self.trial_idxs = trial_idxs

        self.view_type_list = view_type_list

        self.img_type = img_type

        self.mask_side = mask_side  # (sess_ind, H, W)
        if self.mask_side is not None:
            self.mask_side = torch.tensor(self.mask_side).float()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, frames, path, selected_folder, view_type, use_transform):
        X = []
        frame_prefix = selected_folder.split('\\')[2] + '-{}-'.format(view_type)
        for i in frames:
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

        z = torch.tensor(self.sess_inds[index]).long()
        trial_type_label = torch.tensor(self.trial_type_labels[index]).long()
        trial_idx = self.trial_idxs[index]

        # Load data
        if len(self.view_type_list) == 1:
            X = self.read_images(self.frames[index], self.data_path, folder, self.view_type_list[0],
                                 self.transform_list[0])  # (seq, C, H, W)

            if self.mask_side is not None:
                X = X * self.mask_side[self.sess_inds[index]]

            return X, trial_type_label, z, trial_idx


        else:
            X_side = self.read_images(self.frames[index], self.data_path, folder, self.view_type_list[0],
                                      self.transform_list[0])  # (seq, C, H, W)
            X_bottom = self.read_images(self.frames[index], self.data_path, folder, self.view_type_list[1],
                                        self.transform_list[1])  # (seq, C, H, W)

            # In general, X_side and X_bottom may have different h and w.

            if self.mask_side is not None:
                X_side = X_side * self.mask_side[self.sess_inds[index]]

            return X_side, X_bottom, trial_type_label, z, trial_idx


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
        frame_prefix = selected_folder.split('/')[2] + '-{}-'.format(view_type)
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
        f3 = folder.split('/')[2]
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


## ---------------------- end of Dataloaders ---------------------- ##


def train(configs, model, device, train_loader, optimizer, epoch, loss_fct, task_type, x_avg_dict=None, v_std_dict=None,
          cd_trial_avg=None):
    # set model as training mode
    model.train()

    if 't_smooth_loss' in configs.keys() and bool(configs['t_smooth_loss']):
        assert x_avg_dict is not None
        assert v_std_dict is not None
        t_smooth_loss_fct = t_smooth_loss(configs, x_avg_dict, v_std_dict)

    if 'cd_cor_loss' in configs.keys() and bool(configs['cd_cor_loss']):
        assert cd_trial_avg is not None
        cd_cor_loss_fct = cd_cor_mse_loss(configs, cd_trial_avg).to(device)

    losses = []
    scores = []
    N_count = 0  # counting total trained sample in one epoch
    begin_time = time.time()

    # CW: train_loader contains data to be trained
    for batch_idx, data in enumerate(train_loader):
        # distribute data to device
        if 'sess_cond' in configs['model_name'] or 'neureg' in configs['model_name'] or 'reg' in configs['model_name']:
            if 'all_views' in configs['model_name']:
                X_side, X_bottom, y, z, trial_type = data
                X_side, X_bottom, y, z, trial_type = X_side.to(device), X_bottom.to(device), y.to(device), z.to(
                    device), trial_type.to(device)

            else:
                X, y, z, trial_type = data
                X, y, z, trial_type = X.to(device), y.to(device), z.to(device), trial_type.to(device)

        else:
            X, y = data
            X, y = X.to(device), y.to(device)

        N_count += y.size(0)

        optimizer.zero_grad()

        if 'sess_cond' in configs['model_name'] or 'neureg' in configs['model_name'] or 'reg' in configs['model_name']:
            if 'all_views' in configs['model_name']:
                output = model(X_side, X_bottom, z)

            else:
                output = model(X, z)  # output has dim = (batch, number of classes)

        else:
            output = model(X)  # output has dim = (batch, number of classes)

        if task_type == 'cls':
            y = y.squeeze()

        loss = loss_fct(output, y)

        losses.append(loss.item())

        if 't_smooth_loss' in configs.keys() and bool(configs['t_smooth_loss']):
            v_penalty = t_smooth_loss_fct(output, z, trial_type)
            (loss + v_penalty).backward()

        elif 'cd_cor_loss' in configs.keys() and bool(configs['cd_cor_loss']):
            cd_cor_loss = cd_cor_loss_fct(output, y, z)
            (loss + cd_cor_loss).backward()

        else:
            loss.backward()

        optimizer.step()

        if task_type == 'cls':
            # to compute accuracy
            y_pred = torch.max(output, 1)[1]  # y_pred != output
            step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
            scores.append(step_score)  # computed on CPU

            # show information
            if (batch_idx + 1) % configs['log_interval'] == 0:
                cur_time = time.time()
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, Accu: {:.2f}% ({:.3f} s)'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), \
                    loss.item(), 100 * step_score, cur_time - begin_time))
                begin_time = time.time()

        elif task_type == 'reg':
            # to compute accuracy
            y_pred = output
            r2 = r2_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
            scores.append(r2)  # computed on CPU

            # show information
            if (batch_idx + 1) % configs['log_interval'] == 0:
                cur_time = time.time()
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, R2: {:.2f} ({:.3f} s)'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), \
                    loss.item(), r2, cur_time - begin_time))
                begin_time = time.time()

        # CW: This is the relevant task type
        elif task_type == 'neureg':
            # to compute accuracy
            N, T, n_comp = output.size()
            y = y.view(-1, T * n_comp)
            y_pred = output.view(-1, T * n_comp)
            r2 = r2_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy(),
                          multioutput='raw_values') # .reshape(T, n_comp) # CW: since it was throwing errors
            scores.append(r2)  # computed on CPU

            # show information
            if (batch_idx + 1) % configs['log_interval'] == 0:
                cur_time = time.time()
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, R2: {:.2f} ({:.3f} s)'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), \
                    loss.item(), r2.mean(), cur_time - begin_time))
                begin_time = time.time()

    return losses, scores


'''
Used for cd_reg_from_videos_all_no_stim_and_pert_trials_diff_timesteps.
'''


def train_with_two_loaders(configs, model, device, train_loader1, train_loader2, pred_timesteps1, pred_timesteps2, \
                           optimizer, epoch, loss_fct, task_type, x_avg_dict=None, v_std_dict=None, cd_trial_avg=None):
    # set model as training mode
    model.train()

    if 't_smooth_loss' in configs.keys() and bool(configs['t_smooth_loss']):
        assert x_avg_dict is not None
        assert v_std_dict is not None
        t_smooth_loss_fct = t_smooth_loss(configs, x_avg_dict, v_std_dict)

    if 'cd_cor_loss' in configs.keys() and bool(configs['cd_cor_loss']):
        assert cd_trial_avg is not None
        cd_cor_loss_fct = cd_cor_mse_loss(configs, cd_trial_avg).to(device)

    losses = []
    scores = []
    N_count = 0  # counting total trained sample in one epoch
    begin_time = time.time()
    for batch_idx, (data1, data2) in enumerate(itertools.zip_longest(train_loader1, train_loader2)):
        # distribute data to device

        if data1 is not None:
            X_side1, X_bottom1, y1, z1, trial_type1 = data1
            X_side1, X_bottom1, y1, z1, trial_type1 = \
                X_side1.to(device), X_bottom1.to(device), y1.to(device), z1.to(device), trial_type1.to(device)

        if data2 is not None:
            X_side2, X_bottom2, y2, z2, trial_type2 = data2
            X_side2, X_bottom2, y2, z2, trial_type2 = \
                X_side2.to(device), X_bottom2.to(device), y2.to(device), z2.to(device), trial_type2.to(device)

        if data1 is not None:
            N_count += y1.size(0)

        if data2 is not None:
            N_count += y2.size(0)

        optimizer.zero_grad()

        if data1 is not None:
            output1 = model(X_side1, X_bottom1, z1, pred_timesteps1)

        if data2 is not None:
            output2 = model(X_side2, X_bottom2, z2, pred_timesteps2)

        loss = []

        if data1 is not None:
            loss.append(loss_fct(output1, y1))

        if data2 is not None:
            loss.append(loss_fct(output2, y2))

        if len(loss) == 0:
            raise ValueError('len(loss) should not be 0.')

        elif len(loss) == 1:
            loss = loss[0]

        elif len(loss) == 2:
            loss = loss[0] + loss[1]

        losses.append(loss.item())

        if 't_smooth_loss' in configs.keys() and bool(configs['t_smooth_loss']):
            v_penalty = t_smooth_loss_fct(output, z, trial_type)
            (loss + v_penalty).backward()

        elif 'cd_cor_loss' in configs.keys() and bool(configs['cd_cor_loss']):
            cd_cor_loss = cd_cor_loss_fct(output, y, z)
            (loss + cd_cor_loss).backward()

        else:
            loss.backward()

        optimizer.step()

        avg_r2 = []

        if data1 is not None:
            N, T1, n_comp = output1.size()
            y1 = y1.view(-1, T1 * n_comp)
            y_pred1 = output1.view(-1, T1 * n_comp)
            r2 = r2_score(y1.cpu().data.squeeze().numpy(), y_pred1.cpu().data.squeeze().numpy(),
                          multioutput='raw_values').reshape(T1, n_comp)

            avg_r2.append(r2)

        if data2 is not None:
            N, T2, n_comp = output2.size()
            y2 = y2.view(-1, T2 * n_comp)
            y_pred2 = output2.view(-1, T2 * n_comp)
            r2 = r2_score(y2.cpu().data.squeeze().numpy(), y_pred2.cpu().data.squeeze().numpy(),
                          multioutput='raw_values').reshape(T2, n_comp)

            avg_r2.append(r2)

        if len(avg_r2) == 0:
            raise ValueError('len(avg_r2) should not be 0.')

        else:
            avg_r2 = cat(avg_r2, 0)

        scores.append(avg_r2)

        # show information
        if (batch_idx + 1) % configs['log_interval'] == 0:
            cur_time = time.time()
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, R2: {:.2f} ({:.3f} s)'.format(
                epoch + 1, N_count, len(train_loader1.dataset), 100. * (batch_idx + 1) / len(train_loader1), \
                loss.item(), avg_r2.mean(), cur_time - begin_time))
            begin_time = time.time()

    return losses, scores


def train_dry_run_v2(configs, model, device, train_loader, optimizer, epoch, loss_fct, task_type, x_avg_dict=None,
                     v_std_dict=None, cd_trial_avg=None):
    # set model as training mode
    model.train()

    losses = []
    scores = []
    N_count = 0  # counting total trained sample in one epoch
    begin_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        # distribute data to device
        if 'sess_cond' in configs['model_name'] or 'neureg' in configs['model_name'] or 'reg' in configs['model_name']:
            if 'all_views' in configs['model_name']:
                X_side, X_bottom, y, z, trial_type = data
                X_side, X_bottom, y, z, trial_type = X_side.to(device), X_bottom.to(device), y.to(device), z.to(
                    device), trial_type.to(device)

            else:
                X, y, z, trial_type = data
                X, y, z, trial_type = X.to(device), y.to(device), z.to(device), trial_type.to(device)

        else:
            X, y = data
            X, y = X.to(device), y.to(device)

        N_count += y.size(0)

        loss = 1.0
        r2 = 1.0
        losses.append(loss)
        scores.append(r2)

        cur_time = time.time()
        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, R2: {:.2f} ({:.3f} s)'.format(
            epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), \
            loss, r2, cur_time - begin_time))
        begin_time = time.time()

        break

    return losses, scores


def validation(configs, model, device, test_loader, best_test_score_mean, loss_fct, task_type, model_save_path=None,
               pred_timesteps=None):
    '''
    Note added on 101620:
    pred_timesteps argument is necessary for cd_reg_from_videos_all_no_stim_and_pert_trials_diff_timesteps,
    as control and pert trials may have different pred_timesteps.
    '''

    # set model as testing mode
    begin_time = time.time()
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for data in test_loader:
            if 'sess_cond' in configs['model_name'] or 'neureg' in configs['model_name'] or 'reg' in configs[
                'model_name']:
                if 'all_views' in configs['model_name']:
                    X_side, X_bottom, y, z, trial_type = data
                    X_side, X_bottom, y, z, trial_type = X_side.to(device), X_bottom.to(device), y.to(device), z.to(
                        device), trial_type.to(device)

                else:
                    X, y, z, trial_type = data
                    X, y, z, trial_type = X.to(device), y.to(device), z.to(device), trial_type.to(device)


            else:
                X, y = data
                X, y = X.to(device), y.to(device)

            if 'sess_cond' in configs['model_name'] or 'neureg' in configs['model_name'] or 'reg' in configs[
                'model_name']:
                if 'all_views' in configs['model_name']:
                    if pred_timesteps is not None:
                        output = model(X_side, X_bottom, z, pred_timesteps)

                    else:
                        output = model(X_side, X_bottom, z)

                else:
                    output = model(X, z)  # output has dim = (batch, number of classes)

            else:
                output = model(X)  # output has dim = (batch, number of classes)

            if task_type == 'cls':
                y = y.squeeze()

            # Multiply by numel so that we are not dividing by numel.
            loss = loss_fct(output, y) * len(y)
            test_loss += loss.item()  # sum up batch loss

            if task_type == 'cls':
                y_pred = output.max(1)[1]  # (y_pred != output) get the index of the max log-probability

            elif task_type == 'reg':
                y_pred = output

            elif task_type == 'neureg':
                # output has size (N, T, n_comp)
                N, T, n_comp = output.size()
                y = y.view(-1, T * n_comp)
                y_pred = output.view(-1, T * n_comp)

            # collect all y and y_pred in all batches
            all_y.append(y)
            all_y_pred.append(y_pred)

    test_loss /= len(test_loader.dataset)

    if task_type == 'cls':
        # compute accuracy
        all_y = torch.cat(all_y, dim=0)
        all_y_pred = torch.cat(all_y_pred, dim=0)

        test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

        chance_level = all_y.cpu().data.squeeze().numpy().astype(float).mean()

    elif task_type == 'reg':
        all_y = torch.cat(all_y, dim=0)
        all_y_pred = torch.cat(all_y_pred, dim=0)
        test_score = r2_score(all_y.cpu().data.numpy(), all_y_pred.cpu().data.numpy())


    elif task_type == 'neureg':
        all_y = torch.cat(all_y, dim=0)
        all_y_pred = torch.cat(all_y_pred, dim=0)
        test_score = r2_score(all_y.cpu().data.numpy(), all_y_pred.cpu().data.numpy(),
                              multioutput='raw_values').reshape(T, n_comp)

    # show information
    cur_time = time.time()

    if task_type == 'cls':
        print('')
        print('Test set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}% (chance: {:.2f}%) ({:.3f} s)'.format(
            len(all_y), test_loss, \
            100 * test_score, 100 * chance_level, cur_time - begin_time))
        print('')

        test_score_mean = test_score


    elif task_type == 'reg':
        print('')
        print('Test set ({:d} samples): Average loss: {:.4f}, R2: {:.2f} ({:.3f} s)'.format(len(all_y), test_loss, \
                                                                                            test_score,
                                                                                            cur_time - begin_time))
        print('')

        test_score_mean = test_score


    elif task_type == 'neureg':
        print('')
        print('Test set ({:d} samples): Average loss: {:.4f}, R2: {:.2f} ({:.3f} s)'.format(len(all_y), test_loss, \
                                                                                            test_score.mean(),
                                                                                            cur_time - begin_time))
        print('')
        print('R2 for each time point and comp')
        for k in range(n_comp):
            prefix = '[Comp {}] '.format(k + 1)
            list_str = ', '.join(['t={}: {:.2f}'.format(t, test_score[t, k]) for t in range(T)])
            cur_str = prefix + list_str
            print(cur_str)
        print('')

        test_score_mean = test_score.mean()

    # save Pytorch models of best record
    if test_score_mean > best_test_score_mean:
        print('current test_score > best_test_score')

        if model_save_path is None:
            cnn_channel_str = '_'.join([str(i) for i in configs['_cnn_channel_list']])
            if 't_smooth_loss' in configs.keys():
                t_smooth_loss_str = 't_smooth_loss_{:.1f}_sigma'.format(configs['t_smooth_loss_sigma']) if bool(
                    configs['t_smooth_loss']) else ''
            else:
                t_smooth_loss_str = ''

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

            model_save_path = os.path.join(configs['models_dir'], img_type_str, neural_time_str, \
                                           'n_pred_comp_{}'.format(configs['n_pred_comp']),
                                           'view_type_{}'.format(configs['view_type'][0]),
                                           'model_name_{}'.format(configs['model_name']), \
                                           'cnn_channel_list_{}'.format(cnn_channel_str), t_smooth_loss_str)

        os.makedirs(model_save_path, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))  # save model
        print("Current model saved!")
        print('')

    return test_loss, test_score


def validation_with_two_loaders(configs, model, device, test_loader1, test_loader2, pred_timesteps1, pred_timesteps2, \
                                best_test_score_mean, loss_fct, task_type, model_save_path=None):
    # set model as testing mode
    begin_time = time.time()
    model.eval()

    test_loss1 = 0
    test_loss2 = 0

    all_y1 = []
    all_y_pred1 = []

    all_y2 = []
    all_y_pred2 = []

    N_count1 = 0
    N_count2 = 0

    with torch.no_grad():
        for (data1, data2) in itertools.zip_longest(test_loader1, test_loader2):

            if data1 is not None:
                X_side1, X_bottom1, y1, z1, trial_type1 = data1
                X_side1, X_bottom1, y1, z1, trial_type1 = \
                    X_side1.to(device), X_bottom1.to(device), y1.to(device), z1.to(device), trial_type1.to(device)

            if data2 is not None:
                X_side2, X_bottom2, y2, z2, trial_type2 = data2
                X_side2, X_bottom2, y2, z2, trial_type2 = \
                    X_side2.to(device), X_bottom2.to(device), y2.to(device), z2.to(device), trial_type2.to(device)

            if data1 is not None:
                output1 = model(X_side1, X_bottom1, z1, pred_timesteps1)

            if data2 is not None:
                output2 = model(X_side2, X_bottom2, z2, pred_timesteps2)

            if data1 is not None:
                test_loss1 += loss_fct(output1, y1).item() * len(y1)
                N_count1 += len(y1)

            if data2 is not None:
                test_loss2 += loss_fct(output2, y2).item() * len(y2)
                N_count2 += len(y2)

            if data1 is not None:
                # output has size (N, T, n_comp)
                N1, T1, n_comp = output1.size()
                y1 = y1.view(-1, T1 * n_comp)
                y_pred1 = output1.view(-1, T1 * n_comp)

                # collect all y and y_pred in all batches
                all_y1.append(y1)
                all_y_pred1.append(y_pred1)

            if data2 is not None:
                # output has size (N, T, n_comp)
                N2, T2, n_comp = output2.size()
                y2 = y2.view(-1, T2 * n_comp)
                y_pred2 = output2.view(-1, T2 * n_comp)

                # collect all y and y_pred in all batches
                all_y2.append(y2)
                all_y_pred2.append(y_pred2)

    test_loss1 /= N_count1
    test_loss2 /= N_count2

    test_loss = (test_loss1 + test_loss2) / 2

    all_y1 = torch.cat(all_y1, dim=0)
    all_y_pred1 = torch.cat(all_y_pred1, dim=0)
    test_score1 = r2_score(all_y1.cpu().data.numpy(), all_y_pred1.cpu().data.numpy(), multioutput='raw_values').reshape(
        T1, n_comp)

    all_y2 = torch.cat(all_y2, dim=0)
    all_y_pred2 = torch.cat(all_y_pred2, dim=0)
    test_score2 = r2_score(all_y2.cpu().data.numpy(), all_y_pred2.cpu().data.numpy(), multioutput='raw_values').reshape(
        T2, n_comp)

    # show information
    cur_time = time.time()

    print('')
    print('<data1>')
    print('Test set ({:d} samples): Average loss: {:.4f}, R2: {:.2f} ({:.3f} s)'.format(len(all_y1), test_loss1, \
                                                                                        test_score1.mean(),
                                                                                        cur_time - begin_time))
    print('')
    print('R2 for each time point and comp')
    for k in range(n_comp):
        prefix = '[Comp {}] '.format(k + 1)
        list_str = ', '.join(['t={}: {:.2f}'.format(t, test_score1[t, k]) for t in range(T1)])
        cur_str = prefix + list_str
        print(cur_str)
    print('')

    print('')
    print('<data2>')
    print('Test set ({:d} samples): Average loss: {:.4f}, R2: {:.2f} ({:.3f} s)'.format(len(all_y2), test_loss2, \
                                                                                        test_score2.mean(),
                                                                                        cur_time - begin_time))
    print('')
    print('R2 for each time point and comp')
    for k in range(n_comp):
        prefix = '[Comp {}] '.format(k + 1)
        list_str = ', '.join(['t={}: {:.2f}'.format(t, test_score2[t, k]) for t in range(T2)])
        cur_str = prefix + list_str
        print(cur_str)
    print('')

    test_score = cat([test_score1, test_score2], 0)  # (T1+T2, n_comp)

    test_score_mean = test_score.mean()

    # save Pytorch models of best record
    if test_score_mean > best_test_score_mean:
        print('current test_score > best_test_score')

        os.makedirs(model_save_path, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))  # save model
        print("Current model saved!")
        print('')

    return test_loss, test_score


def validation_dry_run_v2(configs, model, device, test_loader, best_test_score_mean, loss_fct, task_type,
                          model_save_path=None):
    # set model as testing mode
    begin_time = time.time()
    model.eval()

    test_loss = 0
    test_score = np.random.randn(2, 2)

    with torch.no_grad():
        for data in test_loader:
            if 'sess_cond' in configs['model_name'] or 'neureg' in configs['model_name'] or 'reg' in configs[
                'model_name']:
                if 'all_views' in configs['model_name']:
                    X_side, X_bottom, y, z, trial_type = data
                    X_side, X_bottom, y, z, trial_type = X_side.to(device), X_bottom.to(device), y.to(device), z.to(
                        device), trial_type.to(device)

                else:
                    X, y, z, trial_type = data
                    X, y, z, trial_type = X.to(device), y.to(device), z.to(device), trial_type.to(device)


            else:
                X, y = data
                X, y = X.to(device), y.to(device)

            break

    # show information
    cur_time = time.time()

    print('')
    print('Test set ({:.3f} s)'.format(cur_time - begin_time))

    return test_loss, test_score


def train_choice_pred(configs, model, device, train_loader, optimizer, epoch, loss_fct, return_auc=False):
    # set model as training mode
    model.train()

    losses = []
    scores = []
    if return_auc:
        aucs = []

    N_count = 0  # counting total trained sample in one epoch
    begin_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        # distribute data to device

        '''
        y is trial type label.
        z is sess ind.
        '''
        if 'all_views' in configs['model_name']:
            X_side, X_bottom, y, z = data
            X_side, X_bottom, y, z = X_side.to(device), X_bottom.to(device), y.to(device), z.to(device)

        else:
            X, y, z = data
            X, y, z = X.to(device), y.to(device), z.to(device)

        N_count += y.size(0)

        optimizer.zero_grad()

        if 'sess_cond' in configs['model_name']:

            if 'all_views' in configs['model_name']:
                output = model(X_side, X_bottom, z)
            else:
                output = model(X, z)  # output has dim = (batch, T)

        else:

            if 'all_views' in configs['model_name']:
                output = model(X_side, X_bottom)  # output has dim = (batch, T)
            else:
                output = model(X)  # output has dim = (batch, T)

        loss = loss_fct(output, y[:, None].float().expand_as(output))

        losses.append(loss.item())

        loss.backward()

        optimizer.step()

        # to compute accuracy and roc_auc_score.
        y_pred = (output >= 0).long()  # (batch, T)

        y = y.cpu().data.numpy()  # (batch)
        y_pred = y_pred.cpu().data.numpy()  # (batch, T)
        if return_auc:
            output = output.cpu().data.numpy()  # (batch, T)

        step_score_over_time = []
        for t in range(y_pred.shape[1]):
            step_score_over_time.append(accuracy_score(y, y_pred[:, t]))
        step_score = np.mean(step_score_over_time)

        scores.append(step_score)

        if return_auc:
            step_auc_over_time = []

            for t in range(y_pred.shape[1]):
                if len(np.unique(y)) == 1:
                    # roc_auc_score gives an error if there is only one type of class in y.
                    step_auc_over_time.append(0.5)
                else:
                    step_auc_over_time.append(roc_auc_score(y, output[:, t]))

            step_auc = np.mean(step_auc_over_time)

            aucs.append(step_auc)

        # show information
        if (batch_idx + 1) % configs['log_interval'] == 0:
            cur_time = time.time()
            if return_auc:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, Accu: {:.2f}%, ROC_AUC: {:.2f}% ({:.3f} s)'.format(
                        epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), \
                        loss.item(), 100 * step_score, 100 * step_auc, cur_time - begin_time))
            else:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, Accu: {:.2f}% ({:.3f} s)'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), \
                    loss.item(), 100 * step_score, cur_time - begin_time))

            begin_time = time.time()

    if return_auc:
        return losses, scores, aucs

    else:
        return losses, scores


def train_choice_pred_manual_db(configs, model, device, train_loader, optimizer, epoch, loss_fct, return_auc=False):
    # set model as training mode
    model.train()

    losses = []
    scores = []
    if return_auc:
        aucs = []

    N_count = 0  # counting total trained sample in one epoch
    begin_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        # if batch_idx > 0:
        #     break
        # distribute data to device

        '''
        y is trial type label.
        z is sess ind.
        '''
        if 'all_views' in configs['model_name']:
            X_side, X_bottom, y, z = data
            X_side, X_bottom, y, z = X_side.to(device), X_bottom.to(device), y.to(device), z.to(device)

        else:
            X, y, z = data
            X, y, z = X.to(device), y.to(device), z.to(device)

        N_count += y.size(0)

        optimizer.zero_grad()

        if 'sess_cond' in configs['model_name']:

            if 'all_views' in configs['model_name']:
                output = model(X_side, X_bottom, z)
            else:
                output = model(X, z)  # output has dim = (batch, T) or (batch)

        else:

            if 'all_views' in configs['model_name']:
                output = model(X_side, X_bottom)  # output has dim = (batch, T) or (batch)
            else:
                output = model(X)  # output has dim = (batch, T) or (batch)

        if len(output.size()) == 1:
            loss = loss_fct(output, y.float())
        else:
            loss = loss_fct(output, y[:, None].float().expand_as(output))

        losses.append(loss.item())

        loss.backward()

        optimizer.step()

        '''
        We then manually optimize the best decision boundary. 
        '''
        # manual_db_optimizer(configs, model, output, y, z)
        # best_thr = manual_db_optimizer_v2(configs, model, output, y, z)
        best_thr = manual_db_optimizer_v3(configs, model, output, y, z)

        # to compute accuracy and roc_auc_score.
        y_pred = (output >= 0).long()  # (batch, T) or (batch)

        y = y.cpu().data.numpy()  # (batch)
        y_pred = y_pred.cpu().data.numpy()  # (batch, T) or (batch)
        if return_auc:
            output = output.cpu().data.numpy()  # (batch, T) or (batch)

        if len(y_pred.shape) == 1:
            step_score = accuracy_score(y, y_pred)

        else:
            step_score_over_time = []
            for t in range(y_pred.shape[1]):
                step_score_over_time.append(accuracy_score(y, y_pred[:, t]))
            step_score = np.mean(step_score_over_time)

        scores.append(step_score)

        # Improved score using best_thr
        if len(y_pred.shape) == 1:
            if not np.isnan(best_thr):
                if isinstance(output, torch.Tensor):
                    improved_y_pred = (output >= best_thr).long().cpu().data.numpy()
                else:
                    improved_y_pred = (output >= best_thr).astype(int)

                step_improved_score = accuracy_score(y, improved_y_pred)

            else:
                step_improved_score = np.nan


        else:
            if not np.isnan(best_thr):
                if isinstance(output, torch.Tensor):
                    improved_y_pred = (output >= best_thr).long().cpu().data.numpy()
                else:
                    improved_y_pred = (output >= best_thr).astype(int)

                step_improved_score_over_time = []
                for t in range(y_pred.shape[1]):
                    step_improved_score_over_time.append(accuracy_score(y, improved_y_pred[:, t]))
                step_improved_score = np.mean(step_improved_score_over_time)

            else:
                step_improved_score = np.nan

        if return_auc:
            if len(y_pred.shape) == 1:
                if len(np.unique(y)) == 1:
                    # roc_auc_score gives an error if there is only one type of class in y.
                    step_auc = 0.5
                else:
                    step_auc = roc_auc_score(y, output)

                aucs.append(step_auc)


            else:
                step_auc_over_time = []

                for t in range(y_pred.shape[1]):
                    if len(np.unique(y)) == 1:
                        # roc_auc_score gives an error if there is only one type of class in y.
                        step_auc_over_time.append(0.5)
                    else:
                        step_auc_over_time.append(roc_auc_score(y, output[:, t]))

                step_auc = np.mean(step_auc_over_time)

                aucs.append(step_auc)

        chance_level = 1 - y.astype(float).mean()
        if chance_level < 0.5:
            chance_level = 1 - chance_level

        # show information
        if (batch_idx + 1) % configs['log_interval'] == 0:
            cur_time = time.time()
            if return_auc:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, Accu: {:.2f}%, Improved Accu: {:.2f}% (best_thr: {:.2f}), ROC_AUC: {:.2f}% (chance: {:.2f}) ({:.3f} s)'.format( \
                        epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), \
                        loss.item(), 100 * step_score, 100 * step_improved_score, best_thr, 100 * step_auc,
                        100 * chance_level, cur_time - begin_time))
            else:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, Accu: {:.2f}%, Improved Accu: {:.2f}% (best_thr: {:.2f}), (chance: {:.2f}) ({:.3f} s)'.format( \
                        epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), \
                        loss.item(), 100 * step_score, 100 * step_improved_score, best_thr, 100 * chance_level,
                        cur_time - begin_time))

            begin_time = time.time()

    if return_auc:
        return losses, scores, aucs, best_thr

    else:
        return losses, scores, best_thr


def train_choice_pred_manual_db_subsample_neg(configs, model, device, train_loader, optimizer, epoch, loss_fct,
                                              return_auc=False):
    # set model as training mode
    model.train()

    losses = []
    scores = []
    if return_auc:
        aucs = []

    N_count = 0  # counting total trained sample in one epoch
    begin_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        # if batch_idx > 0:
        #     break
        # distribute data to device

        '''
        y is trial type label.
        z is sess ind.
        '''
        if 'all_views' in configs['model_name']:
            X_side, X_bottom, y, z = data

            n_pos_samples = len(torch.nonzero(y))
            sample_mask = torch.zeros_like(y).byte()
            sample_mask[y == 1] = 1
            sample_mask[torch.nonzero(y == 0)[:n_pos_samples]] = 1

            X_side = X_side[sample_mask]
            X_bottom = X_bottom[sample_mask]
            y = y[sample_mask]
            z = z[sample_mask]

            X_side, X_bottom, y, z = X_side.to(device), X_bottom.to(device), y.to(device), z.to(device)

        else:
            X, y, z = data

            n_pos_samples = len(torch.nonzero(y))
            sample_mask = torch.zeros_like(y).byte()
            sample_mask[y == 1] = 1
            sample_mask[torch.nonzero(y == 0)[:n_pos_samples]] = 1

            X = X[sample_mask]
            y = y[sample_mask]
            z = z[sample_mask]

            X, y, z = X.to(device), y.to(device), z.to(device)

        N_count += y.size(0)

        optimizer.zero_grad()

        if 'sess_cond' in configs['model_name']:

            if 'all_views' in configs['model_name']:
                output = model(X_side, X_bottom, z)
            else:
                output = model(X, z)  # output has dim = (batch, T)

        else:

            if 'all_views' in configs['model_name']:
                output = model(X_side, X_bottom)  # output has dim = (batch, T)
            else:
                output = model(X)  # output has dim = (batch, T)

        loss = loss_fct(output, y[:, None].float().expand_as(output))

        losses.append(loss.item())

        loss.backward()

        optimizer.step()

        '''
        We then manually optimize the best decision boundary. 
        '''
        # manual_db_optimizer(configs, model, output, y, z)
        # best_thr = manual_db_optimizer_v2(configs, model, output, y, z)
        best_thr = manual_db_optimizer_v3(configs, model, output, y, z)

        # to compute accuracy and roc_auc_score.
        y_pred = (output >= 0).long()  # (batch, T)

        y = y.cpu().data.numpy()  # (batch)
        y_pred = y_pred.cpu().data.numpy()  # (batch, T)
        if return_auc:
            output = output.cpu().data.numpy()  # (batch, T)

        step_score_over_time = []
        for t in range(y_pred.shape[1]):
            step_score_over_time.append(accuracy_score(y, y_pred[:, t]))
        step_score = np.mean(step_score_over_time)

        scores.append(step_score)

        # Improved score using best_thr
        if not np.isnan(best_thr):
            if isinstance(output, torch.Tensor):
                improved_y_pred = (output >= best_thr).long().cpu().data.numpy()
            else:
                improved_y_pred = (output >= best_thr).astype(int)

            step_improved_score_over_time = []
            for t in range(y_pred.shape[1]):
                step_improved_score_over_time.append(accuracy_score(y, improved_y_pred[:, t]))
            step_improved_score = np.mean(step_improved_score_over_time)

        else:
            step_improved_score = np.nan

        if return_auc:
            step_auc_over_time = []

            for t in range(y_pred.shape[1]):
                if len(np.unique(y)) == 1:
                    # roc_auc_score gives an error if there is only one type of class in y.
                    step_auc_over_time.append(0.5)
                else:
                    step_auc_over_time.append(roc_auc_score(y, output[:, t]))

            step_auc = np.mean(step_auc_over_time)

            aucs.append(step_auc)

        chance_level = 1 - y.astype(float).mean()
        if chance_level < 0.5:
            chance_level = 1 - chance_level

        # show information
        if (batch_idx + 1) % configs['log_interval'] == 0:
            cur_time = time.time()
            if return_auc:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, Accu: {:.2f}%, Improved Accu: {:.2f}% (best_thr: {:.2f}), ROC_AUC: {:.2f}% (chance: {:.2f}) ({:.3f} s)'.format( \
                        epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), \
                        loss.item(), 100 * step_score, 100 * step_improved_score, best_thr, 100 * step_auc,
                        100 * chance_level, cur_time - begin_time))
            else:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, Accu: {:.2f}%, Improved Accu: {:.2f}% (best_thr: {:.2f}), (chance: {:.2f}) ({:.3f} s)'.format( \
                        epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), \
                        loss.item(), 100 * step_score, 100 * step_improved_score, best_thr, 100 * chance_level,
                        cur_time - begin_time))

            begin_time = time.time()

    if return_auc:
        return losses, scores, aucs, best_thr

    else:
        return losses, scores, best_thr


def manual_db_optimizer(configs, model, output, y, z):
    y = y.cpu().data.numpy()
    output = output.cpu().data.numpy()
    z = z.cpu().data.numpy()
    broadcast_y = np.broadcast_to(y[:, None], (len(y), output.shape[1]))

    if 'sess_cond' in configs['model_name']:
        for s in np.unique(z):
            sess_mask = (z == s)
            if len(np.nonzero(sess_mask)[0]) == 0:
                continue

            cur_y = y[sess_mask]
            cur_output = output[sess_mask]
            cur_broadcast_y = broadcast_y[sess_mask]
            # Check if we have both classes.
            if len(np.unique(cur_y)) < 2:
                continue

            thrs = list(np.unique(cur_output.reshape(-1))) + [np.max(cur_output) + 1]
            thrs.sort()

            best_thr = 0.0
            best_score = float('-inf')
            for thr in thrs:
                cur_y_pred = (cur_output.reshape(-1) >= thr).astype(int)
                cur_score = accuracy_score(cur_broadcast_y.reshape(-1), cur_y_pred)
                if cur_score > best_score:
                    best_thr = thr
                    best_score = cur_score

            with torch.no_grad():
                if configs['use_cuda']:
                    model.module.linear_list1[s].bias.add_(-best_thr)
                else:
                    model.linear_list1[s].bias.add_(-best_thr)


    else:
        if len(np.unique(y)) < 2:
            return

        thrs = list(np.unique(output.reshape(-1))) + [np.max(output) + 1]
        thrs.sort()

        best_thr = 0.0
        best_score = float('-inf')
        for thr in thrs:
            cur_y_pred = (output.reshape(-1) >= thr).astype(int)
            cur_score = accuracy_score(broadcast_y.reshape(-1), cur_y_pred)
            if cur_score > best_score:
                best_thr = thr
                best_score = cur_score

        with torch.no_grad():
            if configs['use_cuda']:
                model.module.linear1.bias.add_(-best_thr)
            else:
                model.linear1.bias.add_(-best_thr)


def manual_db_optimizer_v2(configs, model, output, y, z):
    '''
    Common thr update for all sess-specfic layers.
    '''
    y = y.cpu().data.numpy()
    output = output.cpu().data.numpy()
    z = z.cpu().data.numpy()
    broadcast_y = np.broadcast_to(y[:, None], (len(y), output.shape[1]))

    if len(np.unique(y)) < 2:
        return np.nan

    thrs = list(np.unique(output.reshape(-1))) + [np.max(output) + 1]
    thrs.sort()

    best_thr = 0.0
    best_score = float('-inf')
    for thr in thrs:
        cur_y_pred = (output.reshape(-1) >= thr).astype(int)
        cur_score = accuracy_score(broadcast_y.reshape(-1), cur_y_pred)
        if cur_score > best_score:
            best_thr = thr
            best_score = cur_score

    if 'sess_cond' in configs['model_name']:
        with torch.no_grad():
            if configs['use_cuda']:
                for s in np.unique(z):
                    model.module.linear_list1[s].bias.add_(-best_thr)
            else:
                for s in np.unique(z):
                    model.linear_list1[s].bias.add_(-best_thr)

    else:
        with torch.no_grad():
            if configs['use_cuda']:
                model.module.linear1.bias.add_(-best_thr)
            else:
                model.linear1.bias.add_(-best_thr)

    return best_thr


def manual_db_optimizer_v3(configs, model, output, y, z):
    '''
    Do not update thr. Just return best_thr.
    '''
    y = y.cpu().data.numpy()
    output = output.cpu().data.numpy()
    z = z.cpu().data.numpy()
    if len(output.shape) == 1:
        broadcast_y = y
    else:
        broadcast_y = np.broadcast_to(y[:, None], (len(y), output.shape[1]))

    if len(np.unique(y)) < 2:
        return np.nan

    thrs = list(np.unique(output.reshape(-1))) + [np.max(output) + 1]
    thrs.sort()

    best_thr = 0.0
    best_score = float('-inf')
    for thr in thrs:
        cur_y_pred = (output.reshape(-1) >= thr).astype(int)
        cur_score = accuracy_score(broadcast_y.reshape(-1), cur_y_pred)
        if cur_score > best_score:
            best_thr = thr
            best_score = cur_score

    return best_thr


def validation_choice_pred(configs, model, device, test_loader, best_test_score_mean, loss_fct, model_save_path,
                           return_auc=False):
    # set model as testing mode
    begin_time = time.time()
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    if return_auc:
        all_output = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            '''
            y is trial type label.
            z is sess ind.
            '''
            if 'all_views' in configs['model_name']:
                X_side, X_bottom, y, z = data
                X_side, X_bottom, y, z = X_side.to(device), X_bottom.to(device), y.to(device), z.to(device)

            else:
                X, y, z = data
                X, y, z = X.to(device), y.to(device), z.to(device)

            if 'sess_cond' in configs['model_name']:

                if 'all_views' in configs['model_name']:
                    output = model(X_side, X_bottom, z)
                else:
                    output = model(X, z)  # output has dim = (batch, number of classes)

            else:

                if 'all_views' in configs['model_name']:
                    output = model(X_side, X_bottom)  # output has dim = (batch, number of classes)
                else:
                    output = model(X)  # output has dim = (batch, number of classes)

            # Multiply by numel so that we are not dividing by numel.
            loss = loss_fct(output, y[:, None].float().expand_as(output)) * len(y)
            test_loss += loss.item()  # sum up batch loss

            y_pred = (output >= 0).long()  # (batch, T)

            # collect all y and y_pred in all batches
            all_y.append(y.cpu().data.numpy())
            all_y_pred.append(y_pred.cpu().data.numpy())
            if return_auc:
                all_output.append(output.cpu().data.numpy())

    test_loss /= len(test_loader.dataset)

    # compute accuracy and roc_auc_score.
    all_y = cat(all_y, 0)  # (batch)
    all_y_pred = cat(all_y_pred, 0)  # (batch, T)
    if return_auc:
        all_output = cat(all_output, 0)

    test_score_over_time = []
    for t in range(all_y_pred.shape[1]):
        test_score_over_time.append(accuracy_score(all_y, all_y_pred[:, t]))

    test_score_mean = np.mean(test_score_over_time)
    test_score = np.array(test_score_over_time)

    if return_auc:
        test_auc_over_time = []
        for t in range(all_y_pred.shape[1]):
            if len(np.unique(all_y)) == 1:
                # roc_auc_score gives an error if there is only one type of class in y.
                test_auc_over_time.append(0.5)
            else:
                test_auc_over_time.append(roc_auc_score(all_y, all_output[:, t]))

        test_auc_mean = np.mean(test_auc_over_time)
        test_auc = np.array(test_auc_over_time)

    chance_level = 1 - all_y.astype(float).mean()
    if chance_level < 0.5:
        chance_level = 1 - chance_level

    # show information
    cur_time = time.time()

    if return_auc:
        print('')
        print(
            'Test set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%, ROC_AUC: {:.2f}% (chance: {:.2f}%) ({:.3f} s)'.format(
                len(all_y), test_loss, \
                100 * test_score_mean, 100 * test_auc_mean, 100 * chance_level, cur_time - begin_time))
        print('')

    else:
        print('')
        print('Test set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}% (chance: {:.2f}%) ({:.3f} s)'.format(
            len(all_y), test_loss, \
            100 * test_score_mean, 100 * chance_level, cur_time - begin_time))
        print('')

    # save Pytorch models of best record
    if test_score_mean > best_test_score_mean:
        print('current test_score > best_test_score')

        os.makedirs(model_save_path, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))  # save model
        print("Current model saved!")
        print('')

    if return_auc:
        return test_loss, test_score, test_auc

    else:
        return test_loss, test_score


def validation_choice_pred_manual_db(configs, model, device, test_loader, best_test_score_mean, loss_fct,
                                     model_save_path, best_thr, return_auc=False):
    # set model as testing mode
    begin_time = time.time()
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    all_improved_y_pred = []
    if return_auc:
        all_output = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            '''
            y is trial type label.
            z is sess ind.
            '''
            if 'all_views' in configs['model_name']:
                X_side, X_bottom, y, z = data
                X_side, X_bottom, y, z = X_side.to(device), X_bottom.to(device), y.to(device), z.to(device)

            else:
                X, y, z = data
                X, y, z = X.to(device), y.to(device), z.to(device)

            if 'sess_cond' in configs['model_name']:

                if 'all_views' in configs['model_name']:
                    output = model(X_side, X_bottom, z)
                else:
                    output = model(X, z)  # output has dim = (batch, number of classes)

            else:

                if 'all_views' in configs['model_name']:
                    output = model(X_side, X_bottom)  # output has dim = (batch, number of classes)
                else:
                    output = model(X)  # output has dim = (batch, number of classes)

            # Multiply by numel so that we are not dividing by numel.
            loss = loss_fct(output, y[:, None].float().expand_as(output)) * len(y)
            test_loss += loss.item()  # sum up batch loss

            y_pred = (output >= 0).long()  # (batch, T)
            improved_y_pred = (output >= float(best_thr)).long()  # (batch, T)

            # collect all y and y_pred in all batches
            all_y.append(y.cpu().data.numpy())
            all_y_pred.append(y_pred.cpu().data.numpy())
            all_improved_y_pred.append(improved_y_pred.cpu().data.numpy())
            if return_auc:
                all_output.append(output.cpu().data.numpy())

    test_loss /= len(test_loader.dataset)

    # compute accuracy and roc_auc_score.
    all_y = cat(all_y, 0)  # (batch)
    all_y_pred = cat(all_y_pred, 0)  # (batch, T)
    all_improved_y_pred = cat(all_improved_y_pred, 0)
    if return_auc:
        all_output = cat(all_output, 0)

    # score
    test_score_over_time = []
    for t in range(all_y_pred.shape[1]):
        test_score_over_time.append(accuracy_score(all_y, all_y_pred[:, t]))

    test_score_mean = np.mean(test_score_over_time)
    test_score = np.array(test_score_over_time)

    # improved_score
    test_improved_score_over_time = []
    for t in range(all_improved_y_pred.shape[1]):
        test_improved_score_over_time.append(accuracy_score(all_y, all_improved_y_pred[:, t]))

    test_improved_score_mean = np.mean(test_improved_score_over_time)

    if return_auc:
        test_auc_over_time = []
        for t in range(all_y_pred.shape[1]):
            if len(np.unique(all_y)) == 1:
                # roc_auc_score gives an error if there is only one type of class in y.
                test_auc_over_time.append(0.5)
            else:
                test_auc_over_time.append(roc_auc_score(all_y, all_output[:, t]))

        test_auc_mean = np.mean(test_auc_over_time)
        test_auc = np.array(test_auc_over_time)

        # acc2 is computed by the optimal threshold in the test set.
        test_acc2_over_time = []
        best_test_thr_over_time = []
        for t in range(all_y_pred.shape[1]):
            if len(np.unique(all_y)) == 1:
                test_acc2_over_time.append(1.0)
                best_test_thr_over_time.append(0.0)
            else:
                thrs = list(np.unique(all_output[:, t])) + [np.max(all_output[:, t]) + 1]
                thrs.sort()

                best_acc_at_t = float('-inf')
                best_test_thr = 0.0
                for thr in thrs:
                    cur_y_pred = (all_output[:, t] >= thr).astype(int)
                    cur_score = accuracy_score(all_y, cur_y_pred)
                    if cur_score > best_acc_at_t:
                        best_acc_at_t = cur_score
                        best_test_thr = thr

                test_acc2_over_time.append(best_acc_at_t)
                best_test_thr_over_time.append(best_test_thr)

        test_acc2_mean = np.mean(test_acc2_over_time)

    chance_level = 1 - all_y.astype(float).mean()
    if chance_level < 0.5:
        chance_level = 1 - chance_level

    # show information
    cur_time = time.time()

    if return_auc:
        print('')
        print('Test set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%, Improved Accuracy: {:.2f}% (best_thr: {:.2f}), Best-test-thr Accuracy: {:.2f} \
            (best_test_thr: min {:.2f} / max {:.2f}) ROC_AUC: {:.2f}% (chance: {:.2f}%) ({:.3f} s)'.format(len(all_y),
                                                                                                           test_loss, \
                                                                                                           100 * test_score_mean,
                                                                                                           100 * test_improved_score_mean,
                                                                                                           best_thr,
                                                                                                           100 * test_acc2_mean,
                                                                                                           np.min(
                                                                                                               best_test_thr_over_time),
                                                                                                           np.max(
                                                                                                               best_test_thr_over_time), \
                                                                                                           100 * test_auc_mean,
                                                                                                           100 * chance_level,
                                                                                                           cur_time - begin_time))
        print('')

    else:
        print('')
        print(
            'Test set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%, Improved Accuracy: {:.2f}% (best_thr: {:.2f}) (chance: {:.2f}%) ({:.3f} s)'.format(
                len(all_y), test_loss, \
                100 * test_score_mean, 100 * test_improved_score_mean, best_thr, 100 * chance_level,
                cur_time - begin_time))
        print('')

    # save Pytorch models of best record
    if test_score_mean > best_test_score_mean:
        print('current test_score > best_test_score')

        os.makedirs(model_save_path, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))  # save model
        print("Current model saved!")
        print('')

    if return_auc:
        return test_loss, test_score, test_auc

    else:
        return test_loss, test_score


def pert_pred_analysis(configs, model, device, test_loader):
    # set model as testing mode
    begin_time = time.time()
    model.eval()

    all_trial_type = []
    all_pred = []
    all_pre_sigmoid_score = []
    all_trial_idx = []
    all_sess_inds = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            '''
            y is trial type label.
            z is sess ind.
            '''
            if 'all_views' in configs['model_name']:
                X_side, X_bottom, y, z, trial_idx = data
                # Note that trial_idx is not needed for the forward pass.
                X_side, X_bottom, y, z = X_side.to(device), X_bottom.to(device), y.to(device), z.to(device)

            else:
                X, y, z, trial_idx = data
                # Note that trial_idx is not needed for the forward pass.
                X, y, z = X.to(device), y.to(device), z.to(device)

            if 'sess_cond' in configs['model_name']:

                if 'all_views' in configs['model_name']:
                    output = model(X_side, X_bottom, z)
                else:
                    output = model(X, z)  # output has dim = (batch, number of classes)

            else:

                if 'all_views' in configs['model_name']:
                    output = model(X_side, X_bottom)  # output has dim = (batch, number of classes)
                else:
                    output = model(X)  # output has dim = (batch, number of classes)

            y_pred = (output >= 0).long()  # (batch, T)

            # collect all y and y_pred in all batches
            all_trial_type.append(y.cpu().data.numpy())
            all_pred.append(y_pred.cpu().data.numpy())
            all_pre_sigmoid_score.append(output.cpu().data.numpy())
            all_trial_idx.append(trial_idx.cpu().data.numpy())
            all_sess_inds.append(z.cpu().data.numpy())

    all_trial_type = cat(all_trial_type, 0)
    all_pred = cat(all_pred, 0)
    all_pre_sigmoid_score = cat(all_pre_sigmoid_score, 0)
    all_trial_idx = cat(all_trial_idx, 0)
    all_sess_inds = cat(all_sess_inds, 0)

    return all_trial_type, all_pred, all_pre_sigmoid_score, all_trial_idx, all_sess_inds


def validation_choice_pred_manual_db_subsample_neg(configs, model, device, test_loader, best_test_score_mean, loss_fct, \
                                                   model_save_path, best_thr, return_auc=False,
                                                   save_best_model_on_balanced_set=False):
    # set model as testing mode
    begin_time = time.time()
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    all_improved_y_pred = []
    if return_auc:
        all_output = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # if batch_idx > 0:
            #     break
            '''
            y is trial type label.
            z is sess ind.
            '''
            if 'all_views' in configs['model_name']:
                X_side, X_bottom, y, z = data
                X_side, X_bottom, y, z = X_side.to(device), X_bottom.to(device), y.to(device), z.to(device)

            else:
                X, y, z = data
                X, y, z = X.to(device), y.to(device), z.to(device)

            if 'sess_cond' in configs['model_name']:

                if 'all_views' in configs['model_name']:
                    output = model(X_side, X_bottom, z)
                else:
                    output = model(X, z)  # output has dim = (batch, number of classes)

            else:

                if 'all_views' in configs['model_name']:
                    output = model(X_side, X_bottom)  # output has dim = (batch, number of classes)
                else:
                    output = model(X)  # output has dim = (batch, number of classes)

            # Multiply by numel so that we are not dividing by numel.
            if len(output.size()) == 1:
                loss = loss_fct(output, y.float()) * len(y)
            else:
                loss = loss_fct(output, y[:, None].float().expand_as(output)) * len(y)

            test_loss += loss.item()  # sum up batch loss

            y_pred = (output >= 0).long()  # (batch, T)
            improved_y_pred = (output >= float(best_thr)).long()  # (batch, T)

            # collect all y and y_pred in all batches
            all_y.append(y.cpu().data.numpy())
            all_y_pred.append(y_pred.cpu().data.numpy())
            all_improved_y_pred.append(improved_y_pred.cpu().data.numpy())
            if return_auc:
                all_output.append(output.cpu().data.numpy())

    test_loss /= len(test_loader.dataset)

    # compute accuracy and roc_auc_score.
    all_y = cat(all_y, 0)  # (batch)
    all_y_pred = cat(all_y_pred, 0)  # (batch, T)
    all_improved_y_pred = cat(all_improved_y_pred, 0)
    if return_auc:
        all_output = cat(all_output, 0)

    # score
    if len(all_y_pred.shape) == 1:
        test_score_mean = accuracy_score(all_y, all_y_pred)
        test_score = test_score_mean

    else:
        test_score_over_time = []
        for t in range(all_y_pred.shape[1]):
            test_score_over_time.append(accuracy_score(all_y, all_y_pred[:, t]))

        test_score_mean = np.mean(test_score_over_time)
        test_score = np.array(test_score_over_time)

    # improved_score
    if len(all_y_pred.shape) == 1:
        test_improved_score_mean = accuracy_score(all_y, all_improved_y_pred)

    else:
        test_improved_score_over_time = []
        for t in range(all_improved_y_pred.shape[1]):
            test_improved_score_over_time.append(accuracy_score(all_y, all_improved_y_pred[:, t]))

        test_improved_score_mean = np.mean(test_improved_score_over_time)

    if return_auc:
        if len(all_y_pred.shape) == 1:
            if len(np.unique(all_y)) == 1:
                test_auc_mean = 0.5
            else:
                test_auc_mean = roc_auc_score(all_y, all_output)

            test_auc = test_auc_mean

            # acc2 is computed by the optimal threshold in the test set.
            if len(np.unique(all_y)) == 1:
                test_acc2_mean = 1.0

            else:
                thrs = list(np.unique(all_output)) + [np.max(all_output) + 1]
                thrs.sort()

                best_acc = float('-inf')
                best_test_thr = 0.0
                for thr in thrs:
                    cur_y_pred = (all_output >= thr).astype(int)
                    cur_score = accuracy_score(all_y, cur_y_pred)
                    if cur_score > best_acc:
                        best_acc = cur_score
                        best_test_thr = thr

                test_acc2_mean = best_acc

        else:
            test_auc_over_time = []
            for t in range(all_y_pred.shape[1]):
                if len(np.unique(all_y)) == 1:
                    # roc_auc_score gives an error if there is only one type of class in y.
                    test_auc_over_time.append(0.5)
                else:
                    test_auc_over_time.append(roc_auc_score(all_y, all_output[:, t]))

            test_auc_mean = np.mean(test_auc_over_time)
            test_auc = np.array(test_auc_over_time)

            # acc2 is computed by the optimal threshold in the test set.
            test_acc2_over_time = []
            best_test_thr_over_time = []
            for t in range(all_y_pred.shape[1]):
                if len(np.unique(all_y)) == 1:
                    test_acc2_over_time.append(1.0)
                    best_test_thr_over_time.append(0.0)
                else:
                    thrs = list(np.unique(all_output[:, t])) + [np.max(all_output[:, t]) + 1]
                    thrs.sort()

                    best_acc_at_t = float('-inf')
                    best_test_thr = 0.0
                    for thr in thrs:
                        cur_y_pred = (all_output[:, t] >= thr).astype(int)
                        cur_score = accuracy_score(all_y, cur_y_pred)
                        if cur_score > best_acc_at_t:
                            best_acc_at_t = cur_score
                            best_test_thr = thr

                    test_acc2_over_time.append(best_acc_at_t)
                    best_test_thr_over_time.append(best_test_thr)

            test_acc2_mean = np.mean(test_acc2_over_time)

    chance_level = 1 - all_y.astype(float).mean()
    if chance_level < 0.5:
        chance_level = 1 - chance_level

    # show information
    cur_time = time.time()

    if return_auc:
        print('')
        if len(all_y_pred.shape) == 1:
            print(
                'Test set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%, Improved Accuracy: {:.2f}% (best_thr: {:.2f}), Best-test-thr Accuracy: {:.2f} ROC_AUC: {:.2f}% (chance: {:.2f}%) ({:.3f} s)'.format( \
                    len(all_y), test_loss, \
                    100 * test_score_mean, 100 * test_improved_score_mean, best_thr, 100 * test_acc2_mean, \
                    100 * test_auc_mean, 100 * chance_level, cur_time - begin_time))

        else:
            print(
                'Test set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%, Improved Accuracy: {:.2f}% (best_thr: {:.2f}), Best-test-thr Accuracy: {:.2f} (best_test_thr: min {:.2f} / max {:.2f}), ROC_AUC: {:.2f}% (chance: {:.2f}%) ({:.3f} s)'.format( \
                    len(all_y), test_loss, \
                    100 * test_score_mean, 100 * test_improved_score_mean, best_thr, 100 * test_acc2_mean,
                    np.min(best_test_thr_over_time), np.max(best_test_thr_over_time), \
                    100 * test_auc_mean, 100 * chance_level, cur_time - begin_time))

        print('')

    else:
        print('')
        print(
            'Test set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%, Improved Accuracy: {:.2f}% (best_thr: {:.2f}) (chance: {:.2f}%) ({:.3f} s)'.format(
                len(all_y), test_loss, \
                100 * test_score_mean, 100 * test_improved_score_mean, best_thr, 100 * chance_level,
                cur_time - begin_time))
        print('')

    if not save_best_model_on_balanced_set:
        # save Pytorch models of best record
        if test_score_mean > best_test_score_mean:
            print('current test_score > best_test_score')

            os.makedirs(model_save_path, exist_ok=True)

            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))  # save model
            print("Current model saved!")
            print('')

    '''
    Now, calculate test acc etc. on a balanced set.
    '''
    n_pos_samples = len(np.nonzero(all_y)[0])
    sample_mask = np.zeros(len(all_y)).astype(bool)
    sample_mask[all_y == 1] = True
    sample_mask[np.nonzero(all_y == 0)[0][:n_pos_samples]] = True

    if len(np.nonzero(sample_mask)[0]) == 0:
        print('sample_mask is empty.')
        print('n_pos_samples: ', n_pos_samples)
        print('')
        subsample_test_score = np.nan  # np.nan is a float, not np.ndarray, I confirmed.
        if return_auc:
            subsample_test_auc = np.nan
    else:

        all_y = all_y[sample_mask]  # (batch)
        all_y_pred = all_y_pred[sample_mask]  # (batch, T)
        all_improved_y_pred = all_improved_y_pred[sample_mask]

        if return_auc:
            all_output = all_output[sample_mask]

        # score
        if len(all_y_pred.shape) == 1:
            subsample_test_score_mean = accuracy_score(all_y, all_y_pred)
            subsample_test_score = subsample_test_score_mean

        else:
            test_score_over_time = []
            for t in range(all_y_pred.shape[1]):
                test_score_over_time.append(accuracy_score(all_y, all_y_pred[:, t]))

            subsample_test_score_mean = np.mean(test_score_over_time)
            subsample_test_score = np.array(test_score_over_time)

        # improved_score
        if len(all_y_pred.shape) == 1:
            subsample_test_improved_score_mean = accuracy_score(all_y, all_improved_y_pred)

        else:
            test_improved_score_over_time = []
            for t in range(all_improved_y_pred.shape[1]):
                test_improved_score_over_time.append(accuracy_score(all_y, all_improved_y_pred[:, t]))

            subsample_test_improved_score_mean = np.mean(test_improved_score_over_time)

        if return_auc:
            if len(all_y_pred.shape) == 1:
                if len(np.unique(all_y)) == 1:
                    subsample_test_auc_mean = 0.5
                else:
                    subsample_test_auc_mean = roc_auc_score(all_y, all_output)

                subsample_test_auc = subsample_test_auc_mean

                # acc2 is computed by the optimal threshold in the test set.
                if len(np.unique(all_y)) == 1:
                    subsample_test_acc2_mean = 1.0

                else:
                    thrs = list(np.unique(all_output)) + [np.max(all_output) + 1]
                    thrs.sort()

                    best_acc = float('-inf')
                    best_test_thr = 0.0
                    for thr in thrs:
                        cur_y_pred = (all_output >= thr).astype(int)
                        cur_score = accuracy_score(all_y, cur_y_pred)
                        if cur_score > best_acc:
                            best_acc = cur_score
                            best_test_thr = thr

                    subsample_test_acc2_mean = best_acc

            else:
                test_auc_over_time = []
                for t in range(all_y_pred.shape[1]):
                    if len(np.unique(all_y)) == 1:
                        # roc_auc_score gives an error if there is only one type of class in y.
                        test_auc_over_time.append(0.5)
                    else:
                        test_auc_over_time.append(roc_auc_score(all_y, all_output[:, t]))

                subsample_test_auc_mean = np.mean(test_auc_over_time)
                subsample_test_auc = np.array(test_auc_over_time)

                # acc2 is computed by the optimal threshold in the test set.
                test_acc2_over_time = []
                best_test_thr_over_time = []
                for t in range(all_y_pred.shape[1]):
                    if len(np.unique(all_y)) == 1:
                        test_acc2_over_time.append(1.0)
                        best_test_thr_over_time.append(0.0)
                    else:
                        thrs = list(np.unique(all_output[:, t])) + [np.max(all_output[:, t]) + 1]
                        thrs.sort()

                        best_acc_at_t = float('-inf')
                        best_test_thr = 0.0
                        for thr in thrs:
                            cur_y_pred = (all_output[:, t] >= thr).astype(int)
                            cur_score = accuracy_score(all_y, cur_y_pred)
                            if cur_score > best_acc_at_t:
                                best_acc_at_t = cur_score
                                best_test_thr = thr

                        test_acc2_over_time.append(best_acc_at_t)
                        best_test_thr_over_time.append(best_test_thr)

                subsample_test_acc2_mean = np.mean(test_acc2_over_time)
                subsample_best_test_thr_over_time = np.array(best_test_thr_over_time)

        subsample_chance_level = 1 - all_y.astype(float).mean()
        if subsample_chance_level < 0.5:
            subsample_chance_level = 1 - subsample_chance_level

        # show information
        cur_time = time.time()

        if return_auc:
            print('')
            if len(all_y_pred.shape) == 1:
                print(
                    'Subsampled Test set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%, Improved Accuracy: {:.2f}% (best_thr: {:.2f}), Best-test-thr Accuracy: {:.2f}, ROC_AUC: {:.2f}% (chance: {:.2f}%)'.format( \
                        len(all_y), test_loss, 100 * subsample_test_score_mean,
                                               100 * subsample_test_improved_score_mean, best_thr,
                                               100 * subsample_test_acc2_mean, \
                                               100 * subsample_test_auc_mean, 100 * subsample_chance_level))
            else:
                print(
                    'Subsampled Test set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%, Improved Accuracy: {:.2f}% (best_thr: {:.2f}), Best-test-thr Accuracy: {:.2f} (best_test_thr: min {:.2f} / max {:.2f}), ROC_AUC: {:.2f}% (chance: {:.2f}%)'.format( \
                        len(all_y), test_loss, 100 * subsample_test_score_mean,
                                               100 * subsample_test_improved_score_mean, best_thr,
                                               100 * subsample_test_acc2_mean,
                        np.min(subsample_best_test_thr_over_time), np.max(subsample_best_test_thr_over_time), \
                                               100 * subsample_test_auc_mean, 100 * subsample_chance_level))

            print('')

        else:
            print('')
            print(
                'Subsampled Test set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%, Improved Accuracy: {:.2f}% (best_thr: {:.2f}) (chance: {:.2f}%)'.format( \
                    len(all_y), test_loss, 100 * subsample_test_score_mean, 100 * subsample_test_improved_score_mean,
                    best_thr, 100 * subsample_chance_level))
            print('')

        if save_best_model_on_balanced_set:
            # save Pytorch models of best record
            if subsample_test_score_mean > best_test_score_mean:
                print('current test_score > best_test_score')

                os.makedirs(model_save_path, exist_ok=True)

                torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))  # save model
                print("Current model saved!")
                print('')

    if return_auc:
        if save_best_model_on_balanced_set:
            return test_loss, subsample_test_score, subsample_test_auc
        else:
            return test_loss, test_score, test_auc


    else:
        if save_best_model_on_balanced_set:
            return test_loss, subsample_test_score
        else:
            return test_loss, test_score


def train_dry_run(configs, train_loader):
    N_count = 0  # counting total trained sample in one epoch
    begin_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        # distribute data to device
        if 'sess_cond' in configs['model_name'] or 'neureg' in configs['model_name'] or 'reg' in configs['model_name']:
            if 'all_views' in configs['model_name']:
                X_side, X_bottom, y, z, trial_type = data

            else:
                X, y, z, trial_type = data

        else:
            X, y = data

        N_count += y.size(0)

        # show information
        if (batch_idx + 1) % configs['log_interval'] == 0:
            cur_time = time.time()
            print('Train dry run: [{}/{} ({:.0f}%)] ({:.3f} s)'.format(
                N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), \
                                                    cur_time - begin_time))
            begin_time = time.time()


def validation_dry_run(configs, test_loader):
    # set model as testing mode
    begin_time = time.time()

    with torch.no_grad():
        for data in test_loader:
            if 'sess_cond' in configs['model_name'] or 'neureg' in configs['model_name'] or 'reg' in configs[
                'model_name']:
                if 'all_views' in configs['model_name']:
                    X_side, X_bottom, y, z, trial_type = data

                else:
                    X, y, z, trial_type = data


            else:
                X, y = data

    # show information
    cur_time = time.time()

    print('')
    print('Test dry run ({:d} samples) ({:.3f} s)'.format(len(test_loader.dataset), cur_time - begin_time))
    print('')


def train_choice_pred_dry_run(configs, train_loader):
    N_count = 0  # counting total trained sample in one epoch
    begin_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        # distribute data to device

        '''
        y is trial type label.
        z is sess ind.
        '''
        if 'all_views' in configs['model_name']:
            X_side, X_bottom, y, z = data

        else:
            X, y, z = data

        N_count += y.size(0)

        # show information
        if (batch_idx + 1) % configs['log_interval'] == 0:
            cur_time = time.time()
            print('Train dry run: [{}/{} ({:.0f}%)] ({:.3f} s)'.format(
                N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), cur_time - begin_time))
            begin_time = time.time()


def validation_choice_pred_dry_run(configs, test_loader):
    # set model as testing mode
    begin_time = time.time()

    with torch.no_grad():
        for data in test_loader:

            '''
            y is trial type label.
            z is sess ind.
            '''
            if 'all_views' in configs['model_name']:
                X_side, X_bottom, y, z = data

            else:
                X, y, z = data

    # show information
    cur_time = time.time()

    print('')
    print('Test dry run ({:d} samples) ({:.3f} s)'.format(len(test_loader.dataset), cur_time - begin_time))
    print('')


def neureg_recon(configs, model, device, test_loader, loss_fct):
    # set model as testing mode
    begin_time = time.time()
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    all_trial_idx = []
    with torch.no_grad():
        for data in test_loader:
            if 'all_views' in configs['model_name']:
                X_side, X_bottom, y, z, trial_type, trial_idx = data
                X_side, X_bottom, y, z, trial_type, trial_idx = \
                    X_side.to(device), X_bottom.to(device), y.to(device), z.to(device), trial_type.to(
                        device), trial_idx.to(device)

            else:
                X, y, z, trial_type, trial_idx = data
                X, y, z, trial_type, trial_idx = \
                    X.to(device), y.to(device), z.to(device), trial_type.to(device), trial_idx.to(device)

            if 'all_views' in configs['model_name']:
                output = model(X_side, X_bottom, z)

            else:
                output = model(X, z)  # output has dim = (batch, number of classes)

            # Multiply by numel so that we are not dividing by numel.
            loss = loss_fct(output, y) * len(y)
            test_loss += loss.item()  # sum up batch loss

            # output has size (N, T, n_comp)
            N, T, n_comp = output.size()

            # collect all y and y_pred in all batches
            all_y.append(y)
            all_y_pred.append(output)
            all_trial_idx.append(trial_idx)

    test_loss /= len(test_loader.dataset)

    all_y = torch.cat(all_y, dim=0).cpu().data.numpy()
    all_y_pred = torch.cat(all_y_pred, dim=0).cpu().data.numpy()
    all_trial_idx = torch.cat(all_trial_idx, dim=0).cpu().data.numpy()

    test_score = r2_score(all_y.reshape(-1, T * n_comp), all_y_pred.reshape(-1, T * n_comp),
                          multioutput='raw_values').reshape(T, n_comp)

    # show information
    cur_time = time.time()

    print('')
    print('Test set ({:d} samples): Average loss: {:.4f}, R2: {:.2f} ({:.3f} s)'.format(len(all_y), test_loss, \
                                                                                        test_score.mean(),
                                                                                        cur_time - begin_time))
    print('')
    print('R2 for each time point and comp')
    for k in range(n_comp):
        prefix = '[Comp {}] '.format(k + 1)
        list_str = ', '.join(['t={}: {:.2f}'.format(t, test_score[t, k]) for t in range(T)])
        cur_str = prefix + list_str
        print(cur_str)
    print('')

    return all_y, all_y_pred, all_trial_idx, test_score


def choice_pred_recon(configs, model, device, test_loader, loss_fct):
    '''
    Return:
    all_y: (n_trials)
    all_y_pred: (n_trials, T)
    all_output: (n_trials, T)
    all_trial_idx: (n_trials)
    acc: (T)
    '''
    # set model as testing mode
    begin_time = time.time()
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    all_output = []
    all_trial_idx = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            '''
            y is trial type label.
            z is sess ind.
            '''
            if 'all_views' in configs['model_name']:
                X_side, X_bottom, y, z, trial_idx = data
                X_side, X_bottom, y, z, trial_idx \
                    = X_side.to(device), X_bottom.to(device), y.to(device), z.to(device), trial_idx.to(device)

            else:
                X, y, z, trial_idx = data
                X, y, z, trial_idx \
                    = X.to(device), y.to(device), z.to(device), trial_idx.to(device)

            if 'sess_cond' in configs['model_name']:

                if 'all_views' in configs['model_name']:
                    output = model(X_side, X_bottom, z)
                else:
                    output = model(X, z)  # output has dim = (batch, number of classes)

            else:

                if 'all_views' in configs['model_name']:
                    output = model(X_side, X_bottom)  # output has dim = (batch, number of classes)
                else:
                    output = model(X)  # output has dim = (batch, number of classes)

            # Multiply by numel so that we are not dividing by numel.
            loss = loss_fct(output, y[:, None].float().expand_as(output)) * len(y)
            test_loss += loss.item()  # sum up batch loss

            y_pred = (output >= 0).long()  # (batch, T)

            # collect all y and y_pred in all batches
            all_y.append(y.cpu().data.numpy())
            all_y_pred.append(y_pred.cpu().data.numpy())
            all_output.append(output.cpu().data.numpy())
            all_trial_idx.append(trial_idx.cpu().data.numpy())

    test_loss /= len(test_loader.dataset)

    # compute accuracy and roc_auc_score.
    all_y = cat(all_y, 0)  # (batch)
    all_y_pred = cat(all_y_pred, 0)  # (batch, T)
    all_output = cat(all_output, 0)
    all_trial_idx = cat(all_trial_idx, 0)

    '''
    For computing accuracy, we will throw away surplus negative samples
    so that the test set is balanced. 
    '''
    n_pos_samples = len(np.nonzero(all_y == 1)[0])
    sample_mask = np.zeros(len(all_y)).astype(bool)
    sample_mask[all_y == 1] = True
    neg_sample_inds = np.nonzero(all_y == 0)[0]
    subsampled_neg_sample_inds = np.random.permutation(neg_sample_inds)[:n_pos_samples]
    sample_mask[subsampled_neg_sample_inds] = True

    acc_over_time = []
    for t in range(all_y_pred.shape[1]):
        acc_over_time.append(accuracy_score(all_y[sample_mask], all_y_pred[sample_mask][:, t]))

    acc_mean = np.mean(acc_over_time)
    acc = np.array(acc_over_time)

    chance_level = 1 - all_y[sample_mask].astype(float).mean()
    if chance_level < 0.5:
        chance_level = 1 - chance_level

    # show information
    cur_time = time.time()

    print('')
    print(
        'Test set ({:d} samples / {:d} balanced samples): Average loss: {:.4f}, Accuracy: {:.2f}% (chance: {:.2f}%) ({:.3f} s)'.format( \
            len(all_y), len(all_y[sample_mask]), test_loss, \
            100 * acc_mean, 100 * chance_level, cur_time - begin_time))
    print('')

    return all_y, all_y_pred, all_output, all_trial_idx, acc


## ------------------------ network modules ---------------------- ##

class cnn(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if not '_cnn_channel_list' in self.configs.keys():
            self.configs['_cnn_channel_list'] = [8, 8, 16, 16, 32, 32]
        if not '_maxPoolLayers' in self.configs.keys():
            self.configs["_maxPoolLayers"] = [2, 4]
        if not '_cnn_kernel' in self.configs.keys():
            self.configs["_cnn_kernel"] = 5

        s = 1
        k = configs['_cnn_kernel']
        os = [1] + self.configs['_cnn_channel_list']
        self.conv_list = nn.ModuleList([
            nn.Conv2d(os[i], os[i + 1], k, stride=s, padding=(k - s) // 2) for i in
            range(len(self.configs['_cnn_channel_list']))
        ])

    def forward(self, x):
        for i in range(len(self.configs['_cnn_channel_list'])):
            x = F.relu(self.conv_list[i](x))
            if i in self.configs["_maxPoolLayers"]:
                x = F.max_pool2d(x, 2)
        return x


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


class cnn_bn_frac_stride(nn.Module):
    '''
    difference from cnn_bn (all inspired by DCGAN):
    1. Instead of max pooling, dowmsamplle by fractional stride.
    2. Instead of relu, use leaky relu.

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

        s = 1
        k = configs['_cnn_kernel']
        os = [1] + self.configs['_cnn_channel_list']

        conv_list = []
        for i in range(len(self.configs['_cnn_channel_list'])):

            if i in self.configs['_maxPoolLayers']:
                # frac stride, downsamples by factor of 2.
                conv_list.append(nn.Conv2d(os[i], os[i + 1], 4, stride=2, padding=1))

            else:
                conv_list.append(nn.Conv2d(os[i], os[i + 1], k, stride=s, padding=(k - s) // 2))

        self.conv_list = nn.ModuleList(conv_list)
        self.bn_list = nn.ModuleList([
            nn.BatchNorm2d(os[i + 1]) for i in range(len(self.configs['_cnn_channel_list']))
        ])

    def forward(self, x):
        for i in range(len(self.configs['_cnn_channel_list'])):
            x = F.leaky_relu(self.conv_list[i](x), 0.2)
            x = self.bn_list[i](x)
        return x


class cnn3d_bn(nn.Module):
    '''
    difference from cnn_bn:
    1. Use 3d convolution.
    2. Strided convolution instead of max pool in time direction.
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
        if not '_cnn_kernel_t' in self.configs.keys():
            self.configs["_cnn_kernel_t"] = 3

        s = 1
        k = configs['_cnn_kernel']
        k_t = configs['_cnn_kernel_t']
        os = [1] + self.configs['_cnn_channel_list']

        conv_list = []
        for i in range(len(self.configs['_cnn_channel_list'])):

            if i in self.configs['_maxPoolLayers']:
                # frac stride in time dim, downsamples by factor of 2.
                conv_list.append(nn.Conv3d(os[i], os[i + 1], kernel_size=[2, k, k], stride=[2, s, s],
                                           padding=[0, (k - s) // 2, (k - s) // 2]))

            else:
                conv_list.append(nn.Conv3d(os[i], os[i + 1], kernel_size=[k_t, k, k], stride=s,
                                           padding=[(k_t - s) // 2, (k - s) // 2, (k - s) // 2]))

        self.conv_list = nn.ModuleList(conv_list)
        self.bn_list = nn.ModuleList([
            nn.BatchNorm3d(os[i + 1]) for i in range(len(self.configs['_cnn_channel_list']))
        ])

    def forward(self, x):
        for i in range(len(self.configs['_cnn_channel_list'])):
            x = F.relu(self.conv_list[i](x))
            x = self.bn_list[i](x)
            if i in self.configs["_maxPoolLayers"]:
                x = F.max_pool3d(x, kernel_size=[1, 2, 2])  # Note that we don't do max pool in time dim.
        return x


class lrcn(nn.Module):

    def __init__(self, configs):

        super(lrcn, self).__init__()
        self.configs = configs

        if "_lstmHidden" not in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        self.configs["_seqOut"] = self.configs["_lstmHidden"] * 2

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 64

        if '_clf_num_fc' not in configs.keys():
            self.configs['_clf_num_fc'] = 2

        if self.configs['_clf_num_fc'] == 2:
            if '_ln1_out' not in self.configs.keys():
                self.configs['_ln1_out'] = 64
        else:
            assert '_ln1_out' not in self.configs.keys()

        if 'dropout2' not in self.configs.keys():
            self.configs['dropout2'] = self.configs['dropout']

        self.cnn = cnn(self.configs)
        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen = (self.configs["image_shape"][0] // pool) * (self.configs["image_shape"][1] // pool) * \
                      self.configs["_cnn_channel_list"][-1]

        self.lstm = nn.LSTM(
            input_size=featureVlen,
            hidden_size=self.configs["_lstmHidden"],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.linear1 = nn.Linear(self.configs["_seqOut"], self.configs["_ln1_out"])
        self.linear2 = nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"])

    def forward(self, x):
        bs, seq_length, c, h, w = x.size()
        # assert seq_length == self.configs["seq_length"]
        x = x.view(-1, c, h, w)
        x = self.cnn(x)
        x = x.view(bs, seq_length, -1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        self.lstm.flatten_parameters()
        x, (_, _) = self.lstm(x)

        x = x.view(bs, seq_length, 2, self.configs["_lstmHidden"])
        x = torch.cat((x[:, -1, 0, :], x[:, 0, 1, :]), dim=1)
        x = F.dropout(x, p=self.configs['dropout2'], training=self.training)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class lrcn_sess_cond(nn.Module):
    '''
    We have separate linear layer for each sess.
    '''

    def __init__(self, configs):

        super(lrcn_sess_cond, self).__init__()
        self.configs = configs

        if "_lstmHidden" not in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        self.configs["_seqOut"] = self.configs["_lstmHidden"] * 2

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 64

        if '_clf_num_fc' not in configs.keys():
            self.configs['_clf_num_fc'] = 2

        if self.configs['_clf_num_fc'] == 2:
            if '_ln1_out' not in self.configs.keys():
                self.configs['_ln1_out'] = 64
        else:
            assert '_ln1_out' not in self.configs.keys()

        if 'dropout2' not in self.configs.keys():
            self.configs['dropout2'] = self.configs['dropout']

        self.cnn = cnn(self.configs)
        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen = (self.configs["image_shape"][0] // pool) * (self.configs["image_shape"][1] // pool) * \
                      self.configs["_cnn_channel_list"][-1]

        self.lstm = nn.LSTM(
            input_size=featureVlen,
            hidden_size=self.configs["_lstmHidden"],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.linear1 = nn.Linear(self.configs["_seqOut"], self.configs["_ln1_out"])
        self.sess_cond_linear2 = nn.ModuleList(
            [nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for s in range(self.configs['n_sess'])])

    def forward(self, x, sess_inds):
        '''
        sess_inds: (bs,). It contains integer indices ranging from 0 to n_sess-1.
        '''
        bs, seq_length, c, h, w = x.size()
        # assert seq_length == self.configs["seq_length"]
        x = x.view(-1, c, h, w)
        x = self.cnn(x)
        x = x.view(bs, seq_length, -1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        self.lstm.flatten_parameters()
        x, (_, _) = self.lstm(x)

        x = x.view(bs, seq_length, 2, self.configs["_lstmHidden"])
        x = torch.cat((x[:, -1, 0, :], x[:, 0, 1, :]), dim=1)
        x = F.dropout(x, p=self.configs['dropout2'], training=self.training)
        x = F.relu(self.linear1(x))

        outputs = x.new_zeros([bs, self.configs["_ln2_out"]])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue
            cur_x = x[sess_mask]
            outputs[sess_mask] = self.sess_cond_linear2[s](cur_x)

        return outputs


class lrcn_neureg(nn.Module):

    def __init__(self, configs):
        super(lrcn_neureg, self).__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

        self.cnn = cnn(self.configs)
        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen = (self.configs["image_shape"][0] // pool) * (self.configs["image_shape"][1] // pool) * \
                      self.configs["_cnn_channel_list"][-1]

        self.lstm = nn.LSTM(
            input_size=featureVlen,
            hidden_size=self.configs["_lstmHidden"],
            num_layers=self.configs['lstm_layer_num'],
            batch_first=True,
            bidirectional=True
        )

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer
        self.linear_list2 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer
        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.configs["_ln2_out"], 1) for t in range(len(self.configs['pred_timesteps']))
                ]) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

    def forward(self, x, sess_inds):

        bs, seq_length, c, h, w = x.size()

        x = x.view(-1, c, h, w)
        x = self.cnn(x)
        x = x.view(bs, seq_length, -1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)
        self.lstm.flatten_parameters()  # for multi-gpu using

        x, (_, _) = self.lstm(x)

        x = x[:, self.configs['pred_timesteps'], :]
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))
                for t in range(len(self.configs['pred_timesteps'])):
                    outputs[sess_mask, t, p] = self.linear_list3[s][p][t](x_sub2task[:, t, :]).view(-1)

        return outputs


class lrcn_neureg_bn(nn.Module):
    def __init__(self, configs):
        super(lrcn_neureg_bn, self).__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

        self.cnn = cnn_bn(self.configs)
        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen = (self.configs["image_shape"][0] // pool) * (self.configs["image_shape"][1] // pool) * \
                      self.configs["_cnn_channel_list"][-1]

        self.lstm = nn.LSTM(
            input_size=featureVlen,
            hidden_size=self.configs["_lstmHidden"],
            num_layers=self.configs['lstm_layer_num'],
            batch_first=True,
            bidirectional=True
        )

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer
        self.linear_list2 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer
        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.configs["_ln2_out"], 1) for t in range(len(self.configs['pred_timesteps']))
                ]) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

    def forward(self, x, sess_inds):

        bs, seq_length, c, h, w = x.size()

        x = x.view(-1, c, h, w)
        x = self.cnn(x)
        x = x.view(bs, seq_length, -1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)
        self.lstm.flatten_parameters()  # for multi-gpu using

        x, (_, _) = self.lstm(x)

        x = x[:, self.configs['pred_timesteps'], :]
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))
                for t in range(len(self.configs['pred_timesteps'])):
                    outputs[sess_mask, t, p] = self.linear_list3[s][p][t](x_sub2task[:, t, :]).view(-1)

        return outputs


class lrcn_neureg_bn2(nn.Module):
    '''
    Difference from lrcn_neureg_bn:
    1. We learn the initial state of lstm.
    2. Also, carefully initialize the forget-gate bias term of lstm.
    '''

    def __init__(self, configs):
        super(lrcn_neureg_bn2, self).__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

        self.cnn = cnn_bn(self.configs)
        pool = 2 ** len(self.configs["_maxPoolLayers"])
        self.featureVlen = (self.configs["image_shape"][0] // pool) * (self.configs["image_shape"][1] // pool) * \
                           self.configs["_cnn_channel_list"][-1]

        self.lstm = nn.LSTM(
            input_size=self.featureVlen,
            hidden_size=self.configs["_lstmHidden"],
            num_layers=self.configs['lstm_layer_num'],
            batch_first=True,
            bidirectional=True
        )

        self.lstm_h0 = Parameter(torch.Tensor(2 * self.configs['lstm_layer_num'], self.configs['_lstmHidden']))
        self.lstm_c0 = Parameter(torch.Tensor(2 * self.configs['lstm_layer_num'], self.configs['_lstmHidden']))

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer
        self.linear_list2 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer
        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.configs["_ln2_out"], 1) for t in range(len(self.configs['pred_timesteps']))
                ]) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

        self.init_params()

    def init_params(self):
        init.constant_(self.lstm_h0, 0.0)
        init.constant_(self.lstm_c0, 0.0)

        for k in range(self.configs['lstm_layer_num']):
            init.normal_(getattr(self.lstm, 'weight_ih_l{}'.format(k)), 0.0, 1.0 / math.sqrt(self.featureVlen))
            init.normal_(getattr(self.lstm, 'weight_hh_l{}'.format(k)), 0.0,
                         1.0 / math.sqrt(self.configs['_lstmHidden']))
            init.constant_(getattr(self.lstm, 'bias_ih_l{}'.format(k)), 0.0)
            # We initialize the forget gate bias to 1.0. Note that we need to modify only one of bias_ih or bias_hh.
            init.constant_(getattr(self.lstm, 'bias_ih_l{}'.format(k))[
                           self.configs['_lstmHidden']:2 * self.configs['_lstmHidden']], 1.0)
            init.constant_(getattr(self.lstm, 'bias_hh_l{}'.format(k)), 0.0)

            init.normal_(getattr(self.lstm, 'weight_ih_l{}_reverse'.format(k)), 0.0, 1.0 / math.sqrt(self.featureVlen))
            init.normal_(getattr(self.lstm, 'weight_hh_l{}_reverse'.format(k)), 0.0,
                         1.0 / math.sqrt(self.configs['_lstmHidden']))
            init.constant_(getattr(self.lstm, 'bias_ih_l{}_reverse'.format(k)), 0.0)
            # We initialize the forget gate bias to 1.0. Note that we need to modify only one of bias_ih or bias_hh.
            init.constant_(getattr(self.lstm, 'bias_ih_l{}_reverse'.format(k))[
                           self.configs['_lstmHidden']:2 * self.configs['_lstmHidden']], 1.0)
            init.constant_(getattr(self.lstm, 'bias_hh_l{}_reverse'.format(k)), 0.0)

    def forward(self, x, sess_inds):

        bs, seq_length, c, h, w = x.size()

        x = x.view(-1, c, h, w)
        x = self.cnn(x)
        x = x.view(bs, seq_length, -1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)
        self.lstm.flatten_parameters()  # for multi-gpu using

        lstm_h0 = self.lstm_h0.unsqueeze(1).expand(-1, bs, -1).contiguous()
        lstm_c0 = self.lstm_c0.unsqueeze(1).expand(-1, bs, -1).contiguous()

        x, (_, _) = self.lstm(x, (lstm_h0, lstm_c0))

        x = x[:, self.configs['pred_timesteps'], :]
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))
                for t in range(len(self.configs['pred_timesteps'])):
                    outputs[sess_mask, t, p] = self.linear_list3[s][p][t](x_sub2task[:, t, :]).view(-1)

        return outputs


class lrcn_neureg_bn_all_views(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

        # Note that cnn modules don't care about the image shape.
        self.cnn_list = nn.ModuleList([cnn_bn(self.configs) for k in range(len(self.configs['view_type']))])

        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen_side = (self.configs["image_shape_side"][0] // pool) * (
                self.configs["image_shape_side"][1] // pool) * self.configs["_cnn_channel_list"][-1]
        featureVlen_bottom = (self.configs["image_shape_bottom"][0] // pool) * (
                self.configs["image_shape_bottom"][1] // pool) * self.configs["_cnn_channel_list"][-1]

        self.lstm = nn.LSTM(
            input_size=featureVlen_side + featureVlen_bottom,
            hidden_size=self.configs["_lstmHidden"],
            num_layers=self.configs['lstm_layer_num'],
            batch_first=True,
            bidirectional=True
        )

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer
        self.linear_list2 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer
        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.configs["_ln2_out"], 1) for t in range(len(self.configs['pred_timesteps']))
                ]) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

    def forward(self, x_side, x_bottom, sess_inds):

        n_views = 2
        x_list = [x_side, x_bottom]

        x_all_views_list = []
        for k in range(n_views):
            cur_x = x_list[k]
            bs, seq_length, c, h, w = cur_x.size()
            x = cur_x.view(-1, c, h, w)
            x = self.cnn_list[k](x)
            x = x.view(bs, seq_length, -1)
            x = F.dropout(x, p=self.configs['dropout'], training=self.training)
            x_all_views_list.append(x)

        x = torch.cat(x_all_views_list, 2)

        self.lstm.flatten_parameters()  # for multi-gpu using
        x, (_, _) = self.lstm(x)

        x = x[:, self.configs['pred_timesteps'], :]
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))
                for t in range(len(self.configs['pred_timesteps'])):
                    outputs[sess_mask, t, p] = self.linear_list3[s][p][t](x_sub2task[:, t, :]).view(-1)

        return outputs


class lrcn_neureg_bn_frame_diff(nn.Module):
    '''
    The only difference from lrcn_neureg_bn is that we use the frame difference as input.
    '''

    def __init__(self, configs):
        super(lrcn_neureg_bn_frame_diff, self).__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

        self.cnn = cnn_bn(self.configs)
        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen = (self.configs["image_shape"][0] // pool) * (self.configs["image_shape"][1] // pool) * \
                      self.configs["_cnn_channel_list"][-1]

        self.lstm = nn.LSTM(
            input_size=featureVlen,
            hidden_size=self.configs["_lstmHidden"],
            num_layers=self.configs['lstm_layer_num'],
            batch_first=True,
            bidirectional=True
        )

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer
        self.linear_list2 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer
        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.configs["_ln2_out"], 1) for t in range(len(self.configs['pred_timesteps']))
                ]) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

    def forward(self, x, sess_inds):
        # Take the frame diff.
        x_diff = x[:, 1:] - x[:, :-1]

        # We are repeating the very first x_diff to keep seq_length.
        x = torch.cat([x_diff[:, 0:1], x_diff], 1)

        bs, seq_length, c, h, w = x.size()

        x = x.view(-1, c, h, w)
        x = self.cnn(x)
        x = x.view(bs, seq_length, -1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)
        self.lstm.flatten_parameters()  # for multi-gpu using

        x, (_, _) = self.lstm(x)

        x = x[:, self.configs['pred_timesteps'], :]
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))
                for t in range(len(self.configs['pred_timesteps'])):
                    outputs[sess_mask, t, p] = self.linear_list3[s][p][t](x_sub2task[:, t, :]).view(-1)

        return outputs


class lrcn_neureg_bn_frame_diff2(nn.Module):
    '''
    Difference from lrcn_neureg_bn_frame_diff:
    1. We take the difference of feature vectors of consecutive frames.
    '''

    def __init__(self, configs):
        super(lrcn_neureg_bn_frame_diff2, self).__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

        self.cnn = cnn_bn(self.configs)
        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen = (self.configs["image_shape"][0] // pool) * (self.configs["image_shape"][1] // pool) * \
                      self.configs["_cnn_channel_list"][-1]

        self.lstm = nn.LSTM(
            input_size=featureVlen,
            hidden_size=self.configs["_lstmHidden"],
            num_layers=self.configs['lstm_layer_num'],
            batch_first=True,
            bidirectional=True
        )

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer
        self.linear_list2 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer
        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.configs["_ln2_out"], 1) for t in range(len(self.configs['pred_timesteps']))
                ]) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

    def forward(self, x, sess_inds):
        bs, seq_length, c, h, w = x.size()

        x = x.view(-1, c, h, w)
        x = self.cnn(x)
        x = x.view(bs, seq_length, -1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)
        self.lstm.flatten_parameters()  # for multi-gpu using

        # Take the difference of feature vectors.
        x_diff = x[:, 1:] - x[:, :-1]

        # We are repeating the very first x_diff to keep seq_length.
        x = torch.cat([x_diff[:, 0:1], x_diff], 1)

        x, (_, _) = self.lstm(x)

        x = x[:, self.configs['pred_timesteps'], :]
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))
                for t in range(len(self.configs['pred_timesteps'])):
                    outputs[sess_mask, t, p] = self.linear_list3[s][p][t](x_sub2task[:, t, :]).view(-1)

        return outputs


class lrcn_neureg_bn_static(nn.Module):
    '''
    Difference from lrcn_neureg_bn:
    1. We only use a single frame at the time of the target neural activity to predict it.
    '''

    def __init__(self, configs):
        super(lrcn_neureg_bn_static, self).__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

        self.cnn = cnn_bn(self.configs)
        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen = (self.configs["image_shape"][0] // pool) * (self.configs["image_shape"][1] // pool) * \
                      self.configs["_cnn_channel_list"][-1]

        self.linear0 = nn.Linear(featureVlen, 2 * self.configs["_lstmHidden"])

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer
        self.linear_list2 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer
        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.configs["_ln2_out"], 1) for t in range(len(self.configs['pred_timesteps']))
                ]) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

    def forward(self, x, sess_inds):

        x = x[:, self.configs['pred_timesteps']]

        bs, seq_length, c, h, w = x.size()

        x = x.view(-1, c, h, w)
        x = self.cnn(x)
        x = x.view(bs, seq_length, -1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        x = F.relu(self.linear0(x))

        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))
                for t in range(len(self.configs['pred_timesteps'])):
                    outputs[sess_mask, t, p] = self.linear_list3[s][p][t](x_sub2task[:, t, :]).view(-1)

        return outputs


class lrcn_neureg_bn_time_indp_dec(nn.Module):
    '''
    Difference from lrcn_neureg_bn:
    1. We don't have a separate linear decoder for each time point.
    '''

    def __init__(self, configs):
        super(lrcn_neureg_bn_time_indp_dec, self).__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

        self.cnn = cnn_bn(self.configs)
        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen = (self.configs["image_shape"][0] // pool) * (self.configs["image_shape"][1] // pool) * \
                      self.configs["_cnn_channel_list"][-1]

        self.lstm = nn.LSTM(
            input_size=featureVlen,
            hidden_size=self.configs["_lstmHidden"],
            num_layers=self.configs['lstm_layer_num'],
            batch_first=True,
            bidirectional=True
        )

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer
        self.linear_list2 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer
        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln2_out"], 1) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

    def forward(self, x, sess_inds):

        bs, seq_length, c, h, w = x.size()

        x = x.view(-1, c, h, w)
        x = self.cnn(x)
        x = x.view(bs, seq_length, -1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)
        self.lstm.flatten_parameters()  # for multi-gpu using

        x, (_, _) = self.lstm(x)

        x = x[:, self.configs['pred_timesteps'], :]
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))

                outputs[sess_mask, :, p] = self.linear_list3[s][p](x_sub2task)[
                    ..., 0]  # [...,0] to squeeze the singleton dimension.

        return outputs


class lrcn_neureg_bn_static_frac_stride(nn.Module):
    '''
    Difference from lrcn_neureg_bn_static:
    1. Instead of max pooling, we are doing fractional stride to downsample.
    '''

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

        self.cnn = cnn_bn_frac_stride(self.configs)
        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen = (self.configs["image_shape"][0] // pool) * (self.configs["image_shape"][1] // pool) * \
                      self.configs["_cnn_channel_list"][-1]

        self.linear0 = nn.Linear(featureVlen, 2 * self.configs["_lstmHidden"])

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer
        self.linear_list2 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer
        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.configs["_ln2_out"], 1) for t in range(len(self.configs['pred_timesteps']))
                ]) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

    def forward(self, x, sess_inds):

        x = x[:, self.configs['pred_timesteps']]

        bs, seq_length, c, h, w = x.size()

        x = x.view(-1, c, h, w)
        x = self.cnn(x)
        x = x.view(bs, seq_length, -1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        x = F.relu(self.linear0(x))

        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))
                for t in range(len(self.configs['pred_timesteps'])):
                    outputs[sess_mask, t, p] = self.linear_list3[s][p][t](x_sub2task[:, t, :]).view(-1)

        return outputs


class lrcn_neureg_bn_static2(nn.Module):
    '''
    Differecne from lrcn_neureg_bn_static:
    1. We use a few consecutive frame as additional channels.
    (2. Since the number of input channels is increased, we should increase the output channels of conv layers.)
    '''

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

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

        self.cnn = cnn_bn(self.configs)
        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen = (self.configs["image_shape"][0] // pool) * (self.configs["image_shape"][1] // pool) * \
                      self.configs["_cnn_channel_list"][-1]

        self.linear0 = nn.Linear(featureVlen, 2 * self.configs["_lstmHidden"])

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer
        self.linear_list2 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer
        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.configs["_ln2_out"], 1) for t in range(len(self.configs['pred_timesteps']))
                ]) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

    def forward(self, x, sess_inds):

        bs, _, _, h, w = x.size()
        seq_length = len(self.configs['pred_timesteps'])

        x = x[:, self.input_timesteps].view(-1, self.n_input_frames, h, w)

        x = self.cnn(x)
        x = x.view(bs, seq_length, -1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        x = F.relu(self.linear0(x))

        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))
                for t in range(len(self.configs['pred_timesteps'])):
                    outputs[sess_mask, t, p] = self.linear_list3[s][p][t](x_sub2task[:, t, :]).view(-1)

        return outputs


class lrcn_neureg_bn_static2_conv3d(nn.Module):
    '''
    Differecne from lrcn_neureg_bn_static2:
    1. We use 3d convolution.

    '''

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

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

        self.cnn = cnn3d_bn(self.configs)
        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen = (self.configs["image_shape"][0] // pool) * (self.configs["image_shape"][1] // pool) * \
                      (self.n_input_frames // pool) * self.configs["_cnn_channel_list"][-1]

        self.linear0 = nn.Linear(featureVlen, 2 * self.configs["_lstmHidden"])

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer
        self.linear_list2 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer
        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.configs["_ln2_out"], 1) for t in range(len(self.configs['pred_timesteps']))
                ]) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

    def forward(self, x, sess_inds):

        bs, _, c, h, w = x.size()
        seq_length = len(self.configs['pred_timesteps'])
        assert c == 1

        # If c == 1, this line needs to be modified.
        x = x[:, self.input_timesteps].view(-1, 1, self.n_input_frames, h, w)

        x = self.cnn(x)
        x = x.view(bs, seq_length, -1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        x = F.relu(self.linear0(x))

        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))
                for t in range(len(self.configs['pred_timesteps'])):
                    outputs[sess_mask, t, p] = self.linear_list3[s][p][t](x_sub2task[:, t, :]).view(-1)

        return outputs


class lrcn_neureg_bn_static2_lstm(nn.Module):
    '''
    Differecne from lrcn_neureg_bn_static2:
    1. We use lstm to integrate information in the input frames.

    '''

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

        self.n_input_frames = self.configs['_input_channel']

        # We change this to 1 temporarily to initialize cnn_bn. After initializing it, we change this back to the original value.
        self.configs['_input_channel'] = 1

        # n_pre and n_post_frames = 4 covers +-0.2 s around the bin center (assuming that skip_frame = 10).
        n_pre_frames = (self.n_input_frames - 1) // 2
        n_post_frames = (self.n_input_frames - 1) // 2

        if (self.n_input_frames - 1) % 2 != 0:
            n_pre_frames += 1

        input_timesteps = []
        for k in self.configs['pred_timesteps']:
            input_timesteps.append(np.arange(k - n_pre_frames, k + n_post_frames + 1, 1))

        self.input_timesteps = cat(input_timesteps, 0)

        self.cnn = cnn_bn(self.configs)

        # Change configs['_input_channel'] back to the original value, so that a print of configs dict shows the original values.
        self.configs['_input_channel'] = self.n_input_frames

        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen = (self.configs["image_shape"][0] // pool) * (self.configs["image_shape"][1] // pool) * \
                      self.configs["_cnn_channel_list"][-1]

        self.lstm = nn.LSTM(
            input_size=featureVlen,
            hidden_size=self.configs["_lstmHidden"],
            num_layers=self.configs['lstm_layer_num'],
            batch_first=True,
            bidirectional=True
        )

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer
        self.linear_list2 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer
        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.configs["_ln2_out"], 1) for t in range(len(self.configs['pred_timesteps']))
                ]) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

    def forward(self, x, sess_inds):

        bs, _, c, h, w = x.size()
        pred_length = len(self.configs['pred_timesteps'])

        x = x[:, self.input_timesteps].view(-1, c, h, w)  # (bs*seq_length*n_input_frames, c, h, w)

        x = self.cnn(x)
        x = x.view(bs * pred_length, self.n_input_frames, -1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)
        self.lstm.flatten_parameters()  # for multi-gpu using

        x, (_, _) = self.lstm(x)

        x = x.view(bs * pred_length, self.n_input_frames, 2, self.configs["_lstmHidden"])
        x = torch.cat((x[:, -1, 0, :], x[:, 0, 1, :]), dim=1)  # (bs*pred_length, _lstmHidden)

        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        x = x.view(bs, pred_length, -1)

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))
                for t in range(len(self.configs['pred_timesteps'])):
                    outputs[sess_mask, t, p] = self.linear_list3[s][p][t](x_sub2task[:, t, :]).view(-1)

        return outputs

# CW: No seperate linear decoder for each time point
class lrcn_neureg_bn_static2_all_views(nn.Module):
    '''
    Differecne from lrcn_neureg_bn_static2:
    1. We use all views.
    '''

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

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
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer
        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.configs["_ln2_out"], 1) for t in range(len(self.configs['pred_timesteps']))
                ]) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

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

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))
                for t in range(len(self.configs['pred_timesteps'])):
                    outputs[sess_mask, t, p] = self.linear_list3[s][p][t](x_sub2task[:, t, :]).view(-1)

        return outputs


class lrcn_neureg_bn_static2_lstm_dec(nn.Module):
    '''
    Differecne from lrcn_neureg_bn_static2:
    1. We have an LSTM decoder at the very top layer instead of time-specific decoder. This could potentially learn temporal smoothness
    of the data.

    '''

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64
        if "_lstm_dec_hidden" not in self.configs.keys():
            self.configs["_lstm_dec_hidden"] = 256

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

        self.cnn = cnn_bn(self.configs)
        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen = (self.configs["image_shape"][0] // pool) * (self.configs["image_shape"][1] // pool) * \
                      self.configs["_cnn_channel_list"][-1]

        self.linear0 = nn.Linear(featureVlen, 2 * self.configs["_lstmHidden"])

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer
        self.linear_list2 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

        self.lstm_dec_list = nn.ModuleList([
            nn.ModuleList([
                nn.LSTM(input_size=self.configs["_ln2_out"],
                        hidden_size=self.configs["_lstm_dec_hidden"],
                        batch_first=True,
                        bidirectional=True
                        ) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(2 * self.configs["_lstm_dec_hidden"], 1) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

    def forward(self, x, sess_inds):

        bs, _, _, h, w = x.size()
        seq_length = len(self.configs['pred_timesteps'])

        x = x[:, self.input_timesteps].view(-1, self.n_input_frames, h, w)

        x = self.cnn(x)
        x = x.view(bs, seq_length, -1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        x = F.relu(self.linear0(x))

        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))  # (bs for a given session, T, _ln2_out)

                bs_sess = len(torch.nonzero(sess_mask))

                self.lstm_dec_list[s][p].flatten_parameters()

                lstm_output, (_, _) = self.lstm_dec_list[s][p](x_sub2task)
                outputs[sess_mask, :, p] = self.linear_list3[s][p](lstm_output)[..., 0]

        return outputs


class lrcn_neureg_bn_static2_lstm_dec2(nn.Module):
    '''
    Differecne from lrcn_neureg_bn_static2_lstm_dec:
    1. We learn the lstm initial states.

    '''

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64
        if "_lstm_dec_hidden" not in self.configs.keys():
            self.configs["_lstm_dec_hidden"] = 256

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

        self.cnn = cnn_bn(self.configs)
        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen = (self.configs["image_shape"][0] // pool) * (self.configs["image_shape"][1] // pool) * \
                      self.configs["_cnn_channel_list"][-1]

        self.linear0 = nn.Linear(featureVlen, 2 * self.configs["_lstmHidden"])

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer
        self.linear_list2 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

        self.lstm_dec_list = nn.ModuleList([
            nn.ModuleList([
                nn.LSTM(input_size=self.configs["_ln2_out"],
                        hidden_size=self.configs["_lstm_dec_hidden"],
                        num_layers=1,
                        batch_first=True,
                        bidirectional=True
                        ) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

        self.lstm_h0_list = nn.ModuleList([
            nn.ParameterList([
                Parameter(torch.Tensor(2 * self.configs['lstm_layer_num'], self.configs['_lstm_dec_hidden']))
                for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

        self.lstm_c0_list = nn.ModuleList([
            nn.ParameterList([
                Parameter(torch.Tensor(2 * self.configs['lstm_layer_num'], self.configs['_lstm_dec_hidden']))
                for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(2 * self.configs["_lstm_dec_hidden"], 1) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

        self.init_params()

    def init_params(self):
        for s in range(self.configs['n_sess']):
            for p in range(self.configs['n_pred_comp']):
                init.constant_(self.lstm_h0_list[s][p], 0.0)
                init.constant_(self.lstm_c0_list[s][p], 0.0)

    def forward(self, x, sess_inds):

        bs, _, _, h, w = x.size()
        seq_length = len(self.configs['pred_timesteps'])

        x = x[:, self.input_timesteps].view(-1, self.n_input_frames, h, w)

        x = self.cnn(x)
        x = x.view(bs, seq_length, -1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        x = F.relu(self.linear0(x))

        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))  # (bs for a given session, T, _ln2_out)

                bs_sess = len(torch.nonzero(sess_mask))

                self.lstm_dec_list[s][p].flatten_parameters()

                lstm_h0 = self.lstm_h0_list[s][p].unsqueeze(1).expand(-1, bs_sess, -1).contiguous()
                lstm_c0 = self.lstm_c0_list[s][p].unsqueeze(1).expand(-1, bs_sess, -1).contiguous()

                lstm_output, (_, _) = self.lstm_dec_list[s][p](x_sub2task, (lstm_h0, lstm_c0))
                outputs[sess_mask, :, p] = self.linear_list3[s][p](lstm_output)[..., 0]

        return outputs


class t_smooth_loss(nn.Module):
    def __init__(self, configs, x_avg_dict, v_std_dict):
        super().__init__()
        self.configs = configs

        # x_avg_dict[s][i] is an array of shape (len(self.configs['pred_timesteps']), self.configs['n_pred_comp']).
        # v_std_dict[s][i] is an array of shape (len(self.configs['pred_timesteps'])-1, self.configs['n_pred_comp']).
        self.x_avg_dict = x_avg_dict
        self.v_std_dict = v_std_dict

    def forward(self, model_outputs, sess_inds, trial_type):
        bs = model_outputs.size(0)

        # Compute v_penalty = max(|v-v_avg|- thr, 0).
        n_trial_types_list = list(range(2))

        device = model_outputs.device

        v_penalty = model_outputs.new_zeros([bs, len(self.configs['pred_timesteps']) - 1, self.configs['n_pred_comp']])
        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            for i in n_trial_types_list:
                tt_mask = (trial_type == i)
                sess_tt_mask = sess_mask * tt_mask

                if len(torch.nonzero(sess_tt_mask)) == 0:
                    continue

                cur_outputs = model_outputs[sess_tt_mask]

                v = cur_outputs[:, 1:] - cur_outputs[:, :-1]
                v_avg = torch.tensor(self.x_avg_dict[s][i][1:] - self.x_avg_dict[s][i][:-1]).to(device)
                thr = torch.tensor(2 * self.v_std_dict[s][i]).to(device)
                v_penalty[sess_tt_mask] = F.relu(torch.abs(v - v_avg) - thr)

        return v_penalty.mean()


class lrcn_neureg_bn_static2_bn_all(nn.Module):
    '''
    Differecne from lrcn_neureg_bn_static2:
    1. We use bn at every layer (but let's keep dropout).

    '''

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

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

        self.cnn = cnn_bn(self.configs)
        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen = (self.configs["image_shape"][0] // pool) * (self.configs["image_shape"][1] // pool) * \
                      self.configs["_cnn_channel_list"][-1]

        self.linear0 = nn.Linear(featureVlen, 2 * self.configs["_lstmHidden"])

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer
        self.linear_list2 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer
        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.configs["_ln2_out"], 1) for t in range(len(self.configs['pred_timesteps']))
                ]) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

        self.bn_list1 = nn.ModuleList([
            nn.BatchNorm1d(self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer
        self.bn_list2 = nn.ModuleList([
            nn.ModuleList([
                nn.BatchNorm1d(self.configs["_ln2_out"]) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

    def forward(self, x, sess_inds):

        bs, _, _, h, w = x.size()
        seq_length = len(self.configs['pred_timesteps'])

        x = x[:, self.input_timesteps].view(-1, self.n_input_frames, h, w)

        x = self.cnn(x)
        x = x.view(bs, seq_length, -1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        x = F.relu(self.linear0(x))

        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask])).view(-1, self.configs['_ln1_out'])

            x_sub1task = self.bn_list1[s](x_sub1task).view(-1, len(self.configs['pred_timesteps']),
                                                           self.configs['_ln1_out'])

            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task)).view(-1, self.configs['_ln2_out'])
                x_sub2task = self.bn_list2[s][p](x_sub2task).view(-1, len(self.configs['pred_timesteps']),
                                                                  self.configs['_ln2_out'])

                for t in range(len(self.configs['pred_timesteps'])):
                    outputs[sess_mask, t, p] = self.linear_list3[s][p][t](x_sub2task[:, t, :]).view(-1)

        return outputs


class lrcn_neureg_bn_static2_time_indp_dec(nn.Module):
    '''
    Differecne from lrcn_neureg_bn_static2:
    1. We don't have a separate linear decoder for each time point.

    '''

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

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

        self.cnn = cnn_bn(self.configs)
        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen = (self.configs["image_shape"][0] // pool) * (self.configs["image_shape"][1] // pool) * \
                      self.configs["_cnn_channel_list"][-1]

        self.linear0 = nn.Linear(featureVlen, 2 * self.configs["_lstmHidden"])

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])
        self.linear_list2 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])

        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln2_out"], 1) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])

    def forward(self, x, sess_inds):

        bs, _, _, h, w = x.size()
        seq_length = len(self.configs['pred_timesteps'])

        x = x[:, self.input_timesteps].view(-1, self.n_input_frames, h, w)

        x = self.cnn(x)
        x = x.view(bs, seq_length, -1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        x = F.relu(self.linear0(x))

        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))

                outputs[sess_mask, :, p] = self.linear_list3[s][p](x_sub2task)[
                    ..., 0]  # [...,0] to squeeze the singleton dimension.

        return outputs


class lrcn_neureg_bn_static2_downsample(nn.Module):
    '''
    Differecne from lrcn_neureg_bn_static2:
    1. We assume that the input frames are not yet downsampled from the orignial ones.
    Instead we perform one more max pooling. Really the only difference in terms of codes is
    assert statements in the beginning of __init__().

    '''

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        assert self.configs['image_shape'][
                   0] == 266  # This is the height after cropping: h - (int(h/6)+1), where h = 320.
        assert self.configs['image_shape'][1] == 400  # This is the original width.
        assert len(self.configs['_maxPoolLayers']) == 3

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

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

        self.cnn = cnn_bn(self.configs)
        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen = (self.configs["image_shape"][0] // pool) * (self.configs["image_shape"][1] // pool) * \
                      self.configs["_cnn_channel_list"][-1]

        self.linear0 = nn.Linear(featureVlen, 2 * self.configs["_lstmHidden"])

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer
        self.linear_list2 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer
        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.configs["_ln2_out"], 1) for t in range(len(self.configs['pred_timesteps']))
                ]) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

    def forward(self, x, sess_inds):

        bs, _, _, h, w = x.size()
        seq_length = len(self.configs['pred_timesteps'])

        x = x[:, self.input_timesteps].view(-1, self.n_input_frames, h, w)

        x = self.cnn(x)
        x = x.view(bs, seq_length, -1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        x = F.relu(self.linear0(x))

        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))
                for t in range(len(self.configs['pred_timesteps'])):
                    outputs[sess_mask, t, p] = self.linear_list3[s][p][t](x_sub2task[:, t, :]).view(-1)

        return outputs

# CW: This is the original model
class lrcn_neureg_bn_static2_all_views_time_indp_dec(nn.Module):
    '''
    Difference from lrcn_neureg_bn_static2_all_views:
    1. We don't have a separate linear decoder for each time point.
    '''

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

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
            nn.ModuleList([
                nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for p in
                range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer
        self.linear_list3 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.configs["_ln2_out"], 1) for p in range(self.configs['n_pred_comp'])
            ]) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

    def forward(self, x_side, x_bottom, sess_inds, pred_timesteps=None):

        n_views = 2
        x_list = [x_side, x_bottom]

        '''
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        
        Updated for when pred_timesteps is given. Begin.
        '''
        if pred_timesteps is None:
            seq_length = len(self.configs['pred_timesteps'])

        else:
            seq_length = len(pred_timesteps)

            # n_pre and n_post_frames = 4 covers +-0.2 s around the bin center (assuming that skip_frame = 10).
            n_pre_frames = (self.n_input_frames - 1) // 2
            n_post_frames = (self.n_input_frames - 1) // 2

            if (self.n_input_frames - 1) % 2 != 0:
                n_pre_frames += 1

            input_timesteps = []
            for k in pred_timesteps:
                input_timesteps.append(np.arange(k - n_pre_frames, k + n_post_frames + 1, 1))

            self.input_timesteps = cat(input_timesteps, 0)

        '''
        Updated for when pred_timesteps is given. End.
        '''

        x_all_views_list = []
        for k in range(n_views):
            cur_x = x_list[k] # side view or bottom view
            bs, _, c, h, w = cur_x.size()

            x = cur_x[:, self.input_timesteps].view(-1, self.n_input_frames, h, w)
            x = self.cnn_list[k](x)
            x = x.view(bs, seq_length, -1)
            x = F.dropout(x, p=self.configs['dropout'], training=self.training)
            x_all_views_list.append(x)

        x = torch.cat(x_all_views_list, 2)  # (bs, seq_length, n_views*feaureVlen)

        x = F.relu(self.linear0(x))

        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        '''
        Updated for when pred_timesteps is given. Begin.
        '''

        # TODO: Outputs is the step that will modify what the network produces
        if pred_timesteps is None:
            outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

        else:
            outputs = x.new_zeros([bs, len(pred_timesteps), self.configs['n_pred_comp']])
        '''
        Updated for when pred_timesteps is given. End.
        '''

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            for p in range(self.configs['n_pred_comp']):
                x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))

                outputs[sess_mask, :, p] = self.linear_list3[s][p](x_sub2task)[
                    ..., 0]  # [...,0] to squeeze the singleton dimension.

        return outputs

    '''
    Before I modified forward function to accomodate cd_reg_all_no_stim_and_pert_trials_diff_timesteps.
    The modified version above is still backward-compatible.
    '''

    # def forward(self, x_side, x_bottom, sess_inds):

    #     n_views = 2
    #     x_list = [x_side, x_bottom]

    #     seq_length = len(self.configs['pred_timesteps'])

    #     x_all_views_list = []
    #     for k in range(n_views):
    #         cur_x = x_list[k]
    #         bs, _, c, h ,w = cur_x.size()

    #         x = cur_x[:,self.input_timesteps].view(-1, self.n_input_frames, h, w)
    #         x = self.cnn_list[k](x)
    #         x = x.view(bs,seq_length,-1)
    #         x = F.dropout(x, p=self.configs['dropout'], training=self.training)
    #         x_all_views_list.append(x)

    #     x = torch.cat(x_all_views_list, 2) # (bs, seq_length, n_views*feaureVlen)

    #     x = F.relu(self.linear0(x))

    #     x = F.dropout(x, p=self.configs['dropout'], training=self.training)

    #     outputs = x.new_zeros([bs, len(self.configs['pred_timesteps']), self.configs['n_pred_comp']])

    #     for s in range(self.configs['n_sess']):
    #         sess_mask = (sess_inds == s)

    #         if len(torch.nonzero(sess_mask)) == 0:
    #             continue

    #         x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
    #         for p in range(self.configs['n_pred_comp']):
    #             x_sub2task = F.relu(self.linear_list2[s][p](x_sub1task))

    #             outputs[sess_mask,:,p] = self.linear_list3[s][p](x_sub2task)[...,0] # [...,0] to squeeze the singleton dimension.

    #     return outputs


class lrcn_choice_pred_bn_static2_all_views_time_indp_dec_sess_cond(nn.Module):
    '''
    Difference from lrcn_neureg_bn_static2_all_views_time_indp_dec:
    1. Aside from the obvious differences due to predicting binary choice,
    we don't have two layers at the top corresponding to prediction of different activity components.
    '''

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

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
            nn.Linear(self.configs["_lstmOut"], 1) for s in range(self.configs['n_sess'])
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

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps'])])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            outputs[sess_mask] = self.linear_list1[s](x[sess_mask])[
                ..., 0]  # (bs_for_sess, seq_length). [...,0] to squeeze the singleton dimension.

        return outputs


class lrcn_choice_pred_bn_static2_all_views_time_indp_dec_sess_cond_v2(nn.Module):
    '''
    Difference from lrcn_choice_pred_bn_static2_all_views_time_indp_dec_sess_cond:
    1. We have the same set of sess-dependent layers as in the cd-reg model.
    2. We have dropout layers after all FC layers, because it's easier to overfit for binary prediction.
    '''

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

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
        ])  # Different time steps have different weights on this layer
        self.linear_list3 = nn.ModuleList([
            nn.Linear(self.configs["_ln2_out"], 1) for s in range(self.configs['n_sess'])
        ])  # Different time steps have different weights on this layer

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

        outputs = x.new_zeros([bs, len(self.configs['pred_timesteps'])])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            sub_x = F.relu(self.linear_list1[s](x[sess_mask]))
            sub_x = F.dropout(sub_x, p=self.configs['dropout'], training=self.training)
            sub_x = F.relu(self.linear_list2[s](sub_x))
            sub_x = F.dropout(sub_x, p=self.configs['dropout'], training=self.training)
            outputs[sess_mask] = self.linear_list3[s](sub_x)[
                ..., 0]  # (bs_for_sess, seq_length). [...,0] to squeeze the singleton dimension.

        return outputs


class lrcn_choice_pred_bn_rnn_single_pred_all_views_sess_cond(nn.Module):
    '''
    Difference from lrcn_choice_pred_bn_static2_all_views_time_indp_dec_sess_cond:
    1. We don't make prediction at every time point. Rather we make a single prediction
    for each trial.
    2. We use bi-directional LSTM to integrate information across time.
    '''

    def __init__(self, configs):

        super().__init__()
        self.configs = configs

        # We don't want multiple frames as in cd_reg.
        assert self.configs['_input_channel'] == 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        self.cnn_list = nn.ModuleList([cnn_bn(self.configs) for k in range(len(self.configs['view_type']))])

        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen_side = (self.configs["image_shape_side"][0] // pool) * (
                self.configs["image_shape_side"][1] // pool) * self.configs["_cnn_channel_list"][-1]
        featureVlen_bottom = (self.configs["image_shape_bottom"][0] // pool) * (
                self.configs["image_shape_bottom"][1] // pool) * self.configs["_cnn_channel_list"][-1]

        # This lstm replaces linear0 of lrcn_choice_pred_bn_static2_all_views_time_indp_dec_sess_cond.
        self.lstm = nn.LSTM(
            input_size=featureVlen_side + featureVlen_bottom,
            hidden_size=self.configs["_lstmHidden"],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], 1) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer

    def forward(self, x_side, x_bottom, sess_inds):

        n_views = 2
        x_list = [x_side, x_bottom]

        x_all_views_list = []
        for k in range(n_views):
            x = x_list[k]
            bs, seq_length, c, h, w = x.size()
            x = x.view(-1, c, h, w)
            x = self.cnn_list[k](x)
            x = x.view(bs, seq_length, -1)
            x = F.dropout(x, p=self.configs['dropout'], training=self.training)
            x_all_views_list.append(x)

        x = torch.cat(x_all_views_list, 2)  # (bs, seq_length, featureVlen_side + featureVlen_bottom)

        '''
        Below replaces
        x = F.relu(self.linear0(x))
        in lrcn_choice_pred_bn_static2_all_views_time_indp_dec_sess_cond.
        '''
        self.lstm.flatten_parameters()
        x, (_, _) = self.lstm(x)

        x = x.view(bs, seq_length, 2, self.configs["_lstmHidden"])
        x = torch.cat((x[:, -1, 0, :], x[:, 0, 1, :]), dim=1)  # (bs, _lstmOut)
        '''
        '''

        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        # Note that we are making a single prediction for each trial, not for each timestep.
        outputs = x.new_zeros([bs])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            outputs[sess_mask] = self.linear_list1[s](x[sess_mask])[
                ..., 0]  # (bs_for_sess. [...,0] to squeeze the singleton dimension.

        return outputs


class lrcn_choice_pred_bn_static2_all_views_time_indp_dec(nn.Module):
    '''
    Main differences from lrcn_choice_pred_bn_static2_all_views_time_indp_dec_sess_cond:
    1. No sess-dependent layers.
    '''

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

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

        self.linear1 = nn.Linear(self.configs["_lstmOut"], 1)

    def forward(self, x_side, x_bottom):

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

        outputs = self.linear1(x)[..., 0]  # (bs_for_sess, seq_length). [...,0] to squeeze the singleton dimension.

        return outputs


class lrcn_bn_all_views_coupreg(nn.Module):
    '''
    Same as lrcn_neureg_bn_all_views, except that we predict coupling.
    '''

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if not 'lstm_layer_num' in self.configs.keys():
            self.configs['lstm_layer_num'] = 1

        if not '_lstmHidden' in self.configs.keys():
            self.configs["_lstmHidden"] = 256

        if "_ln1_out" not in self.configs.keys():
            self.configs["_ln1_out"] = 128
        if "_ln2_out" not in self.configs.keys():
            self.configs["_ln2_out"] = 64

        # Note that cnn modules don't care about the image shape.
        self.cnn_list = nn.ModuleList([cnn_bn(self.configs) for k in range(len(self.configs['view_type']))])

        pool = 2 ** len(self.configs["_maxPoolLayers"])
        featureVlen_side = (self.configs["image_shape_side"][0] // pool) * (
                self.configs["image_shape_side"][1] // pool) * self.configs["_cnn_channel_list"][-1]
        featureVlen_bottom = (self.configs["image_shape_bottom"][0] // pool) * (
                self.configs["image_shape_bottom"][1] // pool) * self.configs["_cnn_channel_list"][-1]

        self.lstm = nn.LSTM(
            input_size=featureVlen_side + featureVlen_bottom,
            hidden_size=self.configs["_lstmHidden"],
            num_layers=self.configs['lstm_layer_num'],
            batch_first=True,
            bidirectional=True
        )

        self.configs["_lstmOut"] = 2 * self.configs["_lstmHidden"]  # factor of 2 because it's bidirectional.

        self.linear_list1 = nn.ModuleList([
            nn.Linear(self.configs["_lstmOut"], self.configs["_ln1_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer

        self.linear_list2 = nn.ModuleList([
            nn.Linear(self.configs["_ln1_out"], self.configs["_ln2_out"]) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer

        self.linear_list3 = nn.ModuleList([
            nn.Linear(self.configs["_ln2_out"], 1) for s in range(self.configs['n_sess'])
        ])  # Different sessions have different weights on this layer

    def forward(self, x_side, x_bottom, sess_inds):

        n_views = 2
        x_list = [x_side, x_bottom]

        x_all_views_list = []
        for k in range(n_views):
            cur_x = x_list[k]
            bs, seq_length, c, h, w = cur_x.size()
            x = cur_x.view(-1, c, h, w)
            x = self.cnn_list[k](x)
            x = x.view(bs, seq_length, -1)
            x = F.dropout(x, p=self.configs['dropout'], training=self.training)
            x_all_views_list.append(x)

        x = torch.cat(x_all_views_list, 2)

        self.lstm.flatten_parameters()  # for multi-gpu using
        x, (_, _) = self.lstm(x)

        x = x.view(bs, seq_length, 2, self.configs["_lstmHidden"])
        x = torch.cat((x[:, -1, 0, :], x[:, 0, 1, :]), dim=1)
        x = F.dropout(x, p=self.configs['dropout'], training=self.training)

        outputs = x.new_zeros([bs, ])

        for s in range(self.configs['n_sess']):
            sess_mask = (sess_inds == s)

            if len(torch.nonzero(sess_mask)) == 0:
                continue

            x_sub1task = F.relu(self.linear_list1[s](x[sess_mask]))
            x_sub2task = F.relu(self.linear_list2[s](x_sub1task))
            outputs[sess_mask] = self.linear_list3[s](x_sub2task).view(-1)

        return outputs


class cd_cor_mse_loss(nn.Module):
    def __init__(self, configs, cd_trial_avg):
        super().__init__()
        self.configs = configs
        self.register_buffer('cd_trial_avg', torch.tensor(cd_trial_avg).float())  # (n_sess, n_hemi, T)

    def compute_cd_cor(self, cd_L, cd_R):
        '''
        Args:
        cd_L, cd_R: (n_trials, T)
        Return:
        cd_cor: (n_trials)
        '''
        cd_L = (cd_L - cd_L.mean(1, keepdim=True)) / cd_L.std(1, keepdim=True, unbiased=False)
        cd_R = (cd_R - cd_R.mean(1, keepdim=True)) / cd_R.std(1, keepdim=True, unbiased=False)

        cd_cor = (cd_L * cd_R).mean(1)

        return cd_cor

    def forward(self, pred, y, sess_inds):
        '''
        pred, y: (bs, T, n_comp)
        sess_inds: (bs)
        '''

        # calculate cd_cor from pred.
        cd_L_pred = pred[..., 0]
        if self.configs['n_pred_comp'] % 2 == 0:
            cd_R_pred = pred[..., self.configs['n_pred_comp'] // 2]
        else:
            cd_R_pred = pred[..., self.configs['n_pred_comp'] // 2 + 1]

        # add back cd_trial_avg

        cd_L_pred = cd_L_pred + self.cd_trial_avg[sess_inds][:, 0]
        cd_R_pred = cd_R_pred + self.cd_trial_avg[sess_inds][:, 1]

        cd_cor_pred = self.compute_cd_cor(cd_L_pred, cd_R_pred)

        # calculate cd_cor from y.
        cd_L_y = y[..., 0]
        if self.configs['n_pred_comp'] % 2 == 0:
            cd_R_y = y[..., self.configs['n_pred_comp'] // 2]
        else:
            cd_R_y = y[..., self.configs['n_pred_comp'] // 2 + 1]

        # add back cd_trial_avg

        cd_L_y = cd_L_y + self.cd_trial_avg[sess_inds][:, 0]
        cd_R_y = cd_R_y + self.cd_trial_avg[sess_inds][:, 1]

        cd_cor_y = self.compute_cd_cor(cd_L_y, cd_R_y)

        return (cd_cor_pred - cd_cor_y).pow(2).mean()
