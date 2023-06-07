import numpy as np
import scipy.stats as spst
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import copy as cp
from numbers import Number

import warnings

from .session import Session
from . import decoders as ld
from . import plottools as ptools
from . import utils


class PopulationAnalysis(object):
    '''
    endTrialIdx is exclusive
    '''

    def __init__(self, neural_data, cull=False, cull_trial_start_idx=None, cull_trial_end_idx=None):

        if not isinstance(neural_data, Session):
            raise ValueError("neural_data must be of type Session.")

        self.input_data = neural_data
        self.culled_data = None
        self.culled_trial_mask = None
        self.culled_unit_mask = None

        self.label_by_report = None

        self.train_stim_type = None
        self.train_stim_type = None
        self.train_data = None
        self.test_data = None
        self.train_label = None
        self.test_label = None
        self.train_test_same_stim_type = None
        self.perform_CV = None

        self.cum_train_trial_mask = None
        self.cum_test_trial_mask = None
        self.cum_unit_mask = None

        self.bin_width = None
        self.stride = None
        self.begin_time = None
        self.end_time = None

        self.train_rates = None
        self.test_rates = None
        self.bin_centers = None
        self.n_bins = None

        self.classifier_cls = None
        self.classifiers = None

        # Cull data if asked
        if cull:
            self.cull_data(cull_trial_start_idx, cull_trial_end_idx)
        #if prepare:
        #    self.prepare_data(train_stim_type, test_stim_type)

    def score_input_trial_range(self, alpha = np.arange(0.01,0.2,0.01), start=None, end=None, print_all=True, print_best=False):
        return self.input_data.score_trial_range(alpha, start, end, print_all=print_all, print_best=print_best)

    def __apply_mask_to_cum(self, mask, is_trial=True, is_train=True):
        if is_trial:
            if is_train:
                self.cum_train_trial_mask[self.cum_train_trial_mask] = \
                    np.all([self.cum_train_trial_mask[self.cum_train_trial_mask], mask], axis=0)
            else:
                self.cum_test_trial_mask[self.cum_test_trial_mask] = \
                    np.all([self.cum_test_trial_mask[self.cum_test_trial_mask], mask], axis=0)
        else: #unit
            self.cum_unit_mask[self.cum_unit_mask] = \
                np.all([self.cum_unit_mask[self.cum_unit_mask], mask], axis=0)

    def is_data_culled(self):
        return self.culled_data is not None

    def cull_data(self, stt_tr_idx=None, end_tr_idx=None, verbose=True):
        '''
        For trial ranges start:end (exclusive), retrieve only units that are intact for all trials.
        Also change all other contents of data to be only start:end
        '''
        self.culled_data = self.input_data.deepcopy()
        (self.culled_trial_mask, self.culled_unit_mask) = \
                    self.culled_data.cull_data(stt_tr_idx, end_tr_idx, verbose)
        return

    def are_train_and_test_loaded(self):
        return self.train_data is not None and self.test_data is not None

    def divide_train_to_train_and_test(self, train_proportion=0.6):
        # Copy train data to test data
        np.random.seed(123) #prev: 123
        self.test_data = self.train_data.deepcopy()
        self.cum_test_trial_mask = cp.deepcopy(self.cum_train_trial_mask)

        # Divide up the data set to train and test sets
        n_og_train = self.train_data.n_trials # number of og train set
        n_new_train = int(np.round(n_og_train*train_proportion))
        shuffled_og_train_idx = np.random.permutation(n_og_train)

        new_train_idx = shuffled_og_train_idx[0:n_new_train]
        new_test_idx = shuffled_og_train_idx[n_new_train:]

        # Create input -> new train trial mask, and apply mask
        og_train_2_new_train_mask = np.zeros(n_og_train, dtype=bool)
        og_train_2_new_train_mask[new_train_idx] = True
        self.filter_train_trials(og_train_2_new_train_mask)

        # Create input -> new test trial mask, and apply mask
        og_train_2_new_test_mask = np.zeros(n_og_train, dtype=bool)
        og_train_2_new_test_mask[new_test_idx] = True
        self.filter_test_trials(og_train_2_new_test_mask)

    def load_train_and_test(self, train_stim_type='no stim', test_stim_type=None, train_proportion=0.6, bootstrap=False, location=None):
        '''
        Divides up culled data to train and test data sets based on the indicated stim types.
        If train_stim_type and test_stim_type are the same or test_stim_type is None, then (1-train_proportion) of the training set becomes the test set, and train_proportion of the training set becomes the training set.
        train_proportion is relevant only if train_stim_type and test_stim_type are the same or test_stim_type is None.
        If train_proportion is None, then the same train and test sets are loaded. Later, prediction accuracy will be computed using cross validation.
        '''

        self.train_test_same_stim_type = \
                (test_stim_type is None) or (train_stim_type == test_stim_type)
        if test_stim_type is None:
            test_stim_type = train_stim_type

        self.train_stim_type = train_stim_type
        self.test_stim_type = test_stim_type

        self.perform_CV = \
                self.train_test_same_stim_type and train_proportion is None

        self.cum_train_trial_mask = np.ones(self.input_data.n_trials,dtype=bool)
        self.cum_test_trial_mask = np.ones(self.input_data.n_trials, dtype=bool)
        self.cum_unit_mask = np.ones(self.input_data.n_units, dtype=bool)

        # 1. Make sure that data is culled
        assert self.is_data_culled

        # 2. Apply culled mask to the train and test trial and unit cum masks
        self.__apply_mask_to_cum(self.culled_trial_mask, True, True)
        self.__apply_mask_to_cum(self.culled_trial_mask, True, False)
        self.__apply_mask_to_cum(self.culled_unit_mask, False)
        
        



        # Select train data based on stim type, and modify trial mask
        self.train_data = self.culled_data.deepcopy()
        if train_stim_type == 'uni':
            print('UNI TRIALS')
            stim_train_trial_mask = \
                self.train_data.select_trials(self.train_data.stim_site % 3 != 0)

            self.__apply_mask_to_cum(stim_train_trial_mask, True, True)
        elif train_stim_type is not None:
            stim_train_trial_mask = \
                self.train_data.select_trials_w_stim_type_str(train_stim_type)
            self.__apply_mask_to_cum(stim_train_trial_mask, True, True)

        
        # If specified, filter units by recording location
        if location is not None:
            location_unit_mask = self.train_data.select_units_w_location(location)
            #print('heyyy', location_unit_mask)
            self.__apply_mask_to_cum(location_unit_mask, False)

        # If train and test sets have different stim types, then just load them
        if self.perform_CV or not self.train_test_same_stim_type:
            # Select test data based on stim type, and modify trial mask
            self.test_data = self.culled_data.deepcopy()
            stim_test_trial_mask = \
                    self.test_data.select_trials_w_stim_type_str(test_stim_type)
            self.__apply_mask_to_cum(stim_test_trial_mask, True, False)

        else: #If train and test stim types are the same, then divide up train
            self.divide_train_to_train_and_test(train_proportion)

        if bootstrap:
            self.select_random_train_trials(self.train_data.n_trials, True)
            self.select_random_test_trials(self.test_data.n_trials, True)

        assert self.are_train_and_test_loaded()
        return

    def select_random_train_trials(self, n, replace=False):
        '''
        Randomly selects n train trials
        '''
        if not replace and self.train_data.n_trials < n:
            raise ValueError('If not replacing, n must not be larger than the existing training data set size.')
        mask = np.zeros(self.train_data.n_trials, dtype=bool)
        n_idxes = np.random.choice(self.train_data.n_trials, n, replace=replace)

        if not replace:
            mask[n_idxes] = True
            self.filter_train_trials(mask)
        else: #replace
            self.train_data.select_trials(n_idxes, False)
            mask[np.unique(n_idxes)] = True
            self.__apply_mask_to_cum(mask, True, True) # NOTE: This is currently slightly incorrect!! because mask has to be boolean at the moment, it can't handle redrawn trial indices.

    def select_random_test_trials(self, n, replace=False):
        '''
        Randomly selects n test trials
        '''
        if not replace and self.test_data.n_trials < n:
            raise ValueError('If not replacing, n must not be larger than the existing test data set size.')
        mask = np.zeros(self.test_data.n_trials, dtype=bool)
        n_idxes = np.random.choice(self.test_data.n_trials, n, replace=replace)

        if not replace:
            mask[n_idxes] = True
            self.filter_test_trials(mask)
        else: #replace
            self.test_data.select_trials(n_idxes, False)
            mask[np.unique(n_idxes)] = True
            self.__apply_mask_to_cum(mask, True, False) # NOTE: This is currently slightly incorrect!! because mask has to be boolean at the moment, it can't handle redrawn trial indices.

    def filter_train_trials(self, train_to_filter_mask):
        '''
        train_to_filter_mask must be of length self.train_data.n_trials
        '''
        train_to_filter_mask = np.asarray(train_to_filter_mask)
        if train_to_filter_mask.dtype == bool:
            assert len(train_to_filter_mask) == self.train_data.n_trials
        else:
            raise ValueError('train_to_filter_mask type must be bool')

        applied_mask = self.train_data.select_trials(train_to_filter_mask)
        self.__apply_mask_to_cum(applied_mask, True, True)

    def filter_test_trials(self, test_to_filter_mask):
        '''
        test_to_filter_mask must be of length self.test_data.n_trials
        '''
        test_to_filter_mask = np.asarray(test_to_filter_mask)
        if test_to_filter_mask.dtype == bool:
            assert len(test_to_filter_mask) == self.test_data.n_trials
        else:
            raise ValueError('test_to_filter_mask type must be bool')

        applied_mask = self.test_data.select_trials(test_to_filter_mask)
        self.__apply_mask_to_cum(applied_mask, True, False)

    def filter_units(self, culled_to_filter_mask):
        '''
        test_to_filter_mask must be of length self.test_data.n_trials
        '''
        culled_to_filter_mask = np.asarray(culled_to_filter_mask)
        if culled_to_filter_mask.dtype == bool:
            assert len(culled_to_filter_mask) == self.culled_data.n_units
        else:
            raise ValueError('culled_to_filter_mask type must be bool')

        applied_mask_train = self.train_data.select_units(culled_to_filter_mask)
        applied_mask_test = self.test_data.select_units(culled_to_filter_mask)
        assert applied_mask_train == applied_mask_test
        self.__apply_mask_to_cum(applied_mask_train, False)

    def filter_train_trials_from_input(self, input_to_filter_mask):
        '''
        input_to_filter_mask must be of length self.input_data.n_trials
        '''
        input_to_filter_mask = np.asarray(input_to_filter_mask)
        if input_to_filter_mask.dtype == bool:
            assert len(input_to_filter_mask) == self.input_data.n_trials
        else:
            raise ValueError('input_to_filter_mask type must be bool')

        masked_mask = input_to_filter_mask[self.cum_train_trial_mask]
        self.filter_train_trials(masked_mask)

    def filter_test_trials_from_input(self, input_to_filter_mask):
        '''
        mask must be of length self.input_data.n_trials
        '''
        mask = np.asarray(input_to_filter_mask)
        if input_to_filter_mask.dtype == bool:
            assert len(input_to_filter_mask) == self.input_data.n_trials
        else:
            raise ValueError('input_to_filter_mask type must be bool')

        masked_mask = input_to_filter_mask[self.cum_test_trial_mask]
        self.filter_test_trials(masked_mask)

    def filter_units_from_input(self, input_to_filter_mask):
        '''
        mask must be of length self.input_data.n_units
        '''
        input_to_filter_mask = np.asarray(input_to_filter_mask)
        if input_to_filter_mask.dtype == bool:
            assert len(input_to_filter_mask) == self.input_data.n_units
        else:
            raise ValueError('input_to_filter_mask type must be bool')

        masked_mask = input_to_filter_mask[self.cum_unit_mask]
        self.filter_units(masked_mask)

    def is_data_preprocessed(self):
        return self.bin_width is not None

    # Allows modification of modified_data (in place of culled_data) at will through a new method and of trials used for training and tests beyond stim type selection (utilizing unit_mask and trial_mask below). E.g. Training set could utilize only correct trials.
    def preprocess(self, align_to=None, begin_time=-4.2, end_time=2, bin_width=0.2, stride=0.1, label_by_report=True, even_train_classes=False, even_test_classes=False):
        '''
        if align_to is None, then align all times to task_cue_on_time.
        otherwise, align_to is of length self.input_data.n_trials
        begin_time and end_time set up the bounds on the time bins to compute firing rates in.

        '''
        # Check that input data is properly formed with asserts
        assert self.are_train_and_test_loaded()
        assert align_to is None or len(align_to) == self.input_data.n_trials

        # Copy parameters to class variables.
        self.bin_width = bin_width
        self.stride = stride
        self.begin_time = begin_time
        self.end_time = end_time
        self.label_by_report = label_by_report

        # Align time
        if align_to is None:
            self.train_data.align_time(None)
            self.test_data.align_time(None)
        else:
            self.train_data.align_time(align_to[self.cum_train_trial_mask])
            self.test_data.align_time(align_to[self.cum_test_trial_mask])

        # Create train and test labels as binary int forms
        if self.label_by_report:
            train_label_str = self.train_data.behavior_report_type
            test_label_str = self.test_data.behavior_report_type
        else:
            train_label_str = self.train_data.task_trial_type
            test_label_str = self.test_data.task_trial_type

        # If even number of samples per class is requested, trim the larger class (l/r)
        if even_train_classes:
            r_train_mask = train_label_str == 'r'
            l_train_mask = train_label_str == 'l'
            n_r_train = sum(r_train_mask)
            n_l_train = sum(l_train_mask)

            if n_r_train > n_l_train:
                r_train_mask_int = np.where(r_train_mask)[0]
                if r_train_mask_int.size != 0:
                    r_train_sampled_indices = \
                    np.random.choice(r_train_mask_int, n_l_train, replace=False)
                else:
                    r_train_sampled_indices = []
                r_train_mask_new = np.zeros(len(train_label_str),dtype=bool)
                r_train_mask_new[r_train_sampled_indices] = True
                l_train_mask_new = l_train_mask
            else:
                l_train_mask_int = np.where(l_train_mask)[0]
                if l_train_mask_int.size != 0:
                    l_train_sampled_indices = \
                    np.random.choice(l_train_mask_int, n_r_train, replace=False)
                else:
                    l_train_sampled_indices = []
                l_train_mask_new = np.zeros(len(train_label_str),dtype=bool)
                l_train_mask_new[l_train_sampled_indices] = True
                r_train_mask_new = r_train_mask

            train_mask_new = np.any([r_train_mask_new,l_train_mask_new], axis=0)
            self.filter_train_trials(train_mask_new)
            train_label_str = train_label_str[train_mask_new]

        # If even number of samples per class is requested, trim the larger class (l/r)
        if even_test_classes:
            r_test_mask = test_label_str == 'r'
            l_test_mask = test_label_str == 'l'
            n_r_test = sum(r_test_mask)
            n_l_test = sum(l_test_mask)

            if n_r_test > n_l_test:
                r_test_mask_int = np.where(r_test_mask)[0]
                if r_test_mask_int.size != 0:
                    r_test_sampled_indices = \
                    np.random.choice(r_test_mask_int, n_l_test, replace=False)
                else:
                    r_test_sampled_indices = []
                r_test_mask_new = np.zeros(len(test_label_str),dtype=bool)
                r_test_mask_new[r_test_sampled_indices] = True
                l_test_mask_new = l_test_mask
            else:
                l_test_mask_int = np.where(l_test_mask)[0]
                if l_test_mask_int.size != 0:
                    l_test_sampled_indices = \
                    np.random.choice(l_test_mask_int, n_r_test, replace=False)
                else:
                    l_test_sampled_indices = []
                l_test_mask_new = np.zeros(len(test_label_str),dtype=bool)
                l_test_mask_new[l_test_sampled_indices] = True
                r_test_mask_new = r_test_mask

            test_mask_new = np.any([r_test_mask_new,l_test_mask_new], axis=0)
            self.filter_test_trials(test_mask_new)
            test_label_str = test_label_str[test_mask_new]


        # Compute spike rate for each sliding bin
        self.bin_centers,self.train_rates = \
                utils.sliding_histogram(self.train_data.neural_data, begin_time+bin_width, end_time-bin_width, bin_width, stride, rate=True)
        _,self.test_rates = \
                utils.sliding_histogram(self.test_data.neural_data, begin_time+bin_width, end_time-bin_width, bin_width, stride, rate=True)

        # def sliding_histogram_bk(spikeTimes, begin_time, end_time, bin_width, stride, rate=True, bin_end_time_eps=1e-2):
        #     '''
        #     Calculates the number of spikes for each unit in each sliding bin of width bin_width, strided by stride.
        #     begin_time and end_time are treated as the lower- and upper- bounds for bin centers.
        #     '''
        #     # Calculate sliding bins for finding firing rates
        #     bin_begin_time = begin_time+bin_width/2
        #     bin_end_time = end_time-bin_width/2+bin_end_time_eps
        #     binCenters = np.arange(bin_begin_time, bin_end_time, stride)

        #     binIntervals = np.vstack((binCenters-bin_width/2, binCenters+bin_width/2)).T

        #     # Count number of spikes for each sliding bin
        #     binSpikes = np.asarray([[[np.sum(np.all([trial>=binInt[0], trial<binInt[1]], axis=0)) for binInt in binIntervals] for trial in unit] 
        #         for unit in spikeTimes]).swapaxes(0,-1)

        #     if rate:
        #         return (binCenters,binSpikes/float(bin_width))
        #     return (binCenters,binSpikes)

        # self.bin_centers,self.train_rates = \
        #         sliding_histogram_bk(self.train_data.neural_data, begin_time, end_time, bin_width, stride, rate=True)
        # _,self.test_rates = \
        #         sliding_histogram_bk(self.test_data.neural_data, begin_time, end_time, bin_width, stride, rate=True)


        #print('heyyy', self.test_rates)
        self.n_bins = np.size(self.bin_centers)

        if self.n_bins == 1:
            self.train_rates = np.array([self.train_rates])
            self.test_rates = np.array([self.test_rates])

        # Store label as 0s (l) and 1s (r)
        self.train_label = (train_label_str == 'r').astype(int)
        self.test_label = (test_label_str == 'r').astype(int)

    def plot_preprocessed_units(self, plot_test_set=True):
        if not self.is_data_preprocessed():
            warnings.warn("Data not preprocessed.", RuntimeWarning)
            return
        if plot_test_set:
            data_to_plot = self.test_data
        else:
            data_to_plot = self.train_data

        # Get the unit numbers (in the input data) for the culled units
        unit_numbers = np.where(self.culled_unit_mask)[0]
        for i in range(data_to_plot.n_units):
            data_to_plot.plot_unit(i, include_raster=False, label_by_report=self.label_by_report, title='Unit ' + str(unit_numbers[i]) + ', Culled unit ' + str(i))

    def compute_CD_for_each_bin(self,classifier_cls=[ld.MDC]):
        '''
        For every bin, the coding direction is computed using the classifier_cls of choice.
        Returns the list of classifiers; length = number of bins
        '''
        self.classifier_cls = classifier_cls
        self.classifiers = [[ccls(self.train_rates[bin_i], self.train_label) for bin_i in range(self.n_bins)] for ccls in classifier_cls]

        return self.classifiers

    def compute_pred_accuracy(self, bin_time, CD_time=0, classifier_idx=0):
        if self.classifiers is None:
            print("Coding directions were not computed. Doing so now.")
            _ = self.compute_CD_for_each_bin()

        if (CD_time is not None) and (not isinstance(CD_time, Number)):
            warnings.warn("CD_time must be a number or None. Proceeding as None.",RuntimeWarning)
            CD_time = None

        if CD_time is None:
            CD_time = bin_time

        # find bin right before CD_time
        temp = self.bin_centers-CD_time
        CD_bin_idx = np.argmax(temp[temp<=0])

        # find bin right before proj_bin_time
        temp = self.bin_centers-bin_time
        bin_idx = np.argmax(temp[temp<=0])

        # get CD classifier for the CD index
        classifier = self.classifiers[classifier_idx][CD_bin_idx]

        test_rates_bin = self.test_rates[bin_idx]
        accuracy = classifier.evaluate(test_rates_bin, self.test_label)[0]

        return accuracy
        ## Loop above through multiple bootstraps... hmm


    # TODO: store the trained classifiers
    def plot_pred_accuracy(self, CD_time=None, CV=10):
        '''
        If CD_time is None, then each time bin has its own classifier based on the corresponding time bin in the training data set.
        If CD_time is not None, it must be a number, indicating the time (in seconds) from cue. Then, there is one classifier based on the time bin closest to that indicated time.
        '''

        if self.classifiers is None:
            print("Coding directions were not computed. Doing so now.")
            _ = self.compute_CD_for_each_bin()

        if (CD_time is not None) and (not isinstance(CD_time, Number)):
            warnings.warn("CD_time must be a number or None. Proceeding as None.",RuntimeWarning)
            CD_time = None

        if CD_time is not None:
            # find bin right before CDtime
            temp = self.bin_centers-CD_time
            CD_bin_idx = np.argmax(temp[temp<0])

        #prepare plot and check jitter
        plt.figure()
        utils.prepare_plot_events(self.test_data, verbose=True)

        acc_bins = np.empty(self.n_bins)
        for c_idx,bin_classifiers in enumerate(self.classifiers):

            # One classifier for all time bins.
            if CD_time is not None:
                # Create CD classifier based on the indicated bin (CDtime)
                classifier = bin_classifiers[CD_bin_idx]

                for bin_i in range(self.n_bins):
                    test_rates_bin = self.test_rates[bin_i]
                    acc_bins[bin_i] = classifier.evaluate(test_rates_bin, self.test_label)[0]

            # One classifier for each time bin.
            else: # CD_time is None:
                for bin_i in range(self.n_bins):
                    if self.perform_CV:
                        train_rates_bin = self.train_rates[bin_i]
                        # Holdout 10-fold CV
                        acc_bins[bin_i] = \
                            ld.utils.evaluate_CV(train_rates_bin,self.train_label,self.classifier_cls[c_idx],CV)
                    else:
                        test_rates_bin = self.test_rates[bin_i]
                        classifier = bin_classifiers[bin_i]
                        acc_bins[bin_i] = classifier.evaluate(test_rates_bin, self.test_label)[0]

            plt.plot(self.bin_centers,acc_bins,label=self.classifier_cls[c_idx].__name__)

        if CD_time is None and self.perform_CV:
            plt.ylabel(str(CV) + '-fold CV accuracy')
        else:
            plt.ylabel('Test accuracy')
        plt.xlabel('Time from cue [s]')
        plt.legend(loc="best")
        if CD_time is not None:
            plt.title('Pred accuracy, CD from ' + str(CD_time) + ' s')
        else:
            plt.title('Pred accuracy, CD from each bin')
        plt.show()

    def CD_projection_one_bin(self, proj_bin_time=0, CD_time=None, subtract_mean=True, use_test_trials=True, plot=True, verbose=True):

        if self.classifiers is None:
            print("Coding directions were not computed. Doing so now.")
            _ = self.compute_CD_for_each_bin()

        if (CD_time is not None) and (not isinstance(CD_time, Number)):
            warnings.warn("CD_time must be a number or None. Proceeding as None.",RuntimeWarning)
            CD_time = None

        if CD_time is None:
            CD_time = proj_bin_time

        if use_test_trials:
            rates = self.test_rates
            label = self.test_label
        else:
            rates = self.train_rates
            label = self.train_label


        # find bin right before CD_time
        temp = self.bin_centers-CD_time
        CD_bin_idx = np.argmax(temp[temp<=0])

        # find bin right before proj_bin_time
        temp = self.bin_centers-proj_bin_time
        proj_bin_idx = np.argmax(temp[temp<=0])

        if verbose:
            print ("# Right trials = ", sum(label))
            print ("# Left trials = ", sum(1-label))

        trial_projections = [None]*len(self.classifiers)
        trial_types = [None]*len(self.classifiers)
        behavior_correct = [None]*len(self.classifiers)
        for c_idx,bin_classifiers in enumerate(self.classifiers):
            trial_projections[c_idx] = \
                                    bin_classifiers[CD_bin_idx].project_to_W(\
                                    rates[proj_bin_idx], subtract_mean)

            trial_types[c_idx] = self.test_data.task_trial_type
            behavior_correct[c_idx] = self.test_data.behavior_report

            right_projections = trial_projections[c_idx][label.astype(bool)]
            left_projections = trial_projections[c_idx][~label.astype(bool)]

            if plot:
                fig = plt.figure()
                #ax = fig.add_subplot(111)
                plt.hist(right_projections, label='r', histtype='step')
                plt.hist(left_projections, label='l', histtype='step')
                if subtract_mean:
                    mean_plotted = 0
                else:
                    mean_plotted = np.mean(bin_classifiers[CD_bin_idx].mean)
                plt.axvline(x=mean_plotted, color='k', linestyle='--', linewidth=1)
                plt.xlabel('Projection (a.u.)')
                plt.ylabel('# Trials')
                plt.legend(loc="best")
                plt.title(self.classifier_cls[c_idx].__name__)
                plt.show()

        return trial_projections,trial_types,behavior_correct

    # TODO: store the trained classifier
    def CD_projection(self, CD_time=None, subtract_mean=True):
        '''
        If CD_time is None, then each time bin has its own classifier based on the corresponding time bin in the training data set.
        If CD_time is not None, it must be a number, indicating the time (in seconds) from cue. Then, there is one classifier based on the time bin closest to that indicated time.
        '''

        if self.classifiers is None:
            print("Coding directions were not computed. Doing so now.")
            _ = self.compute_CD_for_each_bin()

        if (CD_time is not None) and (not isinstance(CD_time, Number)):
            warnings.warn("CD_time must be a number or None. Proceeding as None.",RuntimeWarning)
            CD_time = None

        if CD_time is not None:
            # find bin right before CDtime
            temp = self.bin_centers-CD_time
            CD_bin_idx = np.argmax(temp[temp<0])

        for c_idx,bin_classifiers in enumerate(self.classifiers):
            # One classifier for all time bins.
            if CD_time is not None:
                # Create CD classifier based on the indicated bin (CDtime)
                CD_classifier = bin_classifiers[CD_bin_idx]

                # Compute projections onto CD for each trial
                # bins x trials
                trial_projections = \
                        np.empty((self.n_bins,self.test_rates.shape[1]))
                for bin_i in range(self.n_bins):
                    test_bin_rate = self.test_rates[bin_i]
                    trial_projections[bin_i,:] = \
                        CD_classifier.project_to_W(test_bin_rate, subtract_mean)
                    #each trial gets a scalar projection value.

            # One classifier for each time bin.
            else: # CD_time is None:
                # Compute projections onto CD for each trial
                # bins x trials
                trial_projections = \
                        np.empty((self.n_bins,self.test_rates.shape[1]))
                for bin_i in range(self.n_bins):
                    test_rates_bin = self.test_rates[bin_i]
                    CD_classifier = bin_classifiers[bin_i]
                    trial_projections[bin_i,:] = \
                        CD_classifier.project_to_W(test_rates_bin, subtract_mean)
                    #each trial gets a scalar projection value.

            right_projections = trial_projections[:,self.test_label.astype(bool)]
            right_proj_mean = np.mean(right_projections, axis=1)
            right_proj_sem = spst.sem(right_projections,axis=1)

            left_projections = trial_projections[:,np.invert(self.test_label.astype(bool))]
            left_proj_mean = np.mean(left_projections, axis=1)
            left_proj_sem = spst.sem(left_projections,axis=1)

            # plot
            #prepare plot and check jitter
            fig = plt.figure()
            ax = fig.add_subplot(111)
            colors = ptools.generate_color_arrays(2)
            utils.prepare_plot_events(self.test_data, ax=ax, verbose=True)
            ptools.plot_w_error(y=right_proj_mean,x=self.bin_centers,ye=right_proj_sem,color=colors[0],legend='r',show_legend=False,ax=ax)
            ptools.plot_w_error(y=left_proj_mean,x=self.bin_centers,ye=left_proj_sem,color=colors[1],legend='l',show_legend=False,ax=ax)

            #plt.plot(binCenters,right_proj_mean, color=colors[0], label='r'); plt.fill_between(binCenters,right_proj_mean-right_proj_sem,right_proj_mean+right_proj_sem, color=errorColors[0])
            #plt.plot(binCenters,left_proj_mean, color=colors[1], label='l'); plt.fill_between(binCenters,left_proj_mean-left_proj_sem,left_proj_mean+left_proj_sem, color=errorColors[1])
            plt.ylabel('Projection (a.u.)')
            plt.xlabel('Time from cue [s]')
            plt.legend(loc="best")
            plt.title(self.classifier_cls[c_idx].__name__)
            plt.show()

    def W_dot_matrix(self, verbose=False):

        if self.classifiers is None:
            print("Coding directions were not computed. Doing so now.")
            _ = self.compute_CD_for_each_bin()

        if verbose:
            utils.check_event_jitter(self.train_data)

        for c_idx,bin_classifiers in enumerate(self.classifiers):

            W_dots = np.empty((self.n_bins,self.n_bins))
            for i in range(self.n_bins):
                for j in range(self.n_bins):
                    W_i = bin_classifiers[i].W
                    W_j = bin_classifiers[j].W
                    W_dots[i,j] = np.sum(W_i/np.linalg.norm(W_i) * W_j/np.linalg.norm(W_j))

            bin_centers_str = ['%.1f' % bc for bc in self.bin_centers]
            W_dots_df = pd.DataFrame(W_dots, columns=bin_centers_str, index=bin_centers_str)

            ax = plt.gca()
            sns.heatmap(W_dots_df,ax=ax)
            pole_on_idx = np.argmin(np.abs(self.bin_centers - np.mean(self.train_data.task_pole_on_time)))
            pole_off_idx = np.argmin(np.abs(self.bin_centers - np.mean(self.train_data.task_pole_off_time)))
            cue_on_idx = np.argmin(np.abs(self.bin_centers - np.mean(self.train_data.task_cue_on_time)))
            ax.axvline(x=pole_on_idx, color='k', linestyle='--', linewidth=2)
            ax.axvline(x=pole_off_idx, color='k', linestyle='--', linewidth=2)
            ax.axvline(x=cue_on_idx, color='k', linestyle='--', linewidth=2)
            ax.axhline(y=pole_on_idx, color='k', linestyle='--', linewidth=2)
            ax.axhline(y=pole_off_idx, color='k', linestyle='--', linewidth=2)
            ax.axhline(y=cue_on_idx, color='k', linestyle='--', linewidth=2)
            plt.ylabel('Time from cue [s]')
            plt.xlabel('Time from cue [s]')
            plt.title(self.classifier_cls[c_idx].__name__)
            plt.show()

    def do_everything(self, train_stim_type='no stim', test_stim_type=None, train_proportion=0.6, only_correct=False, label_by_report=True, even_classes=True, classifier_cls=[ld.MDC,ld.DFDAC,ld.FDAC], CD_time=None):
        self.cull_data()
        self.load_train_and_test(train_stim_type, test_stim_type, train_proportion)

        if only_correct:
            correct_mask = self.input_data.behavior_report == 1
            self.filter_train_trials_from_input(correct_mask)
            self.filter_test_trials_from_input(correct_mask)

        self.preprocess(label_by_report=label_by_report, even_classes=even_classes)
        _ = self.compute_CD_for_each_bin(classifier_cls=classifier_cls)

        self.plot_preprocessed_units(True)

        self.pred_accuracy(CD_time=None)

        #self.W_dot_matrix()

        self.CD_projection_one_bin(proj_bin_time=0,CD_time=CD_time)

        # Cull data to stt_tr_idx:end_tr_idx and to units that are held
        #if self.is_data_culled():
        #    data_culled_here = False #indicate that data was culled beforehand
        #else: #data was not already culled, and must be culled here.
        #    self.cull_data(stt_tr_idx, end_tr_idx)
        #    if align_to is not None:
        #        align_to = align_to[self.culled_trial_mask]
        #    data_culled_here = True #indicate that data was culled in this func

        self.CD_projection(CD_time=CD_time, subtract_mean=True)

        return
