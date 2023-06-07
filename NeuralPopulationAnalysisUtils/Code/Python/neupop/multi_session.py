import os
import numpy as np
import scipy.stats as spst
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import warnings

from .session import Session
from .pop_analysis import PopulationAnalysis
from . import decoders
from . import plottools as ptools
from . import utils

#Session.get_unique_stim_periods_str()
#Session.get_unique_stim_sites_str()

class MultiSessionAnalysis(object):

    def __init__(self, sessions, remove_early_report=True, remove_ignored=True, verbose=True):
        '''
        remove_early_report and remove_ignored are ignored if sessions is a list of Session objects
        '''
        self.sessions = []
        self.n_sessions = None
        self.verbose = verbose

        self._load_sessions(sessions, remove_early_report, remove_ignored)

    def _load_sessions(self, sessions, remove_early_report, remove_ignored):
        # If sessions is (dir_name, session_type)
        if isinstance(sessions, tuple):
            print('Loading input sessions by directory.')
            self._load_sessions_by_directory(sessions, remove_early_report, remove_ignored)
        # If all list items are (file_name, session_type)
        elif isinstance(sessions, list) and \
                all([isinstance(s, tuple) for s in sessions]):
            print('Loading input sessions by file names.')
            self._load_sessions_by_filename(sessions, remove_early_report, remove_ignored)
        # If all list items are Sessions, then load them as Sessions
        elif isinstance(sessions, list) and \
                all([isinstance(s, Session) for s in sessions]):
            print('Loading input sessions.')
            self._load_sessions_by_Sessions(sessions)
        else:
            raise ValueError('sessions must be of form [Session object 1, Session object 2, ...], [(session_file_name_1, session_type_1), (session_file_name_2, session_type_2), ...], or (directory_name, session_type).')

    def _load_sessions_by_Sessions(self, sessions):
        '''
        sessions is a list of Session objects
        '''
        self.sessions = sessions
        self.n_sessions = len(sessions)

    def _load_sessions_by_filename(self, sessions, remove_early_report, remove_ignored):
        if not all([(len(s)==2 and \
                    isinstance(s[0],str) and \
                    isinstance(s[1],str)) \
                for s in sessions]):
            raise ValueError('If sessions is a list of tuples, the tuples must be of form (file_name, session_type).')

        self._load_sessions_by_Sessions([Session(sf, st, remove_early_report, remove_ignored, self.verbose) for (sf,st) in sessions])

    def _load_sessions_by_directory(self, sessions, remove_early_report, remove_ignored):
        if len(sessions)!=2 or \
                not isinstance(sessions[0],str) or \
                not isinstance(sessions[1],str):
            raise ValueError('If sessions is a tuple, it must be of form (directory_name, session_type).')

        dir_name, session_type = sessions

        # Add '/' at the end if absent
        if dir_name[-1] != '/':
            dir_name = dir_name + '/'

        # Get list of files in directory
        try:
            fnames = os.listdir(dir_name)
            fnames.sort()
        except ValueError:
            print('The directory path is not valid')

        # collect .mat file names
        fnames = [dir_name + f for f in fnames if f[-4:]=='.mat']

        # Create (fname, session_type) tuple list
        sessions = [(fname, session_type) for fname in fnames]
        self._load_sessions_by_filename(sessions, remove_early_report, remove_ignored)

    def plot_all_CD_projections(self, train_stim_type, test_stim_type, train_proportion=0.5, CD_time=0, time_bin_width=0.4, label_by_report=True, correct_trials_only=False, even_train_classes=False, even_test_classes=False, classifier_cls=decoders.MDC, verbose=False):
        for si in range(self.n_sessions):
            if verbose:
                print('\n###Processing session '+str(si+1)+'/'+str(self.n_sessions))
            s = self.sessions[si]
            # Create two PopulationAnalysis objects for each of the train_stim_types.
            pa = PopulationAnalysis(s)
            pa.cull_data(verbose=verbose)
            pa.load_train_and_test(train_stim_type=train_stim_type, test_stim_type=test_stim_type, train_proportion=train_proportion)
            if correct_trials_only:
                pa.filter_train_trials_from_input(s.behavior_report == 1)
                pa.filter_test_trials_from_input(s.behavior_report == 1)
            pa.preprocess(bin_width=time_bin_width, label_by_report=label_by_report, even_train_classes=even_train_classes, even_test_classes=even_test_classes)

            if pa.train_data.n_trials == 0 or pa.test_data.n_trials ==0:
                if verbose:
                    print("Too few train or test trials survived. Skipping session.")
                continue

            _=pa.compute_CD_for_each_bin(classifier_cls=[classifier_cls])
            pa.CD_projection(CD_time=CD_time, subtract_mean=True)




    def projection_behavioral_performance(self, train_stim_type, test_stim_type, train_proportion=0.5, bin_time=0, time_bin_width=0.4, label_by_report=True, correct_trials_only=False, even_train_classes=False, even_test_classes=False, classifier_cls=decoders.MDC, n_proj_bins=10, bootstrap_threshold_per_bin=6, n_bootstrap=50, marker=None, verbose=True):
        all_trial_projections = []
        all_trial_types = []
        all_behavior_report = [] #1: correct, 0: incorrect, -1: ignored
        for si in range(self.n_sessions):
            if verbose:
                print('\n###Processing session '+str(si+1)+'/'+str(self.n_sessions))
            s = self.sessions[si]
            # Create two PopulationAnalysis objects for each of the train_stim_types.
            pa = PopulationAnalysis(s)
            pa.cull_data(verbose=verbose)
            pa.load_train_and_test(train_stim_type=train_stim_type, test_stim_type=test_stim_type, train_proportion=train_proportion)
            if correct_trials_only:
                pa.filter_train_trials_from_input(s.behavior_report == 1)
                pa.filter_test_trials_from_input(s.behavior_report == 1)
            pa.preprocess(begin_time=bin_time-time_bin_width, end_time=bin_time+time_bin_width, bin_width=time_bin_width, label_by_report=label_by_report, even_train_classes=even_train_classes, even_test_classes=even_test_classes)

            # NOTE: Fix this. it should check for ==0. This was a hack.
            if pa.train_data.n_trials <2 or pa.test_data.n_trials <2:
                if verbose:
                    print("Too few train or test trials survived. Skipping session.")
                continue

            _=pa.compute_CD_for_each_bin(classifier_cls=[classifier_cls])
            trial_projections,trial_types,behavior_report=pa.CD_projection_one_bin(proj_bin_time=bin_time, CD_time=None, use_test_trials=True, plot=False, verbose=verbose)
            all_trial_projections.append(trial_projections[0])
            all_trial_types.append(trial_types[0])
            all_behavior_report.append(behavior_report[0])

        all_trial_projections = np.concatenate(all_trial_projections).ravel()
        all_trial_types = np.concatenate(all_trial_types).ravel()
        all_behavior_report = np.concatenate(all_behavior_report).ravel()

        # Filter out nans in the projections and corresponding values
        no_nan_proj_idx = ~np.isnan(all_trial_projections)
        all_trial_projections = all_trial_projections[no_nan_proj_idx]
        all_trial_types = all_trial_types[no_nan_proj_idx]
        all_behavior_report = all_behavior_report[no_nan_proj_idx]

        assert(np.sum(all_behavior_report==-1) == 0)

        #print(np.max(all_trial_projections))
        #print(np.min(all_trial_projections))
        _,bin_edges = np.histogram(all_trial_projections, n_proj_bins)
        bin_assignments = np.digitize(all_trial_projections, bin_edges[1:-1])

        acc_mean_l = np.zeros(n_proj_bins)
        acc_sem_l = np.zeros(n_proj_bins)
        acc_mean_r = np.zeros(n_proj_bins)
        acc_sem_r = np.zeros(n_proj_bins)
        n_l_trials = np.zeros(n_proj_bins)
        n_r_trials = np.zeros(n_proj_bins)

        if verbose:
            print("Bootstrapping bins...")
        for i in range(n_proj_bins):
            indices_in_bin = bin_assignments == i
            bin_trial_projections = all_trial_projections[indices_in_bin]
            bin_trial_types = all_trial_types[indices_in_bin]
            bin_behavior_report = all_behavior_report[indices_in_bin]

            ## Left trials
            left_idx = bin_trial_types == 'l'
            bin_behavior_report_l = bin_behavior_report[left_idx]
            n_l_trials[i] = np.sum(left_idx)

            if n_l_trials[i] < bootstrap_threshold_per_bin: # too few trials for this bin. mark it as such
                acc_mean_l[i] = np.nan
                acc_sem_l[i] = np.nan
            else: # if there are enough trials in the bin, then bootstrap
                bootstrap_accuacy_l = np.zeros(n_bootstrap)
                for bi in range(n_bootstrap):
                    bootstrap_idx_l = np.random.choice(np.sum(left_idx), replace=True)
                    behavior_l_bs = bin_behavior_report_l[bootstrap_idx_l]
                    bootstrap_accuacy_l[bi] = np.mean(behavior_l_bs)
                acc_mean_l[i] = np.mean(bootstrap_accuacy_l)
                acc_sem_l[i] = spst.sem(bootstrap_accuacy_l)

            ## Right trials
            right_idx = bin_trial_types == 'r'
            bin_behavior_report_r = bin_behavior_report[right_idx]
            n_r_trials[i] = np.sum(right_idx)

            if n_r_trials[i] < bootstrap_threshold_per_bin: # too few trials for this bin. mark it as such
                acc_mean_r[i] = np.nan
                acc_sem_r[i] = np.nan
            else: # if there are enough trials in the bin, then bootstrap
                bootstrap_accuacy_r = np.zeros(n_bootstrap)
                for bi in range(n_bootstrap):
                    bootstrap_idx_r = np.random.choice(np.sum(right_idx), replace=True)
                    behavior_r_bs = bin_behavior_report_r[bootstrap_idx_r]
                    bootstrap_accuacy_r[bi] = np.mean(behavior_r_bs)
                acc_mean_r[i] = np.mean(bootstrap_accuacy_r)
                acc_sem_r[i] = spst.sem(bootstrap_accuacy_r)
        if verbose:
            print("Bootstrap complete!")

        bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2

        min_proj = bin_centers[min(np.where(~np.isnan(acc_mean_r))[0][0], np.where(np.isnan(acc_mean_l))[0][0])]
        max_proj = bin_centers[max(np.where(~np.isnan(acc_mean_r))[0][-1], np.where(np.isnan(acc_mean_l))[0][-1])]


        f, ax = plt.subplots(2,1, \
            gridspec_kw={'height_ratios':[2,1]},sharex=True,figsize=(8,6))
        ax[0].axvline(0, color='grey', linestyle='dashed')
        ax[0].axhline(0.5, color='grey', linestyle='dashed')
        ax[0]=ptools.plot_w_error([acc_mean_r,acc_mean_l],x=bin_centers,ye=[acc_sem_r,acc_sem_l],color=None, marker=marker, legend=['r','l'],ylabel='Behavioral performance', ax=ax[0])

        ax[1].axvline(0, color='grey', linestyle='dashed')
        colors = ptools.generate_color_arrays(2)
        ax[1].plot(bin_centers, n_r_trials, color=colors[0])
        ax[1].plot(bin_centers, n_l_trials, color=colors[1])
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[1].set_xlabel('Left <-- Projection (a.u.) --> Right')
        ax[1].set_ylabel('# Trials')

        f.subplots_adjust(hspace=0.001)
        plt.show()

        return (f,acc_mean_l,acc_sem_l,acc_mean_r,acc_sem_r,bin_edges)

    def compare_neural_predictability_and_behavior(self, train_stim_type, test_stim_type, classifier_cls=decoders.MDC, bin_time=0, bin_width=0.4, label_by_report=True, even_train_classes=False, even_test_classes=True, bootstrap_threshold=20, n_bootstrap=30, min_bootsrap=10, bootstrap_train_proportion=0.6, min_train_size=10, min_test_size=5, verbose=True):
        # Filter sessions that have all both stim types
        filtered_sessions = [s for s in self.sessions if \
                                s.exists_stim_type_str(train_stim_type) and \
                                s.exists_stim_type_str(test_stim_type)]

        if len(filtered_sessions) == 0:
            warnings.warn("There are no sessions with both indicated stim types.", RuntimeWarning)

        n_filtered_sessions = len(filtered_sessions)

        acc_mean = np.zeros(n_filtered_sessions)
        acc_sem = np.zeros(n_filtered_sessions)
        behavior_accuracy = np.zeros(n_filtered_sessions)
        sampling_method = np.zeros(n_filtered_sessions)

        for si in range(n_filtered_sessions):
            if verbose:
                print('\n###Processing session '+str(si+1)+'/'+str(n_filtered_sessions))
            s = filtered_sessions[si]
            # Create two PopulationAnalysis objects for each of the train_stim_types.
            pa = PopulationAnalysis(s)
            pa.cull_data(verbose=verbose)

            n_train_trials_max = \
                        pa.culled_data.num_stim_type_str(train_stim_type)
            n_test_trials_max = \
                        pa.culled_data.num_stim_type_str(test_stim_type)

            if n_train_trials_max < min_train_size:
                print("Too few train trials ("+ str(n_train_trials_max) +") survived the culling!")
                continue

            if n_test_trials_max < min_test_size:
                if verbose:
                    print("Too few test trials ("+ str(n_test_trials_max) +") survived the culling!")
                continue

            if train_stim_type == test_stim_type:
                # bootstrap or CV
                n_bootstrap_train = \
                  int(np.round(n_train_trials_max * bootstrap_train_proportion))

                # Bootstrap
                if n_train_trials_max >= bootstrap_threshold and n_bootstrap_train >= min_train_size:
                    if verbose:
                        print("Bootstrapping...")

                    acc = np.zeros(n_bootstrap)
                    # Use multiple bootstraps of train/test divisions (defined by train_proportion)
                    for i in range(n_bootstrap):
                        pa.load_train_and_test(train_stim_type=train_stim_type, test_stim_type=test_stim_type, train_proportion=bootstrap_train_proportion)
                        pa.preprocess(begin_time=bin_time-bin_width, end_time=bin_time+bin_width, bin_width=bin_width, label_by_report=label_by_report, even_train_classes=even_train_classes, even_test_classes=even_test_classes)

                        if verbose and i == 0:
                            print('# train trials: '+str(pa.train_data.n_trials))
                            print('# test trials: '+str(pa.test_data.n_trials))
                        if pa.train_data.n_trials < min_train_size or pa.train_data.n_trials == 0 or pa.test_data.n_trials < min_test_size or pa.test_data.n_trials == 0:
                            if verbose:
                                print("Too few training or test trials survived. Skipping this bootstrap.")
                            acc[i] = np.nan
                            continue

                        _=pa.compute_CD_for_each_bin(classifier_cls=[classifier_cls])
                        acc[i] = pa.compute_pred_accuracy(bin_time=bin_time, CD_time=bin_time, classifier_idx=0)

                    # Compute mean performance and sem
                    acc_nonan = acc[~np.isnan(acc)]
                    if acc_nonan.size < min_bootsrap:
                        print("Skipping this session because not enough bootstrap samples were collected.")
                        continue
                    acc_mean[si] = np.mean(acc_nonan)
                    acc_sem[si] = spst.sem(acc_nonan)
                    behavior_accuracy[si] = np.sum(pa.test_data.behavior_report == 1) / np.sum(pa.test_data.behavior_report != -1)

                    sampling_method[si] = 1

                else:
                    if verbose:
                        print("Holdout CV...")
                    pa.load_train_and_test(train_stim_type=train_stim_type, test_stim_type=test_stim_type, train_proportion=1)
                    pa.preprocess(begin_time=bin_time-bin_width, end_time=bin_time+bin_width, bin_width=bin_width, label_by_report=label_by_report, even_train_classes=even_train_classes, even_test_classes=even_test_classes)

                    print('data set size: '+str(pa.train_data.n_trials))
                    if pa.train_data.n_trials < min_train_size or pa.train_data.n_trials == 0:
                        print("Too few or no trials survived. Skipping this holdout CV.")
                        continue

                    # find bin right before proj_bin_time
                    rates_bin = pa.train_rates[0]

                    # Hold-one-out CV
                    acc_mean[si] = decoders.utils.evaluate_CV(rates_bin, pa.train_label, classifier_cls, CV=len(pa.train_label))
                    # Use train_data for behavioral accuracy since all trials of train_data is used as test data
                    behavior_accuracy[si] = np.sum(pa.train_data.behavior_report == 1) / np.sum(pa.train_data.behavior_report != -1)

                    sampling_method[si] = 2

            else:
                if verbose:
                    print("Bootstrap train and test separately")
                # Note: a little bit of a hack right now.
                pa.load_train_and_test(train_stim_type=train_stim_type, test_stim_type=test_stim_type, bootstrap=False)
                train_test_s = pa.train_data.deepcopy()
                train_test_s.merge_sessions(pa.test_data)

                acc = np.zeros(n_bootstrap)
                for i in range(n_bootstrap):
                    train_test_s_copy = train_test_s.deepcopy()
                    train_test_s_copy.select_random_trials(train_test_s_copy.n_trials, replace=True)
                    pa = PopulationAnalysis(train_test_s_copy)
                    pa.cull_data(verbose=False)
                    pa.load_train_and_test(train_stim_type=train_stim_type, test_stim_type=test_stim_type, bootstrap=False)
                    pa.preprocess(begin_time=bin_time-bin_width, end_time=bin_time+bin_width, bin_width=bin_width, label_by_report=label_by_report, even_train_classes=even_train_classes, even_test_classes=even_test_classes)

                    #if i==0:
                    if verbose:
                        print('# test trials: '+ str(pa.test_data.n_trials))
                        print('# train trials: '+ str(pa.train_data.n_trials))
                    if pa.train_data.n_trials < min_train_size or pa.test_data.n_trials < min_test_size:
                        if verbose:
                            print("Too few training or test trials survived. Skipping this bootstrap.")
                        acc[i] = np.nan
                        continue

                    _=pa.compute_CD_for_each_bin(classifier_cls=[classifier_cls])
                    acc[i] = pa.compute_pred_accuracy(bin_time=bin_time, CD_time=bin_time, classifier_idx=0)

                acc_nonan = acc[~np.isnan(acc)]
                if acc_nonan.size < min_bootsrap:
                    if verbose:
                        print("Skipping this session because not enough bootstrap samples were collected.")
                    continue

                acc_mean[si] = np.mean(acc_nonan)
                acc_sem[si] = spst.sem(acc_nonan)
                behavior_accuracy[si] = np.sum(pa.test_data.behavior_report == 1) / np.sum(pa.test_data.behavior_report != -1)
                sampling_method[si] = 3

        bootstrap_indices = sampling_method == 1
        holdoutCV_indices = sampling_method == 2
        separate_test_indices = sampling_method == 3

        skipped_indices = sampling_method == 0
        print(str(np.sum(skipped_indices)) + ' sessions were skipped')

        plt.figure(figsize=(6,4))
        if train_stim_type == test_stim_type:
            plt.errorbar(acc_mean[bootstrap_indices], behavior_accuracy[bootstrap_indices], xerr=acc_sem[bootstrap_indices], fmt='o', markersize='5')
            plt.scatter(acc_mean[holdoutCV_indices], behavior_accuracy[holdoutCV_indices], c='orange')
            plt.legend(['Hold-out CV', 'Bootstrap (w/ sem)'])
        else:
            plt.errorbar(acc_mean[separate_test_indices], behavior_accuracy[separate_test_indices], xerr=acc_sem[separate_test_indices], fmt='o', markersize='5')
            plt.legend(['Separate train / test trials'])
        plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
        plt.axvline(0.5, color='grey', linestyle='dashed')
        plt.axhline(0.5, color='grey', linestyle='dashed')
        plt.xlabel('Neural decoder performance')
        plt.ylabel('Behavioral performance')
        plt.show()

        return (acc_mean, acc_sem, behavior_accuracy)

    def compare_decoders(self, train_stim_type_1, train_stim_type_2, test_stim_type, classifier_cls=decoders.MDC, bin_time=0, bin_width=0.4, label_by_report=True, even_train_classes=False, even_test_classes=True, bootstrap_threshold=20, n_bootstrap=30, min_bootsrap=10, bootstrap_train_proportion=0.5, min_train_size=10, min_test_size=6, even_train_trials=True, verbose=True):
        # Filter sessions that have all three stim types
        filtered_sessions = [s for s in self.sessions if \
                                s.exists_stim_type_str(train_stim_type_1) and \
                                s.exists_stim_type_str(train_stim_type_2) and \
                                s.exists_stim_type_str(test_stim_type)]

        if len(filtered_sessions) == 0:
            warnings.warn("There are no sessions with all three indicated stim types.", RuntimeWarning)

        # if test_stim_type equals either of the train_stim_types, then
        #  force train_stim_type_1 to be equal to test_stim_type.
        train_test_overlaps = False
        if test_stim_type == train_stim_type_1:
            train_test_overlaps = True
            train_test_stim_type = train_stim_type_1
            other_train_stim_type = train_stim_type_2
        elif test_stim_type == train_stim_type_2:
            train_test_overlaps = True
            train_test_stim_type = train_stim_type_2
            other_train_stim_type = train_stim_type_1

        if not train_test_overlaps:
            raise NotImplementedError("Test stim type not equalling either of the train stim type has not been implemented.")

        n_filtered_sessions = len(filtered_sessions)

        acc_pa1_mean = np.zeros(n_filtered_sessions)
        acc_pa1_sem = np.zeros(n_filtered_sessions)
        acc_pa2_mean = np.zeros(n_filtered_sessions)
        acc_pa2_sem = np.zeros(n_filtered_sessions)


        sampling_method = np.zeros(n_filtered_sessions) #0: skipped, 1: bootstrapped, 2: leave-one-out CV

        # For each filtered session:
        for si in range(n_filtered_sessions):
            if verbose:
                print('\n###Processing session '+str(si+1)+'/'+str(n_filtered_sessions))
            s = filtered_sessions[si]
            # Create two PopulationAnalysis objects for each of the train_stim_types.
            pa1 = PopulationAnalysis(s)
            pa1.cull_data(verbose=verbose)
            pa2 = PopulationAnalysis(s)
            pa2.cull_data(verbose=False)

            n_pa1_train_test_trials = \
                        pa1.culled_data.num_stim_type_str(train_test_stim_type)
            n_pa2_train_trials = \
                        pa2.culled_data.num_stim_type_str(other_train_stim_type)

            if n_pa1_train_test_trials < min_train_size:
                if verbose:
                    print("Too few pa1 train trials ("+ str(n_pa1_train_test_trials) +") survived the culling!")
                continue
            if n_pa2_train_trials < min_train_size:
                if verbose:
                    print("Too few pa2 train trials ("+ str(n_pa2_train_trials) +") survived the culling!")
                continue

            n_pa1_bootstrap_train = \
              int(np.round(n_pa1_train_test_trials*bootstrap_train_proportion))

            n_min_train = min(n_pa1_bootstrap_train, n_pa2_train_trials)

            # If pa1 data set is sufficiently large:
            if n_pa1_train_test_trials >= bootstrap_threshold and n_pa1_bootstrap_train >= min_train_size:
                if verbose:
                    print("Bootstrapping...")

                acc_pa1 = np.zeros(n_bootstrap)
                acc_pa2 = np.zeros(n_bootstrap)
                # Use multiple bootstraps of train/test divisions (defined by train_proportion)
                for i in range(n_bootstrap):
                    pa1.load_train_and_test(train_stim_type=train_test_stim_type, test_stim_type=train_test_stim_type, train_proportion=bootstrap_train_proportion, bootstrap=False)

                    if even_train_trials and n_pa1_bootstrap_train > n_min_train:
                        #Filter random n_min_train train trials for pa1 to match the number of train trials between pa1 and pa2
                        pa1.select_random_train_trials(n_min_train, replace=False)

                    pa1.preprocess(begin_time=bin_time-bin_width, end_time=bin_time+bin_width, bin_width=bin_width, label_by_report=label_by_report, even_train_classes=even_train_classes, even_test_classes=even_test_classes)

                    if i == 0 and verbose:
                        print('test set size: '+str(pa1.test_data.n_trials))
                        print('pa1 train set size: '+str(pa1.train_data.n_trials))
                    if pa1.train_data.n_trials < min_train_size or pa1.train_data.n_trials == 0:
                        if verbose:
                            print("Too few pa1 training trials survived. Skipping this bootstrap.")
                        acc_pa1[i] = np.nan
                        acc_pa2[i] = np.nan
                        continue
                    if pa1.test_data.n_trials < min_test_size or pa1.test_data.n_trials == 0:
                        if verbose:
                            print('test set size: '+str(pa1.test_data.n_trials))
                            print("Too few pa1 test trials survived. Skipping this bootstrap. Consider increasing bootstrap_threshold.")
                        acc_pa1[i] = np.nan
                        acc_pa2[i] = np.nan
                        continue

                    _=pa1.compute_CD_for_each_bin(classifier_cls=[classifier_cls])
                    acc_pa1[i] = pa1.compute_pred_accuracy(bin_time=bin_time, CD_time=bin_time, classifier_idx=0)

                    pa1_test_mask = pa1.cum_test_trial_mask

                    # Evaluate test set data on the train_2 classifier
                    # Question: Should we equalize the number of the classifiers' training data set?
                    pa2.load_train_and_test(train_stim_type=other_train_stim_type, test_stim_type=train_test_stim_type)

                    if even_train_trials and n_pa2_train_trials > n_min_train:
                        #Filter random n_min_train train trials for pa2
                        pa2.select_random_train_trials(n_min_train, replace=False)

                    pa2.filter_test_trials_from_input(pa1_test_mask)
                    pa2.preprocess(begin_time=bin_time-bin_width, end_time=bin_time+bin_width, bin_width=bin_width, label_by_report=label_by_report, even_train_classes=even_train_classes, even_test_classes=even_test_classes)
                    if i==0 and verbose:
                        print('pa2 train set size: '+str(pa2.train_data.n_trials))
                    if pa2.train_data.n_trials < min_train_size or pa2.train_data.n_trials == 0:
                        if verbose:
                            print("Too few pa2 training trials survived. Skipping this bootstrap.")
                        acc_pa1[i] = np.nan
                        acc_pa2[i] = np.nan
                        continue

                    _=pa2.compute_CD_for_each_bin(classifier_cls=[classifier_cls])
                    acc_pa2[i] = pa2.compute_pred_accuracy(bin_time=bin_time, CD_time=bin_time, classifier_idx=0)

                # Compute mean performance and sem
                acc_pa1_nonan = acc_pa1[~np.isnan(acc_pa1)]
                acc_pa2_nonan = acc_pa2[~np.isnan(acc_pa2)]
                if acc_pa1_nonan.size < min_bootsrap or acc_pa2_nonan.size < min_bootsrap:
                    if verbose:
                        print("Skipping this session because not enough bootstrap samples were collected.")
                    continue

                acc_pa1_mean[si] = np.mean(acc_pa1_nonan)
                acc_pa1_sem[si] = spst.sem(acc_pa1_nonan)
                acc_pa2_mean[si] = np.mean(acc_pa2_nonan)
                acc_pa2_sem[si] = spst.sem(acc_pa2_nonan)

                sampling_method[si] = 1

            # Else (if pa1 data set is not large enough):
            else:
                if verbose:
                    print("Holdout CV...")
                # Use leave-one-out CV for prediction accuracy prediction
                # Evaluate test set data on the train_2 classifier
                # Compute mean performance

                pa1.load_train_and_test(train_stim_type=train_test_stim_type, test_stim_type=train_test_stim_type, train_proportion=1)
                pa1.preprocess(begin_time=bin_time-bin_width, end_time=bin_time+bin_width, bin_width=bin_width, label_by_report=label_by_report, even_train_classes=even_train_classes, even_test_classes=even_test_classes)

                if verbose:
                    print('pa1 data set size: '+str(pa1.train_data.n_trials))
                if pa1.train_data.n_trials < min_train_size or pa1.train_data.n_trials == 0:
                    print("Too few or no training trials survived. Skipping this holdout CV.")
                    continue

                # NOTE: I currently fix CD_time to equal bin_time
                # find bin right before CD_time
                #temp = pa1.bin_centers-bin_time
                #CD_bin_idx = np.argmax(temp[temp<=bin_time])
                CD_bin_idx = 0

                # find bin right before proj_bin_time
                #temp = pa1.bin_centers-bin_time
                #bin_idx_1 = np.argmax(temp[temp<=bin_time])
                #rates_bin_pa1 = pa1.train_rates[bin_idx_1]
                rates_bin_pa1 = pa1.train_rates[0]

                pa2.load_train_and_test(train_stim_type=other_train_stim_type, test_stim_type=train_test_stim_type)
                pa2.preprocess(begin_time=bin_time-bin_width, end_time=bin_time+bin_width, bin_width=bin_width, label_by_report=label_by_report, even_train_classes=even_train_classes, even_test_classes=even_test_classes)
                if verbose:
                    print('pa2 whole train set size: '+str(pa2.train_data.n_trials))
                if pa2.train_data.n_trials < min_train_size or pa2.train_data.n_trials == 0:
                    if verbose:
                        print("Too few or no training trials survived. Skipping this holdout CV.")
                    continue
                if even_train_trials and verbose:
                    print('pa2 sampled train set size: '+str(pa1.train_data.n_trials - 1))

                # find bin right before proj_bin_time
                #temp = pa2.bin_centers-bin_time
                #bin_idx_2 = np.argmax(temp[temp<=bin_time])
                #rates_bin_pa2 = pa2.train_rates[bin_idx_2]
                rates_bin_pa2 = pa2.train_rates[0]

                # Hold-one-out CV
                acc_pa1_mean[si], acc_pa2_mean[si] = decoders.utils.evaluate_CV_2(rates_bin_pa1, pa1.train_label, rates_bin_pa2, pa2.train_label, classifier_cls, classifier_cls, even_train_trials=even_train_trials, CV=len(pa1.train_label))

                sampling_method[si] = 2

            if verbose:
                print (pa1.train_stim_type + ' acc mean: '+str(acc_pa1_mean[si]))
                print (pa1.train_stim_type + ' acc sem: '+str(acc_pa1_sem[si]))
                print (pa2.train_stim_type + ' acc mean: '+str(acc_pa2_mean[si]))
                print (pa2.train_stim_type + ' acc sem: '+str(acc_pa2_sem[si]))

        bootstrap_indices = sampling_method == 1
        holdoutCV_indices = sampling_method == 2

        skipped_indices = sampling_method == 0
        if verbose:
            print(str(np.sum(skipped_indices)) + ' sessions were skipped')

        plt.figure(figsize=(6,4))
        plt.errorbar(acc_pa1_mean[bootstrap_indices], acc_pa2_mean[bootstrap_indices], xerr=acc_pa1_sem[bootstrap_indices], yerr=acc_pa2_sem[bootstrap_indices], fmt='o', markersize='5')
        plt.scatter(acc_pa1_mean[holdoutCV_indices], acc_pa2_mean[holdoutCV_indices], c='orange')
        plt.legend(['Hold-out CV', 'Bootstrap (w/ sem)'])
        plt.plot([0, 1], [0, 1], color='grey', linestyle='dashed')
        plt.axvline(0.5, color='grey', linestyle='dashed')
        plt.axhline(0.5, color='grey', linestyle='dashed')
        plt.xlabel(train_test_stim_type + ' classifier')
        plt.ylabel(other_train_stim_type + ' classifier')
        plt.title('Prediction on ' + train_test_stim_type)
        plt.show()

        self.train_test_stim_type = train_test_stim_type
        self.other_train_stim_type = other_train_stim_type
        self.acc_pa1_mean = acc_pa1_mean
        self.acc_pa1_sem = acc_pa1_sem
        self.acc_pa2_mean = acc_pa2_mean
        self.acc_pa2_sem = acc_pa2_sem

        return (acc_pa1_mean, acc_pa1_sem, acc_pa2_mean, acc_pa2_sem)
