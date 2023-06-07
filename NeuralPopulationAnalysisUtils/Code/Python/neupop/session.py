import numpy as np
import scipy.stats as spst
import scipy.io as spio
import matplotlib.pyplot as plt
from matplotlib import gridspec
import copy as cp
import warnings

from . import plottools as ptools
from . import utils

from .session_parsers import ParserNuoLiALM, ParserNuoLiDualALM_states

class Session(object):
    """
    Class for storing and manipulating spike-sorted neural data.

    Parameters
    ----------
    file_name: string
        Name of the file containing the session data to parse

    neural_data: 2d numpy array of shape (n_units, n_trials)
        Each entry is a list containing spike times.

    neural_unit_depth: 1d numpy array of length n_units, dtype=float
        Depth in um (micrometers)

    neural_unit_type: 1d numpy array of length n_units, dtype=string
        Cell type (e.g. putative_pyramidal, putative_interneuron)

    neural_unit_location: 1d numpy array of length n_units, dtype=string
        Recording location (e.g. left_ALM, right_ALM)

    n_units: int
        Number of units in neural_data

    n_trials: int
        Number of trials in neural_data

    spike_present: 2d numpy array of shape (n_units, n_trials)
        Whether there is at least one spike for the unit in the trial.

    unit_held_table: 2d numpy array of shape (n_units, n_trials)
        Whether a unit was held in the trial.
        Usually inferred from spike_present.

    behavior_report: 1d numpy array of length (n_trials)
        Whether the animal responded correctly.
        It is coded as follows:
            -1: ignored
            0 : incorrect
            1 : correct

    behavior_report_type: 1d numpy array of length (n_trials).
        Whether the animal responded with lick right ('r') or left ('l').
        It is coded as follows:
            'i': ignored
            'r': licked right
            'l': licked left

    behavior_early_report: 1d numpy array of length (n_trials).
        Whether the animal improperly responded before cue.
        It is coded as follows:
            0: no early report (proper behavior)
            1: early report (improper behavior)

    task_trial_type: 1d numpy array of length (n_trials).
        Whether the animal was cued to lick right ('r') or left ('l').
        It is coded as follows:
            'r': cued right
            'l': cued left

    task_pole_on_time: 1d numpy array of length (n_trials).
    task_pole_off_time: 1d numpy array of length (n_trials).
        When during the trial the pole came on and off.

    task_cue_on_time: 1d numpy array of length (n_trials).
    task_cue_off_time: 1d numpy array of length (n_trials).
        When during the trial the cue came on and off.

    stim_present: 1d numpy array of length (n_trials).
        Whether a perturbation stim was delivered.

    stim_on_time: 1d numpy array of length (n_trials).
    stim_off_time: 1d numpy array of length (n_trials).
        When during the trial the perturbation stim came on and off.

    stim_period: 1d numpy array of length (n_trials).
        Indicates with an integer code when during the trial the
        perturbation stimulus was delivered.

    stim_site: 1d numpy array of length (n_trials).
        Indicates with an integer code where the perturbation stimulus
        was delivered.

    stim_period_num2str_dict: Python dict
        Maps integer code to string indicating stim period.
        Must match mapping in parse_stim_period.

    stim_period_str2num_dict: Python dict
        Maps string indicating stim period to integer code.
        Must match mapping in parse_stim_period.

    stim_site_num2str_dict: Python dict
        Maps integer code to string indicating stim site.
        Must match mapping in parse_stim_site.

    stim_site_str2num_dict: Python dict
        Maps string indicating stim site to integer code.
        Must match mapping in parse_stim_site.

    time_aligned: bool
        Whether time variables (i.e. task_pole_on_time, task_pole_off_time,
        task_cue_on_time, task_cue_off_time, stim_on_time, stim_off_time)
        are aligned across trials.

    """

    def __init__(self, file_name, parser=ParserNuoLiALM, remove_early_report=True, remove_ignored=True, verbose=False):
        """
        Inputs
        ------
        file_name: string
            Name of file containing the data from a session.

        parser: SessionParser-like
            SessionParser-like class for parsing the file indicated by
            file_name into class variables here.

        remove_early_report: bool
            Whether to remove trials in which the animal responded before cue.

        remove_ignored: bool
            Whether to remove trials in which the animal did not respond.

        verbose: bool
            Whether to print unessential but sometimes helpful information.
        """
        # Initialize class variables
        self.file_name = file_name
        self.neural_data = []
        self.neural_unit_depth = []
        self.neural_unit_type = []
        self.neural_unit_location = []
        self.n_units = []
        self.n_trials = []
        self.spike_present = []
        self.unit_held_table = []
        self.behavior_report = []
        self.behavior_report_type = []
        self.behavior_early_report = []
        self.task_trial_type = []
        self.stim_present = []
        self.stim_period = []
        self.stim_site = []
        self.stim_period_num2str_dict = None
        self.stim_site_num2str_dict = None
        self.stim_period_str2num_dict = None
        self.stim_site_str2num_dict = None
        self.stim_period_selected = None # None if unselected, num if selected
        self.stim_site_selected = None
        self.stim_on_time = []
        self.stim_off_time = []
        self.task_pole_on_time = []
        self.task_pole_off_time = []
        self.task_cue_on_time = []
        self.task_cue_off_time = []
        self.time_aligned = False
        self.state = []

        # Parse session data using parser object and its parse_all() method
        self = parser(self).parse_all()
        self.n_trials = self.behavior_report.shape[0]
        self.parser = parser

        if parser != ParserNuoLiDualALM_states:
            # Storing few stats for neural_data as class variables for quick access
            self.n_units = self.neural_data.shape[0]
            self.n_trials = self.neural_data.shape[1]

            self.spike_present = np.asarray([[np.size(trial)!=0 for trial in unit] for unit in self.neural_data])

            # Infers and stores during what trials each unit (i.e. neurons) was
            # being recorded by electrodes.
            self.unit_held_table = self.__compute_unit_held_table()

        # Construct stim period & site str2num dicts from num2str dicts.
        self.stim_period_str2num_dict = \
             {v:k for k,v in self.stim_period_num2str_dict.items()}
        self.stim_site_str2num_dict = \
             {v:k for k,v in self.stim_site_num2str_dict.items()}



        # If specified by input flags, remove early_report or ignored trials
        if remove_early_report:
            self.select_trials(self.behavior_early_report==0)
        if remove_ignored:
            self.select_trials(self.behavior_report!=-1)

        if verbose:
            # Print string encodings of unique stim periods and stim sites
            print (self.get_unique_stim_periods_str())
            print (self.get_unique_stim_sites_str())
        return

    def __compute_unit_held_table(self):
        """
        For each unit, infers for what trials the unit was held by electrodes.
        Assumes that all trials between the first and the last trials with
        recorded spikes were held.
        Called only during init, and the table (self.unit_held_table) can be
        accessed with self.get_unit_held_table().
        """
        table = np.zeros(self.spike_present.shape, dtype=bool)
        for i in range(self.n_units):
            u_sp = self.spike_present[i]
            fst_true_idx = utils.find_first_true_idx(u_sp)
            lst_true_idx = utils.find_last_true_idx(u_sp)
            if fst_true_idx is not None and lst_true_idx is not None:
                table[i, fst_true_idx:lst_true_idx] = True
        return table

    def get_unit_held_table(self, show_table=False):
        """
        Returns self.unit_held_table
        If show_table == True, the table is plotted as an image
        """
        if show_table:
            plt.imshow(self.unit_held_table, origin='lower', aspect='auto')
        return self.unit_held_table

    def is_fully_held(self):
        """
        returns whether neural_data consists only of fully intact trials
        """
        return (np.all(self.get_unit_held_table()))

    def select_fully_held_units(self):
        """
        Select units that were held for all trials and return the mask of
        selected units (True if selected, False if not)
        """
        held_units_mask = np.all(self.get_unit_held_table(),axis=1)
        self.select_units(held_units_mask)

        assert self.is_fully_held()
        return held_units_mask

    def score_trial_range(self, alpha=np.arange(0.01,0.2,0.01), start=None, end=None, print_all=True, print_best=False):
        """
        Function to help select trials for which subsequent analyses are done.
        Examines and scores a set of trial ranges (only and all contiguous
        sets of trials are examined), and selects the best trial range based
        on a heuristic.

        The goal, roughly, is to find a long trial range during which a
        reasonably large number of units are consistently held.
        Score function is:
        score(start,end) = (# units w/ spikes) + (alpha) * (# trials selected)

        The function returns the best trial range, chosen with a heuristic.
        The heuristic is as follows:
        1. If any trial range has >20 units, throw out all other ranges/alpha.
        2. Compute "product score" = (# units w/ spikes) * (# trials selected)
        3. Find the trial range with highest product score.

        Input
        -----
        alpha: 1d numpy array
            Values of alpha to score trial ranges with.

        start: int
            Index for the first trial considered for examining trial ranges.

        end: int
            Index for the last trial(+1) considered for examining trial ranges.

        print_all: bool
            Whether to print all examined trial ranges (and alphas)

        print_best: bool
            Whether to print the best trial range scored with a heuristic

        Output
        ------
        (best_start_tr_idx, best_end_tr_idx) : (int, int)
            Start and end (exclusive) indices of trial range selected with
            the heuristic described above.

        """
        if start is None:
            start = 0
        if end is None:
            end = self.n_trials
        n_trials_rel = end-start # number of relevant trials given start/end

        # get the unit_held_table and keep only relevant trials
        unit_held_table = self.get_unit_held_table()[:,start:end]

        scores = np.zeros((n_trials_rel, n_trials_rel, np.size(alpha)))
        n_units_w_spikes = np.zeros((n_trials_rel, n_trials_rel)) #start,end (relative)
        for st in range(n_trials_rel): #start relative trial index 0 to end
            for et in range(st, n_trials_rel): #end index from st to end (incl)
                n_trials_sel = et+1 - st
                n_units_w_spikes[st,et] = np.sum(np.all(unit_held_table[:, st:et+1], axis=1))
                scores[st,et,:] = n_units_w_spikes[st,et] + alpha * n_trials_sel

        start_idx_rel = np.argmax(np.amax(scores,axis=1), axis=0)
        end_idx_rel = np.argmax(np.amax(scores,axis=0), axis=0) + 1 #exclusive
        n_trials_sel = end_idx_rel - start_idx_rel
        n_units_sel = n_units_w_spikes[start_idx_rel, end_idx_rel-1]
        start_idx = start_idx_rel + start
        end_idx = end_idx_rel + start

        everything = np.array([start_idx, end_idx, n_trials_sel, n_units_sel]).T
        if False:
            print("start, end (exlusive), n_trials, n_units")
            print(everything)

        # Select the best one with some heuristic:
        if np.amax(n_units_sel) > 20:
            everything = everything[n_units_sel>20,:]
        product_score = everything[:,2]*everything[:,3]
        best_config_idx = np.argmax(product_score)
        best_start_tr_idx = int(everything[best_config_idx,0])
        best_end_tr_idx = int(everything[best_config_idx,1])

        if print_best:
            print("Trial range chosen with heuristic:")
            print("start, end (exlusive), n_trials, n_units")
            print(everything[best_config_idx,:])

        # return best start and end indices
        return (best_start_tr_idx, best_end_tr_idx)


    def cull_data(self, start=None, end=None, verbose=True):
        """
        For trial ranges start:end (exclusive), retrieve only units that are intact for all trials. self.neural_data and associated variables are
        affected.
        return the held trial and unit mask

        Input
        -----
        start: int
            Index for the first trial considered for examining trial ranges.

        end: int
            Index for the last trial(+1) considered for examining trial ranges.

        verbose: bool
            Whether to print the best trial range scored with a heuristic

        Output
        ------
        selected_trial_mask : 1d numpy array of length n_trials
            True for trials that are kept

        selected_unit_mask: 1d numpy array of length n_units
            True for units that are kept
        """

        # If start or end are None, score trials and get the best option.
        if start is None or end is None:
            (start, end) = self.score_trial_range(start=start, end=end, print_all=False, print_best=verbose)

        # Reduce trials down to start:end for all of data
        selected_trial_mask = self.select_trials(np.asarray(range(start,end)))
        selected_unit_mask = self.select_fully_held_units()
        return (selected_trial_mask,selected_unit_mask)

    def get_unique_stim_periods_num(self):
        """
        Return integer codes for all stim periods in the Session.
        """
        return np.unique(self.stim_period)

    def get_unique_stim_sites_num(self):
        """
        Return integer codes for all stim sites in the Session.
        """
        return np.unique(self.stim_site)

    def get_unique_stim_periods_str(self):
        """
        Return string descriptions for all stim periods in the Session.
        """
        return [self.stim_period_num_to_str(x) for x in self.get_unique_stim_periods_num()]

    def get_unique_stim_sites_str(self):
        """
        Return string descriptions for all stim sites in the Session.
        """
        return [self.stim_site_num_to_str(x) for x in self.get_unique_stim_sites_num()]

    def align_time(self, align_to=None):
        """
        Align all timing data such that they are consistent across trials
        relative to one event time.

        Input
        -----
        align_to: 1d numpy array of length (n_trials).
            Time marker to subtract from all timing data for each trial.

        Affected vars
        -------------
        neural_data
        stim_on_time
        stim_off_time
        task_pole_on_time
        task_pole_off_time
        task_cue_on_time
        task_cue_off_time
        time_aligned
        """
        # If align_to is unspecified, default to cue on time
        if align_to is None:
            align_to = self.task_cue_on_time

        # Align to cue time
        if self.parser != ParserNuoLiDualALM_states:
            self.neural_data -= align_to
        self.stim_on_time -= align_to
        self.stim_off_time -= align_to
        self.task_pole_on_time -= align_to
        self.task_pole_off_time -= align_to
        self.task_cue_on_time -= align_to
        if self.task_cue_off_time is not None:
            self.task_cue_off_time -= align_to

        self.time_aligned = True
        return

    def align_time_if_necessary(self, align_to=None):
        """
        Checks if timing data are aligned across trials (self.time_aligned).
        If not, raises a warning and calls align_time.
        """
        if self.time_aligned == False:
            warnings.warn("Event times were not aligned. Aligning to cue on time.", RuntimeWarning)
            self.align_time(align_to)
        return

    def unit_idx_to_mask(self, unit_indices):
        """
        Converts integer encoding of unit indices to mask encoding.
        Does not account for redundant indices in unit_indices.

        Input
        -----
        unit_indices: 1d numpy array
            Specifies unit indices in integer form

        Output
        ------
        mask: 1d numpy array of length n_units
            True for indices indicated by unit_indices
        """
        mask = np.zeros(self.n_units, dtype=bool)
        mask[np.asarray(unit_indices)] = True
        return (mask)

    def trial_idx_to_mask(self, trial_indices):
        """
        Converts integer encoding of trial indices to mask encoding.
        Does not account for redundant indices in trial_indices.

        Input
        -----
        trial_indices: 1d numpy array
            Specifies trial indices in integer form

        Output
        ------
        mask: 1d numpy array of length n_trials
            True for indices indicated by trial_indices
        """
        mask = np.zeros(self.n_trials, dtype=bool)
        mask[np.asarray(trial_indices)] = True
        return (mask)

    def select_units(self, unit_mask):
        """
        Produces slices of the contents of self.
        unit_indices can be a bool mask of length self.n_units or an int array
        indicating the indices of desired units.
        """
        unit_mask = np.asarray(unit_mask)

        if unit_mask.dtype != np.bool: #if not mask form, convert to mask
            mask = np.zeros(self.n_units, dtype=bool)
            mask[unit_mask] = True
            unit_mask = mask
        assert len(unit_mask) == self.n_units

        self.n_units = sum(unit_mask)
        self.neural_data = self.neural_data[unit_mask,:]
        self.spike_present = self.spike_present[unit_mask,:]
        self.unit_held_table = self.unit_held_table[unit_mask,:]
        self.neural_unit_depth = self.neural_unit_depth[unit_mask]
        self.neural_unit_type = self.neural_unit_type[unit_mask]
        self.neural_unit_location = self.neural_unit_location[unit_mask]

        return unit_mask
    
    def select_units_w_location(self, location_str):
        """
        Culls units to those with location encoded by location_str.
        """    
        unit_mask = self.neural_unit_location == location_str
        self.select_units(unit_mask)
        return unit_mask

    def select_trials(self, trial_mask, output_bool_mask=True):
        """
        Produces slices of the contents of self.
        trial_indices can be either a bool mask of length self.n_trials or an int array
        indicating the indices of desired trials.
        """

        trial_mask = np.asarray(trial_mask)

        #if trial_mask.dtype != np.bool: #if not mask form, convert to mask
        #    mask = np.zeros(self.n_trials, dtype=bool)
        #    mask[trial_mask] = True
        #    trial_mask = mask
        #assert len(trial_mask) == self.n_trials

        if trial_mask.dtype == np.bool:
            assert len(trial_mask) == self.n_trials
            self.n_trials = np.sum(trial_mask)
        else:
            assert trial_mask.dtype == int
            assert np.max(trial_mask) < self.n_trials
            old_n_trials = self.n_trials
            self.n_trials = trial_mask.size

        if self.parser != ParserNuoLiDualALM_states:
            self.neural_data = self.neural_data[:,trial_mask]
            self.spike_present = self.spike_present[:,trial_mask]
            self.unit_held_table = self.unit_held_table[:,trial_mask]
        else:
            self.state = self.state[trial_mask]

        self.behavior_report_type = self.behavior_report_type[trial_mask]
        self.behavior_report = self.behavior_report[trial_mask]
        self.behavior_early_report = self.behavior_early_report[trial_mask]
        self.task_trial_type = self.task_trial_type[trial_mask]
        self.stim_present = self.stim_present[trial_mask]
        self.stim_period = self.stim_period[trial_mask]
        self.stim_site = self.stim_site[trial_mask]
        self.stim_on_time = self.stim_on_time[trial_mask]
        self.stim_off_time = self.stim_off_time[trial_mask]
        self.task_pole_on_time = self.task_pole_on_time[trial_mask]
        self.task_pole_off_time = self.task_pole_off_time[trial_mask]
        self.task_cue_on_time = self.task_cue_on_time[trial_mask]

        if self.task_cue_off_time is not None:
            self.task_cue_off_time = self.task_cue_off_time[trial_mask]

        if self.i_good_trials is not None:
            self.i_good_trials = self.i_good_trials[trial_mask]

        # For compatibility... some functions might depend on the output being bool. This doesn't really make sense if there are redundant indices in int-type trial_mask.
        if trial_mask.dtype != np.bool and output_bool_mask:
            mask = np.zeros(old_n_trials, dtype=bool)
            mask[trial_mask] = True
            trial_mask = mask

        return trial_mask

    def select_random_trials(self, n, replace=False):
        """
        Randomly selects n trials to keep and changes relevant class variables.
        """
        if not replace and self.n_trials < n:
            raise ValueError('If not replacing, n must not be larger than the existing number of trials.')
        n_idxes = np.random.choice(self.n_trials, n, replace=replace)
        self.select_trials(n_idxes, False)
        return


    def deepcopy(self):
        return (cp.deepcopy(self))

    def is_stim_selected(self):
        """
        Returns boolean flag indicating whether perturbation stim period and
        site to examine were manually selected and culled.
        """
        return (self.stim_period_selected is not None and self.stim_site_selected is not None)

    def is_stim_present_and_selected(self):
        """
        Returns boolean flag indicating whether perturbation stim period and
        site to examine were manually selected AND whether stim is present
        for all trials in Session.
        """
        if not self.is_stim_selected():
            return False
        return np.all(self.stim_present)

    def first_and_last_spikes(self):
        """
        return the first and last spikes in the Session across all units and
        trials
        """

        self.align_time_if_necessary()

        firstLastCandidates = \
          np.array([np.array([utils.amin_w_blanks(trial),utils.amax_w_blanks(trial)]) for unit in self.neural_data for trial in unit])
        firstSpike = np.nanmin(firstLastCandidates[:,0])
        lastSpike = np.nanmax(firstLastCandidates[:,1])

        return (firstSpike,lastSpike)

    def stim_period_num_to_str(self, stim_period_num):
        """
        Returns string description of stim_period_num
        """
        if stim_period_num not in self.stim_period_num2str_dict:
            raise ValueError('Improper/non-existent stim_period_num.')
        return self.stim_period_num2str_dict[stim_period_num]

    def stim_period_str_to_num(self, stim_period_str):
        """
        Returns integer encoding of stim_period_str
        """
        for key in self.stim_period_str2num_dict.keys():
            if key in stim_period_str:
                return self.stim_period_str2num_dict[key]
        raise ValueError('Unable to decode stim_period_str (e.g. early sample/late sample/early delay/late delay).')

    def stim_period_num_to_mask(self, stim_period_num, verbose=False):
        """
        Returns mask indicating which trials have stim_period_num
        """
        stim_period_mask = self.stim_period == stim_period_num

        if verbose:
            if sum(stim_period_mask) == 0:
                warnings.warn("There is no trial of stim_period: " + self.stim_period_num_to_str(stim_period_num), RuntimeWarning)

        return (stim_period_mask)

    def select_trials_w_stim_period_num(self, stim_period_num):
        """
        Culls trials to those with stim period encoded by stim_period_num
        """
        stim_period_mask = self.stim_period_num_to_mask(stim_period_num)
        trial_mask = self.select_trials(stim_period_mask)
        self.stim_period_selected = stim_period_num
        return trial_mask

    def select_trials_w_stim_period_str(self, stim_period_str):
        """
        Culls trials to those with stim period encoded by stim_period_str
        """
        stim_period_num = self.stim_period_str_to_num(stim_period_str)
        trial_mask = self.select_trials_w_stim_period_num(stim_period_num)
        return trial_mask

    def exists_stim_period_num(self, stim_period_num):
        """
        Returns flag for whether stim period encoded by stim_period_num exists
        in the Session.
        """
        return np.sum(self.stim_period == stim_period_num) > 0

    def exists_stim_period_str(self, stim_period_str):
        """
        Returns flag for whether stim period encoded by stim_period_str exists
        in the Session.
        """
        stim_period_num = self.stim_period_str_to_num(stim_period_str)
        return self.exists_stim_period_num(stim_period_num)

    def stim_site_num_to_str(self, stim_site_num):
        """
        Returns string description of stim_site_num
        """
        if stim_site_num not in self.stim_site_num2str_dict:
            raise ValueError('Improper/non-existent stim_site_num.')
        return self.stim_site_num2str_dict[stim_site_num]

    def stim_site_str_to_num(self, stim_site_str):
        """
        Returns integer encoding of stim_site_str
        """
        for key in self.stim_site_str2num_dict.keys():
            if key in stim_site_str:
                return self.stim_site_str2num_dict[key]
        raise ValueError('Unable to decode stim_site_str (e.g. ipsi ALM/contra ALM/bi ALM/FN/DN).')

    def stim_site_num_to_mask(self, stim_site_num, verbose=False):
        """
        Returns mask indicating which trials have stim_site_num
        """
        stim_site_mask = self.stim_site == stim_site_num

        if verbose:
            if not self.exists_stim_site_num(stim_site_num):
                warnings.warn("There is no trial of stim_site: " + self.stim_site_num_to_str(stim_site_num), RuntimeWarning)

        return (stim_site_mask)

    def select_trials_w_stim_site_num(self, stim_site_num):
        """
        Culls trials to those with stim site encoded by stim_site_num
        """
        stim_site_mask = self.stim_site_num_to_mask(stim_site_num)
        trial_mask = self.select_trials(stim_site_mask)
        self.stim_site_selected = stim_site_num
        return trial_mask

    def select_trials_w_stim_site_str(self, stim_site_str):
        """
        Culls trials to those with stim site encoded by stim_site_str
        """
        stim_site_num = self.stim_site_str_to_num(stim_site_str)
        trial_mask = self.select_trials_w_stim_site_num(stim_site_num)
        return trial_mask

    def exists_stim_site_num(self, stim_site_num):
        """
        Returns flag for whether stim site encoded by stim_site_num exists
        in the Session.
        """
        return np.sum(self.stim_site == stim_site_num) > 0

    def exists_stim_site_str(self, stim_site_str):
        """
        Returns flag for whether stim site encoded by stim_site_str exists
        in the Session.
        """
        stim_site_num = self.stim_site_str_to_num(stim_site_str)
        return self.exists_stim_site_num(stim_site_num)

    def exists_stim_type_str(self, stim_type_str):
        """
        Returns flag for whether stim type encoded by stim_type_str exists
        in the Session.
        stim_type_str includes descriptions of both stim period and stim site.
        """
        stim_period_num = self.stim_period_str_to_num(stim_type_str)
        stim_site_num = self.stim_site_str_to_num(stim_type_str)
        return self.exists_stim_period_num(stim_period_num) and self.exists_stim_site_num(stim_site_num)

    def num_stim_type_str(self, stim_type_str):
        """
        Returns number of trials with stim type encoded by stim_type_str.
        stim_type_str includes descriptions of both stim period and stim site.
        """
        stim_period_num = self.stim_period_str_to_num(stim_type_str)
        stim_site_num = self.stim_site_str_to_num(stim_type_str)
        stim_type_mask = self.stim_type_nums_to_mask(stim_period_num, stim_site_num)
        return np.sum(stim_type_mask)

    def stim_type_nums_to_mask(self, stim_period_num, stim_site_num, verbose=False):
        """
        Returns mask indicating which trials have stim_period_num AND
        stim_site_num
        """
        stim_period_mask = self.stim_period_num_to_mask(stim_period_num)
        stim_site_mask = self.stim_site_num_to_mask(stim_site_num)

        stim_type_mask = np.all([stim_period_mask, stim_site_mask], axis=0)

        if verbose:
            if sum(stim_type_mask) == 0:
                warnings.warn("There is no trial of stim type: " + self.stim_period_num_to_str(stim_period_num) + " " + self.stim_site_num_to_str(stim_site_num), RuntimeWarning)

        return (stim_type_mask)

    def select_trials_w_stim_type_nums(self, stim_period_num, stim_site_num):
        """
        Culls trials to those with stim type encoded by stim_period_num AND
        stim_site_num.
        """
        stim_type_mask = self.stim_type_nums_to_mask(stim_period_num, stim_site_num)

        trial_mask = self.select_trials(stim_type_mask)
        self.stim_period_selected = stim_period_num
        self.stim_site_selected = stim_site_num
        return trial_mask

    def select_trials_w_stim_type_str(self, stim_type_str):
        """
        Culls trials to those with stim type encoded by stim_type_str.
        stim_type_str includes descriptions of both stim period and stim site.
        """
        stim_period_num = self.stim_period_str_to_num(stim_type_str)
        stim_site_num = self.stim_site_str_to_num(stim_type_str)

        trial_mask = self.select_trials_w_stim_type_nums(stim_period_num, stim_site_num)
        return trial_mask
    


    def plot_unit(self, unit_idx, stim_type='no stim', align_to=None, bin_width=0.2, stride=0.1, label_by_report=True, include_raster=True, title=None, save_fig=False, fig_filename='singleUnit.png',verbose=True):
        """
        For each trial type, plots the PSTH with errorbars.
        If stim type is already selected for both period and site, then stim_type is not filtered.

        Input
        -----
        unit_idx: int
            Index of unit in neural_data

        stim_type: string OR (int, int)
            string: description of both stim period and stim site
            (int,int): integer encodings of stim period and stim site

        align_to: None or 1d numpy array of length (n_trials)
            None: aligned to cue on time.
            1d numpy array: time for each trial at which timings are aligned

        bin_width: float
            Time bin width in seconds to use in computing firing rate

        stride: float
            Time duration in seconds with which to stride bins

        label_by_report: bool
            Whether to split trials by by report type (lick direction) or
            cue type.
                True : by report type
                False: by cue type

        include_raster: bool
            Whether to include raster plots for the trial types

        title: None or string

        save_fig: bool

        fig_filename: string

        verbose: bool

        """

        ###########

        # Make a deep copy so that mutations don't affect input
        unit_data = self.deepcopy()

        # Keep only the indicated unit
        unit_data.select_units([unit_idx])

        # Align to cue time
        unit_data.align_time(align_to=align_to)

        if not unit_data.is_stim_selected():
            # Select trials based on stim
            if isinstance(stim_type, str):
                unit_data.select_trials_w_stim_type_str(stim_type)
            elif isinstance(np.array(trial_type), np.ndarray) and len(stim_type) == 2:
                unit_data.select_trials_w_stim_type_nums(stim_type[0],stim_type[1])
            else:
                raise ValueError('stim_type is neither a string nor a length 2 array-like object.')

        # Only keep trials with any spike, and keep only selected trials
        #recorded_mask = [np.size(trial)!=0 for trial in unit_data.neural_data[0]]
        recorded_mask = unit_data.get_unit_held_table()[0]
        unit_data.select_trials(recorded_mask)

        if label_by_report:
            label_str = unit_data.behavior_report_type
        else:
            label_str = unit_data.task_trial_type
        label = (label_str == 'r').astype(int)

        # Prepare plot;
        # Get unique trial types and generate colors
        unique_trial_types = np.array(['r','l']) #here, l or r
        ntt = np.size(unique_trial_types)

        colors = ptools.generate_color_arrays(ntt)
        error_colors = ptools.add_alpha_to_color_arrays(colors)

        if include_raster:
            # Compute how many trials exist per trial type
            n_trials_per_type = np.empty(ntt)
            for i in range(ntt):
                n_trials_per_type[i] = sum(label_str == unique_trial_types[i])
            f, ax = plt.subplots(ntt+1,1, \
                        gridspec_kw={'height_ratios':np.append(n_trials_per_type,np.amax(n_trials_per_type)*1.5)},sharex=True, figsize=(8,6))
            for i in range(ntt+1):
                utils.prepare_plot_events(unit_data, ax[i], verbose)
        else:
            f, ax = plt.subplots(1,1, figsize=(8,3))
            ax = [ax]
            utils.prepare_plot_events(unit_data, ax[0], verbose)

        plt.rcParams.update({'font.size': 12})


        # For each trial type,
        for i in range(ntt):
            ttype = unique_trial_types[i]
            type_mask = label_str == ttype
            unit_ttype = unit_data.deepcopy()
            unit_ttype.select_trials(type_mask) #trials of unit with ttype
            if include_raster:
                ## Because of eventplot bug: if first trial is empty then errors out
                eventplot_trials = unit_ttype.neural_data[0]
                eventplot_colors = np.tile(colors[i],(unit_ttype.n_trials,1))
                if eventplot_trials[0].size == 0:
                    eventplot_trials[0] = np.array([0])
                    eventplot_colors[0,3] = 0
                ax[i].eventplot(positions=eventplot_trials, orientation='horizontal', colors=eventplot_colors)
                ax[i].set_ylabel('Trials')
                ax[i].get_xaxis().set_visible(False)

            (first_spike, last_spike) = unit_ttype.first_and_last_spikes()
            bin_centers,spikes_per_bin = \
                    utils.sliding_histogram(unit_ttype.neural_data, \
                    first_spike+bin_width, last_spike-bin_width, bin_width, stride)

            rate = np.mean(spikes_per_bin/float(bin_width),axis=1)
            rate_sem = spst.sem(spikes_per_bin/float(bin_width),axis=1)

            ptools.plot_w_error(rate,bin_centers,rate_sem,color=colors[i],legend=ttype,show_legend=False,ax=ax[-1])

        ax[-1].legend(loc='best')
        ax[-1].set_ylabel('Firing rate [spikes/s]')
        ax[-1].set_xlabel('Time from cue [s]')
        #ax[-1].set_xlim(-3.5,2)

        if title is None:
            title = 'Unit ' + str(unit_idx) + ' : ' + unit_data.neural_unit_type[0] + ', ' + unit_data.neural_unit_location[0]
        ax[0].set_title(title)

        f.subplots_adjust(hspace=0.001)
        if save_fig:
            f.savefig(fig_filename)

        return
