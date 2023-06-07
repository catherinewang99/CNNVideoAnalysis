import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt


def amin_w_blanks(xs):
    if np.size(xs) == 0:
        return float('nan')
    else:
        return np.amin(xs)

def amax_w_blanks(xs):
    if np.size(xs) == 0:
        return float('nan')
    else:
        return np.amax(xs)


def is_stim_present(data):
    '''
    Assumes that data has already been selected for one stim type, and checks only the value of data['task_stimulation'][0,1]
    '''
    return (not (np.isnan(data['task_stimulation'][0,1]) or \
            data['task_stimulation'][0,1] == 0.0))

def find_first_true_idx(bool_array):
    for i in range(len(bool_array)):
        if bool_array[i] == True:
            return i
        else:
            continue

def find_last_true_idx(bool_array):
    flipped_solution = find_first_true_idx(np.flipud(bool_array))
    if flipped_solution is None:
        return None
    else:
        return len(bool_array) - find_first_true_idx(np.flipud(bool_array)) - 1



def check_event_jitter(data):
    pole_on_time_std = np.std(data.task_pole_on_time)
    if pole_on_time_std>0.1:
        print ("High pole on time jitter. Std of", str(pole_on_time_std))
    pole_off_time_std = np.std(data.task_pole_off_time)
    if pole_off_time_std>0.1:
        print ("High pole off time jitter. Std of", str(pole_off_time_std))
    cue_on_time_std = np.std(data.task_cue_on_time)
    if cue_on_time_std>0.1:
        print ("High cue on time jitter. Std of", str(cue_on_time_std))

    if data.is_stim_present_and_selected():
        stim_on_time_std = np.std(data.stim_on_time)
        if stim_on_time_std>0.1:
            print ("High laser on time jitter. Std of", str(stim_on_time_std))

        stim_off_time_std = np.std(data.stim_off_time)
        if stim_off_time_std>0.1:
            print ("High laser off time jitter. Std of", str(stim_off_time_std))

def prepare_plot_events(data, ax=None, verbose=False):
    # Mark trial events
    if verbose:
        check_event_jitter(data)
    pole_on_time_mean = np.mean(data.task_pole_on_time)
    pole_off_time_mean = np.mean(data.task_pole_off_time)
    cue_on_time_mean = np.mean(data.task_cue_on_time)

    if ax is None:
        ax = plt.gca()
    ax.axvline(x=pole_on_time_mean, color='k', linestyle='--', linewidth=1)
    ax.axvline(x=pole_off_time_mean, color='k', linestyle='--', linewidth=1)
    ax.axvline(x=cue_on_time_mean, color='k', linestyle='--', linewidth=1)

    if data.is_stim_present_and_selected():
        stim_on_time_mean = np.mean(data.stim_on_time)
        stim_off_time_mean = np.mean(data.stim_off_time)
        ax.axvspan(stim_on_time_mean, stim_off_time_mean, alpha=0.2, color='blue') # To mark Stim

    ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
    return

def sliding_histogram(spikeTimes, begin_time, end_time, bin_width, stride, rate=False):
    '''
    Calculates the number of spikes for each unit in each sliding bin of width bin_width, strided by stride.
    begin_time and end_time are treated as the lower- and upper- bounds for bin centers.
    '''
    # Calculate sliding bins for finding firing rates
    binCenters = np.arange(begin_time, end_time+stride, stride)
    binIntervals = np.vstack((binCenters-bin_width/2, binCenters+bin_width/2)).T

    # Count number of spikes for each sliding bin
    binSpikes = np.squeeze(np.asarray([[[np.sum(np.all([trial>binInt[0], trial<=binInt[1]], axis=0)) for binInt in binIntervals] for trial in unit] for unit in spikeTimes]).swapaxes(0,-1))

    if len(binSpikes.shape) == 1:
        binSpikes = np.expand_dims(binSpikes, 0)

    if rate:
        return (binCenters,binSpikes/float(bin_width))
    return (binCenters,binSpikes)

def generate_gaussian_smoothed_trajectory(allTrialSpikeTimes, allTrialZeroTime, minTime, maxTime, binStep, gaussWidthInSteps):

    binEdgeVec = np.arange(minTime, maxTime+binStep, binStep)
    binPointNum = np.size(binEdgeVec)
    binCenterVec = minTime + binStep/2 + 5


    return (sliding_histogram(allTrialSpikeTimes, minTime, maxTime, binStep*gaussWidthInSteps, binStep, rate=True))
