import argparse, os, math, pickle
from all_neurons_reg_from_videos_recon_cv_exp import AllNeuronsRegFromVideosReconCvExp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--compute_pred', action='store_true', default=True)
    parser.add_argument('--cca', action='store_true', default=False)
    parser.add_argument('--error_svd', action='store_true', default=False)

    parser.add_argument('--r2_hist', action='store_true', default=True)
    parser.add_argument('--r2_hist_sep_trial_type_and_hemi', action='store_true', default=False)

    parser.add_argument('--r2_vs_neural_depth', action='store_true', default=False)
    parser.add_argument('--r2_vs_neuron_type', action='store_true', default=False)
    parser.add_argument('--r2_with_std_across_cv', action='store_true', default=False)



    parser.add_argument('--sample_neural_traces', action='store_true', default=False)

    parser.add_argument('--cor_mat', action='store_true', default=False)


    return parser.parse_args()

def main():

    args = parse_args()

    if args.compute_pred:
        exp = AllNeuronsRegFromVideosReconCvExp()
        exp.compute_pred()

    if args.cca:
        exp = AllNeuronsRegFromVideosReconCvExp()
        exp.cca()


    if args.error_svd:
        exp = AllNeuronsRegFromVideosReconCvExp()
        exp.error_svd()



    if args.r2_hist:
        exp = AllNeuronsRegFromVideosReconCvExp()
        exp.compute_r2()

    if args.r2_hist_sep_trial_type_and_hemi:
        exp = AllNeuronsRegFromVideosReconCvExp()
        exp.r2_hist_sep_trial_type_and_hemi()



    if args.r2_vs_neural_depth:
        exp = AllNeuronsRegFromVideosReconCvExp()
        exp.r2_vs_neural_depth()
    
    if args.r2_vs_neuron_type:
        exp = AllNeuronsRegFromVideosReconCvExp()
        exp.r2_vs_neuron_type()

    if args.r2_with_std_across_cv:
        exp = AllNeuronsRegFromVideosReconCvExp()
        exp.r2_with_std_across_cv()

    if args.sample_neural_traces:
        exp = AllNeuronsRegFromVideosReconCvExp()
        exp.sample_neural_traces()
    
    if args.cor_mat:
        exp = AllNeuronsRegFromVideosReconCvExp()
        exp.cor_mat()



if __name__ == '__main__':
    main()

