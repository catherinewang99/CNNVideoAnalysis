import matplotlib as mpl
mpl.use('agg')

import argparse, os, math, pickle
from state_pred_from_videos_analysis_exp import PertPredFromVideosAnalysisExp
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--compute_pred', action='store_true', default=True)

    parser.add_argument('--plot_test_acc_time_course', action='store_true', default=False)

    parser.add_argument('--plot_score_distribution', action='store_true', default=False)

    parser.add_argument('--correlate_roc_auc_vs_bi_stim_cross_hemi_cor_across_sessions', action='store_true', default=False)
    parser.add_argument('--correlate_cls_acc_vs_bi_stim_cross_hemi_cor_across_sessions', action='store_true', default=False)

    parser.add_argument('--correlate_cls_acc_vs_bi_stim_cross_hemi_cor_not_cond_on_trial_type_across_sessions', action='store_true', default=False)
    parser.add_argument('--correlate_cls_acc_vs_bi_stim_cross_hemi_cor_not_cond_on_trial_type_across_sessions_using_guang_bi_stim_cor', action='store_true', default=False)


    parser.add_argument('--compare_bi_stim_cross_hemi_cor_between_high_and_low_score_trials', action='store_true', default=False)
    parser.add_argument('--cross_hemi_cor_type', type=str, default='rank')
    parser.add_argument('--analyzed_time', type=float, default=-0.2)

    parser.add_argument('--compute_score_grad_in_pixel_space', action='store_true', default=False)

    parser.add_argument('--visualize_score_grad_in_pixel_space', action='store_true', default=False)

    parser.add_argument('--sess_type_list', nargs='+', type=str, default=["states"], help='original_guang_one_laser, full_reverse_guang_one_laser, or original_and_full_reverse_guang_one_laser')


    return parser.parse_args()

def main():

    args = parse_args()

    if args.compute_pred:
        exp = PertPredFromVideosAnalysisExp()
        exp.compute_pred(args.sess_type_list)

    if args.plot_test_acc_time_course:
        exp = PertPredFromVideosAnalysisExp()
        exp.plot_test_acc_time_course(args.sess_type_list)

    if args.plot_score_distribution:
        exp = PertPredFromVideosAnalysisExp()
        exp.plot_score_distribution(args.sess_type_list)


    if args.correlate_roc_auc_vs_bi_stim_cross_hemi_cor_across_sessions:
        exp = PertPredFromVideosAnalysisExp()
        exp.correlate_roc_auc_vs_bi_stim_cross_hemi_cor_across_sessions(args.sess_type_list, args.cross_hemi_cor_type, args.analyzed_time)

    if args.correlate_cls_acc_vs_bi_stim_cross_hemi_cor_across_sessions:
        exp = PertPredFromVideosAnalysisExp()
        exp.correlate_cls_acc_vs_bi_stim_cross_hemi_cor_across_sessions(args.sess_type_list, args.cross_hemi_cor_type, args.analyzed_time)




    if args.correlate_cls_acc_vs_bi_stim_cross_hemi_cor_not_cond_on_trial_type_across_sessions:
        exp = PertPredFromVideosAnalysisExp()
        exp.correlate_cls_acc_vs_bi_stim_cross_hemi_cor_not_cond_on_trial_type_across_sessions(args.sess_type_list, args.cross_hemi_cor_type, args.analyzed_time)




    if args.correlate_cls_acc_vs_bi_stim_cross_hemi_cor_not_cond_on_trial_type_across_sessions_using_guang_bi_stim_cor:
        exp = PertPredFromVideosAnalysisExp()
        exp.correlate_cls_acc_vs_bi_stim_cross_hemi_cor_not_cond_on_trial_type_across_sessions_using_guang_bi_stim_cor(args.sess_type_list, args.cross_hemi_cor_type, args.analyzed_time)






    if args.compare_bi_stim_cross_hemi_cor_between_high_and_low_score_trials:
        exp = PertPredFromVideosAnalysisExp()
        exp.compare_bi_stim_cross_hemi_cor_between_high_and_low_score_trials(args.sess_type_list, args.cross_hemi_cor_type, args.analyzed_time)



    if args.compute_score_grad_in_pixel_space:
        exp = PertPredFromVideosAnalysisExp()
        exp.compute_score_grad_in_pixel_space(args.sess_type_list)


    if args.visualize_score_grad_in_pixel_space:
        exp = PertPredFromVideosAnalysisExp()
        exp.visualize_score_grad_in_pixel_space(args.sess_type_list)




if __name__ == '__main__':
    main()

