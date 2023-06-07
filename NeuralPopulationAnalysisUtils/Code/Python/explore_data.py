import os
os.getcwd()
os.chdir('Code/Python')

from neupop import Session as Session
from neupop import PopulationAnalysis as PA
from neupop import MultiSessionAnalysis as MS
from neupop import decoders as ld
from neupop import session_parsers as parsers

# import data
#sd = Session('../../Data/ALMRecording_ALMStim_140918/ANM257772_20141121.mat', session_type='ALM')
sd = Session('../../Data/ALMRecording_ALMStim_140918/ANM257772_20141121.mat', parser=parsers.ParserNuoLiALM)

# plot single unit
sd.plot_unit(13, 'early delay left ALM', include_raster=True, label_by_report=True)


sd = Session('../../Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC6/BAYLORGC6_2018_03_05_spkwin2ms.mat', parser=parsers.ParserNuoLiDualALM)

# plot single unit
sd.plot_unit(13, 'early delay left ALM', include_raster=True, label_by_report=True)

#####
# population analysis
pa = PA(sd)

pa.do_everything(train_stim_type='no stim', test_stim_type=None, train_proportion=0.6, only_correct=False, label_by_report=True, even_classes=False, CD_time=0)

pa.score_input_trial_range()
pa.cull_data(11,257)
pa.load_train_and_test(train_stim_type='no stim', test_stim_type='early delay ipsi ALM', train_proportion=0.6) #early delay ipsi ALM
#pa.filter_train_trials_from_input(ed.behavior_report == 1) #correct only
#pa.filter_test_trials_from_input(ed.behavior_report == 1) #correct only
pa.preprocess(bin_width=0.4, label_by_report=True, even_test_classes=True)
#pa.plot_preprocessed_units(True)

_=pa.compute_CD_for_each_bin(classifier_cls=[ld.MDC])
#pa.W_dot_matrix()
trial_projections,trial_types,behavior_correct=pa.CD_projection_one_bin(proj_bin_time=0,CD_time=None)


pa.plot_pred_accuracy()

pa.plot_preprocessed_units()

pa.CD_projection(CD_time=0, subtract_mean=True)


########
msTest = MS(('../../Data/test', 'ALM'), verbose=True)
(test_acc_pa1_mean, test_acc_pa1_sem, test_acc_pa2_mean, test_acc_pa2_sem) = msTest.compare_decoders('early delay ipsi ALM', 'no stim', 'early delay ipsi ALM', classifier_cls=ld.MDC, bootstrap_threshold=20, n_bootstrap=10, bootstrap_train_proportion=0.6, min_train_size=10, verbose=False)






import os
os.getcwd()
os.chdir('Code/Python')

#import neupop as npu
from neupop import Session as Session
from neupop import PopulationAnalysis as PA
from neupop import MultiSessionAnalysis as MS
from neupop import decoders as ld

msALM = MS(('../../Data/ALMRecording_ALMStim_140918', 'ALM'), verbose=False)

#### TODO: SOLVE the mask problem when calling load_train_and_test with bootstrap=True

ax,acc_mean_l,acc_sem_l,acc_mean_r,acc_sem_r,bin_edges = msALM.projection_pred_accuracy(train_stim_type='no stim', test_stim_type='early delay ipsi ALM', train_proportion=0.5, bin_time=0, time_bin_width=0.4, label_by_report=True, correct_trials_only=False, even_train_classes=False, even_test_classes=True, classifier_cls=ld.MDC, n_proj_bins=20, bootstrap_threshold_per_bin=5, n_bootstrap=50, marker='o', verbose=False)

(acc_pa1_mean, acc_pa1_sem, acc_pa2_mean, acc_pa2_sem) = msALM.compare_decoders('early delay ipsi ALM', 'no stim', 'early delay ipsi ALM', classifier_cls=ld.MDC, bootstrap_threshold=25, n_bootstrap=30, bootstrap_train_proportion=0.5, min_train_size=10, min_test_size=6, even_train_trials=True, verbose=False)

(acc_mean, acc_sem, behavior_accuracy) = msALM.compare_neural_predictability_and_behavior('no stim', 'early delay ipsi ALM', classifier_cls=ld.MDC, bootstrap_threshold=25, n_bootstrap=20, min_bootsrap=8, bootstrap_train_proportion=0.5, min_train_size=10, min_test_size=6, verbose=False)







ax,acc_mean_l,acc_sem_l,acc_mean_r,acc_sem_r,bin_edges = msALM.projection_pred_accuracy(train_stim_type='no stim', test_stim_type='early delay bi ALM', train_proportion=0.5, bin_time=0, time_bin_width=0.4, label_by_report=True, correct_trials_only=False, even_train_classes=False, even_test_classes=True, classifier_cls=ld.MDC, n_proj_bins=20, bootstrap_threshold_per_bin=5, n_bootstrap=50, marker='o', verbose=False)

(acc_pa1_mean, acc_pa1_sem, acc_pa2_mean, acc_pa2_sem) = msALM.compare_decoders('early delay bi ALM', 'no stim', 'early delay bi ALM', classifier_cls=ld.MDC, bootstrap_threshold=25, n_bootstrap=30, bootstrap_train_proportion=0.5, min_train_size=10, min_test_size=6, even_train_trials=True, verbose=False)

(acc_mean, acc_sem, behavior_accuracy) = msALM.compare_neural_predictability_and_behavior('no stim', 'early delay bi ALM', classifier_cls=ld.MDC, bootstrap_threshold=25, n_bootstrap=20, min_bootsrap=8, bootstrap_train_proportion=0.5, min_train_size=10, min_test_size=6, verbose=False)






from neupop import Session as Session
from neupop import PopulationAnalysis as PA
from neupop import MultiSessionAnalysis as MS
from neupop import decoders as ld

msCb = MS(('../../Data/ALMRecording_CbStim_171207', 'Cb'), verbose=False)

_ = msCb.projection_behavioral_performance(train_stim_type='no stim', test_stim_type='early delay FN', train_proportion=0.5, bin_time=0, time_bin_width=0.4, label_by_report=True, correct_trials_only=False, classifier_cls=ld.MDC, n_proj_bins=20, bootstrap_threshold_per_bin=4, n_bootstrap=50, marker='o', verbose=False)

_ = msCb.projection_behavioral_performance(train_stim_type='no stim', test_stim_type='early delay FN', train_proportion=0.5, bin_time=-0.2, time_bin_width=0.4, label_by_report=True, correct_trials_only=False, classifier_cls=ld.MDC, n_proj_bins=20, bootstrap_threshold_per_bin=4, n_bootstrap=50, marker='o', verbose=False)


(acc_pa1_mean, acc_pa1_sem, acc_pa2_mean, acc_pa2_sem) = msCb.compare_decoders('early delay FN', 'no stim', 'early delay FN', classifier_cls=ld.MDC, bootstrap_threshold=25, n_bootstrap=30, bootstrap_train_proportion=0.5, min_train_size=10, min_test_size=6, even_train_trials=True, verbose=False)

_ = msCb.compare_decoders('early delay FN', 'no stim', 'early delay FN', classifier_cls=ld.MDC, bin_time=-0.2, bin_width=0.4, bootstrap_threshold=25, n_bootstrap=30, bootstrap_train_proportion=0.5, min_train_size=10, min_test_size=6, even_train_trials=True, verbose=False)



(acc_mean, acc_sem, behavior_accuracy) = msCb.compare_neural_predictability_and_behavior('no stim', 'no stim', classifier_cls=ld.MDC, bootstrap_threshold=25, n_bootstrap=20, min_bootsrap=8, bootstrap_train_proportion=0.5, min_train_size=10, min_test_size=6, verbose=False)


(acc_mean, acc_sem, behavior_accuracy) = msCb.compare_neural_predictability_and_behavior('no stim', 'early delay FN', classifier_cls=ld.MDC, bootstrap_threshold=25, n_bootstrap=20, min_bootsrap=8, bootstrap_train_proportion=0.5, min_train_size=10, min_test_size=6, verbose=False)

(acc_mean, acc_sem, behavior_accuracy) = msCb.compare_neural_predictability_and_behavior('no stim', 'early delay FN', classifier_cls=ld.MDC, bin_time=-0.2, bin_width=0.4, bootstrap_threshold=25, n_bootstrap=20, min_bootsrap=8, bootstrap_train_proportion=0.5, min_train_size=10, min_test_size=6, verbose=False)



msCb.plot_all_CD_projections('no stim', 'no stim', classifier_cls=ld.MDC, CD_time=-0.2, verbose=False)

msCb.plot_all_CD_projections('no stim', 'early delay FN', classifier_cls=ld.MDC, CD_time=-0.2, verbose=False)




from neupop import Session as Session
from neupop import PopulationAnalysis as PA
from neupop import MultiSessionAnalysis as MS
from neupop import decoders as ld

msALMCb = MS(('../../Data/ALMRecording_ALMCbStim_180118', 'ALMCb'), verbose=False)

_ = msALMCb.projection_behavioral_performance(train_stim_type='no stim', test_stim_type='early delay ipsi ALM', train_proportion=0.5, bin_time=-0.2, time_bin_width=0.4, label_by_report=True, correct_trials_only=False, even_train_classes=False, even_test_classes=False, classifier_cls=ld.MDC, n_proj_bins=20, bootstrap_threshold_per_bin=4, n_bootstrap=50, marker='o', verbose=False)

_ = msALMCb.compare_decoders('early delay ipsi ALM', 'no stim', 'early delay ipsi ALM', classifier_cls=ld.MDC, bin_time=-0.2, bootstrap_threshold=25, n_bootstrap=30, bootstrap_train_proportion=0.5, min_train_size=10, min_test_size=6, even_train_trials=True, verbose=False)

_ = msALMCb.compare_neural_predictability_and_behavior('no stim', 'early delay ipsi ALM', classifier_cls=ld.MDC, bin_time=-0.2, bootstrap_threshold=25, n_bootstrap=20, min_bootsrap=8, bootstrap_train_proportion=0.5, min_train_size=10, min_test_size=6, verbose=False)





_ = msALMCb.projection_behavioral_performance(train_stim_type='no stim', test_stim_type='early delay bi ALM', train_proportion=0.5, bin_time=-0.2, time_bin_width=0.4, label_by_report=True, correct_trials_only=False, even_train_classes=False, even_test_classes=False, classifier_cls=ld.MDC, n_proj_bins=20, bootstrap_threshold_per_bin=4, n_bootstrap=50, marker='o', verbose=False)

_ = msALMCb.compare_decoders('early delay bi ALM', 'no stim', 'early delay bi ALM', classifier_cls=ld.MDC, bin_time=-0.2, bootstrap_threshold=25, n_bootstrap=30, bootstrap_train_proportion=0.5, min_train_size=10, min_test_size=6, even_train_trials=True, verbose=False)

_ = msALMCb.compare_neural_predictability_and_behavior('no stim', 'early delay bi ALM', classifier_cls=ld.MDC, bin_time=-0.2, bootstrap_threshold=25, n_bootstrap=20, min_bootsrap=8, bootstrap_train_proportion=0.5, min_train_size=10, min_test_size=6, verbose=False)






_ = msALMCb.projection_behavioral_performance(train_stim_type='no stim', test_stim_type='early delay FN', train_proportion=0.5, bin_time=-0.2, time_bin_width=0.4, label_by_report=True, correct_trials_only=False, even_train_classes=False, even_test_classes=False, classifier_cls=ld.MDC, n_proj_bins=20, bootstrap_threshold_per_bin=3, n_bootstrap=50, marker='o', verbose=False)

_ = msALMCb.compare_decoders('early delay FN', 'no stim', 'early delay FN', classifier_cls=ld.MDC, bin_time=-0.2, bootstrap_threshold=25, n_bootstrap=30, bootstrap_train_proportion=0.5, min_train_size=10, min_test_size=6, even_train_trials=True, verbose=False)

_ = msALMCb.compare_neural_predictability_and_behavior('no stim', 'early delay FN', classifier_cls=ld.MDC, bin_time=-0.2, bootstrap_threshold=25, n_bootstrap=20, min_bootsrap=8, bootstrap_train_proportion=0.5, min_train_size=10, min_test_size=6, verbose=False)


msALMCb.plot_all_CD_projections('no stim', 'no stim', classifier_cls=ld.MDC, CD_time=-0.2, verbose=False)

msALMCb.plot_all_CD_projections('no stim', 'early delay FN', classifier_cls=ld.MDC, CD_time=-0.2, verbose=False)



_ = msALMCb.projection_behavioral_performance(train_stim_type='no stim', test_stim_type='early delay FN', train_proportion=0.5, bin_time=-0.2, time_bin_width=0.4, label_by_report=True, correct_trials_only=False, even_train_classes=False, even_test_classes=False, classifier_cls=ld.FDAC, n_proj_bins=20, bootstrap_threshold_per_bin=3, n_bootstrap=50, marker='o', verbose=False)

_ = msALMCb.compare_decoders('early delay FN', 'no stim', 'early delay FN', classifier_cls=ld.FDAC, bin_time=-0.2, bootstrap_threshold=25, n_bootstrap=30, bootstrap_train_proportion=0.5, min_train_size=10, min_test_size=6, even_train_trials=True, verbose=False)

_ = msALMCb.compare_neural_predictability_and_behavior('no stim', 'early delay FN', classifier_cls=ld.FDAC, bin_time=-0.2, bootstrap_threshold=25, n_bootstrap=20, min_bootsrap=8, bootstrap_train_proportion=0.5, min_train_size=10, min_test_size=6, verbose=False)


msALMCb.plot_all_CD_projections('no stim', 'early delay FN', classifier_cls=ld.DFDAC, CD_time=-0.2, verbose=False)


# TODO
# 4. Holdout accuracy (averaged across delay period?) for each session as a bar plot

# 10. Split left and right trials when scattering neural predictability vs behabioral accuracy.

# Convert masks from bool to int


# DONE
# Select data to use (principled or by eye)
# Implement Sw = I classifier
# Separate training vs testing
# 1. Fix how the dotted lines and the colored regions are drawn.
# 2. Performance plot (e.g. Fig 3 bottom row)
# 3. unit W pairwise multiplication across time bins
# 6. Equal representation of right vs left when training and testing
# 8. Create option of evening train set size between pa1 and pa2.
# 9. Train only the desired bin.




# Notes
# Prediction accuracy should drop during perturbation, then recover or not recover
# What we can learn:
# If 1. CD trained on unpereturbed can predict on perturbed:
#       Then, there is residual information that's in the same dimensions
#          despite the perturbation.
# If 2. CD trained on unperturbed cannot predict on perturbed,
#           but CD trained on perturbed can predict on perturbed:
#       Then, there is residual information, but not in the same dimensions,
#           and the brain might need to change the way it "reads" the neurons
# If 3. CD trained on either cannot predict on perturbed:
#       Then, the perturbation destroyed the information.
# Coding direction (trained on end of delay period) and projection
