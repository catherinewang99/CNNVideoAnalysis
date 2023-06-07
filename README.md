Written by Catherine (8/16/2022)

Data storage:

ALYSE DATA (2022):
	Video png's are spread out over three disks due to size:

		C: Contains some of the VGAT videos, mostly problematic trials

		D: 
			- Contains all the relevant sessions in /video/
			- /oldvids/ is temporary storage because of faster transfer speeds

		E: under new_video_storage, the problematic VGLUT sessions are dumped here
__________________________________________________________________________________________________________

Steps to pre-process data + train model to predict neural data:

1. Run make_f1_f2_list. This will make a list of all the relevant sessions from the raw video file. Creates f1_f2_list.pkl pickle file that is used subsequently.

2. Run split_frames.py, with relevant modifications to split_frames_configs.json. This converts the aforementioned raw video files into folders of images for each trial within 
each session. There should be 1000 frames for each side and bottom trial (2000 frames for one trial in total).

3. Run alm_preprocessing: creates the sorted_filenames.obj object file from the neural data folder. Creates culled_session.obj file that contains neural data from session.
	- change Parser depending on need ('ParserNuoLiALM' is good for 2p)
	- Doesn't require neuronal_data4_sorted_filenames.obj object file. Commented out in the code.
	- Main function aligns the neural data by predefined bins (start, end times, bin width, stride)
	- Creates no stim, left/right stim, and bi stim masks to distinguish neural trials
	* removed pert_normal filtering because it was taking out all the neurons - modified data_prefix variable to avoid some preprocessing steps

4. Run all_neurons_time_traces_main.py. Uses culled_session.obj to load file names and store neural data information.
	* all_neurons_time_traces2p_main.py for 2p data
	- default bin width is 0.4, can be 0.1 for higher sampling rate. Include in data prefix and prep_kwargs so.
	- calls ALMDataset object from alm_dataset.py script
	- IMPORTANT: Filters the neurons so that the full trial average is above some firing rate threshold.
	- saves the neurons into all_neurons_{}.obj
	* changed no_stim to all stim (includes right, left and no stim trials) but didn't modify underlying variable names

5. Run all_neurons_reg_from_videos_cv.py to train network
	- make sure all_neurons_reg_configs.json is configured correctly, ie. neural begin and end times
	- reduce max workers to accomodate for system capabilities
	- bugs: if frames is somehow negative, will produce 'E3-BAYLORAT46-201121-176-side--0039.png' instead of 'E3-BAYLORAT46-201121-176-side-00039.png'. Fix in configs
		* tricky issue: determining correct params for begin_frame, end_frame, and skip_frame

* Data note: Removed AT44 201003 and 201013 because of lack of trials - error in step 2

Steps to test trained model:

1. Configure all_neurons_reg_recon_configs.json. Make sure all the variables match up with the ones used in all_neurons_reg_configs.json.

2. Change relevant inputs to true in all_neurons_reg_from_videos_recon_cv_main.py, which calls all_neurons_reg_from_videos_recon_cv_exp.py.

__________________________________________________________________________________________________________


Steps to pre-process and train model to predict stim/no stim:

0. Run alm_preprocessing.py. Same as step #3 above.

1. Prepare pert_pred_configs.json. Make sure that make_f1_f2_list.py is configured correctly with sess_type to grab all relevant sessions. 
	Also relevant: stim type and inst_trial_type. Correspond respectively to stim side and instructed lick dir (0 left, 1 right)

2. Run pert_pred_from_videos.py to train model

3. Run pert_pred_from_videos_main.py to generate predictions and get r2 scores/
__________________________________________________________________________________________________________


Steps to pre-process and train model to predict behavioral state:

1. Run steps #1 and #2 from first section to generate png frames from the videos

2. Run alm_preprocessing_for_states.py to get the masks for different behavioral states from GLM HMM.

3. Run state_pred_from_videos.py to train model to generate state predictions.

4. Run state_pred_from_videos_analysis_main.py to generate predictions.
