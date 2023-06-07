from .session_parser import SessionParser
import numpy as np
import scipy.io as spio

class ParserNuoLiALMCb(SessionParser):
    '''
    SessionParser for Nuo Li's ALM+Cerebellar perturbation data encoding,
        which comes in a .mat file.
    '''
    def __init__(self, session):
        self.s = session

        self.input = spio.loadmat(self.s.file_name, squeeze_me=True)
        return

    def parse_neural_data(self):
        '''
        neural_data: 2d numpy array of shape (n_units, n_trials).
            Each entry is a list containing spike times.
        '''
        # The list comp makes unit a 2D array so that we can index trials
        self.s.neural_data = np.asarray([unit for unit in self.input['neuron_single_units']])
        # Properly wrap single-spike trials into singletons
        for unit in self.s.neural_data:
            for i in range(len(unit)):
                if isinstance(unit[i],float):
                    unit[i] = np.array([unit[i]])
        return

    def parse_neural_unit_info(self):
        '''
        neural_unit_depth: 1d numpy array of length n_units, dtype=float
            Depth in um (micrometers)

        neural_unit_type: 1d numpy array of length n_units, dtype=string
            Cell type (e.g. putative_pyramidal, putative_interneuron)

        neural_unit_location: 1d numpy array of length n_units, dtype=string
            Recording location (e.g. left_ALM, right_ALM)
        '''
        unit_info = np.asarray([unit for unit in self.input['neuron_unit_info']])
        self.s.neural_unit_depth = np.asarray([x[0] for x in unit_info])
        self.s.neural_unit_type = np.asarray([x[1] for x in unit_info])
        self.s.neural_unit_location = np.asarray(self.input['recording_area'])
        return

    def parse_behavior_report(self):
        '''
        behavior_report: 1d numpy array of length (n_trials)
            Whether the animal responded correctly.
        It is coded as follows:
            -1: ignored
            0 : incorrect
            1 : correct
        '''
        self.s.behavior_report = self.input['behavior_report']
        return

    def parse_behavior_early_report(self):
        '''
        behavior_early_report: 1d numpy array of length (n_trials).
            Whether the animal improperly responded before cue.
        It is coded as follows:
            0: no early report (proper behavior)
            1: early report (improper behavior)
        '''
        self.s.behavior_early_report = self.input['behavior_early_report']
        return

    def parse_task_trial_type(self):
        '''
        task_trial_type: 1d numpy array of length (n_trials).
            Whether the animal was cued to lick right ('r') or left ('l').
        It is coded as follows:
            'r': cued right
            'l': cued left
        '''
        self.s.task_trial_type = np.empty_like(self.input['task_trial_type'])
        self.s.task_trial_type[self.input['task_trial_type']=='y'] = 'r'
        self.s.task_trial_type[self.input['task_trial_type']=='n'] = 'l'
        return

    def parse_behavior_report_type(self):
        '''
        behavior_report_type: 1d numpy array of length (n_trials).
            Whether the animal responded with lick right ('r') or left ('l').
        It is coded as follows:
            'i': ignored
            'r': licked right
            'l': licked left
        '''
        self.s.behavior_report_type = np.empty(self.s.behavior_report.shape, dtype=str)
        self.s.behavior_report_type[self.s.behavior_report==-1] = 'i'
        self.s.behavior_report_type[np.all([self.s.behavior_report==1, self.s.task_trial_type=='r'],axis=0)] = 'r'
        self.s.behavior_report_type[np.all([self.s.behavior_report==0, self.s.task_trial_type=='r'],axis=0)] = 'l'
        self.s.behavior_report_type[np.all([self.s.behavior_report==1, self.s.task_trial_type=='l'],axis=0)] = 'l'
        self.s.behavior_report_type[np.all([self.s.behavior_report==0, self.s.task_trial_type=='l'],axis=0)] = 'r'
        return

    def parse_task_pole_times(self):
        '''
        task_pole_on_time: 1d numpy array of length (n_trials).
        task_pole_off_time: 1d numpy array of length (n_trials).
            When during the trial the pole came on and off.
        '''
        self.s.task_pole_on_time = self.input['task_pole_time'][:,0]
        self.s.task_pole_off_time = self.input['task_pole_time'][:,1]


    def parse_task_cue_times(self):
        '''
        task_cue_on_time: 1d numpy array of length (n_trials).
        task_cue_off_time: 1d numpy array of length (n_trials).
            When during the trial the cue came on and off.
        '''
        if len(np.shape(self.input['task_cue_time'])) == 1:
            self.s.task_cue_on_time = self.input['task_cue_time']
            self.s.task_cue_off_time = None
        else:
            self.s.task_cue_on_time = self.input['task_cue_time'][:,0]
            self.s.task_cue_off_time = self.input['task_cue_time'][:,1]
        return

    def parse_stim_present(self):
        '''
        stim_present: 1d numpy array of length (n_trials).
            Whether a perturbation stim was delivered.
        '''
        stim_period_input = self.input['task_stimulation'][:,1]
        self.s.stim_present = ~np.isnan(stim_period_input)
        return

    def parse_stim_times(self):
        '''
        stim_on_time: 1d numpy array of length (n_trials).
        stim_off_time: 1d numpy array of length (n_trials).
            When during the trial the perturbation stim came on and off.
        '''
        self.s.stim_on_time = self.input['task_stimulation'][:,2]
        self.s.stim_off_time = self.input['task_stimulation'][:,3]
        return

    def parse_stim_period(self):
        '''
        stim_period: 1d numpy array of length (n_trials).
            Indicates with an integer code when during the trial the
            perturbation stimulus was delivered.
        '''
        stim_period_input = self.input['task_stimulation'][:,1]
        self.s.stim_period = np.zeros_like(stim_period_input)
        self.s.stim_period[stim_period_input == 1] = 1 #early sample
        self.s.stim_period[stim_period_input == 2] = 3 #early delay
        return

    def parse_stim_period_num2str_dict(self):
        '''
        stim_period_num2str_dict: Python dict mapping integer code to string
            indicating stim period. Must match mapping in parse_stim_period.
        '''
        # Association between number and string encodings of stim period
        self.s.stim_period_num2str_dict = {0:'no stim', 1:'early sample', 3:'early delay'}
        return

    def parse_stim_site(self):
        '''
        stim_site: 1d numpy array of length (n_trials).
            Indicates with an integer code where the perturbation stimulus
            was delivered.
        '''
        stim_site_input = self.input['task_stimulation'][:,4]

        self.s.stim_site = np.zeros_like(stim_site_input)
        self.s.stim_site[stim_site_input == 1] = 1 #ipsi ALM silencing
        self.s.stim_site[stim_site_input == 2] = 3 #bi ALM silencing
        self.s.stim_site[stim_site_input == 3] = 4 #Cb fastigial silencing
        return

    def parse_stim_site_num2str_dict(self):
        '''
        stim_site_num2str_dict: Python dict mapping integer code to string
            indicating stim site. Must match mapping in parse_stim_site.
        '''
        # Association between number and string encodings of stim site
        self.s.stim_site_num2str_dict = {0:'no stim', 1:'ipsi ALM', 3:'bi ALM', 4:'FN'}
        return
