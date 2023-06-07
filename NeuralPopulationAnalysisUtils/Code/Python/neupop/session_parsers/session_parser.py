from abc import ABC, abstractmethod

class SessionParser(ABC):
    '''
    Abstract class for parsing all necessary parameters in a Session object.
    Each subparser method should populate one or two class variables of Session
        as indicated in comments.
    Subparser methods must be overwritten when inheritted.
    parse_all() calls all existing subparser methods and returns the Session.
    '''
    def __init__(self, session):
        self.s = session
        # Additional init code can be included.
        return

    def parse_all(self):
        '''
        Parses and stores all necessary parameters in session using subparsers
        Returns the Session object (self.s)
        '''
        self.parse_neural_data()
        self.parse_neural_unit_info()
        self.parse_behavior_report()
        self.parse_behavior_early_report()
        self.parse_task_trial_type()
        self.parse_behavior_report_type()
        self.parse_task_pole_times()
        self.parse_task_cue_times()
        self.parse_stim_present()
        self.parse_stim_times()
        self.parse_stim_period()
        self.parse_stim_period_num2str_dict()
        self.parse_stim_site()
        self.parse_stim_site_num2str_dict()
        self.parse_other()
        self.parse_i_good_trials()
        self.parse_state()

        return self.s

    @abstractmethod
    def parse_neural_data(self):
        '''
        neural_data: 2d numpy array of shape (n_units, n_trials).
            Each entry is a list containing spike times.
        '''
        pass

    @abstractmethod
    def parse_neural_unit_info(self):
        '''
        neural_unit_depth: 1d numpy array of length n_units, dtype=float
            Depth in um (micrometers)

        neural_unit_type: 1d numpy array of length n_units, dtype=string
            Cell type (e.g. putative_pyramidal, putative_interneuron)

        neural_unit_location: 1d numpy array of length n_units, dtype=string
            Recording location (e.g. left_ALM, right_ALM)
        '''
        pass

    @abstractmethod
    def parse_behavior_report(self):
        '''
        behavior_report: 1d numpy array of length (n_trials)
            Whether the animal responded correctly.
        It is coded as follows:
            -1: ignored
            0 : incorrect
            1 : correct
        '''
        pass

    @abstractmethod
    def parse_behavior_early_report(self):
        '''
        behavior_early_report: 1d numpy array of length (n_trials).
            Whether the animal improperly responded before cue.
        It is coded as follows:
            0: no early report (proper behavior)
            1: early report (improper behavior)
        '''
        pass

    @abstractmethod
    def parse_behavior_report_type(self):
        '''
        behavior_report_type: 1d numpy array of length (n_trials).
            Whether the animal responded with lick right ('r') or left ('l').
        It is coded as follows:
            'i': ignored
            'r': licked right
            'l': licked left
        '''
        pass

    @abstractmethod
    def parse_task_trial_type(self):
        '''
        task_trial_type: 1d numpy array of length (n_trials).
            Whether the animal was cued to lick right ('r') or left ('l').
        It is coded as follows:
            'r': cued right
            'l': cued left
        '''
        pass

    @abstractmethod
    def parse_task_pole_times(self):
        '''
        task_pole_on_time: 1d numpy array of length (n_trials).
        task_pole_off_time: 1d numpy array of length (n_trials).
            When during the trial the pole came on and off.
        '''
        pass

    @abstractmethod
    def parse_task_cue_times(self):
        '''
        task_cue_on_time: 1d numpy array of length (n_trials).
        task_cue_off_time: 1d numpy array of length (n_trials).
            When during the trial the cue came on and off.
        '''
        pass

    @abstractmethod
    def parse_stim_present(self):
        '''
        stim_present: 1d numpy array of length (n_trials).
            Whether a perturbation stim was delivered.
        '''
        pass

    @abstractmethod
    def parse_stim_times(self):
        '''
        stim_on_time: 1d numpy array of length (n_trials).
        stim_off_time: 1d numpy array of length (n_trials).
            When during the trial the perturbation stim came on and off.
        '''
        pass

    @abstractmethod
    def parse_stim_period(self):
        '''
        stim_period: 1d numpy array of length (n_trials).
            Indicates with an integer code when during the trial the
            perturbation stimulus was delivered.
        '''
        pass

    @abstractmethod
    def parse_stim_period_num2str_dict(self):
        '''
        stim_period_num2str_dict: Python dict mapping integer code to string
            indicating stim period. Must match mapping in parse_stim_period.
        '''
        pass

    @abstractmethod
    def parse_stim_site(self):
        '''
        stim_site: 1d numpy array of length (n_trials).
            Indicates with an integer code where the perturbation stimulus
            was delivered.
        '''
        pass

    @abstractmethod
    def parse_stim_site_num2str_dict(self):
        '''
        stim_site_num2str_dict: Python dict mapping integer code to string
            indicating stim site. Must match mapping in parse_stim_site.
        '''
        pass

    def parse_other(self):
        '''
        Method for parsing parameters not included in other subparsers.
        Not required.
        '''
        pass
    

    @abstractmethod
    def parse_i_good_trials(self):
        pass

    @abstractmethod
    def parse_state(self):
        '''
        State: either 1 and 2 or 1 through 3, indicating mouse behavioral state
        '''
        pass