import numpy as np

from ephys.mini_analyses import mini_event_dataclasses as MEDC
from dataclasses import dataclass, field

class Reader():
    def __init__(self, data: MEDC.Mini_Event_Summary):
        """__init__ Provide access to the parts of the Mini_Event_Summary data structure
        This class provides readers that return some specific parts of the class to
        make analysis and subsequent plotting easier.

        The Mini_Event_Summary class has the following structure:
            dt_seconds: float = 2e-5  # seconds
            filtering: object = field(default_factory=Filtering)

            spont_dur: float = 0.1  # seconds
            events: list
            onsets: Union[List, np.ndarray]: list of onsets
            ok_onsets: # onset times for detected events with artifact rejection
            peakindices: # peak indices (not denoised) for detected events

            smpkindex: # iist of indices of smoothed peaks
            smoothed_peaks: list of the amplitudes of the smoothed peaks
            amplitudes:  event amplitudes - not smoothed
            Qtotal:  # charge for each event

            Arrays related to detected events by AJ or CB algorithm
            crit: # peak times (not denoised) for detected events
            scale:# peak times (not denoised) for detected events

            individual_events: bool = False  - determine whether individual events were analyzed

            average: (dataclass AverageEvent) Holding information about averaged events
            average25: AverageEvent # average of lower 25th percentile
            average75: AverageEvent # average of upper 25th percentile
            average_spont: AverageEvent # average of spontaneous events
            average_evoked: AverageEvent # average of evoked events

            allevents:  # array of all events (2d, trace, event waveform)
                 trial x events in trial
            all_event_indices: # list of indices (trial, trace) linking all events to parent trace
            isolated_event_trace_list: # list linking isolated (clean) events to parent trace list
            spontaneous_event_trace_list: list linking spontaneous events to allevents trace list
            evoked_event_trace_list: list linking evoked events to allevents trace list
            artifact_event_list: # list of artifacts

            
        The AverageEvent data class has:
            averaged: bool = False  # set flags in case of no events found
            avgeventtb: Union[List, np.ndarray] = field(  # time base for the event
                default_factory=def_empty_list
            )
            avgevent: Union[List, np.ndarray] = field(  # the event
                default_factory=def_empty_list
            )
            Nevents: int = 0  # number of events that were averaged
            avgnpts: int = 0  # number of points in the array
            fitted: bool = False  # Set True if data has been fitted
            fitted_tau1: float = np.nan  # rising time constant for 2-exp fit
            fitted_tau2: float = np.nan  # falling time constant for 2-exp fit
            fitted_tau3: float = np.nan  # slow superimposed event
            fitted_tau4: float = np.nan
            fitted_tau_ratio: float = np.nan
            risepower: float = np.nan
            best_fit: object = None  # best fit trace
            amplitude: float = np.nan  # amplitude from the fit
            amplitude2: float = np.nan  # amplitude for slow event
            avg_fiterr: float = np.nan  # fit error
            risetenninety: float = np.nan  # rise time (seconds), 10-90 %
            decaythirtyseven: float = np.nan  # fall time to 37% of peak

            
        The pkl files that are created by the mini analysis have a top dictionary with the following:
        dict_keys(['analysis_parameters', 'engine', 'method', 'Qr', 'Qb', 'ZScore', 'I_max',
          'positions', 'stimtimes', 'events', 
          'dataset', 'sign', 'rate', 'ntrials', 'analysisdatetime'])

        Parameters
        ----------
        data : Mini_Event_Summary
            dataclass that holds the Mini event data summary after analysis.

        """
        self.data = data

    def get_ZScore(self):
        return self.data['ZScore']
    
    def get_Qr(self):
        return self.data['Qr']
    
    def get_Qb(self):     
        return self.data['Qb']
    
    def get_I_max(self):
        return self.data['I_max']
    
    def get_positions(self):
        return self.data['positions']
    
    def get_stimtimes(self):
        return self.data['stimtimes']
    
    def get_events(self):
        """get_events 

        Returns
        -------
        all of the events, listed by trial
        """
        return self.data['events']
    
    def get_taus(self, trial):
        trial_events = self.get_events()[trial]
        print(trial_events.average)
        exit() 
        
    def get_ntrials(self):
        return self.data['ntrials']
    
    def get_sample_rate(self, trial):
        trial_events = self.get_events()[trial]
        return trial_events.dt_seconds
    
    def get_trial_events(self, trial, trace_list):
        trial_events = self.get_events()[trial]
        events = []
        tb = None
        for itrace in range(len(trace_list)):
            index = trial_events.all_event_indices[itrace]
            event = trial_events.allevents[index]
            events.append(event)
            if itrace == 0:
                tb=np.linspace(0., len(event)/trial_events.dt_seconds, len(event))
        return np.array(tb), np.array(events)
    
    def find_trace_in_indices(self, itrace, indices):
        for i, j in indices:
            if i == itrace:
                return i
        else:
            return None
        
    def get_trial_event_onset_times(self, trial, trace_list):
        trial_events = self.get_events()[trial]
        eventtimes = []
        for itrace in range(len(trace_list)):
            # print(trial_events.all_event_indices)
            # print("len event indices: ", len(trial_events.all_event_indices), " itrace: ", itrace)
            itr = self.find_trace_in_indices(itrace, trial_events.all_event_indices)
            if itr is None:
                events = []
            else:
                # index = trial_events.all_event_indices[itr]
                events = trial_events.onsets[itr] # [index]
            eventtimes.append(np.array(events)*trial_events.dt_seconds)
        return eventtimes

    def get_trial_event_smpks_times(self, trial, trace_list):
        trial_events = self.get_events()[trial]
        eventtimes = []
        for itrace in range(len(trace_list)):
            itr = self.find_trace_in_indices(itrace, trial_events.all_event_indices)
            if itr is None:
                events = []
            else:
                # index = trial_events.all_event_indices[itr]
                events = trial_events.smpkindex[itr] # [index]
            eventtimes.append(np.array(events)*trial_events.dt_seconds)
        return eventtimes

    def get_trial_event_amplitudes(self, trial, trace_list):
        trial_events = self.get_events()[trial]
        event_amplitudes = []
        for itrace in range(len(trace_list)):
            amplitudes = trial_events.amplitudes[itrace]
            event_amplitudes.append(amplitudes)
        return event_amplitudes

    def get_onsets(self):
        return self.data['events']['onsets']


