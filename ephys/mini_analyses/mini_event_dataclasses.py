"""
Define some data classes for the mini event analysis. This module includes:
Filtering: a class to keep track of what kind of filtering has been done, and the 
values.
AverageEvent: A class to hold information about averaged mini events (usually for a single protocol)
Mini_Event_Summary: A class to hold information about all the mini events detected for a single
protocol.
"""
from dataclasses import dataclass, field
from typing import Union, List
import numpy as np


@dataclass
class Filtering:
    enabled: bool=True
    LPF_applied: bool = False
    LPF_type: str = "ba"  # sos for second order or "ba" (standard)
    LPF_frequency: Union[float, None] = None

    HPF_applied: bool = False
    HPF_frequency: Union[float, None] = None
    HPF_type: str = "ba"
    
    Notch_applied: bool = False
    Notch_frequencies: Union[float, None] = None
    Notch_Q: float=None
    
    Detrend_applied: bool = True 
    Detrend_type: str = "meegkit"  # or "scipy"
    Detrend_order: int = 5



def def_empty_list():
    return [] 


def def_empty_list2D():
    return [[None]]


@dataclass
class AverageEvent:
    """
    The AverageEvent class holds averaged events
    This might be used for different kinds of events (amplitude distributions, etc.)
    """

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


@dataclass
class Mini_Event_Summary:
    """
    The Mini_Event_Summary dataclass holds the results of the
    individual events that were detected,
    as well as the results of various fits
    and the averge fit
    This usually applies to a single protocol (including across trials)
    """

    dt_seconds: float = 2e-5  # seconds
    filtering: object = field(default_factory=Filtering)

    spont_dur: float = 0.1  # seconds

    onsets: Union[List, np.ndarray] = field(  # onset indices for detected events
        default_factory=def_empty_list2D
    )
    ok_onsets: Union[
        List, np.ndarray
    ] = field(  # onset times for detected events with artifact rejection
        default_factory=def_empty_list2D
    )
    peakindices: Union[
        List, np.ndarray
    ] = field(  # peak indices (not denoised) for detected events
        default_factory=def_empty_list
    )
    smpkindex: Union[List, np.ndarray] = field(  # peak indices for smoothed peaks
        default_factory=def_empty_list
    )
    smoothed_peaks: Union[List, np.ndarray] = field(  # smoothed peaks (amplitudes)
        default_factory=def_empty_list
    )
    amplitudes: Union[List, np.ndarray] = field(  # event amplitudes
        default_factory=def_empty_list
    )
    Qtotal: Union[List, np.ndarray] = field(  # charge for each event
        default_factory=def_empty_list
    )
    crit: Union[
        List, np.ndarray
    ] = field(  # peak times (not denoised) for detected events
        default_factory=def_empty_list
    )
    scale: Union[
        List, np.ndarray
    ] = field(  # peak times (not denoised) for detected events
        default_factory=def_empty_list
    )
    individual_events: bool = False
    average: object = field(default_factory=AverageEvent)  # average
    average25: object = field(
        default_factory=AverageEvent
    )  # average of lower 25th percentile
    average75: object = field(
        default_factory=AverageEvent
    )  # average of upper 25th percentile
    average_spont: object = field(
        default_factory=AverageEvent
    )  # average of spontaneous events
    average_evoked: object = field(
        default_factory=AverageEvent
    )  # average of spontaneous events

    allevents: Union[
        List, np.ndarray
    ] = field(  # array of all events (2d, trace, event waveform)
        default_factory=def_empty_list
    )
    all_event_indices: Union[
        List, None
    ] = field(  # list linking all events to parent trace
        default_factory=def_empty_list
    )
    # list linking isolated (clean) events to parent trace list
    isolated_event_trace_list: Union[
        List, None
    ] = field(  
        default_factory=def_empty_list
    )
    spontaneous_event_trace_list: Union[
        List, None
    ] = field(  
        default_factory=def_empty_list
    )
    evoked_event_trace_list: Union[
        List, None
    ] = field(  
        default_factory=def_empty_list
    )

@dataclass
class Artifacts:
    """
    dataclass to hold some information about artifact timing (events at fixed times such as stimulation, shutters, etc)
    """
    starts : Union[ # list of starting times for artifacts
        List, None
    ] = field(  
        default_factory=def_empty_list
    )

    durations : Union[
        List, None
    ] = field(  # list of artifact durations
        default_factory=def_empty_list
    )
