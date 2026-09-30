"""
Define some data classes for the mini event analysis. This module includes:
Filtering: a class to keep track of what kind of filtering has been done, and the 
values.
AverageEvent: A class to hold information about averaged mini events (usually for a single protocol)
Mini_Event_Summary: A class to hold information about all the mini events detected for a single
protocol.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, List, Optional
import numpy as np

def def_empty_list():
    return [] 



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
    Notch_frequencies: Union[float, list, str, None] = None
    Notch_Q: float=30.
    
    Detrend_enable: bool=True
    Detrend_applied: bool = False
    Detrend_method: Union[str, None] = "meegkit"  # or "scipy" or None
    Detrend_order: int = 5


    

def def_empty_list2D():
    return [[None]]

def def_mini_events():
    return Mini_Event()

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
    stdevent: Union[List, np.ndarray] = field(  # the event
        default_factory=def_empty_list
    )
    varevent: Union[List, np.ndarray] = field(  # the event
        default_factory=def_empty_list
    )
    Nevents: int = 0  # number of events that were averaged
    avgnpts: int = 0  # number of points in the array
    fitted: bool = False  # Set True if data has been fitted
    fitted_n_taus: int = 1  # number of rise/fall functions in the fit
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
class Mini_Event:
    """Define parameters of a single event, including the waveform itself
    This is a proposed structure for refactoring.
    """
    tag: tuple = (0,0)  # (trace, event#)
    onset_time: float=np.nan
    onset_index: int= 0
    peak_index: int=0
    smpk_index: int=0
    smoothed_peak: float=np.nan
    amplitude: float=np.nan
    Qtotal: float=np.nan
    baseline: float=np.nan
    event: np.ndarray = field(  # the event waveform
        default_factory=def_empty_list
    )
    classification: str="" # 'isolated', 'toosmall', 'evoked', 'spontaneous', 'artifact'

    
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
    events: list=field(default_factory=def_mini_events)
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
    average: AverageEvent = field(default_factory=AverageEvent)  # average

    average25: AverageEvent = field(
        default_factory=AverageEvent
    )  # average of lower 25th percentile
    average75: AverageEvent = field(
        default_factory=AverageEvent
    )  # average of upper 25th percentile
    average_spont: AverageEvent = field(
        default_factory=AverageEvent
    )  # average of spontaneous events
    average_evoked: AverageEvent = field(
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
    clean_event_onsets_list: Union[
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
    artifact_event_list: Union[
        List, None
    ] = field(
        default_factory=def_empty_list
    )
    direct_responses: Union[List, None] = field(  # per-trace list of DirectResponseRecord
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

def def_notch():
    """defaults for notch frequencies"""
    return [60.0, 120.0, 180.0, 240.0]


def def_stimtimes():
    return {"starts": [0.3], "durations": [3e-4]}

def def_steptimes():
    return {"starts": [0.6], "durations": [1e-2]}

def def_twin_base():
    return [0.0, 0.295]


def def_twin_response():
    return [0.301, 0.325]


def def_taus():
    return [0.0002, 0.005, 0.005, 0.030]

def def_shutter_artifacts():
    # delays to shutter artifacts, durations of shutter artifacts
    return  {"starts": [0.055], "durations": [0.005]}

def def_stim_artifacts():
    # delays to stim artifacts, durations of stim artifacts
    return  {"starts": [0.1, 0.3, 0.5, 0.7, 0.9], "durations": [0.0005, 0.0005, 0.0005, 0.0005, 0.0005]}

def def_analysis_window():
    return [0.0, 0.999]


@dataclass
class DirectSeparationPars:
    """
    Parameters controlling optional separation of direct (non-synaptic) responses
    from synaptic events. Design (revised from the original ephys-jl per-event
    classifier after benchmarking showed AJ/CB correctly detect fast synaptic
    events riding on a slow baseline without confusing them for it -- the direct
    response is a trace-level slow background, not itself a detected event):

      1. Detect fast events normally (AJ/CB, unchanged).
      2. Subtract every detected event's own contribution from the trace using
         the detector's own template, scaled to each event's measured amplitude.
      3. Fit the stimulus-locked window of what remains (now dominated by any
         slow direct component plus noise, fast content already removed).
      4. Subtract that fit from the ORIGINAL trace (not the event-subtracted
         residual) in the stimulus window.
      5. Re-run detection on the corrected trace for final measurements.

    Inactive unless AnalysisPars.direct_separation.enabled is True.
    """
    enabled: bool = False
    detect_hpf: float = 100.0  # Hz; high-pass cutoff applied ONLY to the event-detection
    # step so AJ/CB do not fire on the slow direct response (0 disables). The slow direct is
    # below this; synaptic PSC content is above it. Validated: same-sign spurious detections
    # 24->4 with spontaneous/evoked recall preserved.
    direct_window_start: float = 0.0  # seconds after stimulus to start the fit window
    direct_window_duration: float = 0.120  # seconds; long enough for a typical direct response
    # to decay within the window (a shorter window leaves a step discontinuity at the window
    # edge where the un-decayed direct response is cut off); tune to the data's direct kinetics.
    fit_mode: str = "spline"  # or "template"
    mindy: float = 7.5e-8  # derivative threshold (A/s); see estimate_robust_derivative
    derivative_window_points: int = 9  # ~0.18 ms at dt=20 us; tune relative to tau1
    spline_smooth_factor: float = 1.0  # multiplies the noise-estimated spline smoothing factor
    # template fit_mode ("template") parameters:
    risepower: int = 2  # exponent n on the (1-exp(-t/tau_rise)) rising phase
    template_ndecay: int = 1  # number of decay time constants (1 or 2) in the direct template
    min_direct_charge: float = 1.0e-14  # A*s; fits below this magnitude are not recorded as a direct response
    # --- Phase 2: multi-trace (cross-trace) direct-template estimation ---
    multitrace: bool = False  # if True, estimate a direct-shape template across a map's traces
    # and fit only its amplitude(+shift) per trace, instead of a free per-trace fit
    use_qr_selection: bool = False  # select direct-containing traces by evoked charge Qr
    # (Phase 2 future); when False, select by the per-trace fitted-direct charge
    direct_select_charge_factor: float = 3.0  # k: a trace is "direct" if its fitted-direct
    # charge exceeds median + k*MAD across the map's traces
    n_shape_groups: int = 1  # number of direct-shape clusters/templates (1 = single template)
    template_form: str = "data"  # "data" (averaged waveform) or "parametric" (fit to the average)


def def_empty_list_of_lists():
    return [[]]


@dataclass
class DirectResponseRecord:
    """
    Records a fitted direct (non-synaptic) response for a single stimulus on
    a single trace, kept for downstream use (e.g. as training/guidance data
    for a future ML-based direct/synaptic separation classifier) beyond just
    correcting the trace in-place.
    """
    trace: int = 0  # index of the trace (map spot) this record belongs to
    stim_index: int = 0  # which stimulus in stimtimes this record corresponds to
    onset_time: float = np.nan  # seconds, absolute (window start = stim + direct_window_start)
    duration: float = np.nan  # seconds, length of the fitted window actually used
    charge: float = np.nan  # A*s, integral of the fitted direct waveform over its window
    peak_amplitude: float = np.nan  # A, peak value of the fitted direct waveform
    fit_mode: str = ""


@dataclass
class AnalysisPars:
    """
    Data class that holds most of the analysis parameters
    This class is also passed to the plotMapdata routines
    (display_one_map).
    """

    dt_seconds: float = 2e-5  # sample rate in seconds
    spotsize: float = 42e-6  # spot size in meters
    direct_separation: object = field(
        default_factory=DirectSeparationPars
    )  # optional direct/synaptic response separation; inert unless .enabled is True
    baseline_flag: bool = False
    baseline_subtracted: bool = False
    filters: object = field(default_factory=Filtering)
    zscore_threshold: float = 1.96  # threshold for zscore
    # artifact suppression: Shutter and stim are artifacts at fixed times
    artifact_suppression: bool = False  # flag enabling suppression of artifacts
    shutter_artifacts: dict = field(
        default_factory=def_shutter_artifacts
    )  # time of shutter electrical artifact
    stim_artifacts: dict = field(
        default_factory = def_stim_artifacts
    )
    stimdur: Union[float, None] = None  # duration of stimuli
    # artifact removeal by derivative method (do not use with fast EPSCs)
    
    artifact_derivative: bool = (
        False  # flag enabling artifact suppression based on derivative of trace
    )
    sd_thr: float = 3.0  # threshold in sd for derivative-based artifact suppression.
    
    post_analysis_artifact_rejection: bool = False  # clean up AFTER analysis
    artifact_filename: Union[Path, None] = None  # file with artifact "traces" to subtract
    artifact_path: Union[Path, None] = None  # path to the artifact files
    LaserBlueTimes: Union[dict, None] = None
    LEDTimes: Union[dict, None] = None
    artifactData: Union[dict, None] = None
    artifact_scale: Union[float, None] = None
    artifact_autoscale: bool = True
    artifact_epoch: Union[int, None] = None   # which artifact file to use (divided by time epochs )
    
    ar_tstart: float = 0.10  # starting time for VC or IC stimulus pulse
    ar_tend: float = 0.015  # end time for VC or IC stimulus pulse
    time_zero: float = 0.0  # in seconds, where we start the trace
    time_zero_index: int = 0
    time_end: float = 1.0  # in seconds, end time of the trace
    time_end_index: int = 0  # needs to be set from the data
    stimtimes: dict = field(default_factory=def_stimtimes)
    currentsteptimes: dict = field(default_factory=def_steptimes)
    spont_deadtime: float = (
        0.010  # time after trace onset before counting spont envents
    )
    direct_window: float = 0.0005  # window after stimulus for direct response
    response_window: float = 0.015  # window end for response (response is between direct and response), seconds
    twin_base: list = field(
        default_factory=def_twin_base
    )  # time windows baseline measures
    twin_resp: list = field(
        default_factory=def_twin_response
    )  # time windows for repeated responses
    analysis_window: list = field(
        default_factory=def_analysis_window
    )  # window for data analysis
    # taus = [0.5, 2.0]
    template_tmax = 0.0025 # seconds
    template_pre_time:float = 0.0 # seconds
    risepower: float = 4.0
    taus: list = field(
        default_factory=def_taus
    )  # initial taus for fitting detected events
    threshold: float = 3.0  # default threshold for CB, AJ or ZC
    sign: int = -1  # negative for EPSC, positive for IPSC
    datatype: str = "V"  # data type - VC or IC
    stepi: float = (
        25.0  # step size for stacked traces, in pA (25 default for cc; 2 for cc)
    )
    scale_factor: float = 1.0  # scale factore for data (convert to pA or mV,,, )
    overlay_scale: float = 0.0

    global_SD: float = 0.0  # raw global SD
    global_mean: float = 0.0  # raw global mean
    global_trim_scale: float = 3.0
    global_trimmed_SD: float = 0.0  # raw global trimeed SD with outliers > 3.0
    global_trimmed_median: float = 0.0


@dataclass
class AnalysisData:
    """
    Data class that holds the analysis data separate from the parameters
    and metadata
    This class is also made available to the plotMapdata routines
    """

    timebase: Union[None, np.ndarray] = None  # the time base corresponding to the clean data
    data_clean: Union[None, np.ndarray] = None  # data with filtering, detrending, artifact removal
    raw_data_averaged: Union[None, np.ndarray] = None  # average of raw data
    raw_timebase: Union[None, np.ndarray] = None  # raw data time base
    photodiode: Union[None, np.ndarray] = None  # photodiode current trace (monitor)
    photodiode_timebase: Union[None, np.ndarray] = None # photodiode time base matching current trace
    MA: Union[object, None] = None  # point to minanalysis instance used for analysis



    