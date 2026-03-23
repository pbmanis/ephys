import pickle
import warnings
from dataclasses import dataclass
from typing import Union

import matplotlib.pyplot as mpl
import numpy as np
from pylibrary.plotting import plothelpers as PH
from pylibrary.tools import cprint
from sklearn.linear_model import LinearRegression

import ephys.tools.exp_estimator_lmfit as exp_estimator_lmfit

CP = cprint.cprint

""" Adaptation index calculations

    Adaptation indices are calculated a number of different ways in the literature.
    This module is a compilation of several different methods.

"""

# all_adapt_indices = [adaptation_index_MKX19, adaptation_index_AllenInst, adaptation_first_last_isi, adaptation_index_exponential]


def adaptation_index_MKX19(spk_lat, trace_duration: float = 1.0, minimum_spikes: int = 4):
    """adaptation_index Compute an adaptation index based on Manis, Kasten and Xie, PLoS One, 2019
    This measure simply takes the difference between the spike counts in the first and second halves of the trace,
    and normalizes it by the total number of spikes.
    The adaptation index goes from:
    1 (only a spike in the frst half of the stimulus;
    no spikes in second half)
    to 0 (rate in the first and second half are identical)
    to -1 (all the spikes are in the second half)

    Parameters
    ----------
    spk_lat : _type_
        _description_
    trace_duration : float, optional
        _description_, by default 1.0
    minimum_spikes : int, optional
        minimum number of spikes required to compute the adaptation index, by default 4

    Returns
    -------
    float
        adaptation index as described above
    """
    if len(spk_lat) < minimum_spikes:
        return np.nan
    ai = (-2.0 / len(spk_lat)) * np.sum((spk_lat / trace_duration) - 0.5)
    return ai


def adaptation_index_AllenInst(
    spike_data: Union[list, np.ndarray], trace_duration: float = 1.0, minimum_spikes: int = 4, input: str = "spike_latencies"
):
    """
    Allen Institute ephys SDK version (using normalized diff of ISIs)
    Computes the adaptation index for one list of spike times (or ISIs)
    using the method described by the Allen Institute.
    ----------
    spike_data : array
        spike latencies or ISIs, already trimmed to current step window
        (start time should be 0 or start of the step; the end time should be trace duration,
        but might be shorter if gathering data from traces with different durations).
    trace_duration : float, optional
        duration of the current step, in seconds, used to trim the spike latencies to the
        step window, by default 1.0
    minimum_spikes : int, optional
        minimum number of spikes required to compute the adaptation index, by default 4
    input: str, optional
        type of input data, either "spike_latencies" or "isis", by default "spike_latencies".
        If "spike_latencies", the function will compute ISIs from the spike latencies.

    Returns
    -------
    float
        adaptation index as described above
    """
    assert input in ["spike_latencies", "isis"], "Input must be either 'spike_latencies' or 'isis'"

    if len(spike_data) < minimum_spikes:
        return np.nan

    # note: isis must be used here, not spike latencies. convert as intended.
    if input == "spike_latencies":
        # window to spikes during the stimulus period
        spike_data = spike_data[(spike_data >= 0.0) & (spike_data < trace_duration)]
        isis = np.diff(spike_data)
    elif input == "isis":
        # assumes the intended isis have already been provided.
        isis = spike_data
    else:
        raise ValueError("Invalid input type. Must be 'spike_latencies' or 'isis'.")

    # Allen Institute version:
    # The ratio is computed from the normalized difference between consecutive ISIs,
    # where the normalized difference is defined as (ISI_n+1 - ISI_n) / (ISI_n+1 + ISI_n).
    # The overall adaptation index is then calculated as the mean of these normalized differences
    # across all pairs of consecutive ISIs. Values close to 0 indicate little to no adaptation,
    # positive values indicating increasing ISIs (adaptation),
    # and negative values indicating decreasing ISIs (acceleration).
    if np.allclose((isis[1:] + isis[:-1]), 0.0):
        return np.nan

    norm_diffs = (isis[1:] - isis[:-1]) / (isis[1:] + isis[:-1])
    norm_diffs[(isis[1:] == 0) & (isis[:-1] == 0)] = 0.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
        adaptation_index = np.nanmean(norm_diffs)  # get mean across the trace
    return adaptation_index


def adaptation_index_AllenInst_spktimes(
    spike_data: Union[list, np.ndarray], trace_duration: float = 1.0, minimum_spikes: int = 4
):
    """
    Allen Institute ephys SDK version (norm_diff)
    Computes the adaptation index for one list of spike times (or ISIs)
    using the method described by the Allen Institute.
    ----------
    spike_data : array
        spike latenciesalready trimmed to current step window
        (start time should be 0 or start of the step; the end time should be trace duration,
        but might be shorter if gathering data from traces with different durations).
    trace_duration : float, optional
        duration of the current step, in seconds, used to trim the spike latencies to the
        step window, by default 1.0
    minimum_spikes : int, optional
        minimum number of spikes required to compute the adaptation index, by default 4

    Returns
    -------
    float
        adaptation index as described above
    """

    spike_data = spike_data[(spike_data >= 0.0) & (spike_data < trace_duration)]
    if len(spike_data) < minimum_spikes:
        return np.nan

    # Allen Institute version:
    # The ratio is computed from the normalized difference between consecutive ISIs,
    # where the normalized difference is defined as (ISI_n+1 - ISI_n) / (ISI_n+1 + ISI_n).
    # The overall adaptation index is then calculated as the mean of these normalized differences
    # across all pairs of consecutive ISIs. Values close to 0 indicate little to no adaptation,
    # positive values indicating increasing ISIs (adaptation),
    # and negative values indicating decreasing ISIs (acceleration).
    # if np.allclose((spike_data[1:] + spike_data[:-1]), 0.0):
    #     return np.nan

    norm_diffs = (spike_data[1:] - spike_data[:-1]) / (spike_data[1:] + spike_data[:-1])
    norm_diffs[(spike_data[1:] == 0) & (spike_data[:-1] == 0)] = 0.0
    # print("norm_diffs: ", norm_diffs)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
        adaptation_index = np.nanmean(norm_diffs)  # get mean across the trace
    return adaptation_index


def adaptation_first_last_isi(spk_lat, trace_duration: float = 1.0, minimum_spikes: int = 4):
    """adaptation_first_last_isi
    Compute an adaptation index as the ratio of the mean of the last 3 ISIs
    to the first ISI.
    The spike latency array assumes that the start of the current step
    is at time 0.
    Parameters
    ----------
    spk_lat : array
        spike latencies
    trace_duration : float, optional
        duration of the current step, in seconds, used to trim the spike latencies to the step window, by default 1.0
    minimum_spikes : int, optional
        minimum number of spikes required to compute the adaptation index, by default 4

    Returns
    -------
    float
        adaptation index as described above
    """

    spk_lat = spk_lat[(spk_lat < trace_duration) & (spk_lat >= 0.0)]
    if len(spk_lat) < minimum_spikes:
        return np.nan

    first_isi = spk_lat[1] - spk_lat[0]
    last_isi = np.mean(np.diff(spk_lat[-3:]))
    if first_isi == 0:
        return np.nan
    adapt_index = last_isi / first_isi
    return adapt_index


def adaptation_index_exponential(spk_lat, trace_duration: float = 1.0, minimum_spikes: int = 4):
    """adaptation_index_exponential
    Compute an adaptation index by fitting an exponential function to the
    instantaneous firing rate.
    The function is:
    rate(t) = A * exp(-t/tau) + B
    where:
        A is the amplitude of the exponential decay,
        tau is the time constant of the decay
        B is the "steady-state" firing rate. The trace duration should be long
        enough that this can be estimated from the data.


    Parameters
    ----------
    spk_lat : array
        spike latencies
    trace_duration : float, optional
        duration of the current step, in seconds, used to trim the spike latencies to the step window, by default 1.0
    minimum_spikes : int, optional
        minimum number of spikes required to compute the adaptation index, by default 4
    """

    spk_lat = spk_lat[(spk_lat < trace_duration) & (spk_lat >= 0.0)]
    if len(spk_lat) < minimum_spikes:
        return np.nan

    # compute instantaneous firing rate as the inverse of the ISI
    isis = np.diff(spk_lat)
    inst_rate = 1.0 / isis
    time_base_fit = spk_lat[:-1]  # time points corresponding to the instantaneous rate
    LME = exp_estimator_lmfit.LMexpFit()
    t_fit = time_base_fit - time_base_fit[0]
    LME.initial_estimator(t_fit, inst_rate, verbose=False)
    # print(f"estm: DC: {LME.DC:8.3f}, A1: {LME.A1:8.3f}, R1: {LME.R1:8.3f}")
    fit = LME.fit1(t_fit, inst_rate, plot=False, verbose=False)
    # print(f"Fit:  DC: {fit.params['DC'].value:8.3f}, A1: {fit.params['A1'].value:8.3f}, R1: {fit.params['R1'].value:8.3f}")
    fit_curve = LME.exp_decay1(
        x=t_fit,
        DC=fit.params["DC"].value,
        A1=fit.params["A1"].value,
        R1=fit.params["R1"].value,
    )
    # define adaptation index as the ratio of the amplitude of the exponential decay to the steady-state rate
    if fit.params["DC"].value == 0:  # rate decays to zero, so adaptation index is not defined.
        return np.nan
    adapt_index = fit.params["DC"].value / (fit.params["A1"].value + fit.params["DC"].value)
    if adapt_index < -20.0:
        return np.nan
    return adapt_index


# function is not called anywhere here.
def compute_adaptation_index(
    method: str,
    spikes,
    trace_delay: float = 0.15,
    trace_duration: float = 1.0,
    rate_bounds: list = [20.0, 40.0],
    minimum_spikes: int = 4,
):
    """compute_adapt_index
    Compute the adaptation index for a set of spikes
    The adaptation index is the ratio of the last interspike interval
    to the first interspike interval.

    Parameters
    ----------
    method: str
        method to compute adaptation index. Must be one of "MKX19" or "AllenInst"
    spikes : dictionary
        a dictionary of lists of spike times, with 0 time at the start of the stimulus
        The dictionary key is the trace number (sweep).
        Each sweep holds a list of the spike times (in seconds) for that trace.
    trace_delay: float, seconds
        delay to the start of the stimulus, in seconds. This is subtracted from the spike
        latencies to get the time from the start of the stimulus.
    trace_duration: float, seconds
        duration of the current step, in seconds, used to trim the spike latencies to the step window.
    rate_bounds: list of two floats, in Hz (spikes per second)
        lower and upper bounds on the firing rate for which to compute the adaptation index.
    minimum_spikes: int (default 4)
        minimum number of spikes required to compute the adaptation index.


    Returns
    -------
    adapt_indices, adapt_rates (tuple)
        adaptation indices, rates at which index was computed
    """

    recnums = list(spikes.keys())
    adapt_rates = []
    adapt_indices = []

    for rec in recnums:
        spikelist = list(spikes[rec].keys())
        spk_lat = np.array([spk for spk in spikelist if spk is not None])
        # check if enough spikes
        if spk_lat is None or len(spk_lat) < minimum_spikes:
            continue

        # clip spike latencies to the window
        spk_lat -= trace_delay
        spk_lat = spk_lat[spk_lat < trace_duration and spk_lat >= 0.0]
        n_spikes = len(spk_lat)
        if n_spikes < minimum_spikes:
            continue
        rate = n_spikes / (spk_lat[-1] - spk_lat[0])
        if rate < rate_bounds[0] or rate > rate_bounds[1]:
            # CP("y", f"Adaption calculation: rec: {rec:d} failed rate limit: {rate:.1f}, {rate_bounds!s}, spk_lat: {spk_lat!s}")
            continue
        # else:
        #     CP("c", f"Adaption calculation: rec: {rec:d} rate: {rate:.1f}, spk_lat: {spk_lat!s} PASSED")
        adapt_rates.append(rate)
        match method:
            case "MKX19":
                adapt_indices.append(adaptation_index_MKX19(spk_lat, trace_duration, minimum_spikes=minimum_spikes))
            case "AllenInst":
                adapt_indices.append(
                    adaptation_index_AllenInst(spk_lat, trace_duration, input="spike_latencies", minimum_spikes=minimum_spikes)
                )
            case "FirstLastISI":
                adapt_indices.append(adaptation_first_last_isi(spk_lat, trace_duration, minimum_spikes=minimum_spikes))
    #    CP("y", f"Adaptation index: {adapt_indices[-1]:6.3f}, rate: {rate:6.1f}")

    return adapt_indices, adapt_rates


@dataclass
class SpikeData:
    """A class to hold the spike data and adaption calculations
    for a single sweep (record - recnum).

    """

    recnum: int
    spk_lat: (
        np.ndarray
    )  # spike latencies, in seconds, relative to the start of the stimulus step (i.e., 0 time is the start of the step)
    nspikes: int = 0
    rate: float = np.nan  # firing rate in Hz (spikes per second)
    AI_adapt_index: float = np.nan
    adapt_index_MKX19: float = np.nan
    adapt_index_AllenInst: float = np.nan
    adapt_index_AllenInst_spktimes: float = np.nan
    adapt_index_FirstLastISI: float = np.nan
    adapt_index_exponential: float = np.nan

    def print_head(self):
        """Print the header for the adaptation index table."""
        print(
            f"{'RecNum':>6s} {'NSpikes':>8s} {'Rate':>8s} {'AI_Adapt':>10s} {'MKX19':>10s} {'AllenInst':>12s} {'AllenInst_SpkTimes':>18s} {'FirstLastISI':>14s} {'ExpFit':>10s}"
        )

    def print_row(self):
        """Print the data for this sweep in a row of the adaptation index table."""
        print(
            f"{self.recnum:6d} {self.nspikes:8d} {self.rate:8.1f} {self.AI_adapt_index:10.5f} {self.adapt_index_MKX19:10.5f}",
            end="",
        )
        print(
            f"{self.adapt_index_AllenInst:10.5f} {self.adapt_index_AllenInst_spktimes:18.5f} {self.adapt_index_FirstLastISI:14.5f} {self.adapt_index_exponential:10.5f}"
        )

def linfit(x, y, xrange=[-0.2, 0.2]):
    """linfit
    Fit a line to the data and return the slope and intercept.
    Parameters
    ----------
    x : array
        x values
    y : array
        y values

    Returns
    -------
    slope, intercept (tuple)
        slope and intercept of the fitted line
    """
    lr = LinearRegression()
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    notnan = np.where(~np.isnan(x) & ~np.isnan(y))[0]
    x = x[notnan]
    y = y[notnan]

    lr.fit(x, y)
    r_2 = lr.score(x, y)
    x_fit = np.linspace(xrange[0], xrange[1], 100).reshape(-1, 1)
    y_fit = lr.predict(x_fit)
    return lr.coef_[0][0], lr.intercept_[0], r_2, x_fit, y_fit

def tests(ncells:int=5, minimum_spikes:int=4, plot_flag:bool=False):
    import allensdk.ephys.ephys_extractor as ephys_extractor
    import allensdk.ephys.extract_cell_features as extract_cell_features
    import allensdk.ephys.feature_extractor as ephys_feature_extractor
    import h5py
    from allensdk.core.cell_types_cache import CellTypesCache

    # from pynwb import NWBHDF5IO

    ctc = CellTypesCache(manifest_file="cell_types/manifest.json")
    cells = ctc.get_cells()
    axarr = []
    if plot_flag:
        r, c = PH.getLayoutDimensions(ncells)
        P = PH.regular_grid(rows=r, cols=c, order="rowsfirst", figsize=(11, 8.5),
                            )
        axarr = P.axarr.ravel()  # in order, linear
        labeled = [False] * ncells
    print(f"# of cells in manifest: {len(cells)}")
    cell_ids =  range(ncells)  # [0, 1, 2, 3]

    comparisions = ["MKX19", "AllenInst", "AllenInst_SpkTimes", "FirstLastISI", "ExpFit"]
    cell_data = {}
    adapt_data = {}


    for icell in cell_ids:
        print(f"\n\n{'='*80}")
        print(f"Cell {cells[icell]['id']:d}")
        cell_specimen_id = cells[icell]["id"]  # will be downloaded the first time only
        data_set = ctc.get_ephys_data(cell_specimen_id)
        # print("Spike times: ", data_set.get_sweep(4)["response"])
        adapt_data[cell_specimen_id] = []
        this_cell_data = []
        h5py._errors.unsilence_errors()
        # Get sweep numbers for "Long Square"
        sweep_numbers = data_set.get_sweep_numbers()

        sweep_types = []
        for sweep_number in sweep_numbers:
            sweep_metadata = data_set.get_sweep_metadata(sweep_number)
            # print(f"Sweep {sweep_number} metadata: {sweep_metadata['aibs_stimulus_name']!s}")
            sweep_types.append(sweep_metadata["aibs_stimulus_name"])
        # sweeps_used = list(set(sweep_types)) # get unique sweep types

        use_data_type = b"Long Square"
        sweeps_by_type = {use_data_type: []}
        # if plot_flag:
        #     fig, ax = mpl.subplots(2, 1)

        # accumulate traces
        t_sweep = []
        v_sweep = []
        i_sweep = []
        for sweep_number in sweep_numbers:
            sweep_metadata = data_set.get_sweep_metadata(sweep_number)
            if sweep_metadata["aibs_stimulus_name"] != b"Long Square":
                continue
            sweep_data = data_set.get_sweep(sweep_number)
            sweeps_by_type[use_data_type].append(sweep_number)
            index_range = sweep_data["index_range"]
            i_cmd = sweep_data["stimulus"][0 : index_range[1] + 1]  # in A
            v_rec = sweep_data["response"][0 : index_range[1] + 1]  # in V
            i_cmd *= 1e12  # to pA
            v_rec *= 1e3  # to mV
            sampling_rate = sweep_data["sampling_rate"]  # in Hz
            t = np.arange(0, len(v_rec)) * (1.0 / sampling_rate)
            this_cell_data.append(
                {"id": cell_specimen_id, "sweep": sweep_number, "t": t, "i": i_cmd, "v": v_rec}
            )
            t_sweep.append(t)
            v_sweep.append(v_rec)
            i_sweep.append(i_cmd)

            if plot_flag:
                # axarr[icell].plot(t, i_cmd, label=f"Sweep {sweep_number}", linewidth=0.5)
                axarr[icell].plot(t, v_rec, label=f"Sweep {sweep_number}", linewidth=0.5)
                axarr[icell].set_xlim(0.95, 2.05)
                axarr[icell].set_ylim(-110, 50)
                if not labeled[icell]:
                    axarr[icell].text(0.025, 1.00, f"Cell {cell_specimen_id:d}", transform=axarr[icell].transAxes, fontsize=7)
                    axarr[icell].set_xlabel("Time (s)", fontsize=6)
                    axarr[icell].set_ylabel("Voltage (mV)", fontsize=6)
                    axarr[icell].tick_params(axis='both', which='major', labelsize=6)
                    labeled[icell] = True

        # get data using AI cell ephys extractor
        long_ext = ephys_extractor.extractor_for_nwb_sweeps(data_set, sweeps_by_type[use_data_type])
        long_ext.process_spikes()  # analyze the spike data, but returns None
        for i in range(len(long_ext._sweeps)):
            # print(long_ext._sweeps[i].sweep_feature_keys())
            sweep_num = sweep_numbers[i]
            ddict = long_ext._sweeps[i].as_dict()
            nspikes = len(ddict["spikes"])
            if nspikes < 4:
                continue
            spt = [spk["peak_t"] for spk in ddict["spikes"]]
            spt = np.array(spt) - 1.0  # adust for delay to stimulus onset
            avg_rate = nspikes / (spt[-1] - spt[0])
            # avg_rate = long_ext._sweeps[i].sweep_feature("avg_rate")
            adapt = (
                long_ext._sweeps[i].sweep_feature("adapt")
                if "adapt" in long_ext._sweeps[i].sweep_feature_keys()
                else np.nan
            )
            # if np.isnan(adapt):
            #     print(f"Sweep {i:3d} N={nspikes:5d} adapt: {adapt:9.6f}, 'rate: {avg_rate:8.3f}")
            # else:
            #     print(f"Sweep {i:3d} N={nspikes:5d} adapt: <no data>, rate: {avg_rate:8.3f}")
            if avg_rate < 15.0 or avg_rate > 22.5:
                continue
            adapt_data[cell_specimen_id].append(
                SpikeData(
                    recnum=sweep_num,
                    spk_lat=spt,
                    nspikes=nspikes,
                    rate=avg_rate,
                    AI_adapt_index=adapt if not np.isnan(adapt) else np.nan,
                    adapt_index_MKX19=adaptation_index_MKX19(spt, trace_duration=1.0, minimum_spikes=minimum_spikes),
                    adapt_index_AllenInst=adaptation_index_AllenInst(
                        spike_data=spt, trace_duration=1.0, minimum_spikes=minimum_spikes,
                        input="spike_latencies"
                    ),
                    adapt_index_AllenInst_spktimes=adaptation_index_AllenInst_spktimes(
                        spike_data=spt, trace_duration=1.0, minimum_spikes=minimum_spikes
                    ),
                    adapt_index_exponential=adaptation_index_exponential(spt, trace_duration=1.0, minimum_spikes=minimum_spikes),
                    adapt_index_FirstLastISI=adaptation_first_last_isi(spt, trace_duration=1.0, minimum_spikes=minimum_spikes),
                )
            )

            cell_data[cell_specimen_id] = this_cell_data
        print(f"Cell {cell_specimen_id:d}")
        if len(adapt_data[cell_specimen_id]) > 0:
               adapt_data[cell_specimen_id][0].print_head()

        max_rate = 0.0
        for iad in range(len(adapt_data[cell_specimen_id])):
            max_rate = np.max((max_rate, adapt_data[cell_specimen_id][iad].rate))
            if (
                adapt_data[cell_specimen_id][iad].rate == 0
                or max_rate > adapt_data[cell_specimen_id][iad].rate
            ):
                continue
            else:
                adapt_data[cell_specimen_id][iad].print_row()
                # print(f"Adaptation data: {adapt_data!s}")
    # summarize adaptation across cells

    with open("adaptation_indices.pkl", "wb") as f:
        pickle.dump(adapt_data, f)
    

    if plot_flag:
        mpl.show()


def plot_adaptation_indices(adapt_data, plot_flag=False):
    if adapt_data is None:
        with open("adaptation_indices.pkl", "rb") as f:
            adapt_data = pickle.load(f)

    f, ax = mpl.subplots(2, 3, figsize=(10, 8))
    ax[0,0].set_title("MKX19")
    ai_ai = []
    ai_mkx = []
    ai_ai2 = []
    ai_spk = []
    ai_flisi = []
    ai_exp = []
    for cell in adapt_data.keys():
        print(f"\n\nCell {cell:d}")
        ai_mkx.extend([adapt_data[cell][iad].adapt_index_MKX19 for iad in range(len(adapt_data[cell]))])
        ai_ai.extend([adapt_data[cell][iad].AI_adapt_index for iad in range(len(adapt_data[cell]))])
        ai_ai2.extend([adapt_data[cell][iad].adapt_index_AllenInst for iad in range(len(adapt_data[cell]))])
        ai_spk.extend([adapt_data[cell][iad].adapt_index_AllenInst_spktimes for iad in range(len(adapt_data[cell]))])
        ai_flisi.extend([adapt_data[cell][iad].adapt_index_FirstLastISI for iad in range(len(adapt_data[cell]))])
        ai_exp.extend([adapt_data[cell][iad].adapt_index_exponential for iad in range(len(adapt_data[cell]))])
        nans = np.where(np.array(ai_exp) < -20.0)[0]
        for inan in nans:
            ai_exp[inan] = np.nan
    colors = range(len(ai_ai))
    colormap = 'gist_rainbow'
    print("\nMKX")
    ax[0,0].scatter(ai_ai, ai_mkx, c=colors, cmap = colormap, label=f"MKX")
    slope, intercept, r_2, x_fit, y_fit = linfit(np.array(ai_ai).reshape(-1, 1), np.array(ai_mkx))
    print("mkx: ", slope, intercept, r_2)
    ax[0,0].plot(x_fit, y_fit, 'r--', label=f"Fit: y={slope:.2f}x+{intercept:.2f}, R²={r_2:.2f}")

    print("\nAllenInst")
    ax[0,1].set_title("AllenInst")
    xr = [np.nanmin(ai_ai), np.nanmax(ai_ai)]
    ax[0,1].scatter(ai_ai, ai_ai2, c=colors, cmap = colormap, label=f"AllenInst_isi")
    slope, intercept, r_2, x_fit, y_fit = linfit(np.array(ai_ai), np.array(ai_ai2), xrange=xr)
    print("allen inst: ", slope, intercept, r_2)
    ax[0,1].plot(x_fit, y_fit, 'r--', label=f"Fit: y={slope:.2f}x+{intercept:.2f}, R²={r_2:.2f}")
    
    print("\nAllen Sptimes")
    ax[0,2].set_title("AllenInst_SpkTimes")
    ax[0,2].scatter(ai_ai, ai_spk, c=colors, cmap = colormap, label=f"AllenInst_SpkTimes")
    slope, intercept, r_2, x_fit, y_fit = linfit(np.array(ai_ai), np.array(ai_spk), xrange=xr)
    print("allen inst spk times: ", slope, intercept, r_2)
    ax[0,2].plot(x_fit, y_fit, 'r--', label=f"Fit: y={slope:.2f}x+{intercept:.2f}, R²={r_2:.2f}")
    
    print("\nFirstLastISI")
    ax[1,0].set_title("FirstLastISI")
    ax[1,0].scatter(ai_ai, ai_flisi, c=colors, cmap = colormap, label=f"FirstLastISI")
    slope, intercept, r_2, x_fit, y_fit = linfit(np.array(ai_ai), np.array(ai_flisi), xrange=xr)
    print("first last isi: ", slope, intercept, r_2)
    ax[1,0].plot(x_fit, y_fit, 'r--', label=f"Fit: y={slope:.2f}x+{intercept:.2f}, R²={r_2:.2f}")
    
    print("\nExpFit")
    ax[1,1].set_title("ExpFit")
    ax[1,1].scatter(ai_ai, ai_exp, c=colors, cmap = colormap, label=f"ExpFit")
    slope, intercept, r_2, x_fit, y_fit = linfit(np.array(ai_ai), np.array(ai_exp), xrange=xr)
    print("exp fit: ", slope, intercept, r_2)
    ax[1,1].plot(x_fit, y_fit, 'r--', label=f"Fit: y={slope:.2f}x+{intercept:.2f}, R²={r_2:.2f}")
    
    print("\nFLISI vs EXP")
    ax[1,2].set_title("FLISI vs EXP")
    ax[1,2].scatter(ai_flisi, ai_exp, c=colors, cmap = colormap, label=f"FLISI vs ExpFit")
    slope, intercept, r_2, x_fit, y_fit = linfit(np.array(ai_flisi), np.array(ai_exp), xrange=[np.nanmin(ai_flisi), np.nanmax(ai_flisi)])
    print("flisi vs exp fit: ", slope, intercept, r_2)
    ax[1,2].plot(x_fit, y_fit, 'r--', label=f"Fit: y={slope:.2f}x+{intercept:.2f}, R²={r_2:.2f}")
    axs = ax.ravel()
    for a in axs:
        a.legend()

    mpl.show()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute adaptation indices for a set of cells.")
    parser.add_argument("--test", action="store_true", help="Whether to run the tests")
    parser.add_argument("--ncells", type=int, default=5, help="Number of cells to analyze")
    parser.add_argument("--plot_traces", action="store_true", help="Whether to plot the traces during the tests")
    parser.add_argument("--plot_comparisons", action="store_true", help="Whether to plot the comparision of the adaptation indices after the tests")
    args = parser.parse_args()              
    
    if args.test:
        tests(ncells=args.ncells  , plot_flag=args.plot_traces)
    
    if args.plot_comparisons:
        plot_adaptation_indices(None, plot_flag=True)