"""
ephys.tools.spike_shape_timecourse

This module provides tools to analyze the time course of spike shapes
through current pulses in an IV protocol.
It includes functions to fit the changes in spike half-widths over time
using a double-exponential model, and to plot the results.
It also produces a summary dataframe of spike half-widths categorized by age groups.

Rise and fall rates, and AP thresholds are also analyzed.

To use:
Make sure the experiment configuration file is accessible. Set the experiment name and cell type.

8/27/2025 Paul B. Manis, Ph.D.
Refactored, and better handling of fitting and exclusions: 14 Jan 2026. PBM.s

"""

import argparse
import datetime
from dataclasses import dataclass, field
from pathlib import Path
import pprint
from typing import Tuple, Union

import ephys.tools.categorize_ages as CA
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from ephys.tools import check_inclusions_exclusions as CIE
from ephys.tools import get_configuration
from lmfit import Model
from matplotlib.backends.backend_pdf import PdfPages
from pylibrary.plotting import plothelpers as PH
from pylibrary.tools import cprint

CP = cprint.cprint


@dataclass
class fitResults:
    tau1: float = np.nan
    amp1: float = np.nan
    tau2: float = np.nan
    amp2: float = np.nan
    dc: float = np.nan
    fits: list[object] = field(default_factory=list)


@dataclass
class CellData:
    cell_id: str
    age_category: str
    age: str
    sex: str
    mean_hw: float = np.nan
    mean_dvdt_rise: float = np.nan
    mean_dvdt_fall: float = np.nan
    mean_AP_threshold: float = np.nan
    # halfwidths:
    halfwidth: fitResults = field(default_factory=fitResults)
    dvdt_rising: fitResults = field(default_factory=fitResults)
    dvdt_falling: fitResults = field(default_factory=fitResults)
    ap_threshold: fitResults = field(default_factory=fitResults)
    # amp1: float = np.nan
    # tau1: float = np.nan
    # amp2: float = np.nan
    # tau2: float = np.nan
    # dc: float = np.nan


@dataclass
class Figure:
    fig_initiated: bool = False
    ax: mpl.Axes = None
    figure: mpl.figure = None
    do_fig: bool = False



def double_exp_rise(
    x: np.ndarray,
    dc: float = 0.0,
    a1: float = 0.0,
    r1: float = 0.001,
    a2: float = 0.0,
    r2: float = 0.1,
) -> np.ndarray:
    """double_exp _summary_
    Function to compute a double exponential decay plus offset.
    This is to fit data with rising time course (therefore, 1-exp))

    Parameters
    ----------
    x : np.array
        Time array
    dc : float
        baseline offset
    a1 : float
        ampllitude of first exponential
    r1 : float
        rate constant for a1
    a2 : float
        amplitude of second exponential
    r2 : float
        rate constant for a2
    Returns
    -------
    np.array
        The value of the double exponential function at each point in x
        given the values of dc, a1, r1, a2, r2.
    """
    return dc + (a1 * (1 - np.exp(-x / r1)) + a2 * (1 - np.exp(-x / r2)))


def double_exp_fall(
    x: np.ndarray,
    dc: float = 0.0,
    a1: float = 0.0,
    r1: float = 0.001,
    a2: float = 0.0,
    r2: float = 0.1,
) -> np.ndarray:
    """double_exp _summary_
    Function to compute a double exponential decay plus offset.
    This is to fit data with falling time course

    Parameters
    ----------
    x : np.array
        Time array
    dc : float
        baseline offset
    a1 : float
        ampllitude of first exponential
    r1 : float
        rate constant for a1
    a2 : float
        amplitude of second exponential
    r2 : float
        rate constant for a2
    Returns
    -------
    np.array
        The value of the double exponential function at each point in x
        given the values of dc, a1, r1, a2, r2.
    """
    f = dc + ((a1 * (np.exp(-x / r1)) + a2 * (np.exp(-x / r2))))
    return f


def get_spikedata(
    spike_trace, measure: str, start_time: float, window: list, junction_potential: float
) -> Tuple[list, list, list, list]:
    """get_spikedata _extracts spike measure data from a single spike trace structure
    and computes the steady-state values, and fits the time course of the
    measure change to a double-exponential function.

    Parameters
    ----------
    spike_trace : spike structure
        _description_
    measure : str
        name of the measure to extract ('hw', 'rise', 'fall', 'ap_thr')
    start_time : float
        start time of the current pulse
    window : list
        time window to consider for steady-state values
    junction_potential : float
        junction potential to adjust AP threshold values

    Returns
    -------
    Tuple[list, list, list, list]
        latencies, values, and steady-state latencies and values for the analyzed trace
    """
    latencies = []
    values = []
    ss_values = []
    ss_latency = []
    # print("Spike Trace: ",  spike_trace)
    for j, sn in enumerate(spike_trace):  # for each spike in the trace
        if spike_trace[sn].__getattribute__(measure) is None:  # no spikes
            continue
        if measure == "halfwidth" and (
            1e3 * spike_trace[sn].__getattribute__("halfwidth") > 1.0
        ):  # long HW is artifact in analysis
            continue
        latency = spike_trace[sn].AP_latency - start_time
        value = spike_trace[sn].__getattribute__(measure)  # always get the raw value
        if measure == "AP_begin_V":
            value += junction_potential  # adjust for JP
        values.append(value)
        latencies.append(latency)
        if latency >= window[0] and latency <= window[1]:  # only accumulate in the ss window
            ss_values.append(value)
            ss_latency.append(latency)
    values = np.array(values)
    latencies = np.array(latencies)
    ss_values = np.array(ss_values)
    ss_latency = np.array(ss_latency)
    return latencies, values, ss_latency, ss_values

def check_excluded(
    d,
    experiment: dict,
) -> bool:
    """check_exclusions Check if the cell/protocol should be excluded from analysis
    based on the experiment configuration.
    Here we also check the steady-state IV exclusions
    Uses include_exclude from the ephys.tools.check_inclusions_exclusions module.

    Parameters
    ----------
    d : dictionary of analyzed values from pickle file
        includes 'IV', and 'Spikes' entries.
    experiment : dict
        Parameters for the experiment being analyzed (from configuration file)
    """
    ivs = list(d["IV"].keys())
    validivs, additional_ivs, additional_iv_records = CIE.include_exclude(
        str(d["cell_id"]),
        inclusions=experiment["includeIVs"],
        exclusions=experiment["excludeIVs"],
        allivs=ivs,
        verbose = False,
    )
    # print("ex: valid ivs after first pass: ", validivs)
    # run a second level of exclusion when ss spike parameters are not stable
    validivs, additional_ivs, additional_iv_records = CIE.include_exclude(
        d["cell_id"],
        inclusions=experiment["includeIVs"],
        exclusions = experiment["exclude_steady_state_spike_shapeIVs"],
        allivs=validivs,
        verbose = True,
        )
    print("ex: valid ivs after steady-state spike shape exclusion: ", validivs)
    return validivs

def fit_spike_measures(
    d,
    protocol: str,
    measure: str,
    experiment: dict,
    window: list = [0.4, 0.5],
) -> dict:
    """fit_spike_measures Fit the time course of the spike measure changes through
        the stimulus duration. handles multiple protocols at once.
        Plots the time course of the halfwidths, dv/dt rise and fall, and AP threshold
        for EACH spike detected during the IV protocol, for the rate window specified.
    pr
        Does ONE cell (data is in d), and ONE protocol (in protocol)

        Parameters
        ----------
        d : dictionary of analyzed values from pickle file
            includes 'IV', and 'Spikes' entries.
        measure : str
            name of the measure to plot ('halfwidth', 'dvdt_rising', 'dvdt_falling', 'AP_begin_V')
        rate_limits : list (list of two floats)
            firing rate limits for inclusion, Hz.
        protocol_start_times : dict
            start times for each protocol
        durations : dict
            durations for each protocol
        junction_potential : float  # used to correct AP threshold values
            junction potential to adjust AP threshold values
        experiment : dict
            Parameters for the experiment being analyzed
        window : list, optional
            time window to consider for steady-state values, by default [0.4, 0.5]

        Returns
        -------
        dictionary with : tr_latency, tr_values, fit_values, ss_values, sign
    """
    verbose = False

    print("Fitting spike measure time courses for measure: ", measure)
    assert measure in [
        "halfwidth",
        "dvdt_rising",
        "dvdt_falling",
        "AP_begin_V",
    ], f"Invalid measure name: {measure}"
    if d["Spikes"] is None:
        print("No spike data found in the data structure")
        return None
    sign = 1.0
    if measure in ["dvdt_falling", "AP_begin_V"]:
        sign = -1.0
    fit_func = {
        "halfwidth": double_exp_rise,
        "dvdt_rising": double_exp_fall,
        "dvdt_falling": double_exp_fall,
        "AP_begin_V": double_exp_fall,
    }
    prots = list(d["Spikes"].keys())
    if protocol not in prots:
        raise ValueError(f"Protocol {protocol} not found in data for fitting.")    

    # get parameters from the experiment dict
    protocol_start_times = experiment["Protocol_start_times"]
    durations = experiment["protocol_durations"]
    rate_limits = experiment["spike_rate_limits"]  # firing rate limits for inclusion, Hz.
    junction_potential = experiment["junction_potential"] * 1e-3  # convert to Volts

    # first run through the protocols (top level)

    ivs = d["IV"][protocol]
    spike_traces = d["Spikes"][protocol]["spikes"]
    pname = str(Path(protocol).name)
    ss_values = []
    ss_latency = []
    fit_values = []
    tr_latency = []
    tr_values = []
    if pname[:-4] in protocol_start_times:
        start_time = protocol_start_times[pname[:-4]]
    else:
        start_time = 0

    if pname[:-4] in durations:
        dur = durations[pname[:-4]]
    else:
        raise ValueError(f"Duration for protocol {pname[:-4]} not found in experiment configuration.")
        # continue

    # within each protocol, go through the traces with spikes
    # and get the values
    for i, spike_trace in enumerate(spike_traces):
        latencies, values, sslatency, ssvalues = get_spikedata(
            spike_traces[spike_trace], measure, start_time, window, junction_potential
        )
        if verbose:
            print("\nTrace ", i, " has ", len(latencies), " spikes")
            print("latencies: ", len(latencies), latencies)
            print("ss latency: ", len(sslatency), sslatency)
        if len(sslatency) < 2:  # need at least 2 spikes to compute rate
            # print("      less than 2 spikes")
            continue
        mean_spike_rate = (1.0 / np.diff(sslatency)).mean()  # firing rate in Hz
        if mean_spike_rate < rate_limits[0] or mean_spike_rate > rate_limits[1]:
            if verbose:
                print("      mean spike rate: ", mean_spike_rate, " outside limits ", rate_limits)
            continue  # skip traces outside the rate limits
        else:
            if verbose:
                print("      mean spike rate: ", mean_spike_rate, " inside limits ", rate_limits)
            tr_latency.append(latencies)
            tr_values.append(values)
            ss_latency.append(sslatency)
            ss_values.append(ssvalues)
        # and get fits to rising exponential
    #  prepare fitting with double-exponential : ONE for each protocol (average spike data across traces)
    try:
        tr_latency_array = np.concatenate(tr_latency)
        tr_values_array = np.concatenate(tr_values)
        ss_latency_array = np.concatenate(ss_latency)
        ss_values_array = np.concatenate(ss_values)
    except ValueError:
        CP("m", f"        No valid spikes found for protocol {protocol}")
        CP("m", f"        tr_latency[pname]: {tr_latency}")
        return None
    mean_ss_rate = np.mean(1.0 / np.diff(ss_latency_array))
    tr_values_array = tr_values_array * sign
    if verbose:
        print("Mean firing rate at steady-state: ", mean_ss_rate)
        print("\nFitting function: ", fit_func[measure])
    a_init = 100.0
    if fit_func[measure] == double_exp_rise:
        # estimate the rise as the difference from the first value to the mean of the last 5 values
        dc_init = tr_values_array[0]
        a_init = np.mean(tr_values_array[-5:]) - dc_init
    elif fit_func[measure] == double_exp_fall:
        # estimate the fall as the difference from the first value to the mean of the last 5 values
        dc_init = np.mean(tr_values_array[:-5])
        a_init = tr_values_array[0] - dc_init
    if verbose:
        print(f"Initial values of measure {measure}: ", "dc: ", dc_init, "a1, a2: ", a_init)
    d2model = Model(fit_func[measure])
    if a_init == 0:
        a_init = 1e-6
    limits = {
        "dc": (0.0, 1000.0),
        "a1": (0.0, a_init),
        "r1": (0.0, 0.5),
        "a2": (0.0, a_init),
        "r2": (0.4, 2.0),
        }
    minr1 = 0.02
    minr2 = 0.5
    dc_min = 0.0
    dc_max = 1000.
    if measure == 'halfwidth':
        amax = 200e-6
        minr1 = 0.01
    if measure == "AP_begin_V":
        amax = 0.020
        minr1 = 0.005 
        minr2 = 1.0
        dc_init = -0.55
        dc_min = -0.1
        dc_max = 0.1

    else:
        minr1 = 0.005
        amax = 1000.

    params = d2model.make_params(
        dc={"value": dc_init, "min": dc_min, "max": dc_max},
        a1={"value": a_init, "min": 0., "max": amax},
        r1={"value": minr1, "min": 0., "max": 0.1},
        a2={"value": 0, "min": 0., "max": amax},
        r2={"value": minr2*1.2, "min": minr2, "max": 3.0},
    )
    x = np.linspace(0, dur, 500)
    y = fit_func[measure](x, **params.valuesdict())
    if verbose:
        mpl.plot(x, sign * y, "k--")  # initial conditions
        mpl.plot(
            tr_latency_array,
            sign * tr_values_array,
            "bo",
            markersize=3,
        )
        mpl.ylabel(f"{measure}")
    if verbose:
        print(f"\nParams for {measure}")
        for p in params:
            print(f"    {p}: {params[p].value}  ({params[p].min}, {params[p].max})")
    fit_values = d2model.fit(tr_values_array, params, x=tr_latency_array)
    xf = np.linspace(0, dur, 500)
    yf = fit_values.eval(x=xf)
    if verbose:
        mpl.plot(xf, sign * yf, "r--")
        mpl.title(f"Fit for protocol {pname}")
        # mpl.show()
        print(fit_values.fit_report())
        # exit()

    if verbose:
        print("\nFit for protocol ", protocol, ": ")
        print(fit_values.fit_report())
        
    if verbose:
        mpl.show()
    return {
        "tr_latency": tr_latency,
        "tr_values": tr_values,
        "fit_values": fit_values,
        "ss_values": ss_values,
        "sign": sign,
    }  # return the fits for each protocol


def average_fits(fits: dict):
    """average_fits _summary_
    Averages the fit parameters across protocols.
    Parameters
    ----------
    fits : dict
        dictionary of fit results from fit_spike_measures
    fig : Figure
        Figure object to plot on
    Returns
    -------
    dict
        dictionary of averaged fit parameters
    """

    tau1 = []
    amp1 = []
    tau2 = []
    amp2 = []
    dc = []
    for f in fits.keys():
        print(
            "        Fit for protocol ",
            f,
            ": ",
        )
        if isinstance(fits[f], list) and len(fits[f]) == 0:
            continue
        tau_1 = fits[f].best_values.get("r1", np.nan)
        amp_1 = fits[f].best_values.get("a1", np.nan)
        tau_2 = fits[f].best_values.get("r2", np.nan)
        amp_2 = fits[f].best_values.get("a2", np.nan)
        dc0 = fits[f].best_values.get("dc", np.nan)
        # force order of the values so that tau_1 is always the smaller time constant
        # note that the values are inverted.
        if tau_2 < tau_1:
            tau_1, tau_2 = tau_2, tau_1  # swap
            amp_1, amp_2 = amp_2, amp_1
        tau1.append(tau_1)
        amp1.append(amp_1)
        tau2.append(tau_2)
        amp2.append(amp_2)
        dc.append(dc0)
    # print("FITS: ",tau1, tau2)
    if fits is None or len(fits) == 0:
        fits = {"tau1": np.nan, "amp1": np.nan, "tau2": np.nan, "amp2": np.nan, "dc": 0}
    else:
        tau1 = np.nanmean(tau1)
        tau2 = np.nanmean(tau2)
        amp1 = np.nanmean(amp1)
        amp2 = np.nanmean(amp2)
        dc = np.nanmean(dc)
        fits = {"tau1": tau1, "amp1": amp1, "tau2": tau2, "amp2": amp2, "dc": dc}
    # print("FITS returned: ", fits)
    return fits


def _make_label(pn, ivs, measure):
    Rs = ivs["CCComp"]["CCBridgeResistance"] * 1e-6
    Cp = ivs["CCComp"]["CCNeutralizationCap"] * 1e12
    Rs_Cp_label = f"{pn}: Rs={Rs:.1f}, Cp={Cp:.1f}"
    ylabel = "unknown"
    if measure == "hw":
        ylabel = "AP Halfwidth (us)"
    elif measure == "rise":
        ylabel = "AP dV/dt rise (V/s)"
    elif measure == "fall":
        ylabel = "AP dV/dt fall (V/s)"
    elif measure == "ap_thr":
        ylabel = "AP Threshold (V)"
    return Rs_Cp_label, ylabel


def plot_spike_measure(
    d,
    experiment: dict,
    filename: Path,
    measure="halfwidth",
    plot_fits: bool = True,
    ax: mpl.Axes = None,
) -> Union[dict, None]:
    """plot_spike_measure Plot the time course of the spike measure changes through
    the stimulus duration.
    Plots the time course of a specific measure for EACH spike detected during
    the IV protocol, for traces in which the spike rate falls with
    the specified rate window.

    Parameters
    ----------
    d : dictionary of analyzed values from pickle file
        includes 'IV', and 'Spikes' entries.
    filename : Path
        _description_
    measure : str, optional
        name of the measure to plot ('halfwidth', 'dvdt_rising', 'dvdt_falling', 'AP_begin_V'), by default "halfwidth"
    ax : mpl.Axes, optional
        Axes to plot on, by default None

    Returns
    -------
    sc : float
        scale factor used for plotting (for halfwidths, 1e6 to convert to us)
    """
    # measure = "AP_begin_V"
    assert measure in [
        "halfwidth",
        "dvdt_rising",
        "dvdt_falling",
        "AP_begin_V",
    ], "Invalid measure name"
    scales = {"halfwidth": 1e6, "dvdt_rising": 1.0, "dvdt_falling": -1.0, "AP_begin_V": 1.0e3}
    ylimits = {
        "halfwidth": (0, None),
        "dvdt_rising": (0, 1000),
        "dvdt_falling": (800, 0),
        "AP_begin_V": (-0.060 * scales["AP_begin_V"], -0.040 * scales["AP_begin_V"]),
    }
    musec = r"$\mu$ s"
    ylabels = {
        "halfwidth": f"AP Halfwidth ({musec})",
        "dvdt_rising": "AP dV/dt rise (V/s)",
        "dvdt_falling": "AP dV/dt fall (V/s)",
        "AP_begin_V": "AP Threshold (mV)",
    }
    if ax is None:
        f, ax = mpl.subplots(1, 1)
        f.text(0.95, 0.02, datetime.datetime.now(), fontsize=6, transform=f.transFigure, ha="right", va="bottom")
    if measure == 'halfwidth':
        ax.set_title(f"{str(Path(*filename.parts[-3:])):s}")
    if d["Spikes"] is None:
        ax.text(
            0.5,
            0.5,
            s="No spikes found",
            ha="center",
            va="center",
            fontdict={"color": "red", "size": 20},
        )
        return None

    # screen inclusions and exclusions of protocols.
    valid_ivs = check_excluded(d, experiment)
    # get the ivs and protocols matching the keys in the data
    prots = [p for p in valid_ivs if p in list(d["Spikes"].keys())]
    spike_shape_prots = []
    # filter down to only those protocols that are in the FI_spike_shape_protocols list
    for ip, protoname in enumerate(prots):
        pname = str(Path(protoname).name)[:-4]
        if pname in list(experiment["FI_spike_shape_protocols"].keys()):
            spike_shape_prots.append(protoname)
           
    CP("m", f"Remaining IV protocols for plotting: {spike_shape_prots}")
    protocol_start_times = experiment["Protocol_start_times"]
    durations = experiment["protocol_durations"]

    # Screen the protocols first.
    # 1. Is the protocol in the list of protocols we want to use ? 
    # 2. Select the protocol with the lowest Rs value for analysis.
    # Secondarily, if there are 2 protocols with the same Rs, select the one with the most
    # traces in the required firing range.
    any_protocols = False
    usable_protocols = []
    best_Rs = 100.0 # initialize
    pnames = []
    Rs_s = []
    latencies = {}
    values = {}
    fit_values = {}
    ss_values = {}
    ss_latencies = {}
    datasign = {}

    for ip, protoname in enumerate(spike_shape_prots):
        pname = str(Path(protoname).name)[:-4]
        if pname not in experiment["FI_spike_shape_protocols"]:
            continue
        ivs = d["IV"][protoname]
        Rs = ivs["CCComp"]["CCBridgeResistance"] * 1e-6
        if Rs > experiment["maximum_access_resistance"]:  # limit to lower access.
            continue
        pnames.append(pname)
        Rs_s.append(Rs)
        if Rs < best_Rs:
            best_Rs = Rs
            usable_protocols = [protoname]
        elif Rs == best_Rs:  # same Rs, add to list
            usable_protocols.append(protoname)
    CP("m", f"    Using protocol(s) with best Rs for plotting: {usable_protocols}, {best_Rs}")

    used_protocols = []
    for ip, protoname in enumerate(usable_protocols):
        pname = str(Path(protoname).name)[:-4]
        ivs = d["IV"][protoname]
        start_time = protocol_start_times.get(pname, 0.0)
        dur = durations.get(pname, None)
  
        fit_result = fit_spike_measures(
            d,
            protocol = protoname,
            measure=measure,
            experiment=experiment,
            window=[0.4, 0.5],
        )
        if fit_result is None:
            continue
        latencies[pname], values[pname], fit_values[pname], ss_values[pname], datasign[pname] = [x for k, x in fit_result.items()]
        # print("    Latencies from protocols: ", latencies.keys())
        # print("pname: ", pname, "protoname: ", protoname)
        proto_num = Path(protoname).name
        if len(latencies) == 0:
            continue
        any_protocols = True

        lats = np.concatenate(latencies[pname])
        vals =  np.concatenate(values[pname]) * scales[measure]
        
        colors = sns.color_palette("husl", max(len(prots), 3))
        if protoname.find("1nA") > 0:
            color = colors[1]
        elif protoname.find("4nA") > 0:
            color = colors[2]
        else:
            color = colors[0]
        Rs_Cp_label, labely = _make_label(protoname, ivs, measure)

        ax.plot(lats, vals, "o", color=color, markersize=1, label=labely, linewidth=0.35)
        if plot_fits:
            xf = np.linspace(0, dur, 500)
            yf = datasign[pname] * fit_values[pname].eval(x=xf) * scales[measure]
            ax.plot(xf, yf, "r-", linewidth=0.5)
    # label plot if there was no data to plot
    if not any_protocols:
        ax.text(
            0.5,
            0.5,
            s="No valid spike data found for protocols",
            ha="center",
            va="center",
            fontdict={"color": "red", "size": 12},
        )
        return None
    
    ax.set_ylim(ylimits[measure])
    ax.set_xlim(-0.020, 1.0)
    ax.set_xlabel("AP Latency (s)")
    ax.set_ylabel(ylabels[measure])
    if measure == 'halfwidth':
        ax.text(
            0.98,
            0.02,
            f"Rs: {best_Rs:.2f} MOhm\nProtocols: {', '.join(usable_protocols)}",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=6,
        )
    # ax.legend(fontsize=5)
    return {
        "latencies": latencies,
        "values": values,
        "fit_values": fit_values,
        "ss_values": ss_values,
        "sign": datasign,
        "Rs": best_Rs,
        "protocols": used_protocols,
    }


def post_stimulus_spikes(d):
    f, ax = mpl.subplots(len(d["Spikes"].keys()), 3, figsize=(10, 8))

    for axi, k in enumerate(d["Spikes"].keys()):
        print("protocol, data type: ", k, type(d["Spikes"][k]))
        print("    Spike array keys in protocol: ", d["Spikes"][k].keys())

        print("   pulse duration: ", d["Spikes"][k]["pulseDuration"])
        print("   poststimulus spike window: ", d["Spikes"][k]["poststimulus_spike_window"])
        print("  tstart: ", d["Spikes"][k]["poststimulus_spikes"])
        # print(d['Spikes'][k]['poststimulus_spikes'])
        for i, ivdata in d["IV"].items():
            print("\n   ", i, "\n", ivdata["RMP"], ivdata["RMPs"])

        print("    poststimulus spikes: ", d["Spikes"][k]["poststimulus_spikes"])
        for i in range(len(d["Spikes"][k]["poststimulus_spikes"])):
            nsp = len(d["Spikes"][k]["poststimulus_spikes"][i])
            iinj = d["Spikes"][k]["FI_Curve"][0][i]
            if nsp > 0:
                ax[axi, 0].plot(
                    d["Spikes"][k]["poststimulus_spikes"][i],
                    [iinj] * nsp,
                    marker="o",
                    markersize=2,
                    linestyle="None",
                )
            if nsp > 1:
                dur = (
                    d["Spikes"][k]["poststimulus_spikes"][i][-1]
                    - d["Spikes"][k]["poststimulus_spikes"][i][0]
                )
                ax[axi, 1].plot(iinj * 1e9, dur * 1e3, marker="o", markersize=2, linestyle="None")
                ax[axi, 1].set_xlabel("current (nA)")
                ax[axi, 1].set_ylabel("duration (ms)")
                rate = np.mean(1.0 / np.diff(d["Spikes"][k]["poststimulus_spikes"][i]))
                ax[axi, 2].set_title(f"{k} rate: {rate:.2f} Hz")
                ax[axi, 2].set_ylabel("rate (Hz)")
                ax[axi, 2].set_xlabel("current (nA)")
                ax[axi, 2].plot(iinj * 1e9, rate, marker="o", markersize=2, linestyle="None")
    mpl.tight_layout()
    mpl.show()


def print_LCS_spikes(d):
    # print(d['Spikes'][k]['spikes'].keys())
    # print("   ", k, "LCS: ", d['Spikes'][k]['LowestCurrentSpike'])
    for i, k in enumerate(d["Spikes"].keys()):
        if len(d["Spikes"][k]["LowestCurrentSpike"]) == 0:
            continue
        tr = d["Spikes"][k]["LowestCurrentSpike"]["trace"]
        dt = d["Spikes"][k]["LowestCurrentSpike"]["dt"]
        sn = d["Spikes"][k]["LowestCurrentSpike"]["spike_no"]
        # print("spike values for trace: ", tr, d['Spikes'][k]['spikes'][tr][sn])
        print("LCS spike data: ")

        print("LCS keys: ", d["Spikes"][k]["LowestCurrentSpike"].keys())
        print("   ", k, "LCS HW: ", d["Spikes"][k]["LowestCurrentSpike"]["AP_HW"])
        print("   ", k, "LCS AHP Depth: ", d["Spikes"][k]["LowestCurrentSpike"]["AHP_depth_V"])
        print("   ", k, "LCS AP Peak: ", d["Spikes"][k]["LowestCurrentSpike"]["AP_peak_V"])
        print(
            "   ", k, "LCS AP peak height: ", d["Spikes"][k]["LowestCurrentSpike"]["AP_peak_V"]
        )  #  - d['Spikes'][k]['LowestCurrentSpike']['AP_thr_V'])
        vpk = (
            d["Spikes"][k]["spikes"][tr][sn].peak_V * 1e3
        )  # peak value from the pkl file trace spike number
        vth = d["Spikes"][k]["spikes"][tr][sn].AP_begin_V * 1e3  # threshold value from the pkl file
        icurr = d["Spikes"][k]["spikes"][tr][sn].current * 1e12  # current injection value
        tro = d["Spikes"][k]["LowestCurrentSpike"]["AHP_depth_V"]

        print(
            "   ",
            f"{k:s}, trace: {tr:d} spike #: {sn:d}  peak V: {vpk:5.1f}, thr V: {vth:5.1f}, AP Height: {vpk-vth:5.1f}",
        )
        print(
            f"          AP Trough: {tro:f} current: {icurr:6.1f}"
        )  # confirm that the threshold value is the same

    print("   ", k, "AP1HW: ", d["Spikes"][k]["AP1_HalfWidth"])


def read_pkl_file(filename):

    filename = Path(filename)
    print(filename.is_file())
    d = pd.read_pickle(filename, compression="gzip")
    return d


def build_figure_framework():
    x = -0.1
    y = 1.07

    yht = 0.18
    xp = [0.08, 0.45, 0.72]
    yp = [0.975 - (n + 1) * 0.22 for n in range(4)]
    xw = [0.30, 0.22, 0.22]
    yh = [yht] * 4
    sizer = {}
    for ix in range(3):
        for iy in range(4):
            key = chr(65 + iy) + str(ix + 1)
            sizer[key] = {
                "pos": [xp[ix], xw[ix], yp[iy], yh[iy]],
                "labelpos": (x, y),
            }

    gr = [(a, a + 1, 0, 1) for a in range(0, 12)]  # just generate subplots - shape does not matter
    axmap = dict(zip(sizer.keys(), gr))

    P = PH.arbitrary_grid(sizer=sizer, showgrid=False, label=True, figsize=(11, 8.5))
    # PH.show_figure_grid(P)
    # mpl.show()
    # exit()
    return P


def compute_measures_all_cells(adpath, exptname, celltype, experiment, specific_cell_list: list):
    datadir = Path(adpath, exptname, celltype)
    files = list(datadir.glob("*_IVs.pkl"))
    # fig.do_fig = True

    # steady-state halfwidths
    age_cats = experiment["age_categories"]
    # build dict to hold results sorted by group
    group_hws_ss = {
        "Preweaning": [],
        "Pubescent": [],
        "Young Adult": [],
        "Mature Adult": [],
        "Old Adult": [],
        "ND": [],
    }
    dlist = []
    # prepare for PDF output if needed
    with open("spike_shape_timecourse_skipped.txt", 'w') as f:
        f.write(f"Spike shape timecourse analysis skipped files log\n")
        f.write(f"Date: {datetime.datetime.now()}\n\n")

    with PdfPages("spk_hwidths.pdf") as pdf:
        for nf, f in enumerate(files):
            # print(f.name)
            if specific_cell_list is not None:
                if f.name not in specific_cell_list:
                    continue
            print(f"#{nf:d} {f}")
            d = read_pkl_file(f)
            # if nf > 20:
            #     break

            if d["Spikes"] is None:
                CP('m', f"        No spike data found, skipping file.")
                continue
            
            fig, ax = mpl.subplots(4, 1, figsize=(6, 8))
            spf = build_figure_framework()
            cell_data = {}
            fits = {}
            avdict = {}
            d["age_category"] = CA.get_age_category(d["age"], age_cats)
            for factor in ["cell_id", "age_category", "age", "sex"]:
                cell_data[factor] = d[factor]
            for axn, measure in zip(ax, ["halfwidth", "dvdt_rising", "dvdt_falling", "AP_begin_V"]):
                sp_measures = plot_spike_measure(  # fit_spike_measures(
                    d,
                    filename=f,
                    measure=measure,
                    experiment=experiment,
                    ax=axn,
                )
                if sp_measures is None:
                    fits[measure] = fitResults(
                        tau1=np.nan,
                        amp1=np.nan,
                        tau2=np.nan,
                        amp2=np.nan,
                        dc=np.nan,
                        fits=[],
                    )
                    continue
                
                else:
                    tr_latency, tr_values, fit_values, ss_values, datasign, best_Rs, usable_protocols = [
                        v for k, v in sp_measures.items()
                    ]
                    print("fit values for measure ", measure, ": ", fit_values)
                    avfits = average_fits(fit_values)
                    # print("ss_values: ", measure, ss_values)
                    y= [x for k, x in ss_values.items() if len(x) > 0][0]
                    # print(y)
                    y = np.concatenate(y)
                    ss_mean = np.mean(y)
                    # print("Mean steady-state value for measure ", measure, ": ", ss_mean)
                    # exit()
                    # print("Average fits for measure ", measure, ": ", avfits)
                    avdict[measure] = avfits
                    fits[measure] = fitResults(
                        tau1=avfits["tau1"],
                        amp1=avfits["amp1"],
                        tau2=avfits["tau2"],
                        amp2=avfits["amp2"],
                    )
                for x in ["dc", "tau1", "amp1", "tau2", "amp2"]:
                    colname = measure + "_" + x
                    cell_data[colname] = avfits[x]
                col2 = measure + "_steadystate"
                # print("\n", measure, ss_values)
                # print(ss_values.values())
                if len(ss_values) == 0:
                    cell_data[col2] = np.nan
                else:
                    cell_data[col2] = ss_mean # np.mean([x for x in ss_values.values()])
                cell_data['Rs_MOhm'] = best_Rs
                cell_data['Protocols'] = ','.join(usable_protocols)
            dlist.append(cell_data)
            # if fig.do_fig and fig.fig_initiated:
            mpl.suptitle(f.stem)
            pdf.savefig(fig)

    # if fig.do_fig and fig.fig_initiated:
    mpl.close()

    pprint.pprint(dlist)
    df = pd.DataFrame(dlist)
    print(df.head(20))

    # plot_ss_hws(experiment)
    # save to a file, using the local data path (not the RAID drive)
    stats_dir = experiment.get("R_statistics_summaries", None)
    local_dir = Path(experiment.get("localdatapath", "."), stats_dir)
    if stats_dir is not None:
        df.to_csv(Path(local_dir, "spike_steady_state_halfwidths_11-Jan-2026.csv"))
        print(
            "Saved spike halfwidth summary to: ",
            Path(local_dir, "spike_steady_state_halfwidths_11-Jan-2026.csv"),
        )


def bar_and_scatter(
    df: pd.DataFrame, x: str, y: str, hue: str, experiment: dict, ax: mpl.Axes, scf: float = 1.0
):
    hue_category = "age_category"
    plot_order = experiment["plot_order"]["age_category"]
    plot_colors = experiment.get("plot_colors", {})
    palette = plot_colors["symbol_colors"]
    bar_color = plot_colors.get("bar_background_colors", None)
    # bar_order = plot_colors.get("age_category", None)
    # line_colors =plot_colors.get("line_plot_colors", None)
    edgecolor = (plot_colors["symbol_edge_color"],)
    linewidth = (plot_colors["symbol_edge_width"],)
    # print("Y is : ", y)
    df[y] = df[y] * scf
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=x,
        hue_order=plot_order,
        palette=plot_colors["bar_background_colors"],
        ax=ax,
        order=plot_order,
        saturation=float(plot_colors["bar_saturation"]),
        width=plot_colors["bar_width"],
        orient="v",
        showfliers=False,
        linewidth=plot_colors["bar_edge_width"],
        zorder=50,
        dodge=experiment["dodge"][hue_category],
        # clip_on=False,
    )
    sns.stripplot(
        data=df,
        x=x,
        y=y,
        hue=x,
        order=plot_order,
        palette=plot_colors["symbol_colors"],
        edgecolor=edgecolor,
        linewidth=linewidth,
        size=plot_colors["symbol_size"] * 1.5,
        alpha=1.0,
        ax=ax,
        zorder=100,
        clip_on=False,
    )


def plot_amp_tau(experiment: dict, measure: str, ax1, ax2):
    stats_dir = experiment.get("R_statistics_summaries", None)
    local_dir = Path(experiment.get("localdatapath", "."), stats_dir)

    df = pd.read_csv(Path(local_dir, "spike_steady_state_halfwidths_11-Jan-2026.csv"))
    df = df[df["age_category"] != "ND"]

    hw_ax = [ax1, ax2]
    # hw_fig, hw_ax = mpl.subplots(5, 1, figsize=(6, 8))
    # sns.barplot(data=df, x="age_category", y="mean_hw", hue="age_category",
    #             palette=bar_color, alpha=0.33,
    #             order=bar_order, ax=hw_ax)
    # bar_and_scatter(
    #     df, x="age_category", y=measure + "_steadystate", hue="age_category", experiment=experiment, ax=hw_ax[0]
    # )
    # hw_ax[0].set_ylim(0, 0.5)
    # hw_ax[0].set_ylabel("AP Halfwidth (s)")
    delta = r"$\Delta$"
    usec = r"$\mu$s"

    if measure in ["AP_begin_V"]:
        ylims = (0, 20)
        taulims = (0, 0.25)
        scf = 1.0
        ylabel = f"{delta} AP Threshold (mV)"

    elif measure in ["dvdt_falling"]:
        ylims = (-200, 20)
        taulims = (0, 0.25)
        scf = -1.0e-3
        ylabel = f"{delta} Falling dV/dt (V/s)"
    
    elif measure in ["dvdt_rising"]:
        ylims = (-400, 0)
        taulims = (0, 0.25)
        scf = -1.0e-3
        ylabel = f"{delta} Rising dV/dt (V/s)"
    
    elif measure in ["halfwidth"]:
        ylims = (0, 100)
        taulims = (0, 0.5)
        scf = 1e3
        ylabel = f"{delta} AP Halfwidth ({usec})"
    
    df[measure + "_amp1"] *= scf
    # =================
    bar_and_scatter(
        df,
        x="age_category",
        y=measure + "_tau1",
        hue="age_category",
        experiment=experiment,
        ax=hw_ax[0],
    )
    # hw_ax[0].set_ylim(ylims)
    tau = r"$\tau$"
    hw_ax[0].set_ylabel(f"{tau} (s)")
    hw_ax[0].set_ylim(taulims)

    # =================
    bar_and_scatter(
        df,
        x="age_category",
        y=measure + "_amp1",
        hue="age_category",
        experiment=experiment,
        ax=hw_ax[1],
        scf=1e3,
    )
    hw_ax[1].set_ylim(ylims)
    hw_ax[1].set_ylabel(ylabel)

    # =================
    # bar_and_scatter(
    #     df, x="age_category", y=measure + "_tau2", hue="age_category", experiment=experiment, ax=hw_ax[3]
    # )
    # hw_ax[3].set_ylim(0, 20.0)
    # hw_ax[3].set_ylabel("Slow rate (s)")

    # # =================
    # bar_and_scatter(
    #     df, x="age_category", y=measure + "_amp2", hue="age_category", experiment=experiment, ax=hw_ax[4]
    # )
    # hw_ax[4].set_ylim(-0.50, 0.5)
    # hw_ax[4].set_ylabel("Slow amplitude")

    # =================
    # lab = ["A", "B", "C", "D", "E"]
    # for iax, ax in enumerate(hw_ax):
    #     PH.nice_plot(ax, direction="out", ticklength=3, position=-0.03)
    #     ax.set_xlabel("")
    #     ax.text(-0.1, 1.05, s=lab[iax], transform=ax.transAxes, fontsize=18, fontweight="bold")
    # mpl.tight_layout
    # mpl.show()

csv_file = "spike_steady_state_halfwidths_11-Jan-2026.csv"

def plot_ss_hws(experiment, ax=None):
    if ax is None:
        f, ax = mpl.subplots(1, 1, figsize=(4,4))
        ax = [ax] # make it a list for future compatability
    stats_dir = experiment.get("R_statistics_summaries", None)
    local_dir = Path(experiment.get("localdatapath", "."), stats_dir)
    PH.nice_plot(ax[0], direction="out", ticklength=3, position=-0.03)
    df = pd.read_csv(Path(local_dir, csv_file))
    df = df[df["age_category"] != "ND"]
    measure = "halfwidth_steadystate"
    df['hw'] = df['halfwidth_dc'] + df['halfwidth_amp1'] + df['halfwidth_amp2']
    df['hw'] = df['hw'] * 1e6
    ylims = (0, 800)
    scf = 1e6
    usec = r"$\mu$s"
    ylabel = f"Steady-state AP Halfwidth ({usec})"
    df[measure] = df[measure] * scf  # convert to us
   
    # =================
    bar_and_scatter(
        df,
        x="age_category",
        y="hw", # measure,
        hue="age_category",
        experiment=experiment,
        ax=ax[0],
    )
    ax[0].set_ylim(ylims)
    ax[0].set_ylabel(ylabel)

    mpl.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Analyze spike half-width time course through IV protocols"
    )
    parser.add_argument(
        "-c",
        "--compute",
        action="store_true",
        help="Compute spike half-widths for all cells and save summary",
    )
    parser.add_argument(
        "-s",
        "--summary",
        action="store_true",
        help="Plot spike half-width summary from saved data",
    )
    parser.add_argument(
        "-f3",
        action="store_true",
        help="Plot supplemental figure 3",
    )

    args = parser.parse_args()

    configpath = Path("/Users/pbmanis/Desktop/Python/RE_CBA/config/experiments.cfg")
    exptname = "CBA_Age"
    celltype = "pyramidal"
    datasets, experiments = get_configuration.get_configuration(configpath)
    experiment = experiments["CBA_Age"]
    adpath = experiment["analyzeddatapath"]
    stats_dir = experiment.get("R_statistics_summaries", None)
    local_dir = Path(experiment.get("localdatapath", "."), stats_dir)
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local directory does not exist: {local_dir!s}")

    if args.compute:
        specific_cell_list = None 
        # ['2023_01_23_S0C0_pyramidal_IVs.pkl',
        #         '2023_10_27_S1C1_pyramidal_IVs.pkl',
        #                       '2024_08_21_S1C2_pyramidal_IVs.pkl',
        #                       '2023_09_11_S1C1_pyramidal_IVs.pkl',  # for testing specific cells
        #         ]

        compute_measures_all_cells(adpath, exptname, celltype, experiment, specific_cell_list=specific_cell_list)
        exit()

    if args.summary:
        plot_ss_hws(experiment)
        exit()

    if args.f3:
        P = build_figure_framework()

        # single cell:
        df = pd.read_csv(Path(local_dir, csv_file))
        df = df[df["cell_id"] == "2023.09.06_000/slice_000/cell_001"]
        pyrdatapath = Path(adpath, exptname, celltype, "2023_09_06_S0C1_pyramidal_IVs.pkl")
        d = read_pkl_file(filename=pyrdatapath)
        ivs = list(d["Spikes"].keys())
        print(d.keys())

        # print(d['IV'].keys())
        # prots = list(d['IV'].keys())
        # print(d['IV'][prots[0]].keys())
        # print(d['IV'][prots[0]]['tauh_bovera'], d['IV'][prots[0]]['tauh_tau'],d['IV'][prots[0]]['tauh_Gh'])
        meas_dict = {
            "A1": "halfwidth",
            "B1": "dvdt_rising",
            "C1": "dvdt_falling",
            "D1": "AP_begin_V",
        }
        i = 0
        lets = ["A", "B", "C", "D"]
        for k, v in meas_dict.items():
            print("measure: v): ", v)
            res = plot_spike_measure(
                d,
                experiment=experiment,
                filename=pyrdatapath,
                measure=v,
                ax=P.axdict[k],
            )

            plot_amp_tau(
                experiment, measure=v, ax1=P.axdict[lets[i] + "2"], ax2=P.axdict[lets[i] + "3"]
            )
            i += 1

        mpl.show()
    # post_stimulus_spikes(d)
    # print_LCS_spikes(d)
