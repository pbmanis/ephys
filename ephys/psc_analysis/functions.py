import numpy as np
import matplotlib.pyplot as mpl
from dataclasses import dataclass
from typing import Union, List, Tuple
import MetaArray as EM  # need to use this version for Python 3

@dataclass
class IAnalysis:
    Vcmd: Union[np.ndarray, None] = None
    i_mean_index: int = -1
    i_data: int = -1
    i_tb: int = -1
    i_argmin: int = -1


def mean_I_analysis(
    clamps:object,
    region=None,
    t0=0.0,
    mode="mean",
    baseline=None,
    intno=0,
    nint=1,
    reps=[0],
    slope=False,
    slopewin=None,
):
    """
    Get the mean or min or baseline current in a window
    Works with the current Clamps object

    Parameters
    ----------
    region: tuple, list or numpy array with 2 values (default: None)
        start and end time of a trace used to measure the RMP across
        traces. Note that if slopewin is set, it may replace the region

    t0: float (default=0.5)
        start time for the mean current analysis, offset from the region ("deadwin")

    mode: str (default='mean')
        How to measure the value (valid values: 'mean', 'baseline' both compute mean,
        'min' gives the minimum current in the window.

    baseline: np.array (default None)
        an array of precomputed baseline values to subtract from the data; one value
        per trace

    intno: int (default=0)
        first trace to do in a group

    nint: int (default=1)
        # of traces to skip (e.g., for repeats or different values across the array)

    reps: list (default=[0])
        # of repetitions (used to reshape data in computation)

    slope: bool (default=True)
        set to subtract a slope from trace

    slopewin: list or np.array of 2 elements (default=None)
        Time window to use to compute slope, [start, stop], in seconds

    Return
    ------
    the mean current in the window

    Stores computed mean current in the variable "name".
    """
    if region is None:
        raise ValueError(
            "PSPSummary, mean_I_analysis requires a region beginning and end to measure the current"
        )
    results = IAnalysis()

    analysis_region = region.copy()
    if slope and slopewin is not None:
        region = slopewin

    data1 = clamps.traces["Time" : region[0] : region[1]]
    # print('data shape: ', data1.shape)

    # data2 = clamps.traces.view(np.ndarray)
    rgn = [int(region[i] / clamps.sample_interval) for i in range(len(region))]
    results.V_cmd = (
        clamps.cmd_wave[:, rgn[0] : rgn[1]].mean(axis=1).view(np.ndarray)
    )
    tb = np.arange(
        0, data1.shape[1] * clamps.sample_interval, clamps.sample_interval
    )
    # tb2 = np.arange(0, data2.shape[1]*clamps.sample_interval, clamps.sample_interval)
    data1 = data1.view(np.ndarray)

    if baseline is not None:
        data1 = np.array([data1[i] - baseline[i] for i in range(data1.shape[0])])
        # data2 = np.array([data2[i]-baseline[i] for i in range(data2.shape[0])])

    # subtract the "baseline" from the beginning of the interval to the end.
    if slope:
        data1 = slope_subtraction(tb, data1, region, mode=mode)
        # if not slope and slopewin is not None: # just first slope point to align current
        #     data1 = results.slope_subtraction(tb, data1, region, mode='point')
        print("slope, slopewin: ", slope, slopewin, mode)

    sh = data1.shape
    if nint > 1:
        dindx = range(intno, sh[0], nint)
        if data1.ndim == 3:
            data1 = data1[dindx, :, :]
        elif data1.ndim == 2:
            data1 = data1[dindx, :]
        else:
            raise ValueError("Data must have 2 or 3 dimensions")
    # print(sh, data1.shape, nint)
    results.i_mean_index = None
    results.i_data = data1.mean(axis=0)
    results.i_tb = tb + region[0]

    nx = int(sh[0] / len(reps))

    if mode in ["mean", "baseline"]:  # just return the mean value
        i_mean = data1.mean(
            axis=1
        )  # all traces, average over specified time window
        if nint == 1:
            nx = int(sh[0] / len(reps))
            try:
                i_mean = np.reshape(
                    i_mean, (len(reps), nx)
                )  # reshape by repetition
            except:
                return i_mean
        i_mean = i_mean.mean(axis=0)  # average over reps
        return i_mean, results

    elif mode == "min":

        # mpl.plot(data1.T)
        # mpl.show()

        i_mina = data1.min(axis=1)  # all traces, average over specified time window
        if nint == 1:
            nx = int(sh[0] / len(reps))
            try:
                i_mina = np.reshape(
                    i_mina, (len(reps), nx)
                )  # reshape by repetition
            except:
                raise ValueError("Reshape failed on min")

        i_min = i_mina.min(axis=0)  # average over reps
        mpl.plot(i_min)
        mpl.show()
        results.i_argmin = np.argmin(i_mina, axis=0)
        # return i_min

        # dfilt = data1 # scipy.signal.savgol_filter(data1, 5, 2, axis=1, mode='nearest')
        # print('data1.shpae 0: ', data1.shape)
        # ist = int(t0/clamps.sample_interval) # points in deadwin
        # ien = int((analysis_region[1]-analysis_region[0])/clamps.sample_interval)
        # print('region 0: ', analysis_region[0])
        # print('analysis time: ', ist*clamps.sample_interval, ien*clamps.sample_interval)
        # print('dfilt shape: ', dfilt.shape)
        # print('ist, ien: ', ist, ien)
        # i_min = dfilt[:, ist:ien].min(axis=1)  # all traces, get the minimum over specified time window
        # print('nint: ', nint)
        # print('imin shape: ', i_min.shape)
        # print('reps: ', nreps)
        # if nint == 1:
        #     nx = int(sh[0]/nreps)
        #     print('nx: ', nx)
        #     print(i_min.shape)
        #     print((nreps, nx))
        #     try:
        #         i_min = np.reshape(i_min, (nreps, nx))  # reshape by repetition
        #         print('rehsape ok')
        #     except:
        #         print('reshape failed!!!!')
        # i_min = i_min.mean(axis=0)  # average over reps
        print("imin shape: ", i_min.shape)
        # data2 = np.reshape(data1, (nreps, nx, data1.shape[1])).mean(axis=0)
        # print(data2.shape)
        # f, ax = mpl.subplots(1,1)
        # sns.set_palette("coolwarm_r",data2.shape[0])
        # cp = sns.color_palette("muted",data2.shape[0])
        # device = 'Stim0'
        # # stim_I = np.array(results.AR.sequence[(device, 'command.PulseTrain_amplitude')])*1e6
        # for i in range(data2.shape[0]):
        #     ax.plot(data2[i,:], color = cp[i])
        # data3 = np.reshape(data1, (nreps, nx, data1.shape[1]))
        # for j in range(data3.shape[0]):
        #     for i in range(data3.shape[1]):
        #         csel = i
        #         print('csel: ', csel)
        #         ax.plot(data3[j, i,:], color = cp[csel], linestyle='--', linewidth=1, alpha=1)
        # mpl.show()
        # # print(ist, ien)
        # print(dfilt.shape, nreps)
        # dfw = [[]]*nreps
        #  nvs = int(sh[0]/nreps)
        #  print('nreps, nvs: ', nreps, nvs)
        #  for i in range(nreps):
        #      dfw[i] = data1[i*nvs:i*nvs + nvs,: ]
        #  dfw = np.array(dfw)
        #  print(dfw.shape)
        #  dfw = np.array(dfw).mean(axis=0)
        #  # dfw = dfw.reshape((nreps, -1, int(sh[0]/nreps)))
        #  # dfw = dfw.mean(axis=0)
        #  # for i in range(dfw.shape[0]):
        #  #     mpl.plot(dfw[i])
        #  # mpl.show()
        #
        #  # print(dfw.shape, ist, ien)
        #  results.i_argmin = dfw[:, ist:ien].argmin(axis=1) +ist

        return i_min, results

def slope_subtraction(tb, data1, region, mode="mean"):
    """
    Subtract a slope from the data; the slope is calculated from a time region

    Parameters
    ----------
    tb: np.array
        time base, in seconds. Must be of same size as data1 2nd dimension
    data1: np.array
        data array; 2 dimensional (traces x time)
    region: 2 element list or np.array
        time region for computation of the slope, in seconds
    mode: str (default: 'mean')
        Either 'point' (does nothing to the data)
            or 'mean'
    Return
    ------
        slope-subtracted data
    """
    dt = tb[1] - tb[0]
    minX = 0  # int((region[0])/dt)
    maxX = int((region[1] - region[0]) / dt)
    if mode == "point":  # do nothing...
        # for i in range(data1.shape[0]):
        #     data1[i,:] -=  data1[i,0]
        return data1

    for i in range(data1.shape[0]):
        x0 = list(range(minX, minX + 3))
        ml = maxX
        x0.extend(list(range(ml - 10, ml)))
        fdx = np.array([tb[j] for j in x0])
        fdy = np.array([data1[i][j] for j in x0])
        pf = np.polyfit(fdx, fdy, 1)
        bline = np.polyval(pf, tb)
        if bline.shape[0] > data1[i].shape[0]:
            bline = bline[: data1[i].shape[0]]
        if mode != "baseline":
            data1[i, :] -= bline
    return data1

def get_traces(
    clamps:object,
    region:Union[List, Tuple]=None,
    trlist:Union[List, Tuple, None]=None,
    baseline:Union[None, List]=None,
    order:int=0,
    intno:int=0,
    nint:int=1,
    reps:list=[0],
    mode:str="baseline",
    slope:bool=True,
):
    """
    Get the mean current (averages) in a window

    Parameters
    ----------
    region: tuple, list or numpy array with 2 values (default: None)
        start and end time of a trace used to measure the RMP across
        traces.

    Return
    ------
    Nothing

    Stores computed mean current in the variable "name".
    """
    if region is None:
        raise ValueError(
            "PSPSummary, mean_I_analysis requires a region beginning and end to measure the current"
        )

    data1 = clamps.traces["Time" : region[0] : region[1]]

    tb = np.arange(
        0, data1.shape[1] * clamps.sample_interval, clamps.sample_interval
    )
    data1 = data1.view(np.ndarray)
    nreps = len(reps)
    sh = data1.shape

    if nint > 1:
        # if order == 0:
        dindx = range(intno, sh[0], nint)
        data1 = data1[dindx]
    # subtract the "baseline" from the beginning of the interval to the end.
    if slope:
        data1 = slope_subtraction(tb, data1, region, mode=mode)
    if baseline is not None:
        data1 = np.array([data1[i] - baseline[i] for i in range(data1.shape[0])])

    nx = int(sh[0] / nreps)
    if nx < 13:
        nreps = 1

    if order == 0 and nreps > 1:
        try:
            print(
                "gettraces reshaping: data shape, reps, nx, nint: ",
                data1.shape,
                nreps,
                nx,
                data1.shape[0] / nreps,
                nint,
            )
            data2 = np.reshape(data1, (len(reps), nx, -1))
        except:
            print(
                "Failed to reshape: data shape, reps, nx: ",
                data1.shape,
                len(reps),
                nx,
                data1.shape[0] / len(reps),
            )
            if data1.shape[0] > 1:
                data2 = data1
                return data2, tb
            else:
                return None, None
    elif order == 1 or nreps == 1:
        data2 = data1  # np.reshape(data1, (len(reps), nx,   sh[1]))
    data2 = data2.mean(axis=0)
    
    return data2, tb
