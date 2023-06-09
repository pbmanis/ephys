import numpy as np
from typing import Union
import logging


"""Various calculations on traces
"""

Logger = logging.getLogger(__name__)


def ZScore(
    timebase: np.ndarray,
    data: np.ndarray,
    pre_std: Union[float, None] = None,
    pre_mean: Union[float, None] = None,
    twin_base: list = [0, 0.1],
    twin_resp: list = [[0.101, 0.130]],
) -> float:
    """Compute a Z-Score on the currents, comparing
    the mean and standard deviation during the baseline
    with the mean in a response window
    abs(post.mean() - pre.mean()) / pre.std()

    The pre_std/pre_mean may be taken from the current trace, or may be provided
    from an external aggregate value computed across a group of traces.

    Args:
        timebase (np.ndarray): _description_
        data (np.ndarray): _description_
        pre_std (Union[float, None], optional): _description_. Defaults to None.
            if pre_std is not None, pre_mean must also be provided
        pre_mean (Union[float, None], optional): _description_. Defaults to None.
        twin_base (list, optional): _description_. Defaults to [0, 0.1].
        twin_resp (list, optional): _description_. Defaults to [[0.101, 0.130]].

    Returns:
        float: _description_
    """
    if pre_std is not None or pre_mean is not None:
        assert isinstance(pre_std, float) and isinstance(pre_mean, float)

    # check if we need to compute here, or if we use precomputed values
    if pre_std is None:
        timebaseindx = np.where((timebase >= twin_base[0]) & (timebase < twin_base[1]))
        pre_mean = np.nanmean(data[timebaseindx])  # baseline
        pre_std = np.nanstd(data[timebaseindx])

    trindx = np.where((timebase >= twin_resp[0]) & (timebase < twin_resp[1]))
    mpost = np.nanmean(data[trindx])  # response

    try:
        zs = np.fabs((mpost - pre_mean) / pre_std)
    except:
        zs = 0.0
    return zs


def ZScore2D(
    timebase: np.ndarray,
    data: np.ndarray,
    pre_std: Union[float, None] = None,
    pre_mean: Union[float, None] = None,
    twin_base: list = [0, 0.1],
    twin_resp: list = [[0.101, 0.130]],
):
    """Compute ZScore over a group of traces in 2D

    Args:
        timebase (np.ndarray): _description_
        data (np.ndarray): _description_
        pre_std (Union[float, None], optional): _description_. Defaults to None.
            if pre_std is not None, pre_mean must also be provided
        pre_mean (Union[float, None], optional): _description_. Defaults to None.
        twin_base (list, optional): _description_. Defaults to [0, 0.1].
        twin_resp (list, optional): _description_. Defaults to [[0.101, 0.130]].

    """
    zscores = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        zscores[i] = ZScore(timebase, data[i, :], pre_std, pre_mean, twin_base, twin_resp)
    return zscores


def grand_mean_std(timebase: np.ndarray, data: np.ndarray, window: list = [0, 0.1]):
    if len(data.shape) != 2:
        raise ValueError(
            "grand_mean_std: Input data must be a 2D array: ntraces x trace"
        )
    trindex = np.where((timebase >= window[0]) & (timebase < window[1]))[0]
    print(trindex[0], trindex[-1])
    grandmean = np.nanmean(data[:, trindex[0]:trindex[-1]])
    grandstd = np.nanstd(data[:, trindex[0]:trindex[-1]])
    return grandmean, grandstd


def Imax(
    timebase: np.ndarray,
    data: np.ndarray,
    twin_base: list = [0, 0.1],
    twin_resp: list = [[0.101, 0.130]],
    sign: int = 1,
) -> float:

    # print(np.min(timebase), np.max(timebase))
    # print(twin_base)
    # print(twin_resp)
    try:
        timebaseindex = np.where((timebase >= twin_base[0]) & (timebase < twin_base[1]))[0]
    # print(timebaseindex)
        trindex = np.where((timebase >= twin_resp[0]) & (timebase < twin_resp[1]))[0]
    # print(trindex)
        mpost = np.nanmax(sign * data[trindex[0]:trindex[-1]])  # response goes negative...
    except:
        Logger.critical("Imax has no data to operation on")
        return 0
    return mpost
