import numpy as np
from typing import Union

"""Various calculations on traces
"""


def ZScore(
    tb: np.ndarray,
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
        tb (np.ndarray): _description_
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
        tbindx = np.where((tb >= twin_base[0]) & (tb < twin_base[1]))
        pre_mean = np.nanmean(data[tbindx])  # baseline
        pre_std = np.nanstd(data[tbindx])

    trindx = np.where((tb >= twin_resp[0]) & (tb < twin_resp[1]))
    mpost = np.nanmean(data[trindx])  # response

    try:
        zs = np.fabs((mpost - pre_mean) / pre_std)
    except:
        zs = 0.0
    return zs


def ZScore2D(
    tb: np.ndarray,
    data: np.ndarray,
    pre_std: Union[float, None] = None,
    pre_mean: Union[float, None] = None,
    twin_base: list = [0, 0.1],
    twin_resp: list = [[0.101, 0.130]],
):
    """Compute ZScore over a group of traces in 2D

    Args:
        tb (np.ndarray): _description_
        data (np.ndarray): _description_
        pre_std (Union[float, None], optional): _description_. Defaults to None.
            if pre_std is not None, pre_mean must also be provided
        pre_mean (Union[float, None], optional): _description_. Defaults to None.
        twin_base (list, optional): _description_. Defaults to [0, 0.1].
        twin_resp (list, optional): _description_. Defaults to [[0.101, 0.130]].

    """
    zscores = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        zscores[i] = ZScore(tb, data[i, :], pre_std, pre_mean, twin_base, twin_resp)
    return zscores


def grand_mean_std(tb: np.ndarray, data: np.ndarray, window: list = [0, 0.1]):
    if len(data.shape) != 2:
        raise ValueError(
            "grand_mean_std: Input data must be a 2D array: ntraces x trace"
        )
    trindex = np.where((tb >= window[0]) & (tb < window[1]))[0]
    print(trindex[0], trindex[-1])
    grandmean = np.nanmean(data[:, trindex[0]:trindex[-1]])
    grandstd = np.nanstd(data[:, trindex[0]:trindex[-1]])
    return grandmean, grandstd


def Imax(
    tb: np.ndarray,
    data: np.ndarray,
    twin_base: list = [0, 0.1],
    twin_resp: list = [[0.101, 0.130]],
    sign: int = 1,
) -> float:

    tbindex = np.where((tb >= twin_base[0]) & (tb < twin_base[1]))[0]
    trindex = np.where((tb >= twin_resp[0]) & (tb < twin_resp[1]))[0]
    mpost = np.nanmax(sign * data[trindex[0]:trindex[-1]])  # response goes negative...
    return mpost
