import numpy as np
import warnings
from typing import Union
import matplotlib
matplotlib.use('qtagg')  # Use the Qt6Agg backend for interactive plotting
import matplotlib.pyplot as mpl


def adaptation_index2(spk_lat: Union[list, np.ndarray]):
        """adaptation_index Compute an adaptation index from eFEL (2025)
        Modified to use Allen Institute ephys SDK version (norm_diff)
        Parameters
        ----------
        spk_lat : array
            spike latencies, already trimmed to current step window.
        trace_duration : float, optional
            _description_, by default 1.0

        Returns
        -------
        float
            adaptation index as described above
        """

        if len(spk_lat) < 4:
            return np.nan
        # eFEL version:

        # clean up the spike latencies to be sure there are no duplicates ?
        # spk_lat = np.unique(spk_lat)  # prevent isi_sub from being zero. how that might happen is a mystery
        # isi_values = spk_lat[1:] - spk_lat[:-1]
        # isi_sum = isi_values[1:] + isi_values[:-1]
        # isi_sub = isi_values[1:] - isi_values[:-1]
        # print("adaptation_index2: ISI SUB: ", isi_sub)
        # nonzeros = np.argwhere(isi_sub != 0.0)
        # adaptation_index = np.mean(isi_sum[nonzeros] / isi_sub[nonzeros])

        # Allen Institute version:
        isis =  np.array(np.diff(spk_lat)) # np.array(spk_lat[1:]) - np.array(spk_lat[:-1])
        print("ISIS: ", isis)
        if np.allclose((isis[1:] + isis[:-1]), 0.0):
            return np.nan
        # (st(i+1) - st(i))/ (st(i+1) + st(i))
        norm_diffs = (isis[1:] - isis[:-1]) / (isis[1:] + isis[:-1])
        print("norm_diffs: ", norm_diffs)
        norm_diffs[(isis[1:] == 0) & (isis[:-1] == 0)] = 0.0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
            adaptation_index = np.nanmean(norm_diffs)
        return adaptation_index

if __name__ == "__main__":
    # test the function with some example spike latencies
    # regular train, 100 Hz
    spk_lat = np.arange(0.1, 1.0, 0.02)  # example spike latencies in seconds

    # print(spk_lat)
    # print(np.diff(spk_lat))
    ai_adaptation_index = adaptation_index2(spk_lat)
    print("Adaptation Index  should be ~0: ", ai_adaptation_index)
    mpl.plot(spk_lat, np.zeros_like(spk_lat), 'o-', label='Spike Latencies1')
    
    spk_lat2 = [0.1, 0.20, 0.31, 0.43, 0.56, 0.70, 0.85, 1.01, 1.18]  # example spike latencies in seconds
    ai_adaptation_index2 = adaptation_index2(spk_lat2)
    print("Adaptation Index  should be >0: ", ai_adaptation_index2)
    mpl.plot(spk_lat2, np.zeros_like(spk_lat2)+1, 'o-', label='Spike Latencies2')
    
    spk_lat3 = [0.1, 0.19, 0.27, 0.34, 0.40, 0.45, 0.49, 0.53, 0.55]  # example spike latencies in seconds
    ai_adaptation_index3 = adaptation_index2(spk_lat3)
    print("Adaptation Index  should be <0: ", ai_adaptation_index3)
    mpl.plot(spk_lat3, np.zeros_like(spk_lat3)+2, 'o-', label='Spike Latencies')
    mpl.legend()
    mpl.show()
