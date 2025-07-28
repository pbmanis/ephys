from pathlib import Path
from dataclasses import dataclass, field
from typing import Union
import numpy as np

""" 
define the MINID datalcass for mini analysis. Derived from the 
ank2_datasets.py file (Maness ank2 project).

Just import this to define the class. 
Returns
-------
nothing


"""



def def_empty_list():
    return []


def def_analysis_window():
    return [0.00, None]


def def_empty_dict():
    return {}


def def_notch_list():
    nl = [
        60.0,
        120.0,
        180.0,
        240.0,
        300.0,
        360.0,
        420.0,
        480.02,
        540.0,
        600.0,
        660.0,
        720.2,
        780.0,
        900.0,
        960.4,
        1020.03,
        1140.04,
        1260.5,
        1380.5,
        1500.6,
        1620.6,
        1740.6,
        1860.6,
        1980.6,
        2100.8,
        2220.8,
        2340.8,
        2460.8,
        2581.0,
        2701.0,
        2821.0,
        2941.0,
        3061.2,
    ]
    nl = np.arange(60, 8000, 60)
    nl = np.concatenate((nl, [728.6, 4000]))
    # nl = [] # [60]
    return nl


"""
data classes/structures:
    IOD: Input/output data protocols, and PPF protocols
    MINID : Mini EPSC data class
"""
default_minamp = 5.0e-12  # minimum amplitude for detection of events, in Amps.


@dataclass
class MINID:
    ID: Union[int, str, None] = None  # animal ID
    sex: str = "ND"
    EX: bool = False  # exclusion flag
    GT: Union[str, None] = None  # Genotype
    SPEC_GT: Union[str, None] = None  # specific genotype (exact genotype all 3 crosses)
    EXPR: str = "ND"  # Ank2 expression : should be F/F or +/+
    TMX: bool = False  # Tamoxifen treatment
    EYFP: str = "ND"  # for not determined
    NG: list = field(default_factory=def_empty_list)  # list of bad traces
    analysis_window: list = field(default_factory=def_analysis_window)
    LPF: Union[float, None] = 2000.0  # low-pass filter setting
    HPF: Union[float, None] = 1
    NotchFilter: bool = True
    NotchFreqs: list = field(default_factory=def_notch_list)
    Notch_Q: float = 60.0
    sign: float = -1.0  # 1 for IPSCs, -1 for EPSCs (depending on RMP, of course.)
    rt: float = 0.001  # EPSC rise time (all values in seconds)
    decay: float = 0.006  # EPSC decay time const
    thr: float = 3.5  # threshold for detection algorithm
    min_event_amplitude: float = default_minamp  # minimum amplitude, in pA
    order: int = 7  # smoothing order for peak identification etc.

