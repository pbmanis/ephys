"""
Provide fitting functions for PSCs:
    1. decay tau only
    2. PSC full fit (1-exp(tau_rise))^4 * exp(tau_fall)

"""

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Union

import lmfit
import MetaArray as EM  # need to use this version for Python 3
import numpy as np
import pandas as pd
from matplotlib import pyplot as mpl
from pylibrary.plotting import plothelpers as PH
from pylibrary.tools import cprint
from pyqtgraph.Qt import QtCore, QtGui

from ..tools import cursor_plot as CP
from ephys.datareaders import acq4read


class PSC_Fitter:


    def __init__(self):
        pass  # nothing to do

    def _fcn_tau(self, params, x, data):
        """Model single exponential"""
        v = params.valuesdict()
        model = v["amp"] * np.exp(-x / v["tau_fall"]) + v["DC"]
        return model - data

    def fitTau(self):
        # create a set of Parameters
        params = lmfit.Parameters()
        params.add(
            "amp",
            value=self.ptreedata.param("Initial Fit Parameters").param("amp").value(),
            min=-self.dmax,
            max=self.dmax,
        )
        params.add(
            "tau_fall",
            value=self.ptreedata.param("Initial Fit Parameters")
            .param("taufall")
            .value(),
            min=1e-4,
            max=1e-1,
        )
        params.add(
            "DC",
            value=self.ptreedata.param("Initial Fit Parameters").param("DC").value(),
            min=-1e3,
            max=1e3,
        )

        t0 = self.T0.value()
        t1 = self.T1.value()
        it0 = int(t0 / self.dt)
        it1 = int(t1 / self.dt)
        if it0 > it1:
            t = it0
            it0 = it1
            it1 = t

        time_zero = int(self.time_zero / self.dt)
        print("timezero: ", time_zero, self.dataX[time_zero])
        # do fit, here with the default leastsq algorithm
        minner = lmfit.Minimizer(
            self._fcn_tau,
            params,
            fcn_args=(self.dataX[it0:it1] - self.dataX[time_zero], self.dataY[it0:it1]),
        )
        self.fitresult = minner.minimize("leastsq")

        # write error report
        lmfit.report_fit(self.fitresult)

    def _fcn_EPSC(self, params, x, data):
        """Model EPSC"""
        v = params.valuesdict()

        model = (
            v["amp"]
            * (((1.0 - np.exp(-x / v["tau_rise"])) ** 4.0) * np.exp(-x / v["tau_fall"]))
            + v["DC"]
        )
        return model - data

    def fitEPSC(self):

        # create a set of Parameters
        params = lmfit.Parameters()
        params.add(
            "amp",
            value=self.ptreedata.param("Initial Fit Parameters").param("amp").value(),
            min=-self.dmax,
            max=self.dmax,
        )
        params.add(
            "tau_rise",
            value=self.ptreedata.param("Initial Fit Parameters")
            .param("taurise")
            .value(),
            min=1e-4,
            max=1e-1,
        )
        params.add(
            "tau_fall",
            value=self.ptreedata.param("Initial Fit Parameters")
            .param("taufall")
            .value(),
            min=1e-4,
            max=1e-1,
        )
        params.add(
            "DC",
            value=self.ptreedata.param("Initial Fit Parameters").param("DC").value(),
            min=-1e3,
            max=1e3,
        )
        dc = np.mean(self.dataY[0:10])
        params.add("DC", value=dc, min=dc - dc * 1, max=dc + dc * 1)
        t0 = self.T0.value()
        t1 = self.T1.value()
        it0 = int(t0 / self.dt)
        it1 = int(t1 / self.dt)
        if it0 > it1:
            t = it0
            it0 = it1
            it1 = t

        # do fit, here with the default leastsq algorithm
        time_zero = int(self.time_zero / self.dt)
        print("timezero: ", time_zero, self.dataX[time_zero])
        print(self.dataX[it0:it1] - self.time_zero)
        print(self.dataY[it0:it1])

        minner = lmfit.Minimizer(
            self._fcn_EPSC,
            params,
            fcn_args=(self.dataX[it0:it1] - self.dataX[time_zero], self.dataY[it0:it1]),
        )
        self.fitresult = minner.minimize(
            method="least_squares",
        )

        # write error report
        lmfit.report_fit(self.fitresult)
