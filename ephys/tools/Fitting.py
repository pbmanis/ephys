#!/usr/bin/env python

"""
Python class wrapper for data fitting.
Includes the following external methods:
getFunctions returns the list of function names (dictionary keys)
FitRegion performs the fitting
Note that FitRegion no longer attempts to plot.

"""
# January, 2009
# Paul B. Manis, Ph.D.
# UNC Chapel Hill
# Department of Otolaryngology/Head and Neck Surgery
# Supported by NIH Grants DC000425-22 and DC004551-07 to PBM.
# Copyright Paul Manis, 2009
#
"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
"""
    Additional Terms:
    The author(s) would appreciate that any modifications to this program, or
    corrections of erros, be reported to the principal author, Paul Manis, at
    pmanis@med.unc.edu, with the subject line "Fitting Modifications".

"""

import warnings
from typing import Union

import numpy as np
import scipy
import scipy.optimize


class Fitting:
    # dictionary contains:
    # name of function: function call, initial parameters, iterations, plot color, then x and y for testing
    # target valutes, names of parameters, contant values, and derivative function if needed.
    #
    def __init__(self):
        self.fitfuncmap = {
            "exp0": (
                self.exp0eval,
                [0.0, 20.0],
                2000,
                "k",
                [0, 100, 1.0],
                [1.0, 5.0],
                ["A0", "tau"],
                None,
                None,
            ),
            "exp1": (
                self.expeval,
                [-60, 3.0, 15.0],
                10000,
                "k",
                [0, 100, 1.0],
                [0.5, 1.0, 5.0],
                ["DC", "A0", "tau"],
                None,
                None,
            ),  # self.expevalprime),
            "expsat": (
                self.expsat,
                [0.0, 1.0, 20.0],
                2000,
                "k",
                [0, 10, 1.0],
                [0.5, 1.0, 5.0],
                ["DC", "A0", "tau"],
                None,
                self.expsatprime,
            ),
            "exptau": (
                self.exptaueval,
                [-60, 3.0, 15.0],
                10000,
                "k",
                [0, 100, 1.0],
                [0.5, 1.0, 5.0],
                ["DC", "A0", "tau"],
                None,
                None,
            ),  # self.expevalprime),
            "expsum": (
                self.expsumeval,
                [0.0, -0.5, 200.0, -0.25, 450.0],
                500000,
                "k",
                [0, 1000, 1.0],
                [0.0, -1.0, 150.0, -0.25, 350.0],
                ["DC", "A0", "tau0", "A1", "tau1"],
                None,
                None,
            ),
            "expsum2": (
                self.expsumeval2,
                [0.0, -0.5, -0.250],
                50000,
                "k",
                [0, 1000, 1.0],
                [0.0, -0.5, -0.25],
                ["A0", "A1"],
                [5.0, 20.0],
                None,
            ),
            "exp2": (
                self.exp2eval,
                [0.0, -0.5, 200.0, -0.25, 450.0],
                500000,
                "k",
                [0, 1000, 1.0],
                [0.0, -1.0, 150.0, -0.25, 350.0],
                ["DC", "A0", "tau0", "A1", "tau1"],
                None,
                None,
            ),
            "exppow": (
                self.exppoweval,
                [
                    0.0,
                    1.0,
                    100,
                ],
                2000,
                "k",
                [0, 100, 0.1],
                [0.0, 1.0, 100.0],
                ["DC", "A0", "tau"],
                None,
                None,
            ),
            "exppulse": (
                self.expPulse,
                [3.0, 2.5, 0.2, 2.5, 2.0, 0.5],
                2000,
                "k",
                [0, 10, 0.3],
                [0.0, 0.0, 0.75, 4.0, 1.5, 1.0],
                ["DC", "t0", "tau1", "tau2", "amp", "width"],
                None,
                None,
            ),
            "boltz": (
                self.boltzeval,
                [0.0, 1.0, -50.0, -5.0],
                5000,
                "r",
                [-130.0, -30.0, 1.0],
                [0.00, 0.010, -100.0, 7.0],
                ["DC", "A0", "x0", "k"],
                None,
                None,
            ),
            "gauss": (
                self.gausseval,
                [1.0, 0.0, 0.5],
                2000,
                "y",
                [-10.0, 10.0, 0.2],
                [1.0, 1.0, 2.0],
                ["A", "mu", "sigma"],
                None,
                None,
            ),
            "line": (
                self.lineeval,
                [1.0, 0.0],
                500,
                "r",
                [-10.0, 10.0, 0.5],
                [0.0, 2.0],
                ["m", "b"],
                None,
                None,
            ),
            "poly2": (
                self.poly2eval,
                [1.0, 1.0, 0.0],
                500,
                "r",
                [0, 100, 1.0],
                [0.5, 1.0, 5.0],
                ["a", "b", "c"],
                None,
                None,
            ),
            "poly3": (
                self.poly3eval,
                [1.0, 1.0, 1.0, 0.0],
                1000,
                "r",
                [0.0, 100.0, 1.0],
                [0.5, 1.0, 5.0, 2.0],
                ["a", "b", "c", "d"],
                None,
                None,
            ),
            "poly4": (
                self.poly4eval,
                [1.0, 1.0, 1.0, 1.0, 0.0],
                1000,
                "r",
                [0.0, 100.0, 1.0],
                [0.1, 0.5, 1.0, 5.0, 2.0],
                ["a", "b", "c", "d", "e"],
                None,
                None,
            ),
            "sin": (
                self.sineeval,
                [-1.0, 1.0, 4.0, 0.0],
                1000,
                "r",
                [0.0, 100.0, 0.2],
                [0.0, 1.0, 9.0, 0.0],
                ["DC", "A", "f", "phi"],
                None,
                None,
            ),
            "boltz2": (
                self.boltzeval2,
                [0.0, 0.5, -50.0, 5.0, 0.5, -20.0, 3.0],
                1200,
                "r",
                [-100.0, 50.0, 1.0],
                [0.0, 0.3, -45.0, 4.0, 0.7, 10.0, 12.0],
                ["DC", "A1", "x1", "k1", "A2", "x2", "k2"],
                None,
                None,
            ),
            "taucurve": (
                self.taucurve,
                [50.0, 300.0, 60.0, 10.0, 8.0, 65.0, 10.0],
                50000,
                "r",
                [-150.0, 50.0, 1.0],
                [0.0, 237.0, 60.0, 12.0, 17.0, 60.0, 14.0],
                ["DC", "a1", "v1", "k1", "a2", "v2", "k2"],
                None,
                self.taucurveder,
            ),
            "FIGrowthExpBreak": (
                self.FIGrowth1,
                [0.0, 100.0, 1.0, 40.0, 200.0],
                2000,
                "k",
                [0, 1000, 50],  # [Fzero, Ibreak, F1amp, F2amp, Irate]
                [0.0, 0.0, 0.0, 10.0, 100.0],
                ["Fzero", "Ibreak", "F1amp", "F2amp", "Irate"],
                None,
                None,
            ),
            "FIGrowthExp": (
                self.FIGrowth2,
                [100.0, 40.0, 200.0],
                2000,
                "k",
                [0, 1000, 50],  # [FIbreak, F2amp, Irate]
                [00.0, 10.0, 100.0],
                ["Ibreak", "F2amp", "Irate"],
                None,
                None,
            ),
            "FIGrowthPower": (
                self.FIPower,
                [100.0, 0.2, 0.5],
                2000,
                "k",
                [0, 1000, 50],  # [c, s, d]
                [0.0, 10.0, 100.0],
                ["Ibreak", "Irate", "IPower"],
                None,
                None,
            ),
            "piecewiselinear3": (
                self.piecewiselinear,
                [10.0, 1.0, 200.0, 2.0, 5, 10],
                200,
                "k",
                [-200.0, 500.0, 50.0],  # def f(x,x0,y0,x1,k1,k2,k3):
                # x0,y0 : first breakpoint
                # x1 : second breakpoint
                # k1,k2,k3 : 3 slopes.
                [10.0, 1.0, 100.0, 5.0, 20.0, 50.0],
                ["Ibreak", "Rate0", "Ibreak1", "Irate1", "Irate2", "Irate3"],
                None,
                None,
            ),
            "piecewiselinear2": (
                self.pwl2,
                [100.0, 5.0, 0.05, 0.02],
                2000,
                "k",
                [40.0, 120, 1.0, 3.0],  # def f(x,x0,y0,k1,k2):
                # x0,y0 : first breakpoint
                # k1,k2 : 2 slopes.
                [0.0, 100.0, 0.5, 5.0],
                ["Ibreak", "Rate0", "Irate1", "Irate2"],
                None,
                None,
            ),
            "piecewiselinear3_old": (
                self.pwl3,
                [100.0, 0.0, 200.0, 0.0, 0.05, 0.02],
                2000,
                "k",
                [40, 0, 120, 0.0, 1.0, 3.0],  # def f(x,x0,y0,x1,k1,k2,k3):
                # x0,y0 : first breakpoint
                # x1 : second breakpoint
                # k1,k2,k3 : 3 slopes.
                [0.0, 0.0, 100.0, 0.0, 0.5, 5.0],
                ["Ibreak", "Rate0", "Ibreak1", "Irate1", "Irate2", "Irate3"],
                None,
                None,
            ),
            "Hill": (
                self.Hill,
                [100., 0.5e-9, 2], # fmax, ic50, n
                2000,
                'k',
                [0, 1e-10, 2.5], # [fmax, ic50, n]  # test values
                ["Fmax", "IC50", "n"],
                None,
                None,
            )
        }
        self.fitSum2Err = 0.0

    def getFunctions(self):
        return self.fitfuncmap.keys()

    def exp0eval(self, p, x, y=None, C=None, sumsq=False):
        """
        Exponential function with an amplitude and 0 offset
        """
        yd = p[0] * np.exp(-x / p[1])
        if y is None:
            return yd
        else:
            if sumsq is True:
                return np.sum((y - yd) ** 2.0)
            else:
                return y - yd

    def expsumeval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Sum of two exponentials with independent time constants and amplitudes,
        and a DC offset
        """
        yd = p[0] + (p[1] * np.exp(-x / p[2])) + (p[3] * np.exp(-x / p[4]))
        if y is None:
            return yd
        else:
            yerr = y - yd
            if weights is not None:
                yerr = yerr * weights
            if sumsq is True:
                return np.sum(yerr**2.0)
            else:
                return yerr

    def expsumeval2(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Sum of two exponentials, with predefined time constants , allowing
        only the amplitudes and DC offset to vary
        """
        yd = p[0] + (p[1] * np.exp(-x / C[0])) + (p[2] * np.exp(-x / C[1]))
        if y is None:
            return yd
        else:
            if sumsq is True:
                return np.sum((y - yd) ** 2.0)
            else:
                return y - yd

    def exptaueval(self, p, x, y=None, C=None, sumsq=True, weights=None):
        """
        Exponential with offset, decay from starting value
        """
        yd = (p[0] + p[1]) - p[1] * np.exp(-x / p[2])
        if y is None:
            return yd
        else:
            if sumsq is True:
                return np.sum((y - yd) ** 2.0)
            else:
                return y - yd

    def expeval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Exponential with offset
        if C has a parameter, the first in the list is treated
        as a gap at the start of the trace that is not included in the
        error function of the fit
        """

        try: 
            yd = p[0] + p[1] * np.exp(-x / p[2])
        except:
            print('p: ', p)
            print(np.min(x), np.max(x))
            raise RuntimeWarning("exp problem")
            exit()

        if y is None:
            return yd
        else:
            if C is not None:
                tgap = C[0]
                igap = int(tgap / (x[1] - x[0]))
            else:
                igap = 0
            if sumsq is True:
                return np.sum((y[igap:] - yd[igap:]) ** 2.0)
            else:
                return y[igap:] - yd[igap:]

    def expevalprime(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Derivative for exponential with offset
        """
        ydp = p[1] * np.exp(-x / p[2]) / (p[2] * p[2])
        yd = p[0] + p[1] * np.exp(-x / p[2])
        if y is None:
            return (yd, ydp)
        else:
            if sumsq is True:
                return np.sum((y - yd) ** 2.0)
            else:
                return y - yd

    def expsat(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Saturing single exponential rise with DC offset:
        p[0] + p[1]*(1-np.exp(-x/p[2]))
        """
        yd = p[0] + p[1] * (1.0 - np.exp(-x * p[2]))
        if y is None:
            return yd
        else:
            if sumsq is True:
                return np.sum((y - yd) ** 2)
            else:
                return y - yd

    def expsatprime(self, p, x):
        """
        derivative for expsat
        """
        #        yd = p[0] + p[1] * (1.0 - np.exp(-x * p[2]))
        ydp = p[1] * p[2] * np.exp(-x * p[2])
        return ydp

    def exppoweval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Single exponential function, rising to a ppower
        """

        if C is None:
            cx = 1.0
        else:
            cx = C[0]
        yd = p[0] + p[1] * (1.0 - np.exp(-x / p[2])) ** cx
        if y is None:
            return yd
        else:
            if sumsq is True:
                return np.sum((y - yd) ** 2)
            else:
                return y - yd

    def exp2eval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        For fit to activation currents...
        """
        yd = (
            p[0]
            + (p[1] * (1.0 - np.exp(-x / p[2])) ** 2.0)
            + (p[3] * (1.0 - np.exp(-x / p[4])))
        )
        if y == None:
            return yd
        else:
            if sumsq is True:
                ss = np.sqrt(np.sum((y - yd) ** 2.0))
                #                if p[4] < 3.0*p[2]:
                #                    ss = ss*1e6 # penalize them being too close
                return ss
            else:
                return y - yd

    #    @autojit
    def expPulse(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """Exponential pulse function (rising exponential with optional variable-length
        plateau followed by falling exponential)
        Parameter p is [yOffset, t0, tau1, tau2, amp, width]
        """
        yOffset, t0, tau1, tau2, amp, width = p
        yd = np.empty(x.shape)
        yd[x < t0] = yOffset
        m1 = (x >= t0) & (x < (t0 + width))
        m2 = x >= (t0 + width)
        x1 = x[m1]
        x2 = x[m2]
        yd[m1] = amp * (1 - np.exp(-(x1 - t0) / tau1)) + yOffset
        amp2 = amp * (1 - np.exp(-width / tau1))  ## y-value at start of decay
        yd[m2] = ((amp2) * np.exp(-(x2 - (width + t0)) / tau2)) + yOffset
        if y == None:
            return yd
        else:
            if sumsq is True:
                ss = np.sqrt(np.sum((y - yd) ** 2.0))
                return ss
            else:
                return y - yd

    def boltzeval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        yd = p[0] + (p[1] - p[0]) / (1.0 + np.exp((x - p[2]) / p[3]))
        if y == None:
            return yd
        else:
            if sumsq is True:
                return np.sqrt(np.sum((y - yd) ** 2.0))
            else:
                return y - yd

    def boltzeval2(self, p, x, y=None, C=None, sumsq=False, weights=None):
        yd = (
            p[0]
            + p[1] / (1 + np.exp((x - p[2]) / p[3]))
            + p[4] / (1 + np.exp((x - p[5]) / p[6]))
        )
        if y == None:
            return yd
        else:
            if sumsq is True:
                return np.sum((y - yd) ** 2.0)
            else:
                return y - yd

    def gausseval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        yd = (p[0] / (p[2] * np.sqrt(2.0 * np.pi))) * np.exp(
            -((x - p[1]) ** 2.0) / (2.0 * (p[2] ** 2.0))
        )
        if y == None:
            return yd
        else:
            if sumsq is True:
                return np.sum((y - yd) ** 2.0)
            else:
                return y - yd

    def Hill(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Hill function, as used in Rothman et al., 2009
        p[0] is the firing rate offset
        p[1] is Fmax (maximal value)
        p[2] is IC50 (point at half max)
        p[3] is n (power/cooperativity)
        Typically, p[0] will be set to 0 and not be allowed to vary.
        """
        # only for x > 0
        xn = np.where(x > 0)[0]
        yd = np.zeros_like(x)
        yd[xn] = p[0] + p[1] / (1.0 + (p[2]/x[xn])**p[3])
        if y is None:
            return yd
        else:
            if sumsq is True:
                return np.sum((y - yd)**2.0)
            else:
                return y - yd
            

    def lineeval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        yd = p[0] * x + p[1]
        if y == None:
            return yd
        else:
            if sumsq is True:
                return np.sum((y - yd) ** 2.0)
            else:
                return y - yd

    def poly2eval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        yd = p[0] * x**2.0 + p[1] * x + p[2]
        if y == None:
            return yd
        else:
            if sumsq is True:
                return np.sum((y - yd) ** 2.0)
            else:
                return y - yd

    def poly3eval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        yd = p[0] * x**3.0 + p[1] * x**2.0 + p[2] * x + p[3]
        if y == None:
            return yd
        else:
            if sumsq is True:
                return np.sum((y - yd) ** 2.0)
            else:
                return y - yd

    def poly4eval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        yd = p[0] * x**4.0 + p[1] * x**3.0 + p[2] * x**2.0 + p[3] * x + p[4]
        if y == None:
            return yd
        else:
            if sumsq is True:
                return np.sum((y - yd) ** 2.0)
            else:
                return y - yd

    def sineeval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        yd = p[0] + p[1] * np.sin((x * 2.0 * np.pi / p[2]) + p[3])
        if y == None:
            return yd
        else:
            if sumsq is True:
                return np.sum((y - yd) ** 2.0)
            else:
                return y - yd

    def taucurve(self, p, x, y=None, C=None, sumsq=True, weights=None):
        """
        HH-like description of activation/inactivation function
        'DC', 'a1', 'v1', 'k1', 'a2', 'v2', 'k2'
        """
        yd = p[0] + 1.0 / (
            p[1] * np.exp((x + p[2]) / p[3]) + p[4] * np.exp(-(x + p[5]) / p[6])
        )
        if y == None:
            return yd
        else:
            if sumsq is True:
                return np.sqrt(np.sum((y - yd) ** 2.0))
            else:
                return y - yd

    def taucurveder(self, p, x):
        """
        Derivative for taucurve
        'DC', 'a1', 'v1', 'k1', 'a2', 'v2', 'k2'
        """
        y = (
            -(
                p[1] * np.exp((p[2] + x) / p[3]) / p[3]
                - p[4] * np.exp(-(p[5] + x) / p[6]) / p[6]
            )
            / (p[1] * np.exp((p[2] + x) / p[3]) + p[4] * np.exp(-(p[5] + x) / p[6]))
            ** 2.0
        )
        #  print 'dy: ', y
        return y

    def FIGrowth1(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Frequency versus current intensity (FI plot) fit
        Linear fit from 0 to breakpoint
        (1-exponential) saturating growth thereafter
        weights is a function! :::
        Parameter p is a list containing: [Fzero, Ibreak, F1amp, F2amp, Irate]
        for I < break:
            F = Fzero + I*F1amp
        for I >= break:
            F = F(break)+ F2amp(1-exp^(-t * Irate))
        """
        Fzero, Ibreak, F1amp, F2amp, Irate = p
        if Ibreak == 0.0:
            Ibreak = np.mean(np.diff(x))
        yd = np.zeros(x.shape)
        m1 = np.where((x < Ibreak) & (x > 0.))[0]
        m2 = x >= Ibreak
        yd[m1] = Fzero + x[m1] * F1amp / Ibreak
        maxyd = np.max(yd)
        # print("ibreak, Irate, maxyd: ", Ibreak, Irate, maxyd, x[m2]*1e9)
        yd[m2] = F2amp * (1.0 - np.exp(-(x[m2] - Ibreak) * Irate)) + maxyd
        if y is None:
            return yd
        else:
            dy = y - yd
            if weights is not None:
                w = weights(dy) / weights(np.max(dy))
            else:
                w = np.ones(len(x))
            #            print('weights: ', w)
            #            xp = np.argwhere(x>0)
            #            w[xp] = w[xp] + 3.*x[xp]/np.max(x)
            if sumsq is True:
                ss = np.sqrt(np.sum((w * dy) ** 2.0))
                return ss
            else:
                return w * dy

    def FIGrowth2(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Frequency versus current intensity (FI plot) fit
        Firing rate of 0 until breakpoint
        exponential growth thereafter

        Parameter p is a list containing: [Ibreak, F2amp, Irate]
        for I < break:
            F = 0 (Fzero and F1amp are 0)
        for I >= break:
            F = F(Ibreak)+ F2amp(1-exp^(-t * Irate))
        """
        Ibreak, F2amp, Irate = p
        yd = np.zeros(x.shape)
        m1 = x < Ibreak
        m2 = x >= Ibreak
        yd[m1] = 0.0  # Fzero + x[m1] * F1amp / Ibreak
        maxyd = np.max(yd)
        yd[m2] = F2amp * (1.0 - np.exp(-(x[m2] - Ibreak) * Irate)) + maxyd
        if y is None:
            return yd
        else:
            dy = y - yd
            w = np.ones(len(x))
            #            xp = np.argwhere(x>0)
            #            w[xp] = w[xp] + 3.*x[xp]/np.max(x)
            if sumsq is True:
                ss = np.sqrt(np.sum((w * dy) ** 2.0))
                return ss
            else:
                return w * dy

    def FIPower(self, p, x, y=None, C=None, sumsq=False, weights=None):
        # fit a sublinear power function to FI curve (just spiking part)
        # y = c + s*x^d
        c, s, d = p  # unpack
        m = x < c / s
        n = x >= c / s
        yd = np.zeros(x.shape[0])
        b = s * x[n] - c
        print("b,d: ", c, s, d, np.max(b), d)
        if all(b >= 0.1):
            yd[n] = np.power(b, d)

        if y is None:
            return yd
        else:
            dy = y - yd
            w = np.ones(len(x))
            #            xp = np.argwhere(x>0)
            #            w[xp] = w[xp] + 3.*x[xp]/np.max(x)
            if sumsq is True:
                ss = np.sqrt(np.sum((w * dy) ** 2.0))
                return ss
            else:
                return w * dy

    def pwl2(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        piecwise linear 2 segment fit (tricky)
        """
        # x0,y0 : first breakpoint
        # k1,k2 : 2 slopes - above and below breakpoint.
        # unpack p
        x0, y0, k1, k2 = p
        yd = (x < x0) * (y0 + k1 * (x - x0)) + (x >= x0) * (y0 + k2 * (x - x0))

        if y is None:
            return yd
        else:
            dy = y - yd
            w = np.ones(len(x))
            #            xp = np.argwhere(x>0)
            #            w[xp] = w[xp] + 3.*x[xp]/np.max(x)
            if sumsq is True:
                ss = np.sqrt(np.sum((w * dy) ** 2.0))
                return ss
            else:
                return w * dy

    def piecewiselinear(self, p, x, y=None, C=None, sumsq=False, weights=None):
        x0, x1, b, k1, k2, k3 = p
        condlist = [x < x0, (x >= x0) & (x < x1), x >= x1]
        funclist = [
            lambda x: k1 * x + b,
            lambda x: k1 * x + b + k2 * (x - x0),
            lambda x: k1 * x + b + k2 * (x - x0) + k3 * (x - x1),
        ]
        yd = np.piecewise(x, condlist, funclist)
        if y is None:
            return yd
        else:
            dy = y - yd
            w = np.ones(len(x))
            if sumsq is True:
                ss = np.sqrt(np.sum((w * dy) ** 2.0))
                return ss
            else:
                return w * dy

    def pwl3(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        piecwise linear 3 segment fit (tricky)
        """
        # x0,y0 : first breakpoint
        # x1 : second breakpoint
        # k1,k2,k3 : 3 slopes.
        # unpack p
        x0, y0, x1, k1, k2, k3 = p
        y1 = y0 + k2 * (x1 - x0)  # for continuity
        yd = (
            (x < x0) * (y0 + k1 * (x - x0))
            + ((x >= x0) & (x < x1)) * (y0 + k2 * (x - x0))
            + (x >= x1) * (y1 + k3 * (x - x1))
        )

        if y is None:
            return yd
        else:
            dy = y - yd
            w = np.ones(len(x))
            #            xp = np.argwhere(x>0)
            #            w[xp] = w[xp] + 3.*x[xp]/np.max(x)
            if sumsq is True:
                ss = np.sqrt(np.sum((w * dy) ** 2.0))
                return ss
            else:
                return w * dy

    def getClipData(self, x, y, t0, t1):
        """
        Return the values in y that match the x range in tx from
        t0 to t1. x must be monotonic increasing or decreasing.
        Allow for reverse ordering."""
        it0 = np.argmin(np.fabs(x - t0))
        it1 = np.argmin(np.fabs(x - t1))
        if it0 > it1:
            t = it1
            it1 = it0
            it0 = t
        return x[it0:it1], y[it0:it1]

    def FitRegion(
        self,
        whichdata:list,
        thisaxis:int,
        tdat:np.ndarray,
        ydat:np.ndarray,
        t0:Union[float, None]=None,
        t1:Union[float, None]=None,
        fitFunc:str="exp1",
        fitFuncDer=None,
        fitPars=None,
        fixedPars=None,
        tgap=0.0, # gap before fitting in seconds
        fitPlot=None,
        plotInstance=None,
        dataType="xy",
        method=None,
        bounds=None,
        weights=None,
        constraints=(),
        capture_error:bool=False,
    ):
        """
        **Arguments**
        ============= ===================================================
        whichdata
        thisaxis
        tdat
        ydat
        t0            (optional) Minimum of time data - determined from tdat if left unspecified
        t1            (optional) Maximum of time data - determined from tdat if left unspecified
        fitFunc       (optional) The function to fit the data to (as defined in __init__). Default is 'exp1'.
        fitFuncDer    (optional) default=None
        fitPars       (optional) Initial fit parameters. Use the values defined in self.fitfuncmap if unspecified.
        fixedPars     (optional) Fixed parameters to pass to the function. Default=None
        fitPlot       (optional) default=None (Not implemented)
        plotInstance  (optional) default=None  pyqtgraph axis instance (not implemented)
        dataType      (optional) Options are ['xy', 'blocks']. Default='xy'
        method        (optional) Options are ['curve_fit', 'fmin', 'simplex', 'Nelder-Mead', 'bfgs',
                                              'TNC', 'SLSQP', 'COBYLA', 'L-BFGS-B']. Default='leastsq'
        bounds        (optional) default=None
        weights       (optional) default=None
        constraints   (optional) default=()
        ============= ===================================================

        To call with tdat and ydat as simple arrays:
        FitRegion(1, 0, tdat, ydat, FitFunc = 'exp1')
        e.g., the first argument should be 1, but this axis is ignored if datatype is 'xy'
        """
        self.fitSum2Err = []
        # if t0 == t1:
        #     if plotInstance is not None
        #         (x, y) = plotInstance.get_xlim()
        #         t0 = x[0]
        #         t1 = x[1]
        if t1 is None:
            t1 = np.max(tdat)
        if t0 is None:
            t0 = np.min(tdat)
        func = self.fitfuncmap[fitFunc]
        if func is None:
            raise ValueError("FitRegion: unknown function %s" % (fitFunc))

        # sanitize
        if isinstance(tdat, list):
            tdat = np.array(tdat)
        if isinstance(ydat, list):
            ydat = np.array(ydat)


        xp = []
        xf = []
        yf = []
        yn = []
        tx = []
        names = func[6]
        if fitPars is None:
            fpars = func[1]
        else:
            fpars = fitPars
        if (
            method == "simplex"
        ):  # remap calls if needed for newer versions of scipy (>= 0.11)
            method = "Nelder-Mead"
        if (
            ydat.ndim == 1 or dataType == "xy" or dataType == "2d"
        ):  # check if 1-d, then "pretend" its only a 1-element block
            nblock = 1
        else:
            nblock = ydat.shape[
                0
            ]  # otherwise, this is the number of traces in the block
        # print 'datatype: ', dataType
        # print 'nblock: ', nblock
        # print 'whichdata: ', whichdata
        for block in range(nblock):
            for record in whichdata:
                if dataType == "blocks":
                    tx, dy = self.getClipData(
                        tdat[block], ydat[block][record, thisaxis, :], t0, t1
                    )
                elif ydat.ndim == 1:
                    tx, dy = self.getClipData(tdat, ydat, t0, t1)
                else:
                    tx, dy = self.getClipData(tdat, ydat[record, :], t0, t1)

                # print 'Fitting.py: block, type, Fit data: ', block, dataType
                # print tx.shape
                # print dy.shape
                tx = np.array(tx)
                tx = tx - t0
                dy = np.array(dy)
                if tgap > 0:
                    dt = np.mean(np.diff(tx))
                    igap = int(tgap/dt)
                    tx = tx[igap:]
                    dy = dy[igap:]
                # print(t0, t1)

                yn.append(names)
                if not any(tx):
                    print("Fitting.py: No data in clipping window")
                    print("    Fitting Window: t0, t1: ", t0, t1)
                    print("    Min and max time in time array: ", np.min(tdat), np.max(tdat))
                    print("    Data lengths: ", len(tdat), len(ydat[record, :]))
                    raise ValueError("Fitting.py: No data in clipping window")
                    continue  # no data in the window...
                ier = 0
                #
                # Different optimization methods are included here. Not all have been tested fully with
                # this wrapper.
                #
                if (
                    method is None or method == "leastsq"
                ):  # use standard leastsq, no bounds
                    plsq, cov, infodict, mesg, ier = scipy.optimize.leastsq(
                        func[0],
                        fpars,
                        args=(tx, dy, fixedPars),
                        full_output=1,
                        maxfev=func[2],
                    )
                    if ier > 4:
                        print("optimize.leastsq error flag is: %d" % (ier))
                        print(mesg)
                elif method == "curve_fit":
                    plsq, cov = scipy.optimize.curve_fit(func[0], tx, dy, p0=fpars)
                    ier = 0
                elif method in [
                    "fmin",
                    "simplex",
                    "Nelder-Mead",
                    "bfgs",
                    "TNC",
                    "SLSQP",
                    "COBYLA",
                    "L-BFGS-B",
                ]:  # use standard wrapper from scipy for those routintes
                    if constraints is None:
                        constraints = ()
                    res = scipy.optimize.minimize(
                        func[0],
                        fpars,
                        args=(tx, dy, fixedPars, True, weights),
                        method=method,
                        jac=None,
                        hess=None,
                        hessp=None,
                        bounds=bounds,
                        constraints=constraints,
                        tol=None,
                        callback=None,
                        options={"maxiter": func[2], "disp": False},
                    )

                    plsq = res.x
                    # print "    method:", method
                    # print "    bounds:", bounds
                    # print( "    result:", plsq)

                # next section is replaced by the code above - kept here for reference if needed...
                # elif method == 'fmin' or method == 'simplex':
                #     plsq = scipy.optimize.fmin(func[0], fpars, args=(tx.astype('float64'), dy.astype('float64'), fixedPars, True),
                #                 maxfun = func[2]) # , iprint=0)
                #     ier = 0
                # elif method == 'bfgs':
                #     plsq, cov, infodict = scipy.optimize.fmin_l_bfgs_b(func[0], fpars, fprime=func[8],
                #                 args=(tx.astype('float64'), dy.astype('float64'), fixedPars, True, weights),
                #                 maxfun = func[2], bounds = bounds,
                #                 approx_grad = True) # , disp=0, iprint=-1)

                else:
                    raise ValueError(
                        "Fitting Method %s not recognized, please check Fitting.py"
                        % (method)
                    )
                xfit = np.linspace(t0, t1, 100)
                yfit = func[0](plsq, xfit - t0, C=fixedPars)
                yy = func[0](plsq, tx, C=fixedPars)  # calculate function
                self.fitSum2Err.append(np.sum((dy - yy) ** 2)) # get error for this fit
                #                print('fit error: ', self.fitSum2Err)
                # if plotInstance is not None:
                #     self.FitPlot(xFit=xfit, yFit=yfit, fitFunc=fund[0],
                #             fitPars=plsq, plotInstance=plotInstance)
                xp.append(plsq)  # parameter list
                xf.append(xfit)  # x plot point list
                yf.append(yfit)  # y fit point list
                self.tx = tx
                self.dy = dy

        return (xp, xf, yf, yn)  # includes names with yn and range of tx

    def FitPlot(
        self,
        xFit=None,
        yFit=None,
        fitFunc="exp1",
        fitPars=None,
        fixedPars=None,
        plotInstance=None,
        color=None,
    ):
        """Plot the fit data onto the fitPlot with the specified "plot Instance".
        if there is no xFit, or some parameters are missing, we just return.
        if there is xFit, but no yFit, then we try to compute the fit with
        what we have. The plot is superimposed on the specified "fitPlot" and
        the color is specified by the function color in the fitPars list.
        """
        return  # not implemented
        if xFit is None or fitPars is None:
            return
        func = self.fitfuncmap[fitFunc]
        if color is None:
            fcolor = func[3]
        else:
            fcolor = color
        if yFit is None:
            yFit = np.zeros((len(fitPars), xFit.shape[1]))
            for k in range(0, len(fitPars)):
                yFit[k] = func[0](fitPars[k], xFit[k], C=fixedPars)
        if fitPlot is None:
            return yFit
        for k in range(0, len(fitPars)):
            if plotInstance is None:
                plotInstancet.plot(xFit[k], yFit[k], color=fcolor)
            else:
                plotInstance.PlotLine(fitPlot, xFit[k], yFit[k], color=fcolor)
        return yFit

    def getFitErr(self):
        """Return the fit error for the most recent fit"""
        return self.fitSum2Err

    def expfit(self, x, y):
        """find best fit of a single exponential function to x and y
        using the chebyshev polynomial approximation.
        returns (DC, A, tau) for fit.

        Perform a single exponential fit to data using Chebyshev polynomial method.
        Equation fit: y = a1 * exp(-x/tau) + a0
        Call: [a0 a1 tau] = expfit(x,y);
        Calling parameter x is the time base, y is the data to be fit.
        Returned values: a0 is the offset, a1 is the amplitude, tau is the time
        constant (scaled in units of x).
        Relies on routines chebftd to generate polynomial coeffs, and chebint to compute the
        coefficients for the integral of the data. These are now included in this
        .py file source.
        This version is based on the one in the pClamp manual: HOWEVER, since
        I use the bounded [-1 1] form for the Chebyshev polynomials, the coefficients are different,
        and the resulting equation for tau is different. I manually optimized the tau
        estimate based on fits to some simulated noisy data. (Its ok to use the whole range of d1 and d0
        when the data is clean, but only the first few coeffs really hold the info when
        the data is noisy.)
        NOTE: The user is responsible for making sure that the passed data is appropriate,
        e.g., no large noise or electronic transients, and that the time constants in the
        data are adequately sampled.
        To do a double exp fit with this method is possible, but more complex.
        It would be computationally simpler to try breaking the data into two regions where
        the fast and slow components are dominant, and fit each separately; then use that to
        seed a non-linear fit (e.g., L-M) algorithm.
        Final working version 4/13/99 Paul B. Manis
        converted to Python 7/9/2009 Paul B. Manis. Seems functional.
        """
        n = 30
        # default number of polynomials coeffs to use in fit
        a = np.amin(x)
        b = np.amax(x)
        d0 = self.chebftd(a, b, n, x, y)  # coeffs for data trace...
        d1 = self.chebint(a, b, d0, n)  # coeffs of integral...
        tau = -np.mean(d1[2:3] / d0[2:3])
        try:
            g = np.exp(-x / tau)
        except:
            g = 0.0
        dg = self.chebftd(
            a, b, n, x, g
        )  # generate chebyshev polynomial for unit exponential function
        # now estimate the amplitude from the ratios of the coeffs.
        a1 = self.estimate(d0, dg, 1)
        a0 = (d0[0] - a1 * dg[0]) / 2.0  # get the offset here
        return (a0, a1, tau)  #

    def estimate(self, c, d, m):
        """compute optimal estimate of parameter from arrays of data"""
        n = len(c)
        a = sum(c[m:n] * d[m:n]) / sum(d[m:n] ** 2.0)
        return a

    # note : the following routine is a bottleneck. It should be coded in C.

    def chebftd(self, a, b, n, t, d):
        """Chebyshev fit; from Press et al, p 192.
        matlab code P. Manis 21 Mar 1999
        "Given a function func, lower and upper limits of the interval [a,b], and
        a maximum degree, n, this routine computes the n coefficients c[1..n] such that
        func(x) sum(k=1, n) of ck*Tk(y) - c0/2, where y = (x -0.5*(b+a))/(0.5*(b-a))
        This routine is to be used with moderately large n (30-50) the array of c's is
        subsequently truncated at the smaller value m such that cm and subsequent
        terms are negligible."
        This routine is modified so that we find close points in x (data array) - i.e., we find
        the best Chebyshev terms to describe the data as if it is an arbitrary function.
        t is the x data, d is the y data...
        """
        bma = 0.5 * (b - a)
        bpa = 0.5 * (b + a)
        inc = t[1] - t[0]
        f = np.zeros(n)
        for k in range(0, n):
            y = np.cos(np.pi * (k + 0.5) / n)
            pos = int(0.5 + (y * bma + bpa) / inc)
            if pos < 0:
                pos = 0
            if pos >= len(d) - 2:
                pos = len(d) - 2
            try:
                f[k] = d[pos + 1]
            except:
                print(
                    "error in chebftd: k = %d (len f = %d)  pos = %d, len(d) = %d\n"
                    % (k, len(f), pos, len(d))
                )
                print("you should probably make sure this doesn't happen")
        fac = 2.0 / n
        c = np.zeros(n)
        for j in range(0, n):
            sum = 0.0
            for k in range(0, n):
                sum = sum + f[k] * np.cos(np.pi * j * (k + 0.5) / n)
            c[j] = fac * sum
        return c

    def chebint(self, a, b, c, n):
        """Given a, b, and c[1..n] as output from chebft or chebftd, and given n,
        the desired degree of approximation (length of c to be used),
        this routine computes cint, the Chebyshev coefficients of the
        integral of the function whose coeffs are in c. The constant of
        integration is set so that the integral vanishes at a.
        Coded from Press et al, 3/21/99 P. Manis (Matlab)
        Python translation 7/8/2009 P. Manis
        """
        sum = 0.0
        fac = 1.0
        con = 0.25 * (b - a)  # factor that normalizes the interval
        cint = np.zeros(n)
        for j in range(1, n - 2):
            cint[j] = con * (c[j - 1] - c[j + 1]) / j
            sum = sum + fac * cint[j]
            fac = -fac
        cint[n - 1] = con * c[n - 2] / (n - 1)
        sum = sum + fac * cint[n - 1]
        cint[0] = 2.0 * sum  # set constant of integration.
        return cint

    # routine to flatten an array/list.
    #
    def flatten(self, l, ltypes=(list, tuple)):
        i = 0
        while i < len(l):
            while isinstance(l[i], ltypes):
                if not l[i]:
                    l.pop(i)
                    if not len(l):
                        break
                else:
                    l[i : i + 1] = list(l[i])
            i += 1
        return l

    # flatten()


# run tests if we are "main"

if __name__ == "__main__":
    pass
#    import matplotlib.pyplot as pyplot
#     import timeit
#     import Fitting
#     import matplotlib as MP
#     MP.use('Qt4Agg')
#     ################## Do not modify the following code
#     # sets up matplotlib with sans-serif plotting...
#     import matplotlib.gridspec as GS
#     # import mpl_toolkits.axes_grid1.inset_locator as INSETS
#     # #import inset_axes, zoomed_inset_axes
#     # import mpl_toolkits.axes_grid1.anchored_artists as ANCHOR
#     # # import AnchoredSizeBar
#
#     stdFont = 'Arial'
#
#     import  matplotlib.pyplot as pylab
#     pylab.rcParams['text.usetex'] = True
#     pylab.rcParams['interactive'] = False
#     pylab.rcParams['font.family'] = 'sans-serif'
#     pylab.rcParams['font.sans-serif'] = 'Arial'
#     pylab.rcParams['mathtext.default'] = 'sf'
#     pylab.rcParams['figure.facecolor'] = 'white'
#     # next setting allows pdf font to be readable in Adobe Illustrator
#     pylab.rcParams['pdf.fonttype'] = 42
#     ##################### to here (matplotlib stuff - touchy!
#
#     Fits = Fitting.Fitting()
# #    x = np.arange(0, 100.0, 0.1)
# #    y = 5.0-2.5*np.exp(-x/5.0)+0.5*np.random.randn(len(x))
# #    (dc, aFit,tauFit) = Fits.expfit(x,y)
# #    yf = dc + aFit*np.exp(-x/tauFit)
#  #   pyplot.figure(1)
#   #  pyplot.plot(x,y,'k')
#   #  pyplot.plot(x, yf, 'r')
#   #  pyplot.show()
#     exploreError = False
#
#     if exploreError is True:
#         # explore the error surface for a function:
#
#         func = 'exp1'
#         f = Fits.fitfuncmap[func]
#         p1range = np.arange(0.1, 5.0, 0.1)
#         p2range = np.arange(0.1, 5.0, 0.1)
#
#         err = np.zeros((len(p1range), len(p2range)))
#         x = np.array(np.arange(f[4][0], f[4][1], f[4][2]))
#         C = None
#         if func == 'expsum2':
#           C = f[7]
#
#
#         # check exchange of tau1 ([1]) and width[4]
#         C = None
#         yOffset, t0, tau1, tau2, amp, width = f[1] # get inital parameters
#         y0 = f[0](f[1], x, C=C)
#         noise = np.random.random(y0.shape) - 0.5
#         y0 += 0.0* noise
#         sh = err.shape
#         yp = np.zeros((sh[0], sh[1], len(y0)))
#         for i, p1 in enumerate(p1range):
#             tau1t = tau1*p1
#             for j, p2 in enumerate(p2range):
#                 ampt = amp*p2
#                 pars = (yOffset, t0, tau1t, tau2, ampt, width) # repackage
#                 err[i,j] = f[0](pars, x, y0, C=C, sumsq = True)
#                 yp[i,j] = f[0](pars, x, C=C, sumsq = False)
#
#         pylab.figure()
#         CS=pylab.contour(p1range*tau1, p2range*width, err, 25)
#         CB = pylab.colorbar(CS, shrink=0.8, extend='both')
#         pylab.figure()
#         for i, p1 in enumerate(p1range):
#             for j, p2 in enumerate(p2range):
#                 pylab.plot(x, yp[i,j])
#         pylab.plot(x, y0, 'r-', linewidth=2.0)
#
#
#     # run tests for each type of fit, return results to compare parameters
#
#     cons = None
#     bnds = None
#
#     signal_to_noise = 100000.
#     for func in Fits.fitfuncmap:
#         if func != 'piecewiselinear3':
#             continue
#         print ("\nFunction: %s\nTarget: " % (func),)
#         f = Fits.fitfuncmap[func]
#         for k in range(0,len(f[1])):
#             print ("%f " % (f[1][k]),)
#         print ("\nStarting:     ",)
#         for k in range(0,len(f[5])):
#             print ("%f " % (f[5][k]),)
#
# #        nstep = 500.0
# #        if func == 'sin':
# #            nstep = 100.0
#         x = np.arange(f[4][0], f[4][1], f[4][2])
#         print('f4: ', f[4])
#         print('x', x)
#         C = None
#         if func == 'expsum2':
#             C = f[7]
#
#         if func == 'exppulse':
#             C = f[7]
#         tv = f[5]
#         y = f[0](f[1], x, C=C)
#         print(x)
#         yd = np.array(y)
#         noise = np.random.normal(0, 0.1, yd.shape)
#         print(yd)
#         my = np.amax(yd)
#         #yd = yd + sigmax*0.05*my*(np.random.random_sample(shape(yd))-0.5)
#         yd += noise*my/signal_to_noise
#         testMethod = 'SLSQP'
#         if func == 'taucurve':
#             continue
#             bounds=[(0., 100.), (0., 1000.), (0.0, 500.0), (0.1, 50.0),
#                 (0., 1000), (0.0, 500.0), (0.1, 50.0)]
#             (fpar, xf, yf, names) = Fits.FitRegion(np.array([1]), 0, x, yd, fitFunc = func, bounds=bounds, method=testMethod)
#         elif func == 'boltz':
#             continue
#             bounds = [(-0.5,0.5), (0.0, 20.0), (-120., 0.), (-20., 0.)]
#             (fpar, xf, yf, names) = Fits.FitRegion(np.array([1]), 0, x, yd, fitFunc = func, bounds=bounds, method=testMethod)
#
#         elif func == 'exp2':
#             bounds=[(-0.001, 0.001), (-5.0, 0.), (1.0, 500.0), (-5.0, 0.0),
#                 (1., 10000.)]
#             (fpar, xf, yf, names) = Fits.FitRegion(np.array([1]), 0, x, yd, fitFunc = func, bounds=bounds, method=testMethod)
#
#         elif func == 'exppulse':
#             # set some constraints to the fitting
#             # yOffset, tau1, tau2, amp, width = f[1]  # order of constraings
#             dt = np.mean(np.diff(x))
#             bounds = [(-5, 5), (-15., 15.), (-2, 2.0), (2-10, 10.), (-5, 5.), (0., 5.)]
#             # cxample for constraints:
#             # cons = ({'type': 'ineq', 'fun': lambda x:   x[4] - 3.0*x[2]},
#             #         {'type': 'ineq', 'fun': lambda x:   - x[4] + 12*x[2]},
#             #         {'type': 'ineq', 'fun': lambda x:   x[2]},
#             #         {'type': 'ineq', 'fun': lambda x:  - x[4] + 2000},
#             #         )
#             cons = ({'type': 'ineq', 'fun': lambda x:   x[3] - x[2] }, # tau1 < tau2
#                 )
#             C = None
#
#             tv = f[5]
#             initialgr = f[0](f[5], x, None )
#             (fpar, xf, yf, names) = Fits.FitRegion(
#                 np.array([1]), 0, x, yd, fitFunc = func, fixedPars = C, constraints = cons, bounds = bounds, method=testMethod)
#             # print xf
#             # print yf
#             # print fpar
#             # print names
#
#         else:
#             initialgr = f[0](f[5], x, None )
#             (fpar, xf, yf, names) = Fits.FitRegion(
#                 np.array([1]), 0, x, yd, fitFunc = func, fixedPars = C, constraints = cons, bounds = bnds, method=testMethod)
#         #print fpar
#         s = np.shape(fpar)
#         j = 0
#         outstr = ""
#         initstr = ""
#         truestr = ""
#         for i in range(0, len(names[j])):
# #            print "%f " % fpar[j][i],
#             outstr = outstr + ('%s = %f, ' % (names[j][i], fpar[j][i]))
#             initstr = initstr + '%s = %f, ' % (names[j][i], tv[i])
#             truestr = truestr + '%s = %f, ' % (names[j][i], f[1][i])
#         print( "\nTrue(%d) : %s" % (j, truestr) )
#         print( "FIT(%d)   : %s" % (j, outstr) )
#         print( "init(%d) : %s" % (j, initstr) )
#         print( "Error:   : %f" % (Fits.fitSum2Err))
#         if func is 'piecewiselinear3':
#             pylab.figure()
#             pylab.plot(np.array(x), yd, 'ro-')
#             pylab.plot(np.array(x), initialgr, 'k--')
#             pylab.plot(xf[0], yf[0], 'b-') # fit
#             pylab.show()
