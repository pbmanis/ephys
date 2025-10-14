""" do fits to single or double exponentials
# using lmfit and some other approaches
# fits are done in in 3 steps:
# 1. Parameter estimation: 
#       DC (offset) and A1 are estimated from the data.
#       T1 is estimated from a linear fit of the log of the data with the minimal
#       value of the data subtracted.
# 2. These values are used to seed a simplex fit
# 3. The simplex fit is used to seed a levenberg-marquardt fit
# 4. Confidence intervals are calculated for the parameters
#
# The 2-exp fit seeds the amplitude and time constant of the second exponential
# as 1/10th the values of the first exponential (which will be the slow and dominant
# exponential). 
# The code is partly based on the example code from the lmfit documentation
# https://lmfit.github.io/lmfit-py/confidence.html

pbm 8/24/2024. How many times have I done this?
"""

import datetime
import warnings
from dataclasses import dataclass
from typing import Callable, Union

import lmfit
import numpy as np
import pyqtgraph as pg
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from .fitmodel import FitModel


@dataclass
class TSeries:
    """Recapitulate TSeries in neuroanalysis from Chase/Campagnola"""

    data: np.ndarray
    time_values: np.ndarray
    t0: float = 0.0

    def copy(self):
        return TSeries(self.data.copy(), self.time_values.copy(), self.t0)


def fit_scale_offset(data, template):
    """Return the scale and offset needed to minimize the sum of squared errors between
    *data* and *template*::

        data ≈ scale * template + offset

    Credit: Clements & Bekkers 1997
    """
    assert data.shape == template.shape
    N = len(data)
    dsum = data.sum()
    tsum = template.sum()
    scale = ((template * data).sum() - tsum * dsum / N) / ((template**2).sum() - tsum**2 / N)
    offset = (dsum - scale * tsum) / N

    return scale, offset


def exp_decay(t, yoffset, yscale, tau, xoffset=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return yoffset + yscale * np.exp(-(t - xoffset) / tau)

def exp_decay2(t, yoffset, yscale, tau, yscale1, tau1, xoffset=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return yoffset + yscale * np.exp(-(t - xoffset) / tau) + yscale1 * np.exp(-(t - xoffset) / tau1)

def estimate_exp_params_CC(data: TSeries):
    """Estimate parameters for an exponential fit to data.

    Parameters
    ----------
    data : TSeries
        Data to fit.

    Returns
    -------
    params : tuple
        (yoffset, yscale, tau, toffset)
    """
    start_y = data.data[: len(data.data) // 100].mean()
    end_y = data.data[-len(data.data) // 10 :].mean()
    yscale = start_y - end_y
    yoffset = end_y
    cs = np.cumsum(data.data - yoffset)
    if yscale > 0:
        tau_i = np.searchsorted(cs, cs[-1] * 0.63)
    else:
        tau_i = len(cs) - np.searchsorted(cs[::-1], cs[-1] * 0.63)
    tau = data.time_values[min(tau_i, len(data) - 1)] - data.time_values[0]
    return yoffset, yscale, tau, data.t0


def normalized_rmse(data: TSeries, params: dict, fn: Callable = exp_decay):
    y = fn(data.time_values, *params)
    return np.mean((y - data.data) ** 2) ** 0.5 / data.data.std()


def best_exp_fit_for_tau(tau: float, x: np.ndarray, y: np.ndarray, std: float = None):
    """Given a curve defined by x and y, find the yoffset and yscale that best fit
    an exponential decay with a fixed tau.

    Parameters
    ----------
    tau : float
        Decay time constant.
    x : array
        Time values.
    y : array
        Data values to fit.
    std : float
        Standard deviation of the data. If None, it is calculated from *y*.

    Returns
    -------
    yscale : float
        Y scaling factor for the exponential decay.
    yoffset : float
        Y offset for the exponential decay.
    err : float
        Normalized root mean squared error of the fit.
    exp_y : array
        The exponential decay curve that best fits the data.

    """
    if std is None:
        std = y.std()
    exp_y = exp_decay(x, tau=tau, yscale=1, yoffset=0)
    yscale, yoffset = fit_scale_offset(y, exp_y)
    exp_y = exp_y * yscale + yoffset
    err = ((exp_y - y) ** 2).mean() ** 0.5 / std
    return yscale, yoffset, err, exp_y


def quantify_confidence(tau: float, memory: dict, data: TSeries) -> float:
    """
    Given a run of best_exp_fit_for_tau, quantify the confidence in the fit.
    """
    # errs = np.array([v[2] for v in memory.values()])
    # std = errs.std()
    # n = len(errs)
    # data_range = errs.max() - errs.min()
    # max_std = (data_range / 2) * np.sqrt((n - 1) / n)
    # poor_variation = 1 - std / max_std

    y = data.data
    x = data.time_values
    err = memory[tau][2]
    scale, offset = np.polyfit(x, y, 1)
    linear_y = scale * x + offset
    linear_err = ((linear_y - y) ** 2).mean() ** 0.5 / y.std()
    exp_like = 1 / (1 + err / linear_err)
    exp_like = max(0, exp_like - 0.5) * 2

    # pv_factor = 1
    # el_factor = 4
    # return ((poor_variation ** pv_factor) * (exp_like ** el_factor)) ** (1 / (pv_factor + el_factor))
    return exp_like


def exp_fit(data: TSeries):
    """Fit *data* to an exponential decay.

    This is a minimization of the normalized RMS error of the fit over the decay time constant.
    Other parameters are determined exactly for each value of the decay time constant.
    """
    xoffset = data.t0
    data = data.copy()
    data.t0 = 0
    tau_init = 0.5 * (data.time_values[-1])
    memory = {}
    std = data.data.std()

    def err_fn(params):
        τ = params[0]
        # keep a record of all tau values visited and their corresponding fits
        if τ not in memory:
            memory[τ] = best_exp_fit_for_tau(τ, data.time_values, data.data, std)
        return memory[τ][2]

    result = minimize(
        err_fn,
        tau_init,
        bounds=[(1e-9, None)],
    )

    tau = float(result.x[0])
    yscale, yoffset, err, exp_y = memory[tau]
    return {
        "fit": (yoffset, yscale, tau),
        "result": result,
        "memory": memory,
        "nrmse": err,
        "confidence": quantify_confidence(tau, memory, data),
        "model": lambda t: exp_decay(t, yoffset, yscale, tau, xoffset),
    }


class Exp(FitModel):
    """Single exponential decay fitting model.

    Parameters are xoffset, yoffset, amp, and tau.
    """

    def __init__(self):
        FitModel.__init__(
            self, self.exp, independent_vars=["x"], nan_policy="omit", method="least-squares"
        )

    @staticmethod
    def exp(x, xoffset, yoffset, amp, tau):
        return exp_decay(x - xoffset, yoffset, amp, tau)

    def fit(self, *args, **kwds):
        kwds.setdefault("method", "nelder")
        return FitModel.fit(self, *args, **kwds)


class ParallelCapAndResist(FitModel):
    @staticmethod
    def current_at_t(t, v_over_parallel_r, v_over_total_r, tau, xoffset=0):
        exp = np.exp(-(t - xoffset) / tau)
        return v_over_total_r * (1 - exp) + v_over_parallel_r * exp

    def __init__(self):
        super().__init__(
            self.current_at_t, independent_vars=["t"], nan_policy="omit", method="least-squares"
        )


class Exp2(FitModel):
    """Double exponential fitting model.

    Parameters are xoffset, yoffset, amp, tau1, and tau2.

        exp2 = yoffset + amp * (exp(-(x-xoffset) / tau1) - exp(-(x-xoffset) / tau2))

    """

    def __init__(self):
        FitModel.__init__(self, self.exp2, independent_vars=["x"])

    @staticmethod
    def exp2(x, xoffset, yoffset, amp, tau1, tau2):
        xoff = x - xoffset
        out = yoffset + amp * (np.exp(-xoff / tau1) - np.exp(-xoff / tau2))
        out[xoff < 0] = yoffset
        return out


# ===========================================================================
# Estimator for exponential fit.
# ===========================================================================


class LMexpFit:
    """
    Class to fit single or double exponential decay functions to data
    """

    def __init__(
        self,
        DC: float = -50.0,
        A1: float = -2.0,
        R1: float = 20.0,
        A2: float = 0.0,
        R2: float = 0.0,
    ):
        """__init__ _summary_

        Parameters
        ----------
        DC : float, optional
            _description_, by default -50.0
        A1 : float, optional
            _description_, by default -2.0
        R1 : float, optional
            _description_, by default 20.0
        A2 : float, optional
            _description_, by default 0.0
        R2 : float, optional
            _description_, by default 0.0
        """
        self.DC = DC
        self.A1 = A1
        self.R1 = R1
        self.A2 = A2
        self.R2 = R2

    def local_filter(self, x: np.ndarray, ipos: int, window_length: int = 5, polyorder: int = 2):
        """local_filter  polynomial filter on a small section of a trace

        Parameters
        ----------
        x : np.ndarray
            _description_
        norder : int, optional
            _description_, by default 3
        window_length : int, optional
            _description_, by default 5

        Returns
        -------
        np.ndarray
            _description_
        """
        # print("x: ", x[ipos-window_length//2:ipos+window_length//2 + 1])
        # print("x[ipos]: ", x[ipos])
        filtered = savgol_filter(
            x[ipos - window_length // 2 : ipos + window_length // 2 + 1],
            window_length=window_length,
            polyorder=polyorder,
        )
        # print("filtered x[ipos]: ", filtered[polyorder-1])
        return filtered[polyorder - 1]

    def initial_estimator(self, x_data: np.ndarray, y_data: np.ndarray, verbose: bool = False):
        # three point method:
        # https://math.stackexchange.com/questions/107079/find-parameters-for-exponential-function-fitting-to-datapoints
        # choose 3 points evenly spaced points in the data (we chose 10%, 50% and 90% of the trace)
        # print(x_data.shape, y_data.shape)
        if not x_data.shape == y_data.shape:
            print(
                "exp_estimator_lmfit: initial_estimator: x_data and y_data must have the same shape"
            )
            print(f"x_data.shape: {x_data.shape}, y_data.shape: {y_data.shape}")
            raise ValueError("x_data and y_data must have the same shape")
        npts = x_data.shape[0]
        i1 = int(npts * 0.1)
        i2 = int(npts * 0.5)
        i3 = int(npts * 0.9)
        # print(i1, i2, i3)
        d = x_data[i3] - x_data[i2]
        y32 = y_data[i3] - y_data[i2]
        # self.local_filter(y_data, ipos=i3) - self.local_filter(y_data, ipos=i2)
        y21 = y_data[i2] - y_data[i1]
        # self.local_filter(y_data, ipos=i2) - self.local_filter(y_data, ipos=i1)
        # make sure the data is monotonically decreasing
        # and prevent trying to take the log of a negative number
        if y32 >= 0.0 and y21 < 0.0:
            y32 = 0.15 * y21
        # if y32 < 0 or y21 < 0:
        #     print("Data is not decreasing")
        #     import pyqtgraph as pg
        #     pg.plot(x_data, y_data)
        if y32 <= 0 and y21 > 0:
            y32 = 0.15 * y21
        # if verbose:
            # print(f"y32: {y32}, y21: {y21}, d: {d}")
        if np.abs(y21) < 1e-6:
            if y21 < 0: 
                sign = -1
            else:
                sign = 1
            y21 = sign * 1e-6
        # print(f"Revisited: y32: {y32}, y21: {y21}, d: {d}")
        # print("y32/21: ", y32 / y21)
        if y21 == 0.0 or y32 == 0.0:
            R1 = 5.
        else:
            R1 = -np.log(y32 / y21) / d

        A1a = y32 / (np.exp(x_data[i3] * R1) - np.exp(x_data[i2] * R1))
        A1b = y21 / (np.exp(x_data[i2] * R1) - np.exp(x_data[i1] * R1))
        A1 = y32 / (np.exp(x_data[i3] * R1) * (np.exp(d * R1) - 1.0))
        DC = y_data[0] / (np.exp(-x_data[0] * R1))
        self.DC = DC
        self.A1 = A1b
        self.R1 = R1
        self.A2 = A1 / 10.0
        self.R2 = 0.1 * R1
        if verbose:
            print(
                f"Initial Estimate: \n   DC: {DC:.4f}, A1: {A1:.4f}, R1: {R1:.4f}, A1a: {A1a:.4f} A1b: {A1b:.4f}"
            )

    # define single and double exponential functions
    # # and functions for the residuals

    def exp_decay1(self, x: float, DC: float, A1: float, R1: float) -> np.ndarray:
        """exp_decay _summary_

        Parameters
        ----------
        x : float
            times
        DC : float
            DC offset
        A1 : float
            amplitude of the exponential
        R1 : float
            inverse of time constant (e.g., rate)

        Returns
        -------
        np.ndarray
            computed values
        """
        return DC + A1 * (1.0 - np.exp(-x * R1))

    def exp_decay2(
        self, x: np.ndarray, DC: float, A1: float, R1: float, A2: float, R2: float
    ) -> np.ndarray:
        """exp_decay2 Double exponential decay function

        Parameters
        ----------
         x : float
            times
        DC : float
            DC offset
        A1 : float
            amplitude of the first (dominant) exponential
        R1 : float
            inverse of time constant (e.g., rate, /ms)
        A2 : float
            amplitude of the second exponential
        R2 : float
            inverse of time constant (e.g., rate, /ms)

        Returns
        -------
        np.ndarray
            computed values
        """
        return DC + A1 * (1.0 - np.exp(-x * R1)) + A2 * (1.0 - np.exp(-x * R2))

    def residual1(self, params, x, y):
        dc = params["DC"]
        a1 = params["A1"]
        t1 = params["R1"]
        return y - self.exp_decay1(x, dc, a1, t1)

    def residual2(self, params, x, y):
        dc = params["DC"]
        a1 = params["A1"]
        t1 = params["R1"]
        a2 = params["A2"]
        t2 = params["R2"]
        return y - self.exp_decay2(x, dc, a1, t1, a2, t2)

    # single exponential fit
    def fit1(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        taum_bounds: Union[list, None] = None,
        plot=False,
        verbose=False,
    ):
        assert x_data.ndim == 1
        self.initial_estimator(x_data, y_data, verbose=verbose)  # results stored in class variables
        assert y_data.ndim == 1
        # Create a model for exponential decay
        model = lmfit.Model(self.exp_decay1)
        # Set the initial parameters
        params = model.make_params(
            DC=self.DC,
            A1=self.A1,
            R1=self.R1,
        )
        if taum_bounds is None:
            params["R1"].min = 0.0  # require positive time constant at least
        else:
            if verbose:
                print("taum_bounds: ", taum_bounds)
            params["R1"].min = 1.0 / taum_bounds[1]
            params["R1"].max = 1.0 / taum_bounds[0]

        # set up lmfit
        mini = lmfit.Minimizer(self.residual1, params, fcn_args=(x_data, y_data))
        # Fit the model to the data
        if verbose:
            print("Simplex Fit 1: ")
        result_simplex = mini.minimize(method="nelder")
        if verbose:
            lmfit.report_fit(result_simplex.params, min_correl=0.5)

        # then solve with Levenberg-Marquardt using the
        # Nelder-Mead solution as a starting point
        if verbose:
            print("Levenberg-Marquardt Fit : ")
        try:  # because it is possible that nan's will appear somewhere.
            result_lm = mini.minimize(method="leastsq", params=result_simplex.params)

            if verbose:
                lmfit.report_fit(result_lm.params, min_correl=0.5)
        except:  # just use the nelder solution
            result_lm = result_simplex
            print("Using simplex Fit : ")
            lmfit.report_fit(result_lm.params, min_correl=0.5)
        # if verbose:
        #     ci, trace = lmfit.conf_interval(mini, result_lm, sigmas=[1, 2], trace=True)
        # if verbose:
        #     lmfit.printfuncs.report_ci(ci)

        if plot:
            app = pg.mkQApp()# Plot the data and the fitted curve
            pw = pg.plot(
                x_data,
                y_data,
                symbol="o",
                brush=pg.mkBrush("y", alpha=0.5),
                symbolSize=3,
                symbolpen=None,
                pen=None,
                title="LMfit: Exponential Decay Fitting",
            )
            # pw.plot(x_data, result_lm.best_fit, pen=pg.mkPen('r', width=2))
            # pw.plot(
            #     x_data,
            #     self.residual1(result_lm.params, x_data, y_data) + y_data,
            #     pen=pg.mkPen("m", width=1),
            # )
            pw.plot(
                x_data,
                self.exp_decay1(
                    x_data,
                    DC=result_lm.params["DC"],
                    A1=result_lm.params["A1"],
                    R1=result_lm.params["R1"],
                ),
                pen=pg.mkPen("r", width=10),
            )
            app.exec()
            # result.plot_fit()

        # import matplotlib.pyplot as plt

        # x = x_data
        # y = y_data
        # residual = self.residual
        # out2 = result_lm

        # # plot confidence intervals (a1 vs t2 and a2 vs t2)
        # fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

        # for i, px in enumerate([["DC", "A1"], ["DC", "R1"]]):
        #     cx, cy, grid = lmfit.conf_interval2d(mini, out2, px[0], px[1], 30, 30)
        #     ctp = axes[i].contourf(cx, cy, grid, np.linspace(0, 1, 11))
        #     fig.colorbar(ctp, ax=axes[i])
        #     axes[i].set_xlabel(px[0])
        #     axes[i].set_ylabel(px[1])

        # # cx, cy, grid = lmfit.conf_interval2d(mini, out2, 'a2', 't2', 30, 30)
        # # ctp = axes[1].contourf(cx, cy, grid, np.linspace(0, 1, 11))
        # # fig.colorbar(ctp, ax=axes[1])
        # # axes[1].set_xlabel('a2')
        # # axes[1].set_ylabel('t2')
        # plt.show()

        # plot dependence between two parameters
        # fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
        # for i, px in enumerate([['a', 'b'], ['a', 'c']]):
        #     cx1, cy1, prob = trace[px[0]][px[0]], trace[px[0]][px[1]], trace[px[0]]['prob']
        #     # cx2, cy2, prob2 = trace['t2']['t2'], trace['t2']['a1'], trace['t2']['prob']

        #     axes[i].scatter(cx1, cy1, c=prob, s=30)
        #     axes[i].set_xlabel('a1')
        #     axes[i].set_ylabel('t2')

        #     # axes[1].scatter(cx2, cy2, c=prob2, s=30)
        #     # axes[1].set_xlabel('t2')
        #     # axes[1].set_ylabel('a1')
        # plt.show()

        return result_lm

    def fit2(self, x_data: np.ndarray, y_data: np.ndarray, plot=True, verbose=False):
        assert x_data.ndim == 1
        assert y_data.ndim == 1

        exp1fit = self.fit1(x_data, y_data, plot=False)  # first get the single exponential fit
        # Create a model for exponential decay
        model = lmfit.Model(self.exp_decay2)
        if exp1fit.params['A1'].value > 1:
            return None  # bad fit
        # Set the initial parameters based on the single exponential fit
        params = model.make_params(
            DC=exp1fit.params["DC"].value,
            A1=exp1fit.params["A1"].value,
            R1=exp1fit.params["R1"].value,
            A2=0.5 * exp1fit.params["A1"].value,
            R2=0.1 * exp1fit.params["R1"].value,
        )
        print(params)
        # set bounds on the parameters
        params["DC"].min = 0.98 * params["DC"].value
        params["DC"].max = 1.02 * params["DC"].value
        params["A1"].min = 2 ** params["A1"].value
        params["A1"].max = 0.0  # 1.05*params["A1"].value
        if params["A1"].min == params['A1'].max:
            params["A1"].min = -1
            params["A1"].max = 1

        params["R1"].min = 1e-4  # 0.9*params["R1"].value
        params["R1"].max = 1.5  # 1.1*params["R1"].value
        params["R2"].min = 1e-3
        params["R2"].max = 1.5

        mini = lmfit.Minimizer(self.residual2, params, fcn_args=(x_data, y_data))
        # Fit the model to the data
        result_simplex = mini.minimize(method="nedler")

        if verbose:
            print("Simplex Fit 2: ")
            lmfit.report_fit(result_simplex.params, min_correl=0.5)
        result_simplex.params["A2"].min = 0.9 * params["A2"].value
        result_simplex.params["A2"].max = 0.0  # 1.5*params["A2"].value
        if result_simplex.params['A2'].min == result_simplex.params['A2'].max:
            result_simplex.params["A2"].min = -1
            result_simplex.params["A2"].max = 1
        result_simplex.params["R2"].min = 1e-4  # 0.8*params["R2"].value
        result_simplex.params["R2"].max = 1.5 * params["R2"].value

        # then solve with Levenberg-Marquardt using the
        # Nelder-Mead solution as a starting point
        result_lm2 = mini.minimize(method="leastsq", params=result_simplex.params)
        print("Levenberg-Marquardt Fit 2: ")
        lmfit.report_fit(result_lm2.params, min_correl=0.5)

        # ci, trace = lmfit.conf_interval(mini, result_lm, sigmas=[1, 2], trace=True)
        # lmfit.printfuncs.report_ci(ci)

        if plot:
            app = pg.mkQApp()
            # Plot the data and the fitted curve
            pw = pg.plot(
                x_data,
                y_data,
                symbol="o",
                brush=pg.mkBrush(pg.mkColor("y"), alpha=1),
                symbolSize=3,
                symbolPen=None,
                pen=None,
                title="LMfit: Exponential Decay Fitting",
            )
            # pw.plot(x_data, result_lm.best_fit, pen=pg.mkPen('r', width=2))
            # pw.plot(
            #     x_data,
            #     self.residual2(result_lm.params, x_data, y_data) + y_data,
            #     pen=pg.mkPen("m", width=1),
            # )
            # pw.plot(
            #     x_data,
            #     self.exp_decay2(
            #         x_data,
            #         DC=result_lm2.params["DC"],
            #         A1=result_lm2.params["A1"],
            #         R1=result_lm2.params["R1"],
            #         A2=result_lm2.params["A2"],
            #         R2=result_lm2.params["R2"],
            #     ),
            #     pen=pg.mkPen("r", width=1),
            # )
            pw.plot(
                x_data,
                self.exp_decay2(
                    x_data,
                    DC=result_simplex.params["DC"],
                    A1=result_simplex.params["A1"],
                    R1=result_simplex.params["R1"],
                    A2=result_simplex.params["A2"],
                    R2=result_simplex.params["R2"],
                ),
                pen=pg.mkPen("g", width=1.5),
            )
            app.exec()
        #     # result.plot_fit()

        # import matplotlib.pyplot as plt

        # x = x_data
        # y = y_data
        # residual = self.residual
        # out2 = result_lm
        # # plot data and best fit
        # # already done in pyqtgraph above
        # # plt.figure()
        # # plt.plot(x, y)
        # # plt.plot(x, residual(out2.params, x, y) + y, '-')
        # # plt.show()

        # # plot confidence intervals (a1 vs t2 and a2 vs t2)
        # fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

        # for i, px in enumerate([["A1", "R2"], ["R1", "R2"]]):
        #     cx, cy, grid = lmfit.conf_interval2d(mini, out2, px[0], px[1], 30, 30)
        #     ctp = axes[i].contourf(cx, cy, grid, np.linspace(0, 1, 11))
        #     fig.colorbar(ctp, ax=axes[i])
        #     axes[i].set_xlabel(px[0])
        #     axes[i].set_ylabel(px[1])

        # # cx, cy, grid = lmfit.conf_interval2d(mini, out2, 'a2', 't2', 30, 30)
        # # ctp = axes[1].contourf(cx, cy, grid, np.linspace(0, 1, 11))
        # # fig.colorbar(ctp, ax=axes[1])
        # # axes[1].set_xlabel('a2')
        # # axes[1].set_ylabel('t2')
        # plt.show()

        # # plot dependence between two parameters
        # # fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
        # # for i, px in enumerate([['a', 'b'], ['a', 'c']]):
        # #     cx1, cy1, prob = trace[px[0]][px[0]], trace[px[0]][px[1]], trace[px[0]]['prob']
        # #     # cx2, cy2, prob2 = trace['t2']['t2'], trace['t2']['a1'], trace['t2']['prob']

        # #     axes[i].scatter(cx1, cy1, c=prob, s=30)
        # #     axes[i].set_xlabel('a1')
        # #     axes[i].set_ylabel('t2')

        # #     # axes[1].scatter(cx2, cy2, c=prob2, s=30)
        # #     # axes[1].set_xlabel('t2')
        # #     # axes[1].set_ylabel('a1')
        # # plt.show()

        return result_lm2

    def fit_report(self, params):
        lmfit.fit_report(params)

def draw_arrow(ax, realdata, fitdata, limit:float=1.0):
    arrow_pos = []
    maxy = limit*np.max(realdata)
    for i, e in enumerate(fitdata): 
        if e > maxy:
            arrow_pos.append(i)
    if len(arrow_pos) > 0:
        for i in arrow_pos:
            ax.arrow(realdata[i], 0.95*maxy, 0.0, 0.05, head_width=0.03*maxy, head_length=0.03*maxy, fc='r', ec='k')

def test_fitting(app):


    import matplotlib.pyplot as mpl
    # test
    app = pg.mkQApp()
    sr = 1e-5
    dur = 0.05

    n = int(dur / sr)
    x_data = np.linspace(0, 0.05, n)  # times in seconds
    dc = -0.065
    timing = {"LME": 0.0, "CCexp": 0.0}
    res = {"LME": [], "CCexp": []}
    yd = []
    lme_y = []
    lme_cc = []
    nx = 8
    amps = np.linspace(2, 10, nx)
    rates = np.linspace(1, 1000, nx)
    amparr = []
    ratearr = []
    for a in amps:
        for r in rates:
            amp = a
            rate = r
            # print("amp: ", amp, "rate: ", rate)
            amparr.append(amp)
            ratearr.append(rate)
            y_data = dc - amp * (1.0 - np.exp(-x_data * rate)) + 0.02 * amp * np.random.randn(n)
            yd.append(y_data)
            tstart = datetime.datetime.now()
            LME = LMexpFit()
            LME.initial_estimator(x_data, y_data, verbose=False)
            result = LME.fit1(x_data, y_data, plot=False, verbose=False)
            tend = datetime.datetime.now()
            timing["LME"] += (tend - tstart).total_seconds()
            res["LME"].append(
                [result.params["DC"].value, result.params["A1"].value, result.params["R1"].value]
            )
            # print(res["LME"][-1])
            lme_y.append(
                exp_decay(
                    x_data,
                    res["LME"][-1][0] + res["LME"][-1][1],
                    -res["LME"][-1][1],
                    1.0 / res["LME"][-1][2],
                )
            )
            # print(result.params)
            tstart = datetime.datetime.now()
            # DC, A1, R1 = estimate_exp_params_CC(TSeries(y_data, x_data))
            rescc = exp_fit(TSeries(y_data, x_data))
            res["CCexp"].append(rescc["fit"])
            tend = datetime.datetime.now()
            timing["CCexp"] += (tend - tstart).total_seconds()
            lme_cc.append(exp_decay(x_data, rescc["fit"][0], rescc["fit"][1], rescc["fit"][2]))
        # 'fit': (yoffset, yscale, tau),
        # 'result': result,
        # 'memory': memory,
        # 'nrmse': err,
        # 'confidence': quantify_confidence(tau, memory, data),
        # 'model': lambda t: exp_decay(t, yoffset, yscale, tau, xoffset),
    fits_lme = np.array(res["LME"])
    fits_cc = np.array(res["CCexp"])
    f, ax = mpl.subplots(4, 2, figsize=(8, 10))
    # top row: traces and fits
    for i, y in enumerate(yd):
        ax[0, 0].plot(x_data, y, linewidth=0.5, color="k", linestyle="-")
        ax[0, 0].plot(x_data, lme_y[i], linestyle="--")
    ax[0, 0].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("Current (pA)")
    ax[0, 0].set_title("LME")
    
    for i, y in enumerate(yd):
        ax[0, 1].plot(x_data, y, linewidth=0.5, color="k", linestyle="-")
        ax[0, 1].plot(x_data, lme_cc[i], linestyle="--")
    ax[0, 1].set_xlabel("Time (s)")
    ax[0, 1].set_ylabel("Current (pA)")
    ax[0, 1].set_title("CCexp")

    for i, y in enumerate(yd):
        if -fits_lme[i, 1]/amparr[i] > 1.01 or -fits_lme[i, 1]/amparr[i] < 0.99:
            ax[1, 0].plot(x_data, y, linewidth=0.5, color="k", linestyle="-")
            ax[1, 0].plot(x_data, lme_y[i], linestyle="--")
    ax[1, 0].set_xlabel("Time (s)")
    ax[1, 0].set_ylabel("Current (pA)")
    ax[1, 0].set_title("LME")
    
    for i, y in enumerate(yd):
        if fits_cc[i, 1]/amparr[i] > 1.01 or fits_cc[i, 1]/amparr[i] < 0.99:
            # print("Amplitude error: ", fits_lme[i, 1], amparr[i])
            ax[1, 1].plot(x_data, y, linewidth=0.5, color="k", linestyle="-")
            ax[1, 1].plot(x_data, lme_cc[i], linestyle="--")
    ax[1, 1].set_xlabel("Time (s)")
    ax[1, 1].set_ylabel("Current (pA)")
    ax[1, 1].set_title("CCexp")


    # middle row: time constants vs fit time constants
    
    ax[2, 0].plot(  # line of equality
        [np.min(rates), np.max(rates)],
        [np.min(rates), np.max(rates)],
        color="r",
        linestyle="--",
        linewidth=0.5,
    )
    draw_arrow(ax[2, 0], ratearr, fits_lme[:, 2])
    ax[2, 0].scatter(ratearr, fits_lme[:, 2], color='b', marker='x', s=3)
    ax[2, 0].set_xlim(0, np.max(ratearr))
    ax[2, 0].set_ylim(0, np.max(ratearr))
    ax[2, 0].set_xlabel("Actual Time Constant")
    ax[1, 0].set_ylabel("LME Time Constant")

    ax[2, 1].plot(  # line of equality
        [np.min(rates), np.max(rates)],
        [np.min(rates), np.max(rates)],
        color="r",
        linestyle="--",
        linewidth=0.5,
    )

    ax[2, 1].scatter(ratearr, 1./fits_cc[:, 2], color='b', marker='x', s=3)    
    draw_arrow(ax[2, 1], ratearr, 1./fits_cc[:, 2])
    ax[2, 1].set_xlim(0, np.max(ratearr))
    ax[2, 1].set_xlabel("Actual Time Constant")
    ax[2, 1].set_ylabel("CC Time Constant")
    

    # bottom row: 
    # amplitud vs fit amplitude

    ax[3, 0].scatter(amparr, -fits_lme[:, 1], color="y", marker="o", s=5)
    ax[3, 0].plot(
    [np.min(amparr), np.max(amparr)],
    [np.min(amparr), np.max(amparr)],
    color="r",
    linestyle="--",
    linewidth=0.5,
    )
    draw_arrow(ax[3, 0], amparr, -fits_lme[:, 1])
    ax[3, 0].set_ylim(0, np.max(amparr))
    ax[3, 0].set_xlim(0, np.max(amparr))
    ax[3, 0].set_xlabel("Actual Amplitude")
    ax[3, 0].set_ylabel("LME Amplitude")
    
    ax[3, 1].plot(
        [np.min(amparr), np.max(amparr)],
        [np.min(amparr), np.max(amparr)],
        color="r",
        linestyle="--",
        linewidth=0.5,
    )
    ax[3, 1].scatter(np.array(amparr), fits_cc[:, 1], color="m", marker="o", s=3)
    ax[3, 1].set_ylim(0, np.max(amparr))
    ax[3, 1].set_xlim(0, np.max(amparr))
    ax[3, 1].set_xlabel("Actual Amplitude")
    ax[3, 1].set_ylabel("CCexp Amplitude")
    draw_arrow(ax[3, 1], amparr, fits_cc[:, 1])
    # 
    
    f.tight_layout()
    mpl.show()
    print("elapsed fitting: ", timing["LME"], timing["CCexp"])

    # print("LME results: ", np.mean(fits[:, 2], axis=0), np.std(fits[:, 2], axis=0))
    # print("CCexp results: ", np.mean(fits[:, 2], axis=0), np.std(fits[:, 2], axis=0))


if __name__ == "__main__":
    test_fitting(app)
    