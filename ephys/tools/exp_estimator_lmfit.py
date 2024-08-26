""" do fits to single or double exponentials
# using lmfit. 
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

import numpy as np
import pyqtgraph as pg
from scipy.signal import savgol_filter
import lmfit


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
        #     print(f"y32: {y32}, y21: {y21}, d: {d}")
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
    def fit1(self, x_data: np.ndarray, y_data: np.ndarray, plot=False, verbose=False):
        assert x_data.ndim == 1
        self.initial_estimator(x_data, y_data, verbose=True)  # results stored in class variables
        assert y_data.ndim == 1
        # Create a model for exponential decay
        model = lmfit.Model(self.exp_decay1)
        # Set the initial parameters
        params = model.make_params(
            DC=self.DC,
            A1=self.A1,
            R1=self.R1,
        )
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
        except: # just use the nelder solution
            result_lm = result_simplex
            print("Using simplex Fit : ")
            lmfit.report_fit(result_lm.params, min_correl = 0.5)
        # if verbose:
        #     ci, trace = lmfit.conf_interval(mini, result_lm, sigmas=[1, 2], trace=True)
        # if verbose:
        #     lmfit.printfuncs.report_ci(ci)

        if plot:
            # Plot the data and the fitted curve
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

    def fit2(self, x_data: np.ndarray, y_data: np.ndarray, plot=False, report=False):
        assert x_data.ndim == 1
        assert y_data.ndim == 1

        exp1fit = self.fit1(x_data, y_data, plot=False)  # first get the single exponential fit
        # Create a model for exponential decay
        model = lmfit.Model(self.exp_decay2)

        # Set the initial parameters based on the single exponential fit
        params = model.make_params(
            DC=exp1fit.params["DC"].value,
            A1=exp1fit.params["A1"].value,
            R1=exp1fit.params["R1"].value,
            A2=0.1 * exp1fit.params["A1"].value,
            R2=0.1 * exp1fit.params["R1"].value,
        )
        # set bounds on the parameters
        params["DC"].min = 0.98 * params["DC"].value
        params["DC"].max = 1.02 * params["DC"].value
        params["A1"].min = 2 ** params["A1"].value
        params["A1"].max = 0.0  # 1.05*params["A1"].value
        params["R1"].min = 1e-4  # 0.9*params["R1"].value
        params["R1"].max = 1.5  # 1.1*params["R1"].value
        mini = lmfit.Minimizer(self.residual2, params, fcn_args=(x_data, y_data))
        # Fit the model to the data
        result_simplex = mini.minimize(method="nedler")

        print("Simplex Fit 2: ")
        lmfit.report_fit(result_simplex.params, min_correl=0.5)
        result_simplex.params["A2"].min = 0.9 * params["A2"].value
        result_simplex.params["A2"].max = 0.0  # 1.5*params["A2"].value
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
                pen=pg.mkPen("g", width=3),
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


if __name__ == "__main__":
    # test
    app = pg.mkQApp()
    sr = 1e-5
    dur = 0.05

    n = int(dur / sr)
    x_data = np.linspace(0, 0.05, n)  # times in seconds
    dc = -0.065
    for a in np.linspace(0.5, 3.5, 3):
        for r in np.linspace(10, 100, 3):
            y_data = dc - a * (1.0 - np.exp(-x_data * r)) + a / 100 * np.random.randn(n)
            LME = LMexpFit()
            LME.initial_estimator(x_data, y_data, verbose=False)
            print(f"\nseed: DC: {dc:8.4f}, A1: {-a:8.4f}, R1: {r:8.4f}")
            print(f"estm: DC: {LME.DC:8.4f}, A1: {LME.A1:8.4f}, R1: {LME.R1:8.4f}")
            result = LME.fit1(x_data, y_data, plot=False, verbose=False)
            print(
                f"Fit:  DC: {result.params['DC'].value:8.4f}, A1: {result.params['A1'].value:8.4f}, R1: {result.params['R1'].value:8.4f}"
            )
            print(LME.fit_report(result.params))
            # # x_data = x_data.reshape(-1, 1)
    # y_data = -0.065 - 0.0025 * (1.0 - np.exp(-x_data * 30.0)) + 0.0001 * np.random.randn(n)
    # y_data2 = y_data + 0.0002 * (1.0 - np.exp(-x_data * (1.0 / 2e-4)))

    # LME = LMexpFit()
    # result = LME.fit1(x_data, y_data, plot=True)
    # print("LMfit values: ")
    # # print(result.best_values)
    # print(LME.fit_report(result.params))

    # result2 = LME.fit2(x_data, y_data2, plot=True)
    # print("LMfit values: ")
    # # print(result.best_values)
    # print(LME.fit_report(result2.params))
