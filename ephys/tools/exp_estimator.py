# from:
# https://stackoverflow.com/questions/3938042/fitting-exponential-decay-with-no-initial-guessing
# seems like a heavy lift for a simple problem
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pyqtgraph as pg
import lmfit


def contour_plot(x, y, z, title="Contour Plot", xlabel="X", ylabel="Y", zlabel="Z", img=None):
    curves = []
    levels = np.linspace(data.min(), data.max(), 10)
    for i in range(len(levels)):
        v = levels[i]
        ## generate isocurve with automatic color selection
        c = pg.IsocurveItem(level=v, pen=(i, len(levels) * 1.5))
        if img is not None:
            c.setParentItem(img)  ## make sure isocurve is always correctly displayed over image
        c.setZValue(10)
        curves.append(c)
    return curves


# Define the model
class ExpDecayModel(nn.Module):
    def __init__(
        self,
        a: float = -50.0,
        b: float = -2.0,
        c: float = 20.0,
    ):
        super(ExpDecayModel, self).__init__()
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))
        self.c = nn.Parameter(torch.tensor(c))

    def forward(self, x):
        return self.a + self.b * (1.0 - torch.exp(-x * self.c))


class LMexpFit:
    def __init__(
        self,
        DC: float = -50.0,
        A1: float = -2.0,
        tau1: float = 20.0,
        A2: float = 0.0,
        tau2: float = 0.0,
    ):
        self.DC = DC
        self.A1 = A1
        self.tau1 = tau1
        self.A2 = A2
        self.tau2 = tau2

    def exp_decay(self, x, DC, A1, tau1):
        return DC + A1 * (1.0 - np.exp(-x * tau1))

    def exp_decay2(self, x, DC, A1, tau1, A2, tau2):
        return DC + A1 * (1.0 - np.exp(-x * tau1)) + A2 * (1.0 - np.exp(-x * tau2))

    def residual(self, params, x, y):
        dc = params["DC"]
        a1 = params["A1"]
        t1 = params["tau1"]
        return y - self.exp_decay(x, dc, a1, t1)

    def residual2(self, params, x, y):
        dc = params["DC"]
        a1 = params["A1"]
        t1 = params["tau1"]
        a2 = params["A2"]
        t2 = params["tau2"]
        return y - self.exp_decay2(x, dc, a1, t1, a2, t2)

    def fit1(self, x_data: np.ndarray, y_data: np.ndarray, plot=False):
        # Create a model for exponential decay
        model = lmfit.Model(self.exp_decay)

        # Set the initial parameters
        params = model.make_params(DC=self.DC, A1=self.A1, tau1=self.tau1)

        mini = lmfit.Minimizer(self.residual, params, fcn_args=(x_data.T[0], y_data.T[0]))
        # Fit the model to the data
        result_simplex = mini.minimize(method="nedler")

        print("Simplex Fit: ")
        lmfit.report_fit(result_simplex.params, min_correl=0.5)
        # then solve with Levenberg-Marquardt using the
        # Nelder-Mead solution as a starting point
        result_lm = mini.minimize(method="leastsq", params=result_simplex.params)
        print("Levenberg-Marquardt Fit: ")
        lmfit.report_fit(result_lm.params, min_correl=0.5)

        ci, trace = lmfit.conf_interval(mini, result_lm, sigmas=[1, 2], trace=True)
        lmfit.printfuncs.report_ci(ci)

        if plot:
            # Plot the data and the fitted curve
            pw = pg.plot(
                x_data.T[0],
                y_data.T[0],
                symbol="o",
                brush=pg.mkBrush("y"),
                symbolSize=5,
                pen=None,
                title="LMfit: Exponential Decay Fitting",
            )
            # pw.plot(x_data.T[0], result_lm.best_fit, pen=pg.mkPen('r', width=2))
            pw.plot(
                x_data.T[0],
                self.residual(result_lm.params, x_data.T[0], y_data.T[0]) + y_data.T[0],
                pen=pg.mkPen("r", width=2),
            )
            pw.plot(
                x_data.T[0],
                self.exp_decay(
                    x_data.T[0],
                    DC=result_lm.params["DC"],
                    A1=result_lm.params["A1"],
                    tau1=result_lm.params["tau1"],
                ),
                pen=pg.mkPen("g", width=2),
            )
            app.exec()
            # result.plot_fit()

        import matplotlib.pyplot as plt

        x = x_data.T[0]
        y = y_data.T[0]
        residual = self.residual
        out2 = result_lm

        # plot confidence intervals (a1 vs t2 and a2 vs t2)
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

        for i, px in enumerate([["DC", "A1"], ["DC", "tau1"]]):
            cx, cy, grid = lmfit.conf_interval2d(mini, out2, px[0], px[1], 30, 30)
            ctp = axes[i].contourf(cx, cy, grid, np.linspace(0, 1, 11))
            fig.colorbar(ctp, ax=axes[i])
            axes[i].set_xlabel(px[0])
            axes[i].set_ylabel(px[1])

        # cx, cy, grid = lmfit.conf_interval2d(mini, out2, 'a2', 't2', 30, 30)
        # ctp = axes[1].contourf(cx, cy, grid, np.linspace(0, 1, 11))
        # fig.colorbar(ctp, ax=axes[1])
        # axes[1].set_xlabel('a2')
        # axes[1].set_ylabel('t2')
        plt.show()

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

    def fit2(self, x_data: np.ndarray, y_data: np.ndarray, plot=False):
        # Create a model for exponential decay
        model = lmfit.Model(self.exp_decay2)

        # Set the initial parameters
        params = model.make_params(
            DC=self.DC, A1=self.A1, tau1=self.tau1, A2=self.A2, tau2=self.tau2
        )

        mini = lmfit.Minimizer(self.residual, params, fcn_args=(x_data.T[0], y_data.T[0]))
        # Fit the model to the data
        result_simplex = mini.minimize(method="nedler")

        print("Simplex Fit: ")
        lmfit.report_fit(result_simplex.params, min_correl=0.5)
        # then solve with Levenberg-Marquardt using the
        # Nelder-Mead solution as a starting point
        result_lm = mini.minimize(method="leastsq", params=result_simplex.params)
        print("Levenberg-Marquardt Fit: ")
        lmfit.report_fit(result_lm.params, min_correl=0.5)

        ci, trace = lmfit.conf_interval(mini, result_lm, sigmas=[1, 2], trace=True)
        lmfit.printfuncs.report_ci(ci)

        if plot:
            # Plot the data and the fitted curve
            pw = pg.plot(
                x_data.T[0],
                y_data.T[0],
                symbol="o",
                brush=pg.mkBrush("y"),
                symbolSize=5,
                pen=None,
                title="LMfit: Exponential Decay Fitting",
            )
            # pw.plot(x_data.T[0], result_lm.best_fit, pen=pg.mkPen('r', width=2))
            pw.plot(
                x_data.T[0],
                self.residual2(result_lm.params, x_data.T[0], y_data.T[0]) + y_data.T[0],
                pen=pg.mkPen("r", width=2),
            )
            pw.plot(
                x_data.T[0],
                self.exp_decay2(
                    x_data.T[0],
                    DC=result_lm.params["DC"],
                    A1=result_lm.params["A1"],
                    tau1=result_lm.params["tau1"],
                    A2=result_lm.params["A2"],
                    tau2=result_lm.params["tau2"],
                ),
                pen=pg.mkPen("g", width=2),
            )
            app.exec()
            # result.plot_fit()

        import matplotlib.pyplot as plt

        x = x_data.T[0]
        y = y_data.T[0]
        residual = self.residual
        out2 = result_lm
        # plot data and best fit
        # already done in pyqtgraph above
        # plt.figure()
        # plt.plot(x, y)
        # plt.plot(x, residual(out2.params, x, y) + y, '-')
        # plt.show()

        # plot confidence intervals (a1 vs t2 and a2 vs t2)
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

        for i, px in enumerate([["A1", "tau2"], ["tau1", "tau2"]]):
            cx, cy, grid = lmfit.conf_interval2d(mini, out2, px[0], px[1], 30, 30)
            ctp = axes[i].contourf(cx, cy, grid, np.linspace(0, 1, 11))
            fig.colorbar(ctp, ax=axes[i])
            axes[i].set_xlabel(px[0])
            axes[i].set_ylabel(px[1])

        # cx, cy, grid = lmfit.conf_interval2d(mini, out2, 'a2', 't2', 30, 30)
        # ctp = axes[1].contourf(cx, cy, grid, np.linspace(0, 1, 11))
        # fig.colorbar(ctp, ax=axes[1])
        # axes[1].set_xlabel('a2')
        # axes[1].set_ylabel('t2')
        plt.show()

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


class TorchFit:
    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0):
        self.a = a
        self.b = b
        self.c = c

    # Define the loss function
    def loss_fn(self, y_pred, y_true):
        return torch.max((y_pred - y_true) ** 2)

    def fit(self, x_data, y_data, plot=False, nepochs=1000, verbose=True, percent_loss_crit=0.5):
        # Convert the data to PyTorch tensors
        x_tensor = torch.from_numpy(x_data).float()
        y_tensor = torch.from_numpy(y_data).float()

        # Initialize the model and optimizer
        self.model = ExpDecayModel(a=self.a, b=self.b, c=self.c)
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        # Train the model
        losses = [np.nan]
        ploss = percent_loss_crit + 1
        for epoch in range(nepochs):
            # Forward pass
            y_pred = self.model(x_tensor)
            loss = self.loss_fn(y_pred, y_tensor)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss every 100 epochs
            if (epoch % 100 == 0) and verbose:
                ploss = 100 * (losses[-1] - loss.item()) / loss.item()
                print(
                    "Epoch {}: Loss = {}".format(epoch, loss.item()),
                    f"Percent Improvement: {ploss}",
                )
                losses.append(loss.item())

            # Stop training if the loss is below a threshold
            if ploss < percent_loss_crit and epoch > 100:
                break

        if plot:
            # Plot the data and the fitted curve

            pw = pg.plot(
                x_data.T[0],
                y_data.T[0],
                symbol="o",
                symbolSize=4,
                pen=None,
                title="pyTorch: Exponential Decay Fitting",
            )
            pw.plot(
                x_data.T[0],
                self.model.forward(x_tensor).detach().numpy().T[0],
                pen=pg.mkPen("r", width=2),
            )

        if verbose:
            # Print the optimized parameters
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name, param.data)

        return self.model.named_parameters()


if __name__ == "__main__":
    # test
    app = pg.mkQApp()
    n = 50
    x_data = np.linspace(0, 50, n) * 1e-3
    x_data.sort()
    x_data = x_data.reshape(-1, 1)
    y_data = -65 - 2.5 * (1.0 - np.exp(-x_data * 60)) + 0.05 * np.random.randn(n, 1)
    y_data2 = y_data + 0.1 * (1.0 - np.exp(-x_data * 1))
    print("xdata.shape: ", x_data.shape)
    print("ydata.shape: ", y_data.shape)
    miny = np.min(y_data)
    maxy = np.max(y_data)
    logy = np.log(y_data - miny + 1)
    pf = np.polyfit(x_data.T[0], logy.T[0], 1)
    print(pf)
    # pw = pg.plot(x_data.T[0], y_data.T[0], pen=None, brush=pg.mkBrush('g', linestyle=None), symbol='o', title='Exponential Decay Fitting')
    # pw = pg.plot(x_data.T[0], logy.T[0], pen=None, brush=pg.mkBrush('g', linestyle=None), symbol='o', title='Exponential Decay Fitting')
    # pw.plot(x_data.T[0], pf[0]*x_data.T[0] + pf[1], pen=pg.mkPen('r', width=2))
    # app.exec()

    LM = LMexpFit(DC=maxy, A1=miny - maxy, tau1=-pf[0], A2=0, tau2=pf[0]/5.0)
    # LM = LMexpFit(a=-50, b=-2, c=20)
    result = LM.fit2(x_data, y_data, plot=True)
    print("LMfit values: ")
    # print(result.best_values)
    print(lmfit.fit_report(result.params))

    # TM = TorchFit(a=maxy, b=miny-maxy, c=-pf[0])
    # model = TM.fit(x_data, y_data, plot=True, verbose=False, nepochs=15000, percent_loss_crit=0.5)
    # # print(list(model))
    # print("Torch fit values: ")
    # u = list(model)
    # for i in range(3):
    #     print(u[i][0], u[i][1].data.view(-1).numpy()[0])

    # app.exec()
