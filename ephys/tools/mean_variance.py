"""
non-stationary noise analysis

"""

from random import random, seed

from lmfit.models import QuadraticModel
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd



class TestGenerator():
    def __init__(self):
        self.set_transistions()
        self.dt = 0.0005
        self.dur = 0.10
        self.delay = 0.005
        self.npts = int(self.dur/self.dt)
        self.time = np.arange(0, self.dt*self.npts, self.dt)
        self.n_trials = 400
        self.n_chans = 50 # # openings in one event
        self.i_chan = 3.5
        self.noise_var = 3.0 # self.i_chan
        self.max_prob = 0.6

    def set_transistions(self):
        self.alpha = 0.0025 # opening rate
        self.beta = 0.005 # closing rate
        self.d0 = 0.0001

    def n_sweeps(self):
        dels = np.random.exponential(scale = self.alpha, size=(self.n_chans, self.n_trials)) + self.delay
        durs = np.random.exponential(scale = self.beta,  size=(self.n_chans, self.n_trials)) + self.delay
        probs = np.random.uniform(size=(self.n_chans, self.n_trials))
        idels = dels/self.dt
        idurs = durs/self.dt
        sweeps = np.zeros((self.n_trials, self.npts))
        for i, s in enumerate(range(sweeps.shape[0])):
            sweeps[i] = np.random.normal(0., scale=self.noise_var, size=self.npts)
            for n in range(self.n_chans):
                if probs[n, i] < self.max_prob:
                    sweeps[i, int(idels[n, i]):(int(idels[n, i] + idurs[n, i]))] += self.i_chan

        return sweeps
    

    def fit_meanvar(self):
        qmod = QuadraticModel()

        pars = qmod.guess(self.var, x=self.mean)
        qfit = qmod.fit(self.var, pars, x=self.mean)
        qfit.params['c'].set( vary=False, value=0.0)
        print(qfit.fit_report(min_correl=0.25))
        """fit parameters:
        y = a*x**2 + b*x + c

        var = -(I**2)/N + i_chan * I
        c should be set to 0
        Therfore, a = -1/N (N = -1/a), and b = i_chan

        """
        NChan = -1.0/qfit.values['a']
        gamma = qfit.values['b']
        Pomax = np.max(self.var)/ (2* gamma * NChan)

        print(f"nchan = {NChan:.3f}  gamma = {gamma:.3e}, Pomax = {Pomax:.2f}")

        return(qfit)

    def run(self):
        sweeps = self.n_sweeps()
        self.sweeps = sweeps
        f, ax = mpl.subplots(3,1)
        for i in range(np.min((2, sweeps.shape[0]))):
            ax[0].plot(self.time, sweeps[i] + self.i_chan*1.2*i)
        self.mean = np.mean(sweeps, axis=0)
        self.var = np.var(sweeps, axis=0)
        ax[1].plot(self.time, self.mean)
        ax[1].plot(self.time, self.var)
        ax[2].scatter(self.mean, self.var ,s=12)
        qfit = self.fit_meanvar()
        ax[2].plot(self.mean, qfit.best_fit, 'r--')
        mpl.show()

        



if __name__ == "__main__":

    tg = TestGenerator()
    tg.run()

