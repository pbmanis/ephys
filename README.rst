ephys overview
==============

ephys is a collection of modules for analysis of electrophysiology data. It current consists of 3 modules:

ephysanalysis
-------------
This module provides basic data reading for acq4, matdatac, datac, and nwb files. 
It also provides "standard" analysis of IV curves, including measures of input resistance,
membrane time constant, etc, and measure of spike rate and spike shapes.

mini_analyses
-------------
This module provides functions for analyzing miniature synaptic potentials and evoked synaptic
potentials and currents. Detection with Clements-Bekkers, and Andrade-Jonas methods are provided,
along with a simple zero-crossing method.

mapanalysistools
----------------
This module provides tools for analyzing laser scanning photostimulation data from acq4.


Requirements
------------
Python 3.7, matplotlib, scipy, numpy, seaborn.
pylibrary (github.com/pbmanis/pylibrary)
and no doubt other things... 

