# mini_analyses
mEPSC/mIPSC analysis routines

Here we provide a set of Python interfaces/code to two synaptic event detection algorithms:

1. Clements, J. D. & Bekkers, J. M. Detection of spontaneous synaptic events with an optimally
    scaled template. Biophys. J. 73, 220–229 (1997).

2. Pernia-Andrade, A. J. et al. A deconvolution-based method with high sensitivity and temporal resolution
   for detection of spontaneous synaptic currents in vitro and in vivo. Biophys J 103, 1429–1439 (2012).
   
The core code is in the mini_methods file, which has a MiniAnalysis class and separate classes for
each of the methods. 

MiniMethods
-----------
mini_methods.py provides a MiniAnalysis class, along with classes for the two algorithms.

The MiniAnalysis class provides overall services that are commonly needed for both of the event detectors, 
including setups, making measurements of the events, fitting them, curating them, and plotting.

Clements Bekkers class can use a numba jit routine to speed things up (there is also a cython version
floating around, but the numba one is easier to deal with).

The Pernia-Andrade et al. method just uses numpy and scipy routines to implement the deconvolution.

Finally, there are some test routines that generate synthetic data, and which exercise the code. 


mini_analysis.py
----------------

This module provides the MiniAnalysis class, which is a high-level wrapper that uses mini_methods to analyze events,
organize the results, and create various summary plots of event distributions.

Utilities
---------
Several modules provide utilities. Best to inspect these; they are often special purpose.


# mapanalysistools
================

This is a small repository that provides some tools for analysis of laser-scanning photostimulation maps. 

getTable: a program that reads the datasummary table (see ephysanalysis), or can look for a protocol directory,
displays the waveforms, and can use the Andreade-Jonas deconvolution algorithm to identify events. 
Events are marked on the traces. Also displayed are the average events, fits, and template, and
a histogram of event times across all trials.

analyzeMapData: a program that is similar to getTable, without the GUI. Generates a matplotlib
display of the map grid (color coded by amplitude), average waveforms, and the event histogram.

plotMapData is the routine that was used in the Zhang et al 2017 paper for the TTX plots (similar to analyzeMapData).

Dependencies
------------

To read acq4 files (videos, scanning protocols, images):  ephysanalysis (https://github.com/pbmanis/ephysanalysis)
To analyze events, minis (provides Andrade-Jonas and Clements Bekkers): minis (https://github.com/pbmanis/mini_analysis)
pylibrary (plotting, findspikes) (https://github.com/pbmanis/pylibrary)

pyqtgraph
matplotlib
seaborn
xlrd
pandas
numpy
re (regular expressions)

Usage:
usage: analyzeMapData.py [-h] [-i] [-o DO_ONE] [-m DO_MAP] [-c] [-v] datadict

mini synaptic event analysis

positional arguments:
  datadict              data dictionary

optional arguments:
  -h, --help            show this help message and exit
  -i, --IV              just do iv
  -o DO_ONE, --one DO_ONE
                        just do one
  -m DO_MAP, --map DO_MAP
                        just do one map
  -c, --check           Check for files; no analysis
  -v, --view            Turn off pdf for single run
  
