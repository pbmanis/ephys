Version 0.7.0a

New in this version:
* Simplified process for analysis of IV/spike data (current clamp IVs).
All analyzed data is stored in Python pkl files in cell-specific subdirectories
in a "dataset" directory.
* A text-formatted configuration file provides project-specific parameters, colors, lists,
directories, etc. This serves both to expose some parameters in the analysis,
and to make sure that parameters that get changed are not hidden in the code.

Note that this is "alpha" code, meaning that it works for me, but also that
errors may occur that have to be identified and solved. 


Ephys
=====

This is an experimental suite of programs for analyzing electrophysiology data,
written in Python. These are not considered appropriate for general use at this
point, but may be a useful starting point. In my lab, these are used for various
aspects of analysis, and so are actively used and under regular development. A
recent refactoring of some of the code (May 2020) may have rendered certain
parts non-functional. The base package is set up to analyze data in the "acq4"
format, but also has routines that translate from an older set of matlab and
C-based formats specific to our lab. 

The analysis part of the package is broken up into 4 parts: ephysanalysis, 
minianalysis, psc_analysis and
mapanalysis, plus a set of tools that make use of these packages.

A GUI is available to help with the ephys analysis.

Requirements: Python3.11+, and modules specified in requirements_local.txt

ephysanalysis
-------------
This module provides tools for reading the lab data formats, for doing basic
analysis of IV prototocols, spike shapes and rate adaptation, some voltage-clamp
analyses (incomplete), and measures of EPSPs/EPSCs from particular protocols It
also provides a routine (dataSummary) that generates tables from the
header/index files of a directory of recordings, to aid in creating
semi-automated analysis of datasets.


mini_analyses
-------------

The MiniAnalysis class provides overall services that are commonly needed for
both of the event detectors, including setups, making measurements of the
events, fitting them, curating them, and plotting.

1. Clements, J. D. & Bekkers, J. M. Detection of spontaneous synaptic events
    with an optimally scaled template. Biophys. J. 73, 220–229 (1997).

2. Pernia-Andrade, A. J. et al. A deconvolution-based method with high
   sensitivity and temporal resolution for detection of spontaneous synaptic
   currents in vitro and in vivo. Biophys J 103, 1429–1439 (2012).


Clements Bekkers class can use a numba jit routine to speed things up (there is
also a cython version floating around, but the numba one is easier to deal
with). This is the classic sliding-template algorithm; it does not work well
with overlapping events. The Pernia-Andrade et al. method uses numpy and scipy
routines to implement a deconvolution approach; it works better with overlapping
events. Two other algorithms are included (zero-crossing,
Richardson-Silverberg), but have not been not fully tested. 

There are some test routines that generate synthetic data, and which exercise
the code. 

The core code is in the mini_methods file, which provides a MiniAnalysis class
and separate classes for each of the methods. A set of test routines (some of
which currently fail because of changes in the underlying datasets) are
included. 

Usage:

mini_analysis.py  (rarely called this way).

This module provides the MiniAnalysis class, which is a high-level wrapper that
uses mini_methods_command and mini_methods to analyze events, organize the
results, and create various summary plots of event distributions.

* Utilities

Several modules provide utilities. Best to inspect these; they are often special
purpose.


# mapanalysistools
================

This is a small modulethat provides some tools for analysis of laser-scanning
photostimulation maps. 

getTable: a program that reads the datasummary table (see ephysanalysis), or can
look for a protocol directory, displays the waveforms, and can use the
Andreade-Jonas deconvolution algorithm to identify events. Events are marked on
the traces. Also displayed are the average events, fits, and template, and a
histogram of event times across all trials.

analyzeMapData: a program that is similar to getTable, without the GUI.
Generates a matplotlib display of the map grid (color coded by amplitude),
average waveforms, and the event histogram. It is designed to be called from a
higher-level analysis program as well (that is current usage).

plotMapData is the routine that was used in the Zhang et al 2017 paper for the
TTX plots (similar to analyzeMapData).

Dependencies
------------

To read acq4 files (videos, scanning protocols, images):  ephysanalysis
(https://github.com/pbmanis/ephysanalysis) To analyze events, minis (provides
Andrade-Jonas and Clements Bekkers): minis
(https://github.com/pbmanis/mini_analysis) pylibrary (plotting, findspikes)
(https://github.com/pbmanis/pylibrary)

pyqtgraph matplotlib seaborn xlrd pandas numpy re (regular expressions)

Usage: usage: analyzeMapData.py [-h] [-i] [-o DO_ONE] [-m DO_MAP] [-c] [-v]
datadict

mini synaptic event analysis

positional arguments: datadict              data dictionary

optional arguments: -h, --help            show this help message and exit -i,
  --IV              just do iv -o DO_ONE, --one DO_ONE just do one -m DO_MAP,
  --map DO_MAP just do one map -c, --check           Check for files; no
  analysis -v, --view            Turn off pdf for single run
  
  
  Installation
  ------------
  A requirements file for doing a local install is provided, along with a bash
  shell script to create an environment. 
  

  Uupdate: Now using pyproject.toml and uv to handle build
  Requirements
------------
Python 3.13, matplotlib, scipy, numpy, seaborn, pyqtgraph.
pylibrary (github.com/pbmanis/pylibrary)
and many other things... 

#Installation
------------
Clone the repository and run the following command in the root directory:
`uv venv`  # build the virtual environment
`source .venv/bin/activate`  # activate the virtual environment
`uv build` # build the package if you want it in that format
`python ephys/ephys_analysis/tests/hh_sim.py` # build the test data .pkl files
# The tests test the mini analysis detection and spike shape analysis routines.
Then run the tests:
`python test.py`


