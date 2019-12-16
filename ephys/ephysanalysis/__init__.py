#!/usr/bin/env python

# Use Semantic Versioning, http://semver.org/
version_info = (0, 2, 2, 'a')
__version__ = "%d.%d.%d%s" % version_info

#print ("apparent version: ", __version__)

import ephys.ephysanalysis.Fitting as Fitting
import ephys.ephysanalysis.Utility as Utility
import ephys.ephysanalysis.acq4read
import ephys.ephysanalysis.MatdatacRead
import ephys.ephysanalysis.DatacReader
import ephys.ephysanalysis.DataPlan
import ephys.ephysanalysis.getcomputer
import ephys.ephysanalysis.RmTauAnalysis
import ephys.ephysanalysis.SpikeAnalysis
import ephys.ephysanalysis.dataSummary
import ephys.ephysanalysis.IVSummary
import ephys.ephysanalysis.VCSummary
import ephys.ephysanalysis.PSCAnalyzer
import ephys.ephysanalysis.boundrect
import ephys.ephysanalysis.poisson_score
import ephys.ephysanalysis.bridge
import ephys.ephysanalysis.cursor_plot
import ephys.ephysanalysis.MakeClamps
import ephys.ephysanalysis.test_notch
import ephys.ephysanalysis.plot_maps
import ephys.ephysanalysis.fix_objscale

import ephys.ephysanalysis.metaarray as MetaArray


