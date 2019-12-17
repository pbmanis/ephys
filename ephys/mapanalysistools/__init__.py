from __future__ import absolute_import
"""
init for mapanalysistools
"""

# Use Semantic Versioning, http://semver.org/
version_info = (0, 1, 0, '')
__version__ = '%d.%d.%d%s' % version_info

import ephys.mapanalysistools.getTable
import ephys.mapanalysistools.analyzeMapData
#import mapanalysistoolsplotMapData  # removed - is an old unstructured version for source information
import ephys.mapanalysistools.functions
import ephys.mapanalysistools.digital_filters
