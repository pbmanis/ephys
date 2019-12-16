from __future__ import absolute_import
"""
init for mapanalysistools
"""

# Use Semantic Versioning, http://semver.org/
version_info = (0, 1, 0, '')
__version__ = '%d.%d.%d%s' % version_info

import mapanalysistools.getTable
import mapanalysistools.analyzeMapData
#import mapanalysistoolsplotMapData  # removed - is an old unstructured version for source information
import mapanalysistools.functions
import mapanalysistools.colormaps
import mapanalysistools.digital_filters
