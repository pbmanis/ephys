from __future__ import absolute_import
"""
init for mapanalysistools
"""

# Use Semantic Versioning, http://semver.org/
version_info = (0, 1, 0, '')
__version__ = '%d.%d.%d%s' % version_info

from . import getTable
from . import analyzeMapData
#import mapanalysistoolsplotMapData  # removed - is an old unstructured version for source information
from . import functions
from . import digital_filters
