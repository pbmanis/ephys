#!/usr/bin/env python
"""
init for minis
"""

# Use Semantic Versioning, http://semver.org/
version_info = (0, 3, 0, '')
__version__ = '%d.%d.%d%s' % version_info
AUDIT_TESTS=False
from . import clements_bekkers
from . import make_table
from . import minis_methods
from . import mini_analysis
from . import mini_summary
from . import mini_summary_plots
from . import clembek
#from . import functions
from ..tools import functions
from ..tools import digital_filters
