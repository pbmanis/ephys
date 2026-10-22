# Claude added 2026-10-22: standalone entry point for the mosaic importer.
"""Run the mosaic browser standalone:  python -m ephys.tools.cn_atlas.mosaic"""

import sys

from .browser import main

if __name__ == "__main__":
    sys.exit(main())
