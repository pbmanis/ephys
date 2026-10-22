# Claude added 2026-10-22: acq4 mosaic import + preliminary-alignment tool.
"""Mosaic import tool for the cochlear-nucleus alignment workflow.

Step 1 (this package, initial cut): read acq4 ``.mosaic`` files found under a
day directory, assemble the tiled image from each tile's ``.index`` metadata,
and display it (with markers / cells / rulers) in a PyQt6 window. Later steps
add a 3D preliminary-alignment / ordering tool and an export path that hands the
spaced image stack to the main atlas aligner.
"""

from .reader import (AssembledMosaic, Cell, MarkerSet, MosaicReader, Ruler,
                     Tile, find_mosaic_files)

__all__ = [
    "AssembledMosaic", "Cell", "MarkerSet", "MosaicReader", "Ruler", "Tile",
    "find_mosaic_files",
]
