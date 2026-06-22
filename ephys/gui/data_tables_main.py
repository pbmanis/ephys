"""
Launcher stub for the DataTables GUI.

This file is the sole entry point (``datatable`` console script).  It
intentionally contains **no application logic** — only the minimum needed to
create the QApplication and start the event loop.

Why the split?
--------------
``pyqtgraph.reload.reloadAll()`` (wired to Tools → Reload) patches class and
function objects **in-place**.  For that to work, the live ``DataTables``
instance ``D`` must be an instance of ``ephys.gui.data_tables.DataTables``,
not ``__main__.DataTables``.  Running this stub as ``__main__`` while
importing ``DataTables`` from ``ephys.gui.data_tables`` guarantees that
``D.__class__`` is always the class that reload patches, so a reload takes
effect immediately without restarting.

Never add application logic here.  Changes here *do* require a restart.
"""

import multiprocessing
import sys
from pathlib import Path

# Qt and application imports — imported here so the stub itself is importable
# without side-effects from data_tables.py's module-level code.
from pyqtgraph.Qt import QtCore, QtWidgets

import ephys.tools.get_configuration as GETCONFIG
from ephys.gui.data_tables import DataTables, config_file_path


def main():
    # Keep D at function scope; the blocking exec() call below keeps this
    # frame (and therefore D) alive for the life of the application.
    print(config_file_path)
    if not Path(config_file_path).is_file():
        print(f"Configuration file not found at: {config_file_path!s}")
        raise ValueError(f"Configuration file not found at: {config_file_path!s}")

    datasets, experiments = GETCONFIG.get_configuration(config_file_path)
    D = DataTables(datasets, experiments)  # must retain pointer or the window dies
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtWidgets.QApplication.instance().exec()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
