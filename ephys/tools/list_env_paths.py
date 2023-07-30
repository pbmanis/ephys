""" List out key environment paths for debugging. 
Sometimes a user will have multiple versions of python installed, or multiple versions of a package installed,
in different locations. This can cause problems with importing the correct package. This script will list out
the paths to the python executable, and the paths to the packages that are important for the ephys tools.

"""

from pathlib import Path
import sys

def list_environment_paths():
    import numpy as np
    import scipy as sp
    import lmfit
    import h5py

    import matplotlib, PyQt6, pyqtgraph

    important_imports = [pyqtgraph, PyQt6, np, sp, lmfit,matplotlib, h5py]

    print("\n============= Environment Paths ============= ")
    executable = Path(sys.executable).resolve()
    ver = sys.version_info
    print(f"      {'Python :':>12s} {sys.version_info[0]:d}.{sys.version_info[1]:d}.{sys.version_info[2]:d} ({sys.version_info[3]:s})", end='')
    print(f"{'  path: ':>6s}", str(executable))
    for imp in important_imports:
        if imp == PyQt6:
            print(f"    {imp.__name__:>12s} : {str(imp.QtCore.PYQT_VERSION_STR):<12s}", end='')
        else:
            print(f"    {imp.__name__:>12s} : {str(imp.__version__):<12s}", end='')
        print(f" {'path: ':>6s}{str(imp.__file__):s}")
    print()

if __name__ == "__main__":
    list_environment_paths()
