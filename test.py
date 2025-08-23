from __future__ import print_function
"""
Run unit tests for minis and spike analysis. This should be run if pytest fails
to help isolate the problems. 

This script is intended to be run from the command line. It will
find the test files in the ephys directory and run them using pytest.
The script will also add the ephys directory to the Python path.

Flags:
 -- audit: run the tests in audit mode, which allows you to compare the
 results with the stored results of a previous run, and either accept
 or reject the results. This can be useful if there is a change in code
 that changes the calculations, or if there is a change in libraries
 that changes the calculations. Usually we find that the differences
 are in the 10^-4 to 10^-5 range (relative), which is acceptable given
 that we are using floating point numbers, and that there is noise
 in the data (although there is not supposed to be noise in the test
 data). 
    -- tensor_flow_test: run the tests in the ephys/tools/tests directory
    for tensor flow installation. This is not supported currently
    --tb=short: use short traceback for errors. This is the default.

This should be run if pytest fails.

"""

import os, sys
from pathlib import Path
import pytest

def main():
    # Make sure we look for minis here first.
    path = Path(__file__).parent
    print("Parent path: ", path)
    sys.path.insert(0, str(path))

    # be sure the simulation data exists, but if not, create it.
    simfile = Path('ephys', 'ephys_analysis', 'tests', 'HHData.pkl')
    if not simfile.is_file():
        print(f"HH simulation data file {simfile} not found. Creating it now.")
        import ephys.ephys_analysis.tests.hh_sim as hh_sim
        hh_sim.main()
    else:
        print(f"HH simulation data file {simfile} found ok.")
    # Allow user to audit tests with --audit flag
    import ephys.ephys_analysis
    import ephys.tools
        # generate test flags
    flags = sys.argv[1:]
    if '--audit' in sys.argv:
        sys.argv.remove('--audit')
        sys.argv.append('-s') # needed for cli-based user interaction
        ephys.mini_analyses.AUDIT_TESTS = True
    if '--tensor_flow_test' in sys.argv:
        sys.argv.remove('--tensor_flow_test')
        sys.argv.append('-s')
        flags.append('ephys/tools/tests/test_tensorflow.py')

    flags.append('-v')
    tb = [flag for flag in flags if flag.startswith('--tb')]
    if len(tb) == 0:
        flags.append('--tb=short')

    add_path = True
    print("flags: ", flags)
    for flag in flags:
        pflag = Path(flag)
        if pflag.is_dir() or pflag.is_file():
            add_path = False
            print("add path was set false")
            break
    if add_path:
        flags.append('ephys/mini_analyses')
        flags.append('ephys/ephys_analysis')
        flags.append('ephys/psc_analysis')

    print("flags: ", flags)
    # ignore the an cache
    # flags.append('--ignore=minis/somedir')

    # Start tests.
    print("Testing with flags: %s" % " ".join(flags))
    pytest.main(flags)


if __name__ == '__main__':
    main()
