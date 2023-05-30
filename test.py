from __future__ import print_function
"""
Run unit tests for minis
"""

import os, sys
from pathlib import Path
import pytest

def main():
    # Make sure we look for minis here first.
    path = Path(__file__).parent
    print("Parent path: ", path)
    sys.path.insert(0, str(path))

    # Allow user to audit tests with --audit flag
    import ephys.ephys_analysis
    if '--audit' in sys.argv:
        sys.argv.remove('--audit')
        sys.argv.append('-s') # needed for cli-based user interaction
        ephys.mini_analyses.AUDIT_TESTS = True

    # generate test flags
    flags = sys.argv[1:]
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
