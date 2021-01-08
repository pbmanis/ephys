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
    sys.path.insert(0, str(path))

    # Allow user to audit tests with --audit flag
    import ephys.ephysanalysis
    if '--audit' in sys.argv:
        sys.argv.remove('--audit')
        sys.argv.append('-s') # needed for cli-based user interaction
        minis.AUDIT_TESTS = True

    # generate test flags
    flags = sys.argv[1:]
    flags.append('-v')
    tb = [flag for flag in flags if flag.startswith('--tb')]
    if len(tb) == 0:
        flags.append('--tb=short')

    add_path = True
    for flag in flags:
        pflag = Path(flag)
        if pflag.is_dir() or pflag.is_file():
            add_path = False
            break
    if add_path:
        flags.append('mini_analyses')

    # ignore the an cache
    # flags.append('--ignore=minis/somedir')

    # Start tests.
    print("Testing with flags: %s" % " ".join(flags))
    pytest.main(flags)


if __name__ == '__main__':
    main()
