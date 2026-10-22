"""setup.py — compile the ephys Cython extension(s) at build time.

Package metadata lives in pyproject.toml (PEP 621). This file exists only to
declare the compiled extension modules so setuptools runs Cython (via
``cythonize``) during the build. Pre-compiling here means ``c_deriv`` is
imported as an ordinary compiled module instead of being JIT-compiled by
``pyximport`` on first use at runtime (which emitted harmless but noisy
build warnings and added a one-time per-environment compile).

Extensions wired in:
  - ``ephys/tools/c_deriv.pyx`` — the derivative / spike-finder used by the
    IV / spike-analysis path.
  - ``ephys/mini_analyses/clembek.pyx`` — the mini-event detector imported by
    ``ephys.mini_analyses``.

The .pyx sources only ``import numpy`` at the Python level (no
``cimport numpy``), so no numpy build headers / include dirs are required.
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        [
            "ephys/tools/c_deriv.pyx",
            "ephys/mini_analyses/clembek.pyx",
        ],
        compiler_directives={"language_level": "3"},
    ),
)
