from setuptools import setup
from setuptools import find_packages
# from setuptools.extension import Extension
# from Cython.Build import cythonize

setup(name='minis',
      version='0.3.0',
      description='Mini event analysis routines and support',
      url='https://',
      author='Paul B. Manis, Ph.D.',
      author_email='pmanis@med.unc.edu',
      license='MIT',
      packages=['minis'],
      # ext_modules = cythonize("minis/clembek.pyx"),
      zip_safe=False,
      )
      