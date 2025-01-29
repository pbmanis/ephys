from setuptools import setup, find_packages
from Cython.Build import cythonize

# Use Semantic Versioning, http://semver.org/
version_info = (0, 7, 5, 'a')
__version__ = '%d.%d.%d%s' % version_info


setup(name='ephys',
      version=__version__,
      description='Methods for analysis of elecrophysiology data: IV and FI curves, minis, LSPS maps',
      url='http://github.com/pbmanis/ephys',
      author='Paul B. Manis',
      author_email='pmanis@med.unc.edu',
      license='MIT',
      packages=find_packages(include=['ephys*']),
      ext_modules=cythonize(["ephys/mini_analyses/clembek.pyx", "ephys/ephys_analysis/c_deriv.pyx"]),
      zip_safe=False,
      entry_points={
          'console_scripts': [
               'datasummary=ephys.tools.data_summary:main',
               'datatable=ephys.gui.data_tables:main',
               'dircheck=ephys.tools.dir_check:main',
               'ma2tiff=ephys.tools.ma2tiff:convertfiles',
               'bridge=ephys.tools.bridge:main',
               'miniviewer=ephys.tools.miniviewer:main',
               'checkrs = ephys.tools.check_rs:main',
               'measure=ephys.tools.cursor_plot:main',
               'matread=ephys.ephys_analysis.MatdatacRead:main',
               'plotmaps=ephys.tools.plot_maps:main',
               'fix_objscale=ephys.tools.fix_objscale:main',
               'analyzemapdata=ephys.mapanalysistools.analyzeMapData:main',
          ]
      },
      classifiers = [
             "Programming Language :: Python :: 3.10+",
             "Development Status ::  Beta",
             "Environment :: Console",
             "Intended Audience :: Neuroscientists",
             "License :: MIT",
             "Operating System :: OS Independent",
             "Topic :: Scientific Software :: Tools :: Python Modules",
             "Topic :: Data Processing :: Neuroscience",
             ],
    )
      