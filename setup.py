# -*- coding: utf-8 -*-
import sys, os
from setuptools import setup
from setuptools.command.test import test as TestCommand



__version__='0.0.2'

# if sys.argv[-1] == 'publish':
#     os.system("python setup.py sdist bdist_wheel register upload")
#     print("  git tag -a %s -m 'version %s'" % (__version__,__version__))
#     print("  git push --tags")
#     sys.exit()

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='meirlop',
      version=__version__,
      description='Motif Enrichment In Ranked Lists Of Peaks',
      long_description=('This project analyzes the relative enrichment of '
      'transcription factor binding motifs found in peaks at the top or bottom of '
      'a given ranking/score. The design is based on MOODS and statsmodels.'),
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: POSIX',
          'Topic :: Software Development :: Libraries'
          'Topic :: Scientific/Engineering :: Bio-Informatics'],
      keywords= ['Motif', 'Enrichment',
          'Bioinformatics'],
      url='https://github.com/npdeloss/meirlop',
      author='Nathaniel Delos Santos',
      author_email='npdeloss@ucsd.edu',
      license='MIT',
      packages=['meirlop'],
      entry_points={
          'console_scripts':[
              'meirlop = meirlop.__main__:main'
          ]
      },
      install_requires=[
                        'biopython',
                        'joblib',
                        'moods-python',
                        'numpy',
                        'pandas',
                        'scikit-learn',
                        'statsmodels',
                        'tqdm'
                        ],
      zip_safe=False,
      download_url='https://github.com/npdeloss/meirlop')

__author__ = 'Nathaniel Delos Santos'
