#!/usr/bin/env python3
import sys
import os
from setuptools.command.install import install as InstallCommandBase
from setuptools import find_packages, setup
import julian

PROJECT_NAME, VERSION = julian.__package__, julian.__version__
VERSION = VERSION.split('-')[0]
REQUIRED_PACKAGES = open('requirements.txt').read().split()
CLASSIFIERS = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ]


if __name__ == '__main__':
    #from numpy.distutils.core import setup

    setup(
          name=PROJECT_NAME,
          maintainer="Julian Developers",
          author="Zex Li",
          author_email="top_zlynch@yahoo.com",
          description="Machine learning algorithms",
          long_description='',
          url='https://github.com/zex',
          license="MIT license",
          version=VERSION,
          install_requires=REQUIRED_PACKAGES,
          classifiers=CLASSIFIERS,
          zip_safe=False,
          packages=find_packages(),
          include_package_data=True,
          data_files=[
              ('julian', ['julian/algo/model.yaml', 'requirements.txt']),
          ],
          keywords=['julian', 'ML', 'classifications', 'tensorflow', 'scipy'],
        )
