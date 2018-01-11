#!/usr/bin/env python3
import sys
import os
from setuptools.command.install import install as InstallCommandBase
from setuptools import find_packages, setup

PROJECT_NAME, VERSION = open('version.txt').read().strip().split(':')
VERSION = VERSION.split('-')[0]
REQUIRED_PACKAGES = open('requirements.txt').read().split()
CLASSIFIERS = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],

def configuration(parent_package='', top_path=None):
    from numpy.distutils.system_info import get_info
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('julian', parent_package, top_path)

    return config


if __name__ == '__main__':
    #from numpy.distutils.core import setup

    setup(
          name=PROJECT_NAME,
          maintainer="Julian Developers",
          author="Zex Li",
          maintainer_email="top_zlynch@yahoo.com",
          description="Machine learning algorithms",
          long_description='',
          url='',
          license="MIT license",
          version=VERSION,
          install_requires=REQUIRED_PACKAGES,
          classifiers=CLASSIFIERS,
          zip_safe=False,
          packages=find_packages(),
          include_package_data=True,
          data_files=[
              ('julian', ['julian/algo/model.yaml']),
          ],
          keywords=['julian', 'ML', 'classifications', 'tensorflow', 'scipy']
          #**configuration(top_path='').todict(),
        )
