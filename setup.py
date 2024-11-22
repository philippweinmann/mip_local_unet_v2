#!/usr/bin/env python

from distutils.core import setup, find_packages

setup(name='mip_unet_v2',
      version='0.1',
      description='My model for the MIP project',
      author='Philipp Weinmann',
      author_email='philipp.weinmann71@gmail.com',
      packages=find_packages(),
      install_requires=[
        # Dependencies go here
        'numpy',
    ],
     )