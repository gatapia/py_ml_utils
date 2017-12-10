#!/usr/bin/env python

from distutils.core import setup

setup(name='pml',
  version='0.1',
  description='PicNet Python Machine Learning Utilities',
  author='Guido Tapia',
  author_email='guido.tapia@picnet.com.au',
  url='http://picnet.com.au/predictive-analytics-service/',
  packages=['pml'],
  setup_requires=['pytest-runner'],
  tests_require=['pytest']
)