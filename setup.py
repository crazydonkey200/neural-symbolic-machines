#!/usr/bin/env python

from setuptools import setup, find_packages
import sys

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='nsm',
    version='0.1.0',
    description='Neural Symbolic Machines',
    long_description=readme,
    license=license,
    python_requires='==2.7',
    packages=find_packages(exclude=('examples')),
    install_requires=reqs.strip().split('\n'),
)
