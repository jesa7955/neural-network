# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='neural-network',
    version='0.1.0',
    description='implementation practice of neural network with computation graph',
    long_description=readme,
    author='Tong Li',
    author_email='litong@logos.t.u-tokyo.ac.jp',
    url='https://kiwi.logos.ic.i.u-tokyo.ac.jp/gitlab/litong/neural-network.git',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

