from setuptools import setup
import os

setup(
	name='composition_utils',
	version='1.03',
	description='utilities for benchmarking models used in Clipper model composition research',
	maintainer='Corey Zumar',
	maintainer_email='czumar@berkeley.edu',
	url='http://clipper.ai',
	packages=[
		"composition_utils", "composition_utils.model", "composition_utils.driver"
	],
	install_requires=[
		'numpy',
		'Pillow'
	])