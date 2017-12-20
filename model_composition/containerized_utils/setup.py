from setuptools import setup
import os

setup(
    name='containerized_utils',
    version='1.0.0',
    description='A ZMQ-based client for querying Clipper',
    maintainer='Corey Zumar',
    maintainer_email='czumar@berkeley.edu',
    url='http://clipper.ai',
    packages=[
        "containerized_utils",
        "containerized_utils.zmq_client",
        "containerized_utils.driver_utils"
    ],
    install_requires=[
        'numpy',
        'pyzmq',
        'futures_then',
        'scipy',
    ])
