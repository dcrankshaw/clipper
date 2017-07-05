from setuptools import find_packages

setup(
    name='clipper_admin',
    version='0.2.0',
    description='Admin commands for the Clipper prediction-serving system',
    maintainer='Dan Crankshaw',
    maintainer_email='crankshaw@cs.berkeley.edu',
    url='http://clipper.ai',
    packages=['clipper_admin'],
    keywords=['clipper', 'prediction', 'model', 'management'],
    install_requires=[
        'requests', 'pyparsing', 'appdirs', 'pprint', 'subprocess32',
        'sklearn', 'numpy', 'scipy', 'pyyaml', 'abc', 'docker',
        'kubernetes'
    ],
    extras_require={
        'TensorFlow': ['tensorflow'],
        'RPython': ['rpy2']
    }
    )
