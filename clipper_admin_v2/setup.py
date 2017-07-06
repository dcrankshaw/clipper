from setuptools import setup

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()

setup(
    name='clipper_admin',
    version='0.2.0',
    description='Admin commands for the Clipper prediction-serving system',
    long_description=readme,
    maintainer='Dan Crankshaw',
    maintainer_email='crankshaw@cs.berkeley.edu',
    url='http://clipper.ai',
    license=license,
    packages=['clipper_admin'],
    keywords=['clipper', 'prediction', 'model', 'management'],
    install_requires=[
        'requests', 'pyparsing', 'appdirs', 'pprint', 'subprocess32',
        'sklearn', 'numpy', 'scipy', 'pyyaml', 'abc', 'docker', 'kubernetes', 'six'
    ],
    extras_require={'TensorFlow': ['tensorflow'],
                    'RPython': ['rpy2']}
)
