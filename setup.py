#!usr/bin/env python

from setuptools import setup
import re

# https://packaging.python.org/discussions/install-requires-vs-requirements/
install_requires = [
    'numpy>=1.21', 'pandas==1.3.5', 'scipy==1.7.3', 'scikit-learn==1.0.2',
    'matplotlib==3.5.1', 'tqdm', 'argparse',
]

with open("README.md", "rb") as f:
    long_descr = f.read().decode("utf-8")

version = re.search(
    '^__version__\\s*=\\s*"(.*)"',
    open('dsp_transport/__version__.py').read(),
    re.M
).group(1)

setup(name='dsp_transport',
      version=version,
      author='Jordan Gilbert',
      license='MIT',
      python_requires='>3.7',
      long_description=long_descr,
      author_email='jtgilbert89@gmail.com',
      install_requires=install_requires,
      zip_safe=False,
      entry_points={
          "console_scripts": [
              'dsp_transport = dsp_transport.dsp_transport:main',
              'plotting = dsp_transport.plotting:main'
          ]
      },
      url='https://github.com/jtgilbert/dynamic-stress-partitioning',
      packages=[
          'dsp_transport'
      ]
      )
