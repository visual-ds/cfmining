# -*- coding: utf-8 -*-
"""
@author:  Marcos M. Raimundo
@email:   marcosmrai@gmail.com
"""
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def read_version():

    # Default value if we cannot find the __version__ field in the init file:
    version = "0.0.1"

    # TODO: Possibly more robust way to find the directory:
    # filename = inspect.getframeinfo(inspect.currentframe()).filename
    # path = os.path.dirname(os.path.abspath(filename))

    init_file = (os.path.dirname(os.path.realpath(__file__)) +
                 "/actionsenum/__init__.py")
    if os.path.exists(init_file):
        with open(init_file, 'r') as f:
            for line in f:
                if line.startswith("__version__"):
                    _, version = line.split("=")
                    version = version.replace('\"', '').replace('\'', '')
                    version = version.strip()
                    break

    return version


params = dict(name="cfmining",
              version=read_version(),
              author="See contributors on ",
              author_email="marcosmrai@gmail.com",
              maintainer="Marcos Medeiros Raimundo",
              maintainer_email="marcosmrai@gmail.com",
              description="""Algorithm to enumerate Pareto-optimal actions
                             to regression problems.""",
              license="Private.",
              keywords="optimization, explainability, actionable",
              url="",
              long_description=read("README.md"),
              package_dir={"": "."},
              packages=["cfmining",
                        ],
              # package_data = {"": ["README.md", "LICENSE"],
              #                 "examples": ["*.py"],
              #                 "tests": ["*.py"],
              #                },
              classifiers=["Development Status :: 3 - Alpha",
                           "Intended Audience :: Developers",
                           "Intended Audience :: Science/Research",
                           "Topic :: Scientific/Engineering",
                           "Topic :: Machine learning"
                           "Programming Language :: Python :: 3.7",
                           ],
              )

try:
    from setuptools import setup

    params["install_requires"] = ['actionable-recourse==0.1.1', 'pandas', 'numpy',
                                  'jupyterlab', 'sortedcontainers', 'line_profiler', 
                                  'eli5', 'lightgbm', 'mip',  'graphviz',
                                  ]
except:
    from distutils.core import setup

setup(**params)
