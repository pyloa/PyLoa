from setuptools import setup, find_packages
import os

# load info from files
setup_directory = os.path.abspath(os.path.dirname(__file__))
readme_file = os.path.join(setup_directory, 'README.md')
version_file = os.path.join(setup_directory, 'pyloa', '_version.py')
requirements_file = os.path.join(setup_directory, 'requirements.txt')
__version__ = None

# load long description for PyPi
with open(readme_file, encoding='utf-8', mode='r') as f:
    __long_description__ = f.read()

# get version number
with open(version_file, encoding='utf-8', mode='r') as fd:
    exec(fd.read())

# get dependencies
with open(requirements_file, encoding='utf-8', mode='r') as f:
    __install_requires__ = [i for i in f.read().strip().split('\n')]

# PyPi classifiers
__classifiers__ = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Education",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
]

setup(
    name="pyloa",
    version=__version__,
    license="MIT",
    author="PyLoa Developers",
    author_email="pyloadevelopers@gmail.com",
    maintainer="PyLoa Developers",
    maintainer_email="pyloadevelopers@gmail.com",
    packages=['pyloa'] + ['pyloa.' + i for i in find_packages('pyloa')],
    provides=['pyloa'],
    python_requires='>=3.5',
    classifiers=__classifiers__,
    install_requires=__install_requires__,
    description="PyLoa: Learning on-line Algorithms with Python",
    long_description=__long_description__,
    long_description_content_type='text/markdown',
    url="https://github.com/pyloa/PyLoa",
    project_urls={
        "Bug Tracker": "https://github.com/pyloa/PyLoa/issues",
        "Documentation": "https://github.com/pyloa/PyLoa/tree/master/pyloa",
        "Source Code": "https://github.com/pyloa/PyLoa/tree/master/pyloa",
    },
    platforms=['any'],
    entry_points={'console_scripts': ['pyloa=pyloa._main:main'], },
    keywords='on-line algorithms, paging, vertex coloring, machine learning',
    zip_safe=False
)
