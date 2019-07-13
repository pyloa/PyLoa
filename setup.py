from setuptools import setup, find_packages
from os import path

# load long description from README.md
setup_directory = path.abspath(path.dirname(__file__))
with open(path.join(setup_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="pyloa",
    version="1.0.1",
    license="MIT",
    author="PyLoa Developers",
    author_email="pyloadevelopers@gmail.com",
    maintainer="PyLoa Developers",
    maintainer_email="pyloadevelopers@gmail.com",
    packages=['pyloa'] + ['pyloa.' + i for i in find_packages('pyloa')],
    provides=['pyloa'],
    python_requires='>=3.5',
    install_requires=['matplotlib==3.0.3', 'scipy==1.2.1', 'tensorflow==1.13.1', 'tqdm==4.31.1', 'numpy==1.16.2'],
    description="PyLoa: Learning on-line Algorithms with Python",
    long_description=long_description,
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
