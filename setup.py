from setuptools import setup, find_packages

setup(
    name="pyloa",
    version="1.0.0",
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
    long_description="A research repository for analyzing the performance of classic on-line algorithms vs modern "
                     "Machine Learning, specifically Reinforcement Learning, approaches.",

    url="https://github.com/TBD",
    project_urls={
        "Bug Tracker": "https://github.com/TBD",
        "Documentation": "https://github.com/TBD",
        "Source Code": "https://github.com/TBD",
    },
    platforms=['any'],
    entry_points={'console_scripts': ['pyloa=pyloa._main:main'], },
    package_data={'pyloa': ['README.md', 'LICENCE', 'examples', 'requirements.txt']},
    keywords='on-line algorithms, paging, vertex coloring, machine learning',
    zip_safe=False
)
