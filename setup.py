from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pitia",
    version="0.1.0",
    author="Stayely",
    author_email="lsszdst@gmail.com",
    description="Assistente virtual com memória e aprendizado contínuo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['pitia'],
    install_requires=[
        "requests>=4.13.4",
        "beautifulsoup4>=2.0.3",
        "google>=2.0.3",
        "nltk>=3.9.1",
        "sumy>=0.11.0",
        "scikit-learn>=1.7.1",
        "numpy>=1.20"
    ],
    entry_points={
        'console_scripts': [
            'pitia=pitia.cli:main',
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Natural Language :: Portuguese",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],

)
