"""Legacy setup.py for bib2 package compatibility."""
from setuptools import setup, find_packages

setup(
    name="bib2",
    version="2.0.0",
    packages=find_packages(include=["bib2*"]),
)
