from setuptools import find_packages, setup
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
with open("requirements.txt") as handle:
    dependencies = list(handle.readlines())

setup(
    
    name="hocmo",
    version="1.0.1",
    license='MIT',
    description = 'A Generalized Higher-Order Correlation Model (HOCMO) tool to generate scores modeling the strength of the relationship between triplicate entities using a tensor-based approach',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author = 'Charles Lu',
    author_email = 'lucharles@wustl.edu',
    install_requires = dependencies,
    packages=find_packages(),
)