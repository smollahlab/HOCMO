from setuptools import find_packages, setup

with open("requirements.txt") as handle:
    dependencies = list(handle.readlines())

setup(
    
    name="hocmo",
    version="1.0.0",
    license='MIT',
    description = 'A Generalized Higher-Order Correlation Model (HOCMO) tool to generate scores modeling the strength of the relationship between triplicate entities using a tensor-based approach',
    author = 'Charles Lu',
    author_email = 'lucharles@wustl.edu',
    install_requires = dependencies,
    packages=find_packages(),
)