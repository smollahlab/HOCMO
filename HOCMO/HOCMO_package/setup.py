from setuptools import find_packages, setup

with open("requirements.txt") as handle:
    dependencies = list(handle.readlines())

setup(
    name="hocmo",
    version="0.0.4",
    install_requires = dependencies,
    packages=find_packages(),
)
