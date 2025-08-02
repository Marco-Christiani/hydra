# hydra_ray_tune_sweeper/setup.py
# type: ignore
from pathlib import Path

from read_version import read_version
from setuptools import find_namespace_packages, setup

setup(
    name="hydra-ray-tune-sweeper",
    version=read_version("hydra_plugins/hydra_ray_tune_sweeper", "__init__.py"),
    author="Marco Christiani",
    author_email="mchristiani2017@gmail.com",
    description="Ray Tune Sweeper for Hydra",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/Marco-Christiani/hydra-ray-tune-sweeper",
    packages=find_namespace_packages(include=["hydra_plugins.*"]),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],
    install_requires=[
        "hydra-core>=1.1.0",
        "ray[tune]>=2.0.0",
        "omegaconf>=2.1.0",
    ],
    include_package_data=True,
)
