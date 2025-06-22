from setuptools import find_packages
from setuptools import setup

setup(
    name="z-zephyr",
    version="0.0.13",
    description="Zephyr is a declarative neural network library on top of JAX allowing for easy and fast neural network designing, creation, and manipulation",
    url="https://github.com/mzguntalan/zephyr",
    author="Marko Zolo Gozano Untalan",
    author_email="mzguntalan@gmail.com",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=["jax>=0.4.28", "jaxlib>=0.4.28"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: GPU",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
)
