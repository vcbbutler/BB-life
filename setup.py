from setuptools import setup, find_packages

setup(
    name="bb-life",
    version="0.1.0",
    description="Conway's Game of Life with GPU acceleration and 3D visualization",
    author="BB-life contributors",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "vispy>=0.6.6",
        "PyQt5>=5.15.0",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "bb-life=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
) 