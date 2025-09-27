from setuptools import setup, find_packages

setup(
    name="model_ir_parser",
    version="0.1.0",
    description="Parser for converting Jiuge model configs to Transformer IR",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # No external dependencies, using only standard library
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)