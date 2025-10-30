"""Setup configuration for UAL Adapter package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ual-adapter",
    version="0.1.0",
    author="Your Name",
    author_email="mehrabi.hamed@outlook.com",
    description="Universal Adapter LoRA for architecture-agnostic model adaptation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hamehrabi/ual-adapter.git",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "isort>=5.10",
            "flake8>=4.0",
            "mypy>=0.990",
            "pre-commit>=2.20",
        ],
        "docs": [
            "sphinx>=4.5",
            "sphinx-rtd-theme>=1.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "ual-adapter=ual_adapter.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ual_adapter": ["configs/*.yaml", "binders/*.json"],
    },
)
