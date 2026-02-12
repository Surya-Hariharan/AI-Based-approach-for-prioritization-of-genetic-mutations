from setuptools import setup, find_packages
import os

# Read the README file for the long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements():
    requirements = []
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements

setup(
    name="genetic-mutation-prioritization",
    version="1.0.0",
    author="Surya Hariharan",
    author_email="your.email@example.com",
    description="AI-Based Platform for Genetic Mutation Prioritization and Pathogenicity Classification",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Surya-Hariharan/AI-Based-approach-for-prioritization-of-genetic-mutations",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "notebooks.*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=21.0.0",
            "isort>=5.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "ensemble": [
            "xgboost>=1.4.0",
            "lightgbm>=3.2.0",
        ],
        "graph": [
            "torch_geometric>=2.0.0",
        ],
        "transformers": [
            "transformers>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "genetic-mutation-ai=run:main",
            "gmai=run:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.md", "*.txt"],
        "configs": ["*.yaml", "*.yml"],
        "frontend": ["templates/*.html", "static/css/*.css", "static/js/*.js"],
    },
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/Surya-Hariharan/AI-Based-approach-for-prioritization-of-genetic-mutations/issues",
        "Source": "https://github.com/Surya-Hariharan/AI-Based-approach-for-prioritization-of-genetic-mutations",
        "Documentation": "https://github.com/Surya-Hariharan/AI-Based-approach-for-prioritization-of-genetic-mutations#readme",
    },
)