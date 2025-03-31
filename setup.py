from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="DeepAllele",
    version="0.1.0",
    author="Mostafavi Lab",
    author_email="mostafavi@cs.washington.edu",
    description="A deep learning framework for predicting allele-specific gene regulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mostafavilabuw/DeepAllele-public",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "h5py",
        "pytorch-lightning",
        "captum",
        "biopython",
        "einops",
        "joblib",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
            "sphinx",
            "black",
            "flake8",
        ],
    }
)
