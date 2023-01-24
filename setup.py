from setuptools import setup

setup(
    name="comic",
    version="0.1",
    description="Convolutional Kernel Networks for Interpretable End-to-End Learning on (Multi-)Omics Data",
    author="Jonas Ditz",
    author_email="jonas.ditz@uni-tuebingen.de",
    packages=["comic"],
    python_requires=">=3.7, <3.10",
    install_requires=[
        "numpy>=1.19.2",
        "scipy>=1.6.1",
        "matplotlib>=3.3.4",
        "pandas>=1.2.3",
        "biopython>=1.78",
        "scikit-learn>=0.24.1",
        "torch>=1.8.1",
        "torchvision>=0.9.1",
        "torchaudio>=0.8.1",
        "pyunlocbox",
        "sphinx",
        "sphinx_bootstrap_theme",
    ],
    zip_safe=False,
)
