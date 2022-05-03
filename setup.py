from setuptools import setup

setup(
    name="comik",
    version="0.1",
    description="Interpretable End-to-End Learning for Graph-Based Data in Healthcare",
    author="Jonas Ditz",
    author_email="jonas.ditz@uni-tuebingen.de",
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
        "pyunlocbox"
    ],
    zip_safe=False
)
