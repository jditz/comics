# COmic

This repository contains the open-source implementation of Convolutional Omics Kernel Networks (COmic) as described
in Ditz et al., "COmic: Convolutional Kernel Networks for Interpretable End-to-End Learning on (Multi-)Omics Data".
COmic is implemented using custom PyTorch layers, hence, can be easily integrated in existing PyTorch-based 
neural network training pipelines. Furthermore, we provide a custom DataSet object that is specifically designed to
handle divers (multi-)omics input data. The COmic package also provide several ready-to-use models that allow users
without a strong background in neural network creation to perform machine learning experiments with COmic models.

## Installation

You can perform a user-specific installation by running

    $ python -m pip install .

from the root of the project. We strongly advise an installation in a virtual environment. You can create and activate 
one by executing the following two commands from the root of the project

    $ python -m venv venv
    $ . venv/bin/activate

If you are using anaconda, you can create a separate environment with the following commands

    $ conda create -n venv python=3.9
    $ conda activate venv

and then performing the installation as usual by running

    (venv) $ python -m pip install .

If you plan to extend the code, then you should perform an editable installation with

    (venv) $ python -m pip install -e .

## Testing

You can run the unit-tests by executing

    $ python -m unittest

from the `tests/` folder of the project. The ground truth needed for the tests is stored in the folder `tests/gound_truth/`.

## Documentation

The documentation is written with `sphinx`. You can build it by running

    $ cd docs && make html

from the root of the project. The entry point for the documentation will be placed in `docs/_build/html/index.html` which you can open with a browser of your choice.

## Scripts

We are providing several scripts together with the main COmic package. These scripts can be used as a starting point to 
develop user-specific COmic experiments. Furthermore, the scripts can be used to re-compute the experiments and analysis 
presented in our corresponding manuscript.

- `scripts/experiments.py` can be used to re-compute the experiments presented in COmic's corresponding manuscript or as a starting point for the development of user-specific experiments
- `scripts/analysis.py` can be used to re-compute the analysis presented in COmic's corresponding manuscript or as a starting point for the development of user-specific analysis
- `scripts/supplement.py` can be used to re-compute the figure in the supplementary material
