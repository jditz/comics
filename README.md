# COmic
This repository contains the open-source implementation of Convolutional Omics Kernel Networks (COmic) as described
in Ditz et al., "COmic: Convolutional Kernel Networks for Interpretable End-to-End Learning on (Multi-)Omics Data",
2022. COmic is implemented using custom PyTorch layers, hence, can be easily integrated in existing PyTorch-based 
neural network training pipelines. Furthermore, we provide a custom DataSet object that is specifically designed to
handle divers (multi-)omics input data. The COmic package also provide several ready-tu-use models that allow users
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