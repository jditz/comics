"""Convolutional Omics Kernel Network

This package implements convolutional omics kernel networks (COmic) as described in Ditz et al., "COmic: Convolutional Kernel 
Networks for Interpretable End-to-End Learning on (Multi-)Omics Data", 2022. COmic is developed with PyTorch.

References:
    models: Module that contains examplary COmic models. These models can be used for quick experiments with COmic.
    layers: Module that implements all custom PyTorch layers needed for the creation of COmic models.
    graph_learner: Module that contains different graph learning frameworks. These frameworks are used to directly learn a 
                   Laplacian from data. 
    data_utils: Module that contains a custom DataSet class for (multi-)omics data. Furthermore, this module contains
                data-specific utility functions.
    utils: Module that contains utility functions.

Author: Jonas C. Ditz
"""
