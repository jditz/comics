import sys
import unittest

import numpy as np
import torch

sys.path.append("..")
from comic.data_utils import OmicsDataset
from comic.layers import PIMKLLayer


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {"__getitem__": __getitem__,})


class TestPIMKLLayer(unittest.TestCase):
    """TestCase for the PIMKLLayer class"""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.column_file = "./ground_truth/omics_column.csv"
        cls.row_file = "./ground_truth/omics_row.csv"
        cls.label_file = "./ground_truth/omics_labels.csv"

        cls.laplacians = [
            (
                np.array([0, 1, 3]),
                np.array([[1, 0, -0.34], [0, 1, -23], [-34, -23, 1]]),
            ),
            (np.array([2, 4]), np.array([[1, -0.72], [-0.72, 1]])),
        ]
