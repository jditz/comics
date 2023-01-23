import sys
import unittest

import torch

sys.path.append("..")
from comic.data_utils import MultiOmicsDataset, OmicsDataset


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {"__getitem__": __getitem__,})


class TestOmicsDataset(unittest.TestCase):
    """TestCase for the OmicsDataset class."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.column_file = "./ground_truth/omics_column.csv"
        cls.row_file = "./ground_truth/omics_row.csv"
        cls.label_file = "./ground_truth/omics_labels.csv"

        cls.true_data = [
            [1.51, 4.40, 3.06, 3.38, 1.60],
            [0.32, 0.68, 4.51, 3.64, 2.95],
            [0.11, 1.72, 4.07, 3.79, 0.12],
            [0.93, 2.05, 1.88, 1.66, 1.76],
            [4.92, 3.67, 2.92, 3.69, 4.24],
        ]
        cls.true_label = ["False", "False", "True", "False", "True"]

    def test_row(self):
        # load the omics file in row format
        data = OmicsDataset(
            self.row_file, self.label_file, omics_format="row", class_labels="True"
        )

        # test whether the data and labels attributes of the class
        # are in the expected form
        self.assertEqual(data.data, self.true_data)
        self.assertEqual(data.labels, self.true_label)

    def test_column(self):
        # load the omics file in column format
        data = OmicsDataset(
            self.column_file,
            self.label_file,
            omics_format="column",
            class_labels="True",
        )

        # test whether the data and labels attributes of the class
        # are in the expected form
        self.assertEqual(data.data, self.true_data)
        self.assertEqual(data.labels, self.true_label)

    def test_row_dataloader(self):
        # read in omics file in row format and create PyTorch DataLoader
        OmicsWithIndices = dataset_with_indices(OmicsDataset)
        data = OmicsWithIndices(
            self.row_file, self.label_file, omics_format="row", class_labels=["True"]
        )
        loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

        # iterate over DataLoader and make sure that the output is correct
        for d, l, i in loader:
            self.assertTrue(
                torch.all(
                    torch.eq(d[0], torch.tensor(self.true_data[i], dtype=torch.float))
                )
            )
            self.assertTrue(
                torch.all(
                    torch.eq(
                        l[0],
                        torch.tensor([self.true_label[i] == "True"], dtype=torch.float),
                    )
                )
            )

    def test_column_dataloader(self):
        # read in omics file in row format and create PyTorch DataLoader
        OmicsWithIndices = dataset_with_indices(OmicsDataset)
        data = OmicsWithIndices(
            self.column_file,
            self.label_file,
            omics_format="column",
            class_labels=["True"],
        )
        loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

        # iterate over DataLoader and make sure that the output is correct
        for d, l, i in loader:
            self.assertTrue(
                torch.all(
                    torch.eq(d[0], torch.tensor(self.true_data[i], dtype=torch.float))
                )
            )
            self.assertTrue(
                torch.all(
                    torch.eq(
                        l[0],
                        torch.tensor([self.true_label[i] == "True"], dtype=torch.float),
                    )
                )
            )


class TestMultiOmicsDataset(unittest.TestCase):
    """TestCase for MultiOmicsDataset class"""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.column_file = "./ground_truth/omics_column.csv"
        cls.row_file = "./ground_truth/omics_row.csv"
        cls.label_file = "./ground_truth/omics_labels.csv"

        cls.true_data = [
            [
                [1.51, 4.40, 3.06, 3.38, 1.60],
                [0.32, 0.68, 4.51, 3.64, 2.95],
                [0.11, 1.72, 4.07, 3.79, 0.12],
                [0.93, 2.05, 1.88, 1.66, 1.76],
                [4.92, 3.67, 2.92, 3.69, 4.24],
            ],
            [
                [1.51, 4.40, 3.06, 3.38, 1.60],
                [0.32, 0.68, 4.51, 3.64, 2.95],
                [0.11, 1.72, 4.07, 3.79, 0.12],
                [0.93, 2.05, 1.88, 1.66, 1.76],
                [4.92, 3.67, 2.92, 3.69, 4.24],
            ],
        ]
        cls.true_label = ["False", "False", "True", "False", "True"]

    def test_multi(self):
        # load multi omics dataset
        data = MultiOmicsDataset(
            data_files=[self.row_file, self.column_file],
            label_file=self.label_file,
            omics_format=["row", "column"],
            class_labels=["True"],
        )

        # test whether the data and labels attributes of the class
        # are in the expected form
        self.assertEqual(data.data, self.true_data)
        self.assertEqual(data.labels, self.true_label)

    def test_dataloader(self):
        # load multi omics dataset
        MultiOmicsWithIndices = dataset_with_indices(MultiOmicsDataset)
        data = MultiOmicsWithIndices(
            data_files=[self.row_file, self.column_file],
            label_file=self.label_file,
            omics_format=["row", "column"],
            class_labels=["True"],
        )

        # create a data loader
        loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

        # iterate over dataloader and test output
        for d, l, i in loader:
            self.assertTrue(
                torch.all(
                    torch.eq(
                        d[0][0], torch.tensor(self.true_data[0][i], dtype=torch.float)
                    )
                )
            )
            self.assertTrue(
                torch.all(
                    torch.eq(
                        d[1][0], torch.tensor(self.true_data[1][i], dtype=torch.float)
                    )
                )
            )
            self.assertTrue(
                torch.all(
                    torch.eq(
                        l[0],
                        torch.tensor([self.true_label[i] == "True"], dtype=torch.float),
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
