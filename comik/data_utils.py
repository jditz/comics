r"""Module that contains function and classes to handle omics data.
"""

import os
import sys

import numpy as np
import torch
from torch.utils import data


class OmicsDataset(data.Dataset):
    r"""Custom Pytorch Dataset class that handles omics datasets. Data has to be provided
    with two files: a csv file with the features and a csv file with labels. The data file
    has to be in a format where the first line is a header containing all sample ids and
    each of the following lines correspond to one feature and contains the values that this
    feature takes for all samples. The labels file has to be in a format where each line
    contains one sample with the first entry being the sample's id and the second entry 
    being the sample's label. Both csv files have to have the same separator.

    Attributes
    ----------
    data : list
        List containing all features
    labels : list
        List containing all labels
    """

    def __init__(
        self,
        data_file: str,
        label_file: str,
        separator: str = ",",
        num_classes: int = 1,
        class_labels=None,
    ):
        r"""Constructor of the OmicsDataset class

        Parameters
        ----------
        data_file : str
            A string that contains the path to the file which stores the sample data.
        label_file : str
            A string that contains the path to the file which stores the labels.
        separator : str
            The string (most time this will be a single character) that separates the
            entries in the csv files.
        num_classes : int
            Number of classes in the prediction problem. If a regression problem has 
            to be solved, set the num_classes parameter to 1 and the class_labels 
            parameter to None.
        class_labels : list
            List that contains the labels of each class in the labels file. The order
            of labels in class_labels will determine their index. For a binary
            classification, only the positive label should be given. If a regression
            problem has to be solved, set the num_classes parameter to 1 and the
            class_labels parameter to None.
        """

        # read in the data file
        features = []
        with open(data_file, "r") as in_data:
            for idx, line in enumerate(in_data):
                linelist = line.strip().split(separator)

                # the line will contain the sample ids, if it is the first line
                if idx == 0:
                    for sample in linelist:
                        features.append((sample, []))

                # each line after the first will contain the value of one feature
                else:
                    for idx2, sample in enumerate(linelist):
                        # skip the first entry since it will contain the id of the feature
                        if idx2 == 0:
                            continue
                        features[idx2 - 1][1].append(sample)

        # read in the labels file
        labels = []
        with open(label_file, "r") as in_label:
            for line in in_label:
                linelist = line.strip().split(separator)
                labels.append((linelist[0], linelist[1]))

        # report an error if the number of samples in the data file does not
        # match the number of samples in the label file
        if len(features) != len(labels):
            raise ValueError(
                "Unequal number of samples in data and label file detected!\n"
                + f"    data file contains {len(features)} samples\n"
                + f"    label file contains {len(labels)} samples\n"
            )

        # make sure that all samples have the same amount of features
        ref_len = len(features[0][1])
        if not all([len(x[1]) == ref_len for x in features]):
            raise ValueError("Samples do not have equal number of features!\n")

        # make sure that the data and labels attributes are in the same order
        features.sort(key=lambda x: x[0])
        labels.sort(key=lambda x: x[0])

        # prepare the data and labels attributes of the class
        self.nb_features = len(features[0][1])
        self.nb_classes = num_classes
        self.sample_ids = []
        self.data = []
        self.labels = []
        for i in range(len(features)):
            self.sample_ids.append(features[i][0])
            self.data.append(features[i][1])
            self.labels.append(labels[i][1])

        # prepare the label on index mapping
        # if the prediction problem is a regression, no class to index mapping is needed
        if num_classes == 1 and class_labels == None:
            self.class_to_idx = None
        # if the prediction problem is a binary classification, only the label
        # of the positive class is needed
        elif num_classes == 1:
            self.class_to_idx = class_labels[0]
        # if the prediction problem is a multi-class classification, the mapping of
        # each class onto an index is realised with a dictionary
        else:
            self.class_to_idx = {}
            for idx, class_label in enumerate(class_labels):
                self.class_to_idx[class_label] = idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r"""Overwritten __getitem__ method from the Dataset class.
        
        When called with an index, returns a data tensor containing the feature values
        of the corresponding sample and a label tensor with the corresponding label.
        
        Parameters
        ----------
            idx : int
                Index of the requested sample

        Returns
        -------
            features : Tensor
                Tensor containing the feature values of the requested sample
            label : Tensor
                Tensor containing the label of the requested sample
        """
        # retrieve the requested data
        features = torch.Tensor(self.nb_features)
        features.data = self.data[idx]

        # retrieve the requested label
        if self.class_to_idx == None:
            label = torch.Tensor(float(self.labels[idx]))
        elif isinstance(self.class_to_idx, dict):
            label = torch.zeros(self.nb_classes)
            label[self.class_to_idx[self.labels[idx]]] = 1
        else:
            if self.labels[idx] == self.class_to_idx:
                label = torch.Tensor([1])
            else:
                label = torch.Tensor([0])

        return features, label
