r"""Module that contains function and classes to handle omics data.
"""

from random import shuffle

import torch
from torch.utils import data


def __load_omics_column_without_label(
    data_file: str, separator: str = ",",
):
    r"""Private function that loads an omics data file in column format.
    
    Parameters
    ----------
    data_file : str
        A string that contains the path to the file which stores the sample data.
    separator : str
        The string (most time this will be a single character) that separates the
        entries in the csv files.

    Returns
    -------
    features : list
        List of tupels containing ids and features of each sample in the dataset.
    """
    # read in the data file
    features = []
    with open(data_file, "r") as in_data:
        for idx, line in enumerate(in_data):
            linelist = line.strip().split(separator)

            # the line will contain the sample ids, if it is the first line
            if idx == 0:
                for sample in linelist:
                    # remove possible unneccessary characters from the id strings
                    sample = sample.strip(" \"'")
                    features.append((sample, []))

            # each line after the first will contain the value of one feature
            else:
                for idx2, sample in enumerate(linelist):
                    # skip the first entry since it will contain the id of the feature
                    if idx2 == 0:
                        continue
                    features[idx2 - 1][1].append(float(sample))

    return features


def __load_omics_row_without_label(
    data_file: str, separator: str = ",",
):
    r"""Private function that loads an omics data file in row format.
    
    Parameters
    ----------
    data_file : str
        A string that contains the path to the file which stores the sample data.
    separator : str
        The string (most time this will be a single character) that separates the
        entries in the csv files.

    Returns
    -------
    features : list
        List of tupels containing ids and features of each sample in the dataset.
    """
    # read in the data file
    features = []
    with open(data_file, "r") as in_data:
        for idx, line in enumerate(in_data):
            linelist = line.strip().split(separator)

            # the line will contain the feature ids, if it is the first line
            if idx == 0:
                continue

            # each line after the first will contain id and features of one sample
            else:
                features.append(
                    (linelist[0], [float(feature) for feature in linelist[1:]])
                )

    return features


def __load_omics_labels(label_file: str, separator: str):
    r"""Private function that loads the labels of an (multi) omics data set.
    
    Parameters
    ----------
    label_file : str
        A string that contains the path to the file which stores the labels.
    separator : str
        The string (most time this will be a single character) that separates the
        entries in the csv files.

    Returns
    -------
    labels : list
        List of tupels containing ids and labels of each sample in the dataset.
    """
    # read in the labels file
    labels = []
    with open(label_file, "r") as in_label:
        for line in in_label:
            linelist = line.strip().split(separator)
            labels.append((linelist[0], linelist[1]))

    return labels


def load_omics_column(
    data_file: str, label_file: str, separator: str = ",",
):
    """Function to read in omics data. Data has to be provided
    with two files: a csv file with the features and a csv file with labels. The data file
    has to be in a format where the first line is a header containing all sample ids and
    each of the following lines correspond to one feature and contains the values that this
    feature takes for all samples. The labels file has to be in a format where each line
    contains one sample with the first entry being the sample's id and the second entry 
    being the sample's label. Both csv files have to have the same separator.
    
    Parameters
    ----------
    data_file : str
        A string that contains the path to the file which stores the sample data.
    label_file : str
        A string that contains the path to the file which stores the labels.
    separator : str
        The string (most time this will be a single character) that separates the
        entries in the csv files.

    Returns
    -------
    features : list
        List of tupels containing ids and features of each sample in the dataset.
    labels : list
        List of tupels containing ids and labels of each sample in the dataset.
    """
    # read in the data file
    features = __load_omics_column_without_label(data_file, separator)

    # read in the labels file
    labels = __load_omics_labels(label_file, separator)

    return features, labels


def load_omics_row(
    data_file: str, label_file: str, separator: str = ",",
):
    """Function to read in omics data. Data has to be provided
    with two files: a csv file with the features and a csv file with labels. The data file
    has to be in a format where the first line is a header containing the ids of each 
    feature and each of the following lines correspond to one samples with the first entry 
    of the line corresponding to the sample id. The labels file has to be in a format where
    each line contains one sample with the first entry being the sample's id and the second
    entry being the sample's label. Both csv files have to have the same separator.
    
    Parameters
    ----------
    data_file : str
        A string that contains the path to the file which stores the sample data.
    label_file : str
        A string that contains the path to the file which stores the labels.
    separator : str
        The string (most time this will be a single character) that separates the
        entries in the csv files.

    Returns
    -------
    features : list
        List of tupels containing ids and features of each sample in the dataset.
    labels : list
        List of tupels containing ids and labels of each sample in the dataset.
    """
    # read in the data file
    features = __load_omics_row_without_label(data_file, separator)

    # read in the labels file
    labels = __load_omics_labels(label_file, separator)

    return features, labels


def load_omics(
    data_file: str,
    label_file: str,
    omics_format: str,
    separator: str = ",",
    raise_unequal_sample_count: bool = False,
):
    r"""Helper function to load single omics data.
    
    Parameters
    ----------
    data_file : str
        A string that contains the path to the file which stores the sample data.
    label_file : str
        A string that contains the path to the file which stores the labels.
    omics_format : str
        A string that indicates the format of the omics data file.
    separator : str
        The string (most time this will be a single character) that separates the
        entries in the csv files.

    Returns
    -------
    features : list
        List of tupels containing ids and features of each sample in the dataset.
    labels : list
        List of tupels containing ids and labels of each sample in the dataset.
    """

    # read in the data and label files
    if omics_format == "column":
        features, labels = load_omics_column(data_file, label_file, separator)
    elif omics_format == "row":
        features, labels = load_omics_row(data_file, label_file, separator)
    else:
        raise ValueError(f"Unknown format of the omics data file: {omics_format}")

    # report an error if the number of samples in the data file does not
    # match the number of samples in the label file or delete all samples
    # that are only in the data file or in the label file
    if len(features) != len(labels):
        if raise_unequal_sample_count:
            raise ValueError(
                "Unequal number of samples in data and label file detected!\n"
                + f"    data file contains {len(features)} samples\n"
                + f"    label file contains {len(labels)} samples\n"
            )
        else:
            # determine the sample ids that are in both, the data file and
            # the label file
            features_ids = set([x[0] for x in features])
            labels_ids = set([x[0] for x in labels])
            common_ids = features_ids.intersection(labels_ids)

            # remove all samples that are not in both files
            features = [x for x in features if x[0] in common_ids]
            labels = [x for x in labels if x[0] in common_ids]

    # make sure that the data and labels are in the same order
    features.sort(key=lambda x: x[0])
    labels.sort(key=lambda x: x[0])

    return features, labels


def load_multi_omics(
    data_files: list,
    label_file: str,
    omics_format: list,
    separators: list,
    separator_label: str = ",",
    raise_unequal_sample_count: bool = True,
):
    r"""Helper function to load multi omics datasets.
    
    Parameters
    ----------
    data_files : list
        A list that contains the path to the files which store the sample data.
    label_file : str
        A string that contains the path to the file which stores the labels.
    omics_format : list
        A list that indicates the format of each of the omics data files.
    separators : list
        The list containing strings (most time this will be a single character) that 
        separate the entries in the csv files.
    separator_label : str
        The string (most time this will be a single character) that separates the
        entries in the labels file.
    raise_unequal_sample_count : bool
        If this flag is set to true, an exception is raised if data file and label
        file contain unequal number of samples. Otherwise, the class will include only
        samples with ids that occure in both, data file and label file.
        
    Returns
    -------
    features : list
        List of lists where each list corresponds to one omics data type and contains
        tupels representing id and features of all samples.
    labels : list
        List of tupels containing ids and labels of each sample in the dataset.
    """
    features = []
    labels = []

    # iterate over all data types and load the data
    for i, data_file in enumerate(data_files):
        # load the current data type
        if omics_format[i] == "column":
            features.append(__load_omics_column_without_label(data_file, separators[i]))
        elif omics_format[i] == "row":
            features.append(__load_omics_row_without_label(data_file, separators[i]))
        else:
            raise ValueError(
                f"Unknown format of the omics data file: {omics_format[i]}"
            )

    # load the labels
    labels = __load_omics_labels(label_file, separator_label)

    # get the number of samples for each data type
    dataset_len = [len(x) for x in features]

    # depending on the raise_unequal_sample_count flag, either raise an error if
    # the number of samples in each data type and the labels file do not match or
    # remove all samples that do not occur in every file
    if len(set(dataset_len)) > 1 or dataset_len[0] != len(labels):
        if raise_unequal_sample_count:
            raise ValueError(
                "Unequal number of samples in data files and label file detected!\n"
                + f"    data files contain {dataset_len} samples\n"
                + f"    label file contains {len(labels)} samples\n"
            )

        else:
            # determine the samples that are present in all data files and the label file
            feature_ids = []
            for feature in features:
                feature_ids.append(set([x[0] for x in feature]))
            label_ids = set([x[0] for x in labels])

            # find the common ids in all files
            common_ids = feature_ids[0]
            for ids in feature_ids:
                common_ids = common_ids.intersection(ids)
            common_ids = common_ids.intersection(label_ids)

            # remove all features that are not in all files
            aux_features = []
            for feature in features:
                aux_features.append([x for x in feature if x[0] in common_ids])
            features = aux_features
            labels = [x for x in labels if x[0] in common_ids]

    # make sure that the data and labels are in the same order
    for feature in features:
        feature.sort(key=lambda x: x[0])
    labels.sort(key=lambda x: x[0])

    return features, labels


class OmicsDataset(data.Dataset):
    r"""Custom Pytorch Dataset class that handles omics datasets.

    Attributes
    ----------
    data : list
        List containing all features
    labels : list
        List containing all labels
    nb_features : int
        Number of features each sample consists of
    nb_classes : int
        Number of classes, i.e. number of different labels, in the dataset. This number will
        be one for regression and can be one for binary classification.
    """

    def __init__(
        self,
        data_file: str,
        label_file: str,
        omics_format: str,
        separator: str = ",",
        num_classes: int = 1,
        class_labels=None,
        raise_unequal_sample_count: bool = True,
        permutation: bool = False,
    ):
        r"""Constructor of the OmicsDataset class

        Parameters
        ----------
        data_file : str
            A string that contains the path to the file which stores the sample data.
        label_file : str
            A string that contains the path to the file which stores the labels.
        omics_format : str
            A string that indicates the format of the omics data file.
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
        raise_unequal_sample_count : bool
            If this flag is set to true, an exception is raised if data file and label
            file contain unequal number of samples. Otherwise, the class will include only
            samples with ids that occure in both, data file and label file.
        """
        # load the features and labels
        features, labels = load_omics(
            data_file, label_file, omics_format, separator, raise_unequal_sample_count
        )

        # prepare the data and labels attributes of the class
        self.num_features = len(features[0][1])
        self.num_classes = num_classes
        self.sample_ids = []
        self.data = []
        self.labels = []
        for i in range(len(features)):
            self.sample_ids.append(features[i][0])
            self.data.append(features[i][1])
            self.labels.append(labels[i][1])

        # if a permutation test on the labels should be performed,
        # shuffle the labels
        if permutation:
            shuffle(self.labels)

        # prepare the label on index mapping
        # if the prediction problem is a regression, no class to index mapping is needed
        if num_classes == 1 and class_labels is None:
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
        r"""Overwritten __len__ method from the Dataset class. Returns the number
        of samples stored by the OmicsDataset object.
        """
        return len(self.sample_ids)

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
        features = torch.Tensor(self.num_features)
        features.data = torch.tensor(self.data[idx])

        # retrieve the requested label
        if self.class_to_idx == None:
            label = torch.Tensor(float(self.labels[idx]))
        elif isinstance(self.class_to_idx, dict):
            label = torch.zeros(self.num_classes)
            label[self.class_to_idx[self.labels[idx]]] = 1
        else:
            if self.labels[idx] == self.class_to_idx:
                label = torch.Tensor([1])
            else:
                label = torch.Tensor([0])

        return features, label


class MultiOmicsDataset(data.Dataset):
    r"""Custom Pytorch Dataset class that handles multi omics datasets.

    Attributes
    ----------
    data : list
        List of lists where each list represent one omics datatype and contains all features
        of this datatype.
    labels : list
        List containing all labels.
    nb_features : list
        Number of features of each data type. Every sample has to have the same amount of features.
    nb_classes : int
        Number of classes, i.e. number of different labels, in the dataset. This number will
        be one for regression and can be one for binary classification.
    """

    def __init__(
        self,
        data_files: list,
        label_file: str,
        omics_format: list,
        separators: str or list,
        num_classes: int = 1,
        class_labels=None,
        raise_unequal_sample_count: bool = True,
        permutation: bool = False,
    ):
        r"""Constructor of the MultiOmicsDataset class

        Parameters
        ----------
        data_files : list
            A list that contains the path to the files which store the sample data.
        label_file : str
            A string that contains the path to the file which stores the labels.
        omics_format : list
            A list that indicates the format of each of the omics data files.
        separators : str or list
            The list containing strings (most time this will be a single character) that 
            separate the entries in the csv files. If all files have the same separator,
            this separator can be given as a single string. If the label file uses the same
            separator as the first data file, not separator for the label file has to be given.
            Otherwise, the separator of the label fils has to be given at the end of the list.
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
        raise_unequal_sample_count : bool
            If this flag is set to true, an exception is raised if data file and label
            file contain unequal number of samples. Otherwise, the class will include only
            samples with ids that occure in both, data file and label file.
        """
        # perform sanity checks
        if len(data_files) != len(omics_format):
            raise ValueError(
                f"Unequal number of files and file formats given: {len(data_files)} files and {len(omics_format)} formats"
            )

        # determine the separators of the different files
        if isinstance(separators, str):
            separator_label = separators
            separators = [separators for i in range(len(data_files))]
        elif len(separators) == len(data_files):
            separator_label = separators[0]
        elif len(separators) == len(data_files) + 1:
            separator_label = separators[-1]
            separators = separators[:-1]
        else:
            raise ValueError(
                f"Unexpected number of separators: {len(separators)}\n"
                + f"    Valid numbers of separators are: 1, {len(data_files)}, or {len(data_files) + 1}"
            )

        # load data and labels
        features, labels = load_multi_omics(
            data_files,
            label_file,
            omics_format,
            separators,
            separator_label,
            raise_unequal_sample_count,
        )

        # store the number of omics data types and, for each data type,
        # store the number of features
        self.num_datatypes = len(data_files)
        self.num_features = []
        for feature in features:
            self.num_features.append(len(feature[0][1]))

        # store the number of classes
        self.num_classes = num_classes

        # prepare the data and label attributes of the class
        self.sample_ids = []
        self.data = [[] for _ in range(self.num_datatypes)]
        self.labels = []

        # iterate over each samples
        for i in range(len(features[0])):

            # store the id of the current sample
            self.sample_ids.append(features[0][i][0])

            # store the label of the current samples
            self.labels.append(labels[i][1])

            # iterate over each data type and store the features belonging to
            # the current samples
            for j, data_type in enumerate(features):
                self.data[j].append(data_type[i][1])

        # if a permutation test on the labels should be performed,
        # shuffle the labels
        if permutation:
            shuffle(self.labels)

        # prepare the label on index mapping
        # if the prediction problem is a regression, no class to index mapping is needed
        if num_classes == 1 and class_labels is None:
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
        r"""Overwritten __len__ method from the Dataset class. Returns the number
        of samples stored by the MultiOmicsDataset object.
        """
        return len(self.sample_ids)

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
            features : list
                List of tensors containing the feature values of the requested sample.
                Each tensor in the list represents one of the omics data types.
            label : Tensor
                Tensor containing the label of the requested sample
        """
        # retrieve the requested data
        features = [torch.Tensor(num_features) for num_features in self.num_features]

        for i in range(self.num_datatypes):
            features[i].data = torch.tensor(self.data[i][idx])

        # retrieve the requested label
        if self.class_to_idx == None:
            label = torch.Tensor(float(self.labels[idx]))
        elif isinstance(self.class_to_idx, dict):
            label = torch.zeros(self.num_classes)
            label[self.class_to_idx[self.labels[idx]]] = 1
        else:
            if self.labels[idx] == self.class_to_idx:
                label = torch.Tensor([1])
            else:
                label = torch.Tensor([0])

        return features, label
