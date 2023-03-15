import argparse
import os
import sys
from multiprocessing import pool

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from comic.data_utils import MultiOmicsDataset, OmicsDataset
from comic.models import GLKNet, GLPIMKLNet, MultiPIMKLNet, PIMKLNet
from comic.utils import ClassBalanceLoss, compute_metrics_classification


def load_args():
    """
    Function to create an argument parser
    """
    parser = argparse.ArgumentParser(description="Comik Experiment Skript")
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help=" Set the random seed."
    )
    parser.add_argument(
        "--type",
        dest="type",
        default="omics",
        type=str,
        choices=["omics", "multiomics", "debug"],
        help="Specify the type of experiment that should be performed.",
    )
    parser.add_argument(
        "--use-cuda",
        dest="use_cuda",
        action="store_true",
        default=False,
        help="Flag that determines if GPU resources are used for the experiment.",
    )
    parser.add_argument(
        "--repeats",
        dest="repeats",
        type=int,
        default=10,
        help="Set the number of repeats for the repeated cross-validation procedure.",
    )
    parser.add_argument(
        "--folds",
        dest="folds",
        type=int,
        default=10,
        help="Set the number of folds for the repeated cross-validation procedure.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=32,
        help="Input batch size for training and testing.",
    )
    parser.add_argument(
        "--epochs",
        dest="num_epochs",
        type=int,
        default=100,
        help="Number of epochs used to train models.",
    )
    parser.add_argument(
        "--network",
        dest="network",
        default="pimkl",
        type=str,
        choices=["pimkl", "adapimkl", "glpimkl", "laplacian", "multipimkl"],
        help="Select the typ of network architecture to be used for the experiment.",
    )
    parser.add_argument(
        "--anchors",
        dest="num_anchors",
        type=int,
        nargs="+",
        default=10,
        help="Number of anchor points of the kernel layer.",
    )
    parser.add_argument(
        "--laplacians",
        dest="laplacians",
        type=str,
        nargs="+",
        default=None,
        help="Complete path to the folder that contains the Laplacians used for the network.",
    )
    parser.add_argument(
        "--data",
        dest="data",
        type=str,
        nargs="+",
        required=True,
        help="Complete path to the file that stores the samples.",
    )
    parser.add_argument(
        "--labels",
        dest="labels",
        type=str,
        required=True,
        help="Complete path to the file that stores the labels.",
    )
    parser.add_argument(
        "--classes",
        dest="classes",
        default=None,
        nargs="+",
        type=str,
        help="Labels of the classes. Only provide label of positive class, if the prediction problem is binary.",
    )
    parser.add_argument(
        "--format",
        dest="omics_format",
        default="column",
        type=str,
        nargs="+",
        help="Format of the omics data file. See documentation of OmicsDataset object for details.",
    )
    parser.add_argument(
        "--outdir",
        dest="outdir",
        default="./output",
        type=str,
        help="Directory where outputs will be stored.",
    )
    parser.add_argument(
        "--loss-beta",
        dest="loss_beta",
        default=0.99,
        type=float,
        help="Beta parameter of the ClassBalanceLoss class",
    )
    parser.add_argument(
        "--loss-gamma",
        dest="loss_gamma",
        default=1.0,
        type=float,
        help="Gamma parameter of the ClassBalanceLoss class",
    )
    parser.add_argument(
        "--pooling",
        dest="pooling",
        action="store_true",
        default=False,
        help="Flag that determines if a pooling layer after the kernel layer will be used",
    )
    parser.add_argument(
        "--attention",
        dest="attention",
        default="normal",
        type=str,
        choices=["none", "normal", "gated"],
        help="Type of attention layer used in the network",
    )
    parser.add_argument(
        "--attention-params",
        dest="attention_params",
        default=[8, 1],
        type=int,
        nargs="+",
        help="Parameters of the attention layer. Two integer values have to be provided. See documentation for details.",
    )

    # parse the arguments
    args = parser.parse_args()

    # GPU will only be used if available
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    # set the random seeds
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # preprocess the parsed argument depending on the type of experiment
    if args.type == "omics":
        # make sure that certain arguments are not given as a list
        args.num_anchors = args.num_anchors[0]
        args.laplacians = args.laplacians[0]
        args.data = args.data[0]
        args.omics_format = args.omics_format[0]

    # determine the number of output states of the network, aka. the number of classes
    #   -> 1 for regression AND binary classification
    if args.classes == None:
        args.num_classes = 1
    else:
        args.num_classes = len(args.classes)

    # make sure that laplacians are provided if an architecure based on PIMKL is selected
    if args.network in ["pimkl", "adapimkl"] and args.laplacians is None:
        raise ValueError(
            "Please provide Laplacians for PIMKL-based network architectures."
        )

    # store the name of the used dataset
    if args.type == "omics":
        args.dataset = args.data.split(os.path.sep)[-1]
        args.dataset = args.dataset.split(".")[0]
    elif args.type == "multiomics":
        args.dataset = "MultiOmics"

    # if an output directory is specified, create the dir structure to store the output of the current run
    args.save_logs = False
    if args.outdir != "":
        args.save_logs = True

        # make sure that the outdir path does not end with a seperator character
        args.outdir = args.outdir.rstrip(os.path.sep)

        if not os.path.exists(args.outdir):
            try:
                os.makedirs(args.outdir)
            except:
                pass

    return args


def read_in_laplacians(dirpath: str, header=None, sep: str = ","):
    """Function to read in predefined Laplacians."""
    # encode directory path in byte
    directory = os.fsencode(dirpath)

    # iterate through all files in the specified folder
    pi_laplacians = []
    for file in os.listdir(directory):

        # decode the filepath from bytes to characters
        filename = os.fsdecode(os.path.join(directory, file))

        # read in the laplacian with pandas
        laplacian = pd.read_csv(filename, header=header, sep=sep)
        laplacian.columns = ["row", "column", "data"]

        # remove all the zero rows
        laplacian = laplacian.loc[laplacian["data"] != 0]

        # get the nodes that are included in the current laplacian
        node_ids = np.array(laplacian["row"].unique())

        # create the laplacian that contains only the relevant nodes
        l_matrix = np.zeros((len(node_ids), len(node_ids)))
        for _, row in laplacian.iterrows():
            l_matrix[
                np.where(node_ids == row["row"]), np.where(node_ids == row["column"])
            ] = row["data"]

        pi_laplacians.append((node_ids, l_matrix))

    return pi_laplacians


def omics_experiment(args):
    """Train a model on a single omics dataset."""

    # load omics data
    data_all = OmicsDataset(
        data_file=args.data,
        label_file=args.labels,
        omics_format=args.omics_format,
        class_labels=args.classes,
        raise_unequal_sample_count=False,
    )

    # get the laplacians
    pi_laplacians = read_in_laplacians(args.laplacians)

    if args.repeats > 1 and args.folds > 1:
        # create repeated stratified cross-validation object
        rskf = RepeatedStratifiedKFold(
            n_repeats=args.repeats, n_splits=args.folds, random_state=args.seed
        )

        # iterate over repeats and folds
        all_performance = []
        mean_performance = []
        for fold, split_idx in enumerate(rskf.split(data_all.data, data_all.labels)):

            # create the model
            if args.network == "laplacian":
                model = GLKNet(
                    num_classes=args.num_classes,
                    num_features=data_all.num_features,
                    num_anchors=args.num_anchors,
                    graph_learner="glasso",
                )
            elif args.network == "pimkl":
                model = PIMKLNet(
                    num_features=data_all.num_features,
                    num_classes=args.num_classes,
                    num_anchors=args.num_anchors,
                    pi_laplacians=pi_laplacians,
                    pooling=args.pooling,
                    attention=args.attention,
                    attention_params=args.attention_params,
                )
            elif args.network == "glpimkl":
                model = GLPIMKLNet(
                    num_features=data_all.num_features,
                    num_classes=args.num_classes,
                    num_anchors=args.num_anchors,
                    pi_laplacians=pi_laplacians,
                    graph_learner="kalofolias",
                )
            else:
                raise NotImplementedError(
                    f"Network archtecture {args.network} is not implemented."
                )

            # split dataset into training and validation samples
            args.train_indices, args.val_indices = split_idx[0], split_idx[1]

            # set arguments for the DataLoader
            loader_args = {}
            if args.use_cuda:
                loader_args = {"num_workers": 1, "pin_memory": True}

            # Creating PyTorch data Subsets using the indices for the current fold
            data_train = Subset(data_all, args.train_indices)
            data_val = Subset(data_all, args.val_indices)

            # create PyTorch DataLoader for training and validation data
            loader_train = DataLoader(
                data_train, batch_size=args.batch_size, shuffle=False, **loader_args
            )
            loader_val = DataLoader(
                data_val, batch_size=args.batch_size, shuffle=False, **loader_args
            )

            # initialize loss function, optimizer, and learning rate scheduler
            args.class_count = [
                sum(
                    [
                        1 if data_all.labels[i] != args.classes[0] else 0
                        for i in args.train_indices
                    ]
                ),
                sum(
                    [
                        1 if data_all.labels[i] == args.classes[0] else 0
                        for i in args.train_indices
                    ]
                ),
            ]
            criterion = ClassBalanceLoss(
                args.class_count,
                args.num_classes,
                "sigmoid",
                args.loss_beta,
                args.loss_gamma,
            )
            # criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-6)
            lr_scheduler = ReduceLROnPlateau(
                optimizer, factor=0.5, patience=4, min_lr=1e-4
            )

            # train the model
            acc, loss = model.sup_train(
                train_loader=loader_train,
                criterion=criterion,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epochs=args.num_epochs,
            )

            # store validation performance for each fold of the current repeat
            if fold % args.folds == 0:
                fold_res = []

            # predict validation data
            y_pred, y_true = model.predict(loader_val, True)
            fold_res.append(compute_metrics_classification(y_true, y_pred))
            print(fold_res[-1], "\n")

            torch.save(
                {
                    "args": args,
                    "state_dict": model.state_dict(),
                    "acc": acc,
                    "loss": loss,
                },
                args.outdir
                + f"{os.path.sep}{args.dataset}_{args.network}_{int((fold/args.folds)%args.repeats)}_{fold%args.folds}.pkl",
            )

            # calculate mean result for the current repeat
            if fold % args.folds == args.folds - 1:
                # concatenate all fold splits into one dataframe
                df_res = pd.concat(fold_res, axis=1)
                all_performance.append(df_res)
                mean_performance.append(df_res.mean(axis=1))

        # concatenate the mean performance of each repeat onto one dataframe
        # and store it
        df_all = pd.concat(all_performance, axis=1)
        df_mean = pd.concat(mean_performance, axis=1)
        df_all.to_csv(
            args.outdir
            + f"{os.path.sep}{args.dataset}_{args.network}_{args.attention}_ {args.num_anchors}_all.csv"
        )
        df_mean.to_csv(
            args.outdir
            + f"{os.path.sep}{args.dataset}_{args.network}_{args.attention}_ {args.num_anchors}_mean.csv"
        )
        print(df_mean)

    # if either folds or repeats are set to 1, train model on whole dataset
    # and store the model
    else:
        # create the model
        if args.network == "laplacian":
            model = GLKNet(
                num_classes=args.num_classes,
                num_features=data_all.num_features,
                num_anchors=args.num_anchors,
                graph_learner="glasso",
            )
        elif args.network == "pimkl":
            model = PIMKLNet(
                num_features=data_all.num_features,
                num_classes=args.num_classes,
                num_anchors=args.num_anchors,
                pi_laplacians=pi_laplacians,
                pooling=args.pooling,
                attention=args.attention,
                attention_params=args.attention_params,
            )
        elif args.network == "glpimkl":
            model = GLPIMKLNet(
                num_features=data_all.num_features,
                num_classes=args.num_classes,
                num_anchors=args.num_anchors,
                pi_laplacians=pi_laplacians,
                graph_learner="kalofolias",
            )
        else:
            raise NotImplementedError(
                f"Network archtecture {args.network} is not implemented."
            )

        # set arguments for the DataLoader
        loader_args = {}
        if args.use_cuda:
            loader_args = {"num_workers": 1, "pin_memory": True}

        # create PyTorch DataLoader for training and validation data
        if args.folds == 1:
            loader_train = DataLoader(
                data_all, batch_size=args.batch_size, shuffle=True, **loader_args
            )
            loader_val = None

            # calculate class counts
            args.class_count = [
                sum([1 if i != args.classes[0] else 0 for i in data_all.labels]),
                sum([1 if i == args.classes[0] else 0 for i in data_all.labels]),
            ]

        else:
            args.train_indices, args.val_indices = train_test_split(
                np.arange(len(data_all.data)),
                stratify=np.array(
                    [1 if i == args.classes else 0 for i in data_all.labels]
                ),
            )

            # Creating PyTorch data Subsets using the indices for the current fold
            data_train = Subset(data_all, args.train_indices)
            data_val = Subset(data_all, args.val_indices)

            # create PyTorch DataLoader for training and validation data
            loader_train = DataLoader(
                data_train, batch_size=args.batch_size, shuffle=False, **loader_args
            )
            loader_val = DataLoader(
                data_val, batch_size=args.batch_size, shuffle=False, **loader_args
            )

            # calculate class counts
            args.class_count = [
                sum(
                    [
                        1 if data_all.labels[i] != args.classes[0] else 0
                        for i in args.train_indices
                    ]
                ),
                sum(
                    [
                        1 if data_all.labels[i] == args.classes[0] else 0
                        for i in args.train_indices
                    ]
                ),
            ]

        # initialize loss function, optimizer, and learning rate scheduler
        criterion = ClassBalanceLoss(
            args.class_count,
            args.num_classes,
            "sigmoid",
            args.loss_beta,
            args.loss_gamma,
        )
        optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-6)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-4)

        # train the model
        acc, loss = model.sup_train(
            train_loader=loader_train,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epochs=args.num_epochs,
            val_loader=loader_val,
        )

        # store the trained model
        torch.save(
            {"args": args, "state_dict": model.state_dict(), "acc": acc, "loss": loss},
            args.outdir
            + f"{os.path.sep}{args.dataset}_{args.network}_{args.attention}.pkl",
        )

        # predict validation data
        if loader_val is not None:
            y_pred, y_true = model.predict(loader_val, True)
            print(y_pred, y_true)
            print(compute_metrics_classification(y_true, y_pred))

        # plot acc and loss
        viz = False
        if viz:
            try:
                # try to import pyplot
                import matplotlib.pyplot as plt

                # show the evolution of the acc and loss
                fig2, axs2 = plt.subplots(2, 2)
                fig2.set_size_inches(w=20, h=10)
                axs2[0, 0].plot(acc["train"])
                axs2[0, 0].set_title("train accuracy")
                axs2[0, 0].set(xlabel="epoch", ylabel="accuracy")
                axs2[0, 1].plot(acc["val"])
                axs2[0, 1].set_title("val accuracy")
                axs2[0, 1].set(xlabel="epoch", ylabel="accuracy")
                axs2[1, 0].plot(loss["train"])
                axs2[1, 0].set_title("train loss")
                axs2[1, 0].set(xlabel="epoch", ylabel="loss")
                axs2[1, 1].plot(loss["val"])
                axs2[1, 1].set_title("val loss")
                axs2[1, 1].set(xlabel="epoch", ylabel="loss")
                plt.show()

            except ImportError:
                print("Cannot import matplotlib.pyplot")

            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise


def multiomics_experiment(args):
    """Train a network on a multi omics dataset."""
    # read in data
    data_all = MultiOmicsDataset(
        data_files=args.data,
        label_file=args.labels,
        omics_format=args.omics_format,
        class_labels=args.classes,
        raise_unequal_sample_count=False,
    )

    # read in the laplacians for each data type
    pi_laplacians = []
    for l_path in args.laplacians:
        pi_laplacians.append(read_in_laplacians(l_path))

    if args.repeats > 1 and args.folds > 1:
        # create repeated stratified cross-validation object
        rskf = RepeatedStratifiedKFold(
            n_repeats=args.repeats, n_splits=args.folds, random_state=args.seed
        )

        # iterate over repeats and folds
        all_performance = []
        mean_performance = []
        for fold, split_idx in enumerate(rskf.split(data_all.data[0], data_all.labels)):

            # create the model
            if args.network == "multipimkl":
                model = MultiPIMKLNet(
                    num_classes=args.num_classes,
                    num_features=data_all.num_features,
                    num_anchors=args.num_anchors,
                    pi_laplacians=pi_laplacians,
                    pooling=args.pooling,
                    attention=args.attention,
                    attention_params=args.attention_params,
                )
            else:
                raise NotImplementedError(
                    f"Network archtecture {args.network} is not implemented for multi omics experiments."
                )

            # split dataset into training and validation samples
            args.train_indices, args.val_indices = split_idx[0], split_idx[1]

            # set arguments for the DataLoader
            loader_args = {}
            if args.use_cuda:
                loader_args = {"num_workers": 1, "pin_memory": True}

            # Creating PyTorch data Subsets using the indices for the current fold
            data_train = Subset(data_all, args.train_indices)
            data_val = Subset(data_all, args.val_indices)

            # create PyTorch DataLoader for training and validation data
            loader_train = DataLoader(
                data_train, batch_size=args.batch_size, shuffle=False, **loader_args
            )
            loader_val = DataLoader(
                data_val, batch_size=args.batch_size, shuffle=False, **loader_args
            )

            # initialize loss function, optimizer, and learning rate scheduler
            args.class_count = [
                sum(
                    [
                        1 if data_all.labels[i] != args.classes[0] else 0
                        for i in args.train_indices
                    ]
                ),
                sum(
                    [
                        1 if data_all.labels[i] == args.classes[0] else 0
                        for i in args.train_indices
                    ]
                ),
            ]
            criterion = ClassBalanceLoss(
                args.class_count,
                args.num_classes,
                "sigmoid",
                args.loss_beta,
                args.loss_gamma,
            )
            # criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-6)
            lr_scheduler = ReduceLROnPlateau(
                optimizer, factor=0.5, patience=4, min_lr=1e-4
            )

            # train the model
            try:
                acc, loss = model.sup_train(
                    train_loader=loader_train,
                    criterion=criterion,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    epochs=args.num_epochs,
                )
            except Exception as err:
                print(f"Error during training:\n    {err}")
                if fold % args.folds == 0:
                    fold_res = []
                continue

            # store validation performance for each fold of the current repeat
            if fold % args.folds == 0:
                fold_res = []

            # predict validation data
            try:
                y_pred, y_true = model.predict(loader_val, True)
                fold_res.append(compute_metrics_classification(y_true, y_pred))
                print(fold_res[-1], "\n")
            except Exception as err:
                print(f"Error during prediction:\n    {err}")
                continue

            torch.save(
                {
                    "args": args,
                    "state_dict": model.state_dict(),
                    "acc": acc,
                    "loss": loss,
                },
                args.outdir
                + f"{os.path.sep}{args.dataset}_{args.network}_{int((fold/args.folds)%args.repeats)}_{fold%args.folds}.pkl",
            )

            # calculate mean result for the current repeat
            if fold % args.folds == args.folds - 1:
                # concatenate all fold splits into one dataframe
                df_res = pd.concat(fold_res, axis=1)
                all_performance.append(df_res)
                mean_performance.append(df_res.mean(axis=1))

        # concatenate the mean performance of each repeat onto one dataframe
        # and store it
        df_all = pd.concat(all_performance, axis=1)
        df_mean = pd.concat(mean_performance, axis=1)
        df_all.to_csv(
            args.outdir + f"{os.path.sep}{args.dataset}_{args.network}_all.csv"
        )
        df_mean.to_csv(
            args.outdir + f"{os.path.sep}{args.dataset}_{args.network}_mean.csv"
        )
        print(df_mean)

    # if either folds or repeats are set to 1, train model on whole dataset
    # and store the model
    else:
        # create the model
        if args.network == "multipimkl":
            model = MultiPIMKLNet(
                num_classes=args.num_classes,
                num_features=data_all.num_features,
                num_anchors=args.num_anchors,
                pi_laplacians=pi_laplacians,
                pooling=args.pooling,
                attention=args.attention,
                attention_params=args.attention_params,
            )
        else:
            raise NotImplementedError(
                f"Network archtecture {args.network} is not implemented for multi omics experiments."
            )

        # set arguments for the DataLoader
        loader_args = {}
        if args.use_cuda:
            loader_args = {"num_workers": 1, "pin_memory": True}

        # create PyTorch DataLoader for training and validation data
        if args.folds == 1:
            loader_train = DataLoader(
                data_all, batch_size=args.batch_size, shuffle=True, **loader_args
            )
            loader_val = None
        else:
            args.train_indices, args.val_indices = train_test_split(
                np.arange(len(data_all.data[0])),
                stratify=np.array(
                    [1 if i == args.classes else 0 for i in data_all.labels]
                ),
            )

            # Creating PyTorch data Subsets using the indices for the current fold
            data_train = Subset(data_all, args.train_indices)
            data_val = Subset(data_all, args.val_indices)

            # create PyTorch DataLoader for training and validation data
            loader_train = DataLoader(
                data_train, batch_size=args.batch_size, shuffle=False, **loader_args
            )
            loader_val = DataLoader(
                data_val, batch_size=args.batch_size, shuffle=False, **loader_args
            )

        # initialize loss function, optimizer, and learning rate scheduler
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-6)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-4)

        # train the model
        acc, loss = model.sup_train(
            train_loader=loader_train,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epochs=args.num_epochs,
            val_loader=loader_val,
        )

        # store the trained model
        torch.save(
            {"args": args, "state_dict": model.state_dict(), "acc": acc, "loss": loss},
            args.outdir + f"{os.path.sep}{args.dataset}_{args.network}.pkl",
        )

        # predict validation data
        y_pred, y_true = model.predict(loader_val)
        print(y_pred, y_true)
        print(compute_metrics_classification(y_true, y_pred))

        # plot acc and loss
        viz = True
        if viz:
            try:
                # try to import pyplot
                import matplotlib.pyplot as plt

                # show the evolution of the acc and loss
                fig2, axs2 = plt.subplots(2, 2)
                fig2.set_size_inches(w=20, h=10)
                axs2[0, 0].plot(acc["train"])
                axs2[0, 0].set_title("train accuracy")
                axs2[0, 0].set(xlabel="epoch", ylabel="accuracy")
                axs2[0, 1].plot(acc["val"])
                axs2[0, 1].set_title("val accuracy")
                axs2[0, 1].set(xlabel="epoch", ylabel="accuracy")
                axs2[1, 0].plot(loss["train"])
                axs2[1, 0].set_title("train loss")
                axs2[1, 0].set(xlabel="epoch", ylabel="loss")
                axs2[1, 1].plot(loss["val"])
                axs2[1, 1].set_title("val loss")
                axs2[1, 1].set(xlabel="epoch", ylabel="loss")
                plt.show()

            except ImportError:
                print("Cannot import matplotlib.pyplot")

            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise


def debugging(args):
    """This function is solely for debugging models"""

    # define the function for the forward hooks
    def printnorm(self, input, output):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        print("\n-------------------------------------------------------------------")
        print("Inside " + self.__class__.__name__ + " forward\n")
        print("    input: ", type(input))
        print("    input[0]: ", type(input[0]))
        print("    output: ", type(output))
        print("")
        if isinstance(input[0], torch.Tensor):
            print("    input size:", input[0].size())
        print("    output size:", output.data.size())
        print("    output norm:", output.data.norm())
        print("    output NaNs:", torch.isnan(output).sum().item())
        if self.__class__.__name__ == "Linear":
            print("    output: ", output)
            print("    output (sigmoid): ", output.sigmoid())
        print("-------------------------------------------------------------------")

    # define the function for the backward hooks
    def printgradnorm(self, grad_input, grad_output):
        print("\n-------------------------------------------------------------------")
        print("Inside " + self.__class__.__name__ + " backward")
        print("Inside class:" + self.__class__.__name__)
        print("")
        print("    grad_input: ", type(grad_input))
        print("    grad_input[0]: ", type(grad_input[0]))
        print("    grad_output: ", type(grad_output))
        print("    grad_output[0]: ", type(grad_output[0]))
        print("")
        if isinstance(grad_input[0], torch.Tensor):
            print("    grad_input size:", grad_input[0].size())
            print("    grad_input norm:", grad_input[0].norm())
        elif isinstance(grad_input, torch.Tensor):
            print("    grad_input size:", grad_input.size())
            print("    grad_input norm:", grad_input.norm())
        else:
            print("    grad_input:", grad_input)
        print("    grad_output size:", grad_output[0].size())
        print("-------------------------------------------------------------------")

    print(
        "\nATTENTION\nYou started debugging mode. Only use this if you "
        "are familiar with PyTorch\nand make sure that you adapted the "
        "debugging function accordingly.\n"
    )
    # create dataset object
    data_type = "multiomics"
    if data_type == "omics":
        # make sure that certain arguments are not given as a list
        args.num_anchors = args.num_anchors[0]
        args.laplacians = args.laplacians[0]
        args.data = args.data[0]
        args.omics_format = args.omics_format[0]

        # store the dataset name
        args.dataset = args.data.split(os.path.sep)[-1]
        args.dataset = args.dataset.split(".")[0]

        data_all = OmicsDataset(
            data_file=args.data,
            label_file=args.labels,
            omics_format=args.omics_format,
            class_labels=args.classes,
            raise_unequal_sample_count=False,
        )

        # get the laplacians
        pi_laplacians = read_in_laplacians(args.laplacians)

        # create training and validation split
        args.train_indices, args.val_indices = train_test_split(
            np.arange(len(data_all.data)),
            stratify=np.array([1 if i == args.classes else 0 for i in data_all.labels]),
        )

        # create the model
        if args.network == "laplacian":
            model = GLKNet(
                num_classes=args.num_classes,
                num_features=data_all.num_features,
                num_anchors=args.num_anchors,
                graph_learner="glasso",
            )
        elif args.network == "pimkl":
            print(args.dataset)
            model = PIMKLNet(
                num_features=data_all.num_features,
                num_classes=args.num_classes,
                num_anchors=args.num_anchors,
                pi_laplacians=pi_laplacians,
                pooling=False,
                attention="normal",
            )
        elif args.network == "glpimkl":
            model = GLPIMKLNet(
                num_features=data_all.num_features,
                num_classes=args.num_classes,
                num_anchors=args.num_anchors,
                pi_laplacians=pi_laplacians,
                graph_learner="kalofolias",
            )
        else:
            raise NotImplementedError(
                f"Network archtecture {args.network} is not implemented."
            )

    elif data_type == "multiomics":
        args.dataset = "MultiOmics"

        data_all = MultiOmicsDataset(
            data_files=args.data,
            label_file=args.labels,
            omics_format=args.omics_format,
            class_labels=args.classes,
            raise_unequal_sample_count=False,
        )

        # read in the laplacians for each data type
        pi_laplacians = []
        for l_path in args.laplacians:
            pi_laplacians.append(read_in_laplacians(l_path))

        # create training and validation split
        args.train_indices, args.val_indices = train_test_split(
            np.arange(len(data_all.data[0])),
            stratify=np.array([1 if i == args.classes else 0 for i in data_all.labels]),
        )

        # create the model
        if args.network == "multipimkl":
            model = MultiPIMKLNet(
                num_classes=args.num_classes,
                num_features=data_all.num_features,
                num_anchors=args.num_anchors,
                pi_laplacians=pi_laplacians,
                pooling=args.pooling,
                attention=args.attention,
                attention_params=args.attention_params,
            )
        else:
            raise NotImplementedError(
                f"Network archtecture {args.network} is not implemented for multi omics experiments."
            )

    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    # set the debug_split flag to True, if you want to get deeper insight into the splitting routine
    debug_split = False
    if debug_split and data_type == "omics":
        args.repeats = 10
        args.folds = 10

        # create repeated stratified cross-validation object
        rskf = RepeatedStratifiedKFold(
            n_repeats=args.repeats, n_splits=args.folds, random_state=args.seed
        )

        # iterate over repeats and folds
        for fold, split_idx in enumerate(rskf.split(data_all.data, data_all.labels)):
            val_idx = split_idx[1]
            train_idx = split_idx[0]
            true_vals_in_split = [
                1 if data_all.labels[i] == args.classes[0] else 0 for i in val_idx
            ]
            true_trains_in_split = [
                1 if data_all.labels[i] == args.classes[0] else 0 for i in train_idx
            ]
            print(f"\nFold {fold}")
            print(
                "Positive samples (train/val): ",
                sum(true_trains_in_split),
                sum(true_vals_in_split),
            )
            print(
                "Negative samples (train/vals): ",
                len(true_trains_in_split) - sum(true_trains_in_split),
                len(true_vals_in_split) - sum(true_vals_in_split),
            )

        return

    # create loader, and other needed objects
    data_train = Subset(data_all, args.train_indices)
    data_val = Subset(data_all, args.val_indices)
    loader_args = {}
    if args.use_cuda:
        loader_args = {"num_workers": 1, "pin_memory": True}
    loader_train = DataLoader(
        data_train, batch_size=args.batch_size, shuffle=False, **loader_args
    )
    loader_val = DataLoader(
        data_val, batch_size=args.batch_size, shuffle=False, **loader_args
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-6)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-4)

    # train the model
    model.sup_train(
        train_loader=loader_train,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.num_epochs,
    )

    # iterate through all layers and register a forward and backward hook
    for name, layer in model.named_modules():
        print("register hooks for layer {}".format(name))
        layer.register_forward_hook(printnorm)
        layer.register_full_backward_hook(printgradnorm)

    # create a single batch of validation data
    valiter = iter(loader_val)
    val_d, val_l, *_ = valiter.next()

    # make a forward and backward pass to trigger the hooks
    out = model(val_d)
    loss = criterion(out, val_l)
    loss.backward()

    print("\n")
    print("pred class: ", out.sigmoid().flatten() > 0.5)
    print("true class: ", val_l.flatten() > 0.5)
    print("\n")

    # predict on validation data
    y_pred, y_true = model.predict(loader_val, True)
    print(compute_metrics_classification(y_true, y_pred))


def main():
    """Main function of the experiment script."""
    args = load_args()

    if args.type == "omics":
        omics_experiment(args)
    elif args.type == "multiomics":
        multiomics_experiment(args)
    elif args.type == "debug":
        debugging(args)


if __name__ == "__main__":
    main()
