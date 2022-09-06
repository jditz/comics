r"""This module implements example networks that provide
different functionality and can be used as ispiration
for new networks utilizing the comik capabilities.
"""

import copy
from timeit import default_timer as timer

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .layers import (
    AttentionLayer,
    GatedAttentionLayer,
    GLKLayer,
    GLPIMKLLayer,
    PIMKLLayer,
)
from .utils import category_from_output, sample_data, sample_data_multiomics


class GLKNet(nn.Module):
    r"""Examplary network that uses the graph learning framework to learn a single
    Laplacian from the anchor points.
    """

    def __init__(
        self,
        num_classes: int,
        num_features: int,
        num_anchors: int,
        graph_learner: str = "kalofolias",
        gl_params: dict = None,
    ):
        r"""Constructor of the GLKNet class

        Parameters
        ----------
        num_classes : int
            Number of classes, i.e. number of output states of the network.
        num_features : int
            Number of nodes in the graph, i.e the dimensionality of the inputs.
        num_anchors : int
            Number of anchor points of the layer.
        graph_learner : str
            Framework that will be used to learn a graph from the optimized
            anchor points. The framework is called after every optimization step,
            i.e. everytime the anchor points changed.
        gl_params : dict
            Dictionary to provide parameter settings for the graph learning framework.
            Available parameters for each framework can be found in the corresponding
            doc strings. If no dictionary is provided, the default parameters will be
            used. The dictionary has to be of the form {'param_name': param_value}.
        """
        super().__init__()
        self.n_classes = num_classes
        self.n_features = num_features
        self.n_anchors = num_anchors

        self.kernel = GLKLayer(num_features, num_anchors, graph_learner, gl_params)
        self.fc = nn.Linear(in_features=num_anchors, out_features=num_classes)

    def forward(self, x_in, proba=False):
        r"""Implementation of the forward pass through the network.

        Parameters
        ----------
        x_in : Tensor
            Input to the model given as a Tensor of size
            (batch_size x self.in_channels x seq_len)
        proba : bool
            Indicates whether the network should produce probabilistic output
        
        Returns
        -------
            Result of a forward pass through the model using the specified input. 
            If 'proba' is set to True, the output will be the probabilities assigned
            to each class by the model.
        """
        x_out = self.kernel(x_in)
        x_out = self.fc(x_out)
        if proba:
            if self.n_classes == 1:
                return x_out.sigmoid()
            else:
                return F.softmax(x_out, dim=1)
        else:
            return x_out

    def init_params(self, data_loader, n_samples, kmeans_init="k-means++"):
        r"""Initialization routine of the model's parameters.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader object (initialized over a comik.OmicsDataset object)
            that is used to access the data.
        n_samples : int
            Number of data points that will be sampled from the data set. These
            samples are used in the clustering algorithm for the initialization
            of the anchor points.
        kmeans_init : str
            This parameter specifies the initialization routine used by the 
            kmeans algorithm.
        """
        # get the data to initialize the weights
        data = sample_data(data_loader, self.n_features, n_samples)

        # initialize the weights
        self.kernel.init_params(random_init=False, data=data, init=kmeans_init)

    def sup_train(
        self,
        train_loader,
        criterion,
        optimizer,
        lr_scheduler=None,
        init_train_loader=None,
        n_samples=100000,
        epochs=100,
        val_loader=None,
        use_cuda=False,
        early_stop=False,
    ):
        r"""Perform supervised training of the model

        This function will first initialize all convolutional layers of the model.
        Afterwards, a normal training routine for ANNs follows. If validation data
        is given, the performance on the validation data is calculate after each
        epoch.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            PyTorch DataLoader that handles training data.
        criterion : PyTorch Loss Object
            Specifies the loss function.
        optimizer : PyTorch Optimizer
            Optimization algorithm used during training.
        lr_scheduler : PyTorch LR Scheduler
            Algorithm used for learning rate adjustments.
        init_train_loader : torch.utils.data.DataLoader
            This DataLoader can be set if different datasets should be
            used for initializing the convolutional motif kernel layer 
            of this model and training the model. Data accessed by this
            DataLoader will be used for initialization of CMK layers.
        n_samples : int
            Number of motifs that will be sampled to initialize anchor points
            using the k-Means algorithm.
        epochs : int
            Number of epochs during training.
        val_loader : torch.utils.data.DataLoader
            PyTorch DataLoader that handles data during the validation phase.
        use_cuda : bool
            Specified whether all computations will be performed on the GPU.
        early_stop : bool
            Specifies if early stopping will be used during training.

        Returns
        -------
        list_acc : list
            List containing the accuracy on the train (and validation) data
            after each epoch.
        list_loss : list
            List containing the loss on the train (and validation) data
            after each epoch.
        """
        # initialize model
        print("Initializing network layers...")
        tic = timer()
        if init_train_loader is None:
            self.init_params(train_loader, n_samples)
        else:
            self.init_params(init_train_loader, n_samples)
        toc = timer()
        print("Finished, elapsed time: {:.2f}min\n".format((toc - tic) / 60))

        # specify the data used for each phase
        #   -> ATTENTION: a validation phase only exists if val_loader is not None
        phases = ["train"]
        data_loader = {"train": train_loader}
        if val_loader is not None:
            phases.append("val")
            data_loader["val"] = val_loader

        # initialize variables to keep track of the epoch's loss, the best loss, and the best accuracy
        epoch_loss = None
        best_epoch = 0
        best_loss = float("inf")
        best_acc = 0

        # iterate over all epochs
        list_acc = {"train": [], "val": []}
        list_loss = {"train": [], "val": []}
        for epoch in range(epochs):
            tic = timer()
            print("Epoch {}/{}".format(epoch + 1, epochs))
            print("-" * 10)

            # set the models train mode to False
            self.train(False)

            # iterate over all phases of the training process
            for phase in phases:

                # if the current phase is 'train', set model's train mode to True and initialize the Learning Rate
                # Scheduler (if one was specified)
                if phase == "train":
                    if lr_scheduler is not None:

                        # if the learning rate scheduler is 'ReduceLROnPlateau' and there is a current loss, the next
                        # lr step needs the current loss as input
                        if isinstance(lr_scheduler, ReduceLROnPlateau):
                            if epoch_loss is not None:
                                lr_scheduler.step(epoch_loss)

                        # otherwise call the step() function of the learning rate scheduler
                        else:
                            lr_scheduler.step()

                        # print the current learning rate
                        print("current LR: {}".format(optimizer.param_groups[0]["lr"]))

                    # set model's train mode to True
                    self.train(True)

                # if the current phase is not 'train', set the model's train mode to False. In this case, the learning
                # rate is irrelevant.
                else:
                    self.train(False)

                # initialize variables to keep track of the loss and the number of correctly classified samples in the
                # current epoch
                running_loss = 0.0
                running_corrects = 0

                # iterate over dataset
                for data, label, *_ in data_loader[phase]:
                    b_size = data.size(0)

                    # if computation should take place on the GPU, send everything to GPU
                    if use_cuda:
                        data = data.cuda()
                        label = label.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward propagation through the model
                    #   -> do not keep track of the gradients if we are in the validation phase
                    if phase == "val":
                        with torch.no_grad():
                            output = self(data)

                            # create prediction tensor
                            pred = label.new_zeros(output.shape)
                            for i in range(output.shape[0]):
                                if self.n_classes >= 2:
                                    pred[i, category_from_output(output[i, :])] = 1
                                else:
                                    pred[i] = output[i] > 0.5

                            # multiclass prediction needs special call of loss function
                            if self.n_classes >= 2:
                                loss = criterion(output, label.argmax(1))
                            else:
                                loss = criterion(output, label)
                    else:
                        output = self(data)

                        # create prediction tensor
                        pred = label.new_zeros(output.shape)
                        for i in range(output.shape[0]):
                            if self.n_classes >= 2:
                                pred[i, category_from_output(output[i, :])] = 1
                            else:
                                pred[i] = output[i] > 0.5

                        # multiclass prediction needs special call of loss function
                        if self.n_classes >= 2:
                            loss = criterion(output, label.argmax(1))
                        else:
                            loss = criterion(output, label)

                    # backward propagate + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                        optimizer.step()
                        self.kernel.learn_graph()

                    # update statistics
                    running_loss += loss.item() * b_size
                    running_corrects += torch.sum(
                        torch.sum(pred == label.data, 1)
                        == label.new_ones(pred.shape[0]) * self.n_classes
                    ).item()

                # calculate loss and accuracy in the current epoch
                epoch_loss = running_loss / len(data_loader[phase].dataset)
                epoch_acc = running_corrects / len(data_loader[phase].dataset)

                # print the statistics of the current epoch
                list_acc[phase].append(epoch_acc)
                list_loss[phase].append(epoch_loss)
                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )

                # deep copy the model
                if (phase == "val") and epoch_loss < best_loss:
                    best_epoch = epoch + 1
                    best_acc = epoch_acc
                    best_loss = epoch_loss

                    # store model parameters only if the generalization error improved (i.e. early stopping)
                    if early_stop:
                        best_weights = copy.deepcopy(self.state_dict())

            toc = timer()
            print("Finished, elapsed time: {:.2f}min\n".format((toc - tic) / 60))

        # report training results
        print("Finish at epoch: {}".format(epoch + 1))
        print(
            "Best epoch: {} with Acc = {:4f} and loss = {:4f}".format(
                best_epoch, best_acc, best_loss
            )
        )

        # if early stopping is enabled, make sure that the parameters are used, which resulted in the best
        # generalization error
        if early_stop:
            self.load_state_dict(best_weights)

        return list_acc, list_loss

    def predict(self, data_loader, proba=False, use_cuda=False):
        r"""Prediction function of the model

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            PyTorch DataLoader that handles data
        proba : bool
            Indicates whether the network should produce probabilistic output.
        use_cuda : bool
            Specified whether all computations will be performed on the GPU
        
        Returns
        -------
        output : torch.Tensor
            The computed output for each sample in the DataLoader. 
        target_output : torch.Tensor
            The real label for each sample in the DataLoader.
        """
        # set training mode of the model to False
        self.train(False)

        # detect the number of samples that will be classified and initialize tensor that stores the targets of each
        # sample
        n_samples = len(data_loader.dataset)

        # iterate over all samples
        batch_start = 0
        for i, (data, label, *_) in enumerate(data_loader):

            batch_size = data.shape[0]

            # transfer sample data to GPU if computations are performed there
            if use_cuda:
                data = data.cuda()

            # do not keep track of the gradients during the forward propagation
            with torch.no_grad():
                batch_out = self(data, proba).data.cpu()

            # initialize tensor that holds the results of the forward propagation for each sample
            if i == 0:
                output = torch.Tensor(n_samples, batch_out.shape[-1])
                target_output = torch.Tensor(n_samples, label.shape[-1]).type_as(label)

            # update output and target_output tensor with the current results
            output[batch_start : batch_start + batch_size] = batch_out
            target_output[batch_start : batch_start + batch_size] = label

            # continue with the next batch
            batch_start += batch_size

        # return the forward propagation results and the real targets
        output.squeeze_(-1)
        return output, target_output


class PIMKLNet(nn.Module):
    r"""Exemplary network implementing the PIMKL framework.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        num_anchors: int,
        pi_laplacians: list,
        pooling: bool = True,
        attention: str = "none",
    ) -> None:
        r"""Constructor of the PIMKLNet class

        Parameters
        ----------
        num_classes : int
            Number of classes, i.e. number of output states of the network.
        num_features : int
            Number of nodes in the graph, i.e the dimensionality of the inputs.
        num_anchors : int
            Number of anchor points of the layer.
        pi_laplacians : list
            List of tuples that contain the information about the pathway-induced kernels.
            The first entry of each tuple contains the indices of the parts of inputs that
            are used for the kernel. the second entry contains the Laplacian matrix of
            the kernel.
        pooling : bool
            If set to True, perform max pooling of the output of each pathway-induced
            kernel evaluation.
        attention : str
            Determine the type of attention layer used in the network.
        """
        super().__init__()

        # initialize attributes
        self.n_features = num_features
        self.n_classes = num_classes
        self.n_anchors = num_anchors
        self.n_pathways = len(pi_laplacians)
        self.pooling = pooling
        self.attention = attention

        # define layers
        self.kernel = PIMKLLayer(num_anchors, pi_laplacians)

        if pooling:
            self.pool = nn.MaxPool1d(kernel_size=num_anchors, stride=num_anchors)
            self.embedding_dim = self.n_pathways
            self.attention_dim = 1
        else:
            self.embedding_dim = num_anchors * self.n_pathways
            self.attention_dim = num_anchors

        if attention == "normal":
            self.attent = AttentionLayer(self.attention_dim, 8, 1)
            self.fc = nn.Linear(self.attention_dim, 1)
        elif attention == "gated":
            self.attent = GatedAttentionLayer(self.attention_dim, 8, 1)
            self.fc = nn.Linear(self.attention_dim, 1)
        else:
            self.fc = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x_in, proba=False):
        r"""Implementation of the forward pass through the network.

        Parameters
        ----------
        x_in : Tensor
            Input to the model given as a Tensor of size
            (batch_size x self.in_channels x seq_len)
        proba : bool
            Indicates whether the network should produce probabilistic output
        
        Returns
        -------
            Result of a forward pass through the model using the specified input. 
            If 'proba' is set to True, the output will be the probabilities assigned
            to each class by the model.
        """
        x_out = self.kernel(x_in)

        if self.pooling:
            x_out = self.pool(x_out)

        if self.attention in ["normal", "gated"]:
            # split each input into the bag of instances
            x_out = x_out.view(-1, self.n_pathways, self.attention_dim)
            x_out = self.attent(x_out)
            x_out = x_out.view(-1, self.attention_dim)

        x_out = self.fc(x_out)

        if proba:
            if self.n_classes == 1:
                return x_out.sigmoid()

            return F.softmax(x_out, dim=1)

        return x_out

    def init_params(self, data_loader, n_samples, kmeans_init="k-means++"):
        r"""Initialization routine of the model's parameters.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader object (initialized over a comik.OmicsDataset object)
            that is used to access the data.
        n_samples : int
            Number of data points that will be sampled from the data set. These
            samples are used in the clustering algorithm for the initialization
            of the anchor points.
        kmeans_init : str
            This parameter specifies the initialization routine used by the 
            kmeans algorithm.
        """
        # get the data to initialize the weights
        data = sample_data(data_loader, self.n_features, n_samples)

        # initialize the weights
        self.kernel.init_params(
            random_init=False, data=data, init=kmeans_init, verbose=False
        )

    def sup_train(
        self,
        train_loader,
        criterion,
        optimizer,
        lr_scheduler=None,
        init_train_loader=None,
        n_samples=100000,
        epochs=100,
        val_loader=None,
        use_cuda=False,
        early_stop=False,
    ):
        r"""Perform supervised training of the model

        This function will first initialize all convolutional layers of the model.
        Afterwards, a normal training routine for ANNs follows. If validation data
        is given, the performance on the validation data is calculate after each
        epoch.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            PyTorch DataLoader that handles training data.
        criterion : PyTorch Loss Object
            Specifies the loss function.
        optimizer : PyTorch Optimizer
            Optimization algorithm used during training.
        lr_scheduler : PyTorch LR Scheduler
            Algorithm used for learning rate adjustments.
        init_train_loader : torch.utils.data.DataLoader
            This DataLoader can be set if different datasets should be
            used for initializing the convolutional motif kernel layer 
            of this model and training the model. Data accessed by this
            DataLoader will be used for initialization of CMK layers.
        n_samples : int
            Number of motifs that will be sampled to initialize anchor points
            using the k-Means algorithm.
        epochs : int
            Number of epochs during training.
        val_loader : torch.utils.data.DataLoader
            PyTorch DataLoader that handles data during the validation phase.
        use_cuda : bool
            Specified whether all computations will be performed on the GPU.
        early_stop : bool
            Specifies if early stopping will be used during training.

        Returns
        -------
        list_acc : list
            List containing the accuracy on the train (and validation) data
            after each epoch.
        list_loss : list
            List containing the loss on the train (and validation) data
            after each epoch.
        """
        # initialize model
        print("Initializing network layers...")
        tic = timer()
        if init_train_loader is None:
            self.init_params(train_loader, n_samples)
        else:
            self.init_params(init_train_loader, n_samples)
        toc = timer()
        print("Finished, elapsed time: {:.2f}min\n".format((toc - tic) / 60))

        # specify the data used for each phase
        #   -> ATTENTION: a validation phase only exists if val_loader is not None
        phases = ["train"]
        data_loader = {"train": train_loader}
        if val_loader is not None:
            phases.append("val")
            data_loader["val"] = val_loader

        # initialize variables to keep track of the epoch's loss, the best loss, and the best accuracy
        epoch_loss = None
        best_epoch = 0
        best_loss = float("inf")
        best_acc = 0

        # iterate over all epochs
        list_acc = {"train": [], "val": []}
        list_loss = {"train": [], "val": []}
        for epoch in range(epochs):
            tic = timer()
            print("Epoch {}/{}".format(epoch + 1, epochs))
            print("-" * 10)

            # set the models train mode to False
            self.train(False)

            # iterate over all phases of the training process
            for phase in phases:

                # if the current phase is 'train', set model's train mode to True and initialize the Learning Rate
                # Scheduler (if one was specified)
                if phase == "train":
                    if lr_scheduler is not None:

                        # if the learning rate scheduler is 'ReduceLROnPlateau' and there is a current loss, the next
                        # lr step needs the current loss as input
                        if isinstance(lr_scheduler, ReduceLROnPlateau):
                            if epoch_loss is not None:
                                lr_scheduler.step(epoch_loss)

                        # otherwise call the step() function of the learning rate scheduler
                        else:
                            lr_scheduler.step()

                        # print the current learning rate
                        print("current LR: {}".format(optimizer.param_groups[0]["lr"]))

                    # set model's train mode to True
                    self.train(True)

                # if the current phase is not 'train', set the model's train mode to False. In this case, the learning
                # rate is irrelevant.
                else:
                    self.train(False)

                # initialize variables to keep track of the loss and the number of correctly classified samples in the
                # current epoch
                running_loss = 0.0
                running_corrects = 0

                # iterate over dataset
                for data, label, *_ in data_loader[phase]:
                    b_size = data.size(0)

                    # if computation should take place on the GPU, send everything to GPU
                    if use_cuda:
                        data = data.cuda()
                        label = label.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward propagation through the model
                    #   -> do not keep track of the gradients if we are in the validation phase
                    if phase == "val":
                        with torch.no_grad():
                            output = self(data)

                            # create prediction tensor
                            pred = label.new_zeros(output.shape)
                            for i in range(output.shape[0]):
                                if self.n_classes >= 2:
                                    pred[i, category_from_output(output[i, :])] = 1
                                else:
                                    pred[i] = output[i] > 0.5

                            # multiclass prediction needs special call of loss function
                            if self.n_classes >= 2:
                                loss = criterion(output, label.argmax(1))
                            else:
                                loss = criterion(output, label)
                    else:
                        output = self(data)

                        # create prediction tensor
                        pred = label.new_zeros(output.shape)
                        for i in range(output.shape[0]):
                            if self.n_classes >= 2:
                                pred[i, category_from_output(output[i, :])] = 1
                            else:
                                pred[i] = output[i] > 0.5

                        # multiclass prediction needs special call of loss function
                        if self.n_classes >= 2:
                            loss = criterion(output, label.argmax(1))
                        else:
                            loss = criterion(output, label)

                    # backward propagate + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                        optimizer.step()

                    # update statistics
                    running_loss += loss.item() * b_size
                    running_corrects += torch.sum(
                        torch.sum(pred == label.data, 1)
                        == label.new_ones(pred.shape[0]) * self.n_classes
                    ).item()

                # calculate loss and accuracy in the current epoch
                epoch_loss = running_loss / len(data_loader[phase].dataset)
                epoch_acc = running_corrects / len(data_loader[phase].dataset)

                # print the statistics of the current epoch
                list_acc[phase].append(epoch_acc)
                list_loss[phase].append(epoch_loss)
                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )

                # deep copy the model
                if (phase == "val") and epoch_loss < best_loss:
                    best_epoch = epoch + 1
                    best_acc = epoch_acc
                    best_loss = epoch_loss

                    # store model parameters only if the generalization error improved (i.e. early stopping)
                    if early_stop:
                        best_weights = copy.deepcopy(self.state_dict())

            toc = timer()
            print("Finished, elapsed time: {:.2f}min\n".format((toc - tic) / 60))

        # report training results
        print("Finish at epoch: {}".format(epoch + 1))
        print(
            "Best epoch: {} with Acc = {:4f} and loss = {:4f}".format(
                best_epoch, best_acc, best_loss
            )
        )

        # if early stopping is enabled, make sure that the parameters are used, which resulted in the best
        # generalization error
        if early_stop:
            self.load_state_dict(best_weights)

        return list_acc, list_loss

    def predict(self, data_loader, proba=False, use_cuda=False):
        r"""Prediction function of the model

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            PyTorch DataLoader that handles data
        proba : bool
            Indicates whether the network should produce probabilistic output.
        use_cuda : bool
            Specified whether all computations will be performed on the GPU
        
        Returns
        -------
        output : torch.Tensor
            The computed output for each sample in the DataLoader. 
        target_output : torch.Tensor
            The real label for each sample in the DataLoader.
        """
        # set training mode of the model to False
        # self.train(False)

        # detect the number of samples that will be classified and initialize tensor that stores the targets of each
        # sample
        n_samples = len(data_loader.dataset)

        # iterate over all samples
        batch_start = 0
        for i, (data, label, *_) in enumerate(data_loader):

            batch_size = data.shape[0]

            # transfer sample data to GPU if computations are performed there
            if use_cuda:
                data = data.cuda()

            # do not keep track of the gradients during the forward propagation
            with torch.no_grad():
                batch_out = self(data, proba).data.cpu()

            # initialize tensor that holds the results of the forward propagation for each sample
            if i == 0:
                output = torch.Tensor(n_samples, batch_out.shape[-1])
                target_output = torch.Tensor(n_samples, label.shape[-1]).type_as(label)

            # update output and target_output tensor with the current results
            output[batch_start : batch_start + batch_size] = batch_out
            target_output[batch_start : batch_start + batch_size] = label

            # continue with the next batch
            batch_start += batch_size

        # return the forward propagation results and the real targets
        output.squeeze_(-1)
        target_output.squeeze_(-1)
        return output, target_output


class GLPIMKLNet(nn.Module):
    r"""Examplary network utilizing the combination of the graph learning
    framework and the PIMKL framework
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        num_anchors: int,
        pi_laplacians: list,
        graph_learner: str = "glasso",
        gl_params: dict = None,
    ) -> None:
        r"""Constructor of the GLPIMKLNet class

        Parameters
        ----------
        num_classes : int
            Number of classes, i.e. number of output states of the network.
        num_features : int
            Number of nodes in the graph, i.e the dimensionality of the inputs.
        num_anchors : int
            Number of anchor points of the layer.
        pi_laplacians : list
            List of tuples that contain the information about the pathway-induced kernels.
            The first entry of each tuple contains the indices of the parts of inputs that
            are used for the kernel. the second entry contains the Laplacian matrix of
            the kernel.
        graph_learner : str
            Framework that will be used to learn a graph from the optimized
            anchor points. The framework is called after every optimization step,
            i.e. everytime the anchor points changed.
        gl_params : dict
            Dictionary to provide parameter settings for the graph learning framework.
            Available parameters for each framework can be found in the corresponding
            doc strings. If no dictionary is provided, the default parameters will be
            used. The dictionary has to be of the form {'param_name': param_value}.
        """
        super().__init__()

        # initialize attributes
        self.n_features = num_features
        self.n_classes = num_classes
        self.n_anchors = num_anchors

        # define layers
        self.kernel = GLPIMKLLayer(num_anchors, pi_laplacians, graph_learner, gl_params)
        self.fc = nn.Linear(num_anchors * len(pi_laplacians), num_classes)

    def forward(self, x_in, proba=False):
        r"""Implementation of the forward pass through the network.

        Parameters
        ----------
        x_in : Tensor
            Input to the model given as a Tensor of size
            (batch_size x self.in_channels x seq_len)
        proba : bool
            Indicates whether the network should produce probabilistic output
        
        Returns
        -------
            Result of a forward pass through the model using the specified input. 
            If 'proba' is set to True, the output will be the probabilities assigned
            to each class by the model.
        """
        x_out = self.kernel(x_in)
        x_out = self.fc(x_out)

        if proba:
            if self.n_classes == 1:
                return x_out.sigmoid()

            return F.softmax(x_out, dim=1)

        return x_out

    def init_params(self, data_loader, n_samples, kmeans_init="k-means++"):
        r"""Initialization routine of the model's parameters.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader object (initialized over a comik.OmicsDataset object)
            that is used to access the data.
        n_samples : int
            Number of data points that will be sampled from the data set. These
            samples are used in the clustering algorithm for the initialization
            of the anchor points.
        kmeans_init : str
            This parameter specifies the initialization routine used by the 
            kmeans algorithm.
        """
        # get the data to initialize the weights
        data = sample_data(data_loader, self.n_features, n_samples)

        # initialize the weights
        self.kernel.init_params(
            random_init=False, data=data, init=kmeans_init, verbose=False
        )

    def sup_train(
        self,
        train_loader,
        criterion,
        optimizer,
        lr_scheduler=None,
        init_train_loader=None,
        n_samples=100000,
        epochs=100,
        val_loader=None,
        use_cuda=False,
        early_stop=False,
    ):
        r"""Perform supervised training of the model

        This function will first initialize all convolutional layers of the model.
        Afterwards, a normal training routine for ANNs follows. If validation data
        is given, the performance on the validation data is calculate after each
        epoch.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            PyTorch DataLoader that handles training data.
        criterion : PyTorch Loss Object
            Specifies the loss function.
        optimizer : PyTorch Optimizer
            Optimization algorithm used during training.
        lr_scheduler : PyTorch LR Scheduler
            Algorithm used for learning rate adjustments.
        init_train_loader : torch.utils.data.DataLoader
            This DataLoader can be set if different datasets should be
            used for initializing the convolutional motif kernel layer 
            of this model and training the model. Data accessed by this
            DataLoader will be used for initialization of CMK layers.
        n_samples : int
            Number of motifs that will be sampled to initialize anchor points
            using the k-Means algorithm.
        epochs : int
            Number of epochs during training.
        val_loader : torch.utils.data.DataLoader
            PyTorch DataLoader that handles data during the validation phase.
        use_cuda : bool
            Specified whether all computations will be performed on the GPU.
        early_stop : bool
            Specifies if early stopping will be used during training.

        Returns
        -------
        list_acc : list
            List containing the accuracy on the train (and validation) data
            after each epoch.
        list_loss : list
            List containing the loss on the train (and validation) data
            after each epoch.
        """
        # initialize model
        print("Initializing network layers...")
        tic = timer()
        if init_train_loader is None:
            self.init_params(train_loader, n_samples)
        else:
            self.init_params(init_train_loader, n_samples)
        toc = timer()
        print("Finished, elapsed time: {:.2f}min\n".format((toc - tic) / 60))

        # specify the data used for each phase
        #   -> ATTENTION: a validation phase only exists if val_loader is not None
        phases = ["train"]
        data_loader = {"train": train_loader}
        if val_loader is not None:
            phases.append("val")
            data_loader["val"] = val_loader

        # initialize variables to keep track of the epoch's loss, the best loss, and the best accuracy
        epoch_loss = None
        best_epoch = 0
        best_loss = float("inf")
        best_acc = 0

        # iterate over all epochs
        list_acc = {"train": [], "val": []}
        list_loss = {"train": [], "val": []}
        for epoch in range(epochs):
            tic = timer()
            print("Epoch {}/{}".format(epoch + 1, epochs))
            print("-" * 10)

            # set the models train mode to False
            self.train(False)

            # iterate over all phases of the training process
            for phase in phases:

                # if the current phase is 'train', set model's train mode to True and initialize the Learning Rate
                # Scheduler (if one was specified)
                if phase == "train":
                    if lr_scheduler is not None:

                        # if the learning rate scheduler is 'ReduceLROnPlateau' and there is a current loss, the next
                        # lr step needs the current loss as input
                        if isinstance(lr_scheduler, ReduceLROnPlateau):
                            if epoch_loss is not None:
                                lr_scheduler.step(epoch_loss)

                        # otherwise call the step() function of the learning rate scheduler
                        else:
                            lr_scheduler.step()

                        # print the current learning rate
                        print("current LR: {}".format(optimizer.param_groups[0]["lr"]))

                    # set model's train mode to True
                    self.train(True)

                # if the current phase is not 'train', set the model's train mode to False. In this case, the learning
                # rate is irrelevant.
                else:
                    self.train(False)

                # initialize variables to keep track of the loss and the number of correctly classified samples in the
                # current epoch
                running_loss = 0.0
                running_corrects = 0

                # iterate over dataset
                for data, label, *_ in data_loader[phase]:
                    b_size = data.size(0)

                    # if computation should take place on the GPU, send everything to GPU
                    if use_cuda:
                        data = data.cuda()
                        label = label.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward propagation through the model
                    #   -> do not keep track of the gradients if we are in the validation phase
                    if phase == "val":
                        with torch.no_grad():
                            output = self(data)

                            # create prediction tensor
                            pred = label.new_zeros(output.shape)
                            for i in range(output.shape[0]):
                                if self.n_classes >= 2:
                                    pred[i, category_from_output(output[i, :])] = 1
                                else:
                                    pred[i] = output[i] > 0.5

                            # multiclass prediction needs special call of loss function
                            if self.n_classes >= 2:
                                loss = criterion(output, label.argmax(1))
                            else:
                                loss = criterion(output, label)
                    else:
                        tic_for = timer()
                        output = self(data)
                        toc_for = timer()

                        # create prediction tensor
                        pred = label.new_zeros(output.shape)
                        for i in range(output.shape[0]):
                            if self.n_classes >= 2:
                                pred[i, category_from_output(output[i, :])] = 1
                            else:
                                pred[i] = output[i] > 0.5

                        # multiclass prediction needs special call of loss function
                        if self.n_classes >= 2:
                            loss = criterion(output, label.argmax(1))
                        else:
                            loss = criterion(output, label)

                    # backward propagate + optimize only if in training phase
                    if phase == "train":
                        tic_back = timer()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                        optimizer.step()
                        toc_back = timer()

                        # learn the new Laplacians from the anchor points
                        tic_graph = timer()
                        for gkl in self.kernel.kernels:
                            gkl.learn_graph()
                        toc_graph = timer()

                    # update statistics
                    running_loss += loss.item() * b_size
                    running_corrects += torch.sum(
                        torch.sum(pred == label.data, 1)
                        == label.new_ones(pred.shape[0]) * self.n_classes
                    ).item()

                # calculate loss and accuracy in the current epoch
                epoch_loss = running_loss / len(data_loader[phase].dataset)
                epoch_acc = running_corrects / len(data_loader[phase].dataset)

                # print the statistics of the current epoch
                list_acc[phase].append(epoch_acc)
                list_loss[phase].append(epoch_loss)
                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )

                # deep copy the model
                if (phase == "val") and epoch_loss < best_loss:
                    best_epoch = epoch + 1
                    best_acc = epoch_acc
                    best_loss = epoch_loss

                    # store model parameters only if the generalization error improved (i.e. early stopping)
                    if early_stop:
                        best_weights = copy.deepcopy(self.state_dict())

            toc = timer()
            print(
                "Finished, elapsed time: {:.2f}min ({:.2}min forward, {:.2}min backward/optim, {:.2}min graph learning)\n".format(
                    (toc - tic) / 60,
                    (toc_for - tic_for) / 60,
                    (toc_back - tic_back) / 60,
                    (toc_graph - tic_graph) / 60,
                )
            )

        # report training results
        print("Finish at epoch: {}".format(epoch + 1))
        print(
            "Best epoch: {} with Acc = {:4f} and loss = {:4f}".format(
                best_epoch, best_acc, best_loss
            )
        )

        # if early stopping is enabled, make sure that the parameters are used, which resulted in the best
        # generalization error
        if early_stop:
            self.load_state_dict(best_weights)

        return list_acc, list_loss

    def predict(self, data_loader, proba=False, use_cuda=False):
        r"""Prediction function of the model

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            PyTorch DataLoader that handles data
        proba : bool
            Indicates whether the network should produce probabilistic output.
        use_cuda : bool
            Specified whether all computations will be performed on the GPU
        
        Returns
        -------
        output : torch.Tensor
            The computed output for each sample in the DataLoader. 
        target_output : torch.Tensor
            The real label for each sample in the DataLoader.
        """
        # set training mode of the model to False
        # self.train(False)

        # detect the number of samples that will be classified and initialize tensor that stores the targets of each
        # sample
        n_samples = len(data_loader.dataset)

        # iterate over all samples
        batch_start = 0
        for i, (data, label, *_) in enumerate(data_loader):

            batch_size = data.shape[0]

            # transfer sample data to GPU if computations are performed there
            if use_cuda:
                data = data.cuda()

            # do not keep track of the gradients during the forward propagation
            with torch.no_grad():
                batch_out = self(data, proba).data.cpu()

            # initialize tensor that holds the results of the forward propagation for each sample
            if i == 0:
                output = torch.Tensor(n_samples, batch_out.shape[-1])
                target_output = torch.Tensor(n_samples, label.shape[-1]).type_as(label)

            # update output and target_output tensor with the current results
            output[batch_start : batch_start + batch_size] = batch_out
            target_output[batch_start : batch_start + batch_size] = label

            # continue with the next batch
            batch_start += batch_size

        # return the forward propagation results and the real targets
        output.squeeze_(-1)
        target_output.squeeze_(-1)
        return output, target_output


class MultiPIMKLNet(nn.Module):
    r"""Exemplary network for a prediction task on a multi omics dataset utilizing
    the PIMKL framework."""

    def __init__(
        self,
        num_classes: int,
        num_features: list,
        num_anchors: list,
        pi_laplacians: list,
        pooling: bool = True,
        attention: str = "none",
    ):
        r"""Constructor of the MultiPIMKLNet class

        Parameters
        ----------
        num_classes : int
            Number of classes, i.e. number of output states of the network.
        num_features : list
            Number of nodes in the graph, i.e the dimensionality of the inputs.
        num_anchors : list
            Number of anchor points of the layer.
        pi_laplacians : list
            List of lists of tuples that contain the information about the pathway-induced kernels.
            The first entry of each tuple contains the indices of the parts of inputs that
            are used for the kernel. the second entry contains the Laplacian matrix of
            the kernel.
        pooling : bool
            If set to True, perform max pooling of the output of each pathway-induced kernel evaluation.
        attention : str
            Determines the type of attention that will be used.
        """
        super().__init__()

        # perform sanity checks
        if len(num_features) != len(num_anchors) != len(pi_laplacians):
            raise ValueError(
                f"num_features, num_anchors, and pi_laplacians have to be of same length"
            )

        # make sure that each omics datatype has the same number of anchor points if an
        # attention layer should be used
        if attention in ["normal", "gated"]:
            if not all([num_anchors[0] == i for i in num_anchors]):
                raise ValueError(
                    "Pathway level attention layer can only be used if all omics datatypes use the same number of anchor points."
                )

        # initialize attributes
        self.n_features = num_features
        self.n_classes = num_classes
        self.n_anchors = num_anchors
        self.n_pathways = [len(i) for i in pi_laplacians]
        self.pooling = pooling
        self.attention = attention

        # define kernel layers
        self.kernels = nn.ModuleList()
        if pooling:
            self.pools = nn.ModuleList()

        self.embedding_dim = 0
        for i in range(len(num_anchors)):
            self.kernels.append(PIMKLLayer(num_anchors[i], pi_laplacians[i]))

            if pooling:
                self.pools.append(
                    nn.MaxPool1d(kernel_size=num_anchors[i], stride=num_anchors[i])
                )
                self.embedding_dim += self.n_pathways[i]
                self.attention_dim = 1
            else:
                self.embedding_dim += num_anchors[i] * self.n_pathways[i]
                self.attention_dim = num_anchors[i]

        # define attention and output layer
        if attention == "normal":
            self.attent = AttentionLayer(self.attention_dim, 8, 1)
            self.out = nn.Linear(sum(self.n_pathways), num_classes)
        elif attention == "gated":
            self.attent = GatedAttentionLayer(self.attention_dim, 8, 1)
            self.out = nn.Linear(sum(self.n_pathways), num_classes)
        else:
            self.out = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x_in, proba=False):
        r"""Implementation of the forward pass through the network.

        Parameters
        ----------
        x_in : list
            Input to the model given as a list of Tensors of size
            (batch_size x num_features)
        proba : bool
            Indicates whether the network should produce probabilistic output
        
        Returns
        -------
            Result of a forward pass through the model using the specified input. 
            If 'proba' is set to True, the output will be the probabilities assigned
            to each class by the model.
        """
        # pass each data type to the correponding PIMKLLayer
        concat_out = []
        for i, aux_in in enumerate(x_in):
            x_out = self.kernels[i](aux_in)

            if self.pooling:
                x_out = self.pools[i](x_out)

            concat_out.append(x_out)

        # concatenate the outputs of each kernel
        x_out = torch.cat(concat_out, dim=1)

        if self.attention in ["normal", "gated"]:
            x_out = x_out.view(-1, sum(self.n_pathways), self.attention_dim)
            x_out = self.attent(x_out)

        # pass concatinated outputs through Linear layer
        x_out = self.out(x_out)

        if proba:
            if self.n_classes == 1:
                return x_out.sigmoid()

            return F.softmax(x_out, dim=1)

        return x_out

    def init_params(self, data_loader, n_samples, kmeans_init="k-means++"):
        r"""Initialization routine of the model's parameters.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader object (initialized over a comik.OmicsDataset object)
            that is used to access the data.
        n_samples : int
            Number of data points that will be sampled from the data set. These
            samples are used in the clustering algorithm for the initialization
            of the anchor points.
        kmeans_init : str
            This parameter specifies the initialization routine used by the 
            kmeans algorithm.
        """
        # get the data to initialize the weights
        data = sample_data_multiomics(data_loader, self.n_features, n_samples)

        # initialize the weights
        for i in range(len(self.n_features)):
            self.kernels[i].init_params(
                random_init=False, data=data[i], init=kmeans_init, verbose=False
            )

    def sup_train(
        self,
        train_loader,
        criterion,
        optimizer,
        lr_scheduler=None,
        init_train_loader=None,
        n_samples=100000,
        epochs=100,
        val_loader=None,
        use_cuda=False,
        early_stop=False,
    ):
        r"""Perform supervised training of the model

        This function will first initialize all convolutional layers of the model.
        Afterwards, a normal training routine for ANNs follows. If validation data
        is given, the performance on the validation data is calculate after each
        epoch.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            PyTorch DataLoader that handles training data.
        criterion : PyTorch Loss Object
            Specifies the loss function.
        optimizer : PyTorch Optimizer
            Optimization algorithm used during training.
        lr_scheduler : PyTorch LR Scheduler
            Algorithm used for learning rate adjustments.
        init_train_loader : torch.utils.data.DataLoader
            This DataLoader can be set if different datasets should be
            used for initializing the convolutional motif kernel layer 
            of this model and training the model. Data accessed by this
            DataLoader will be used for initialization of CMK layers.
        n_samples : int
            Number of motifs that will be sampled to initialize anchor points
            using the k-Means algorithm.
        epochs : int
            Number of epochs during training.
        val_loader : torch.utils.data.DataLoader
            PyTorch DataLoader that handles data during the validation phase.
        use_cuda : bool
            Specified whether all computations will be performed on the GPU.
        early_stop : bool
            Specifies if early stopping will be used during training.

        Returns
        -------
        list_acc : list
            List containing the accuracy on the train (and validation) data
            after each epoch.
        list_loss : list
            List containing the loss on the train (and validation) data
            after each epoch.
        """
        # initialize model
        print("Initializing network layers...")
        tic = timer()
        if init_train_loader is None:
            self.init_params(train_loader, n_samples)
        else:
            self.init_params(init_train_loader, n_samples)
        toc = timer()
        print("Finished, elapsed time: {:.2f}min\n".format((toc - tic) / 60))

        # specify the data used for each phase
        #   -> ATTENTION: a validation phase only exists if val_loader is not None
        phases = ["train"]
        data_loader = {"train": train_loader}
        if val_loader is not None:
            phases.append("val")
            data_loader["val"] = val_loader

        # initialize variables to keep track of the epoch's loss, the best loss, and the best accuracy
        epoch_loss = None
        best_epoch = 0
        best_loss = float("inf")
        best_acc = 0

        # iterate over all epochs
        list_acc = {"train": [], "val": []}
        list_loss = {"train": [], "val": []}
        for epoch in range(epochs):
            tic = timer()
            print("Epoch {}/{}".format(epoch + 1, epochs))
            print("-" * 10)

            # set the models train mode to False
            self.train(False)

            # iterate over all phases of the training process
            for phase in phases:

                # if the current phase is 'train', set model's train mode to True and initialize the Learning Rate
                # Scheduler (if one was specified)
                if phase == "train":
                    if lr_scheduler is not None:

                        # if the learning rate scheduler is 'ReduceLROnPlateau' and there is a current loss, the next
                        # lr step needs the current loss as input
                        if isinstance(lr_scheduler, ReduceLROnPlateau):
                            if epoch_loss is not None:
                                lr_scheduler.step(epoch_loss)

                        # otherwise call the step() function of the learning rate scheduler
                        else:
                            lr_scheduler.step()

                        # print the current learning rate
                        print("current LR: {}".format(optimizer.param_groups[0]["lr"]))

                    # set model's train mode to True
                    self.train(True)

                # if the current phase is not 'train', set the model's train mode to False. In this case, the learning
                # rate is irrelevant.
                else:
                    self.train(False)

                # initialize variables to keep track of the loss and the number of correctly classified samples in the
                # current epoch
                running_loss = 0.0
                running_corrects = 0

                # iterate over dataset
                for data, label, *_ in data_loader[phase]:
                    b_size = data[0].size(0)

                    # if computation should take place on the GPU, send everything to GPU
                    if use_cuda:
                        data = [d.cuda() for d in data]
                        label = label.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward propagation through the model
                    #   -> do not keep track of the gradients if we are in the validation phase
                    if phase == "val":
                        with torch.no_grad():
                            output = self(data, proba=True)

                            # create prediction tensor
                            pred = label.new_zeros(output.shape)
                            for i in range(output.shape[0]):
                                if self.n_classes >= 2:
                                    pred[i, category_from_output(output[i, :])] = 1
                                else:
                                    pred[i] = output[i] > 0.5

                            # multiclass prediction needs special call of loss function
                            if self.n_classes >= 2:
                                loss = criterion(output, label.argmax(1))
                            else:
                                loss = criterion(output, label)
                    else:
                        output = self(data)

                        # create prediction tensor
                        pred = label.new_zeros(output.shape)
                        for i in range(output.shape[0]):
                            if self.n_classes >= 2:
                                pred[i, category_from_output(output[i, :])] = 1
                            else:
                                pred[i] = output[i] > 0.5

                        # multiclass prediction needs special call of loss function
                        if self.n_classes >= 2:
                            loss = criterion(output, label.argmax(1))
                        else:
                            loss = criterion(output, label)

                    # backward propagate + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                        optimizer.step()

                    # update statistics
                    running_loss += loss.item() * b_size
                    running_corrects += torch.sum(
                        torch.sum(pred == label.data, 1)
                        == label.new_ones(pred.shape[0]) * self.n_classes
                    ).item()

                # calculate loss and accuracy in the current epoch
                epoch_loss = running_loss / len(data_loader[phase].dataset)
                epoch_acc = running_corrects / len(data_loader[phase].dataset)

                # print the statistics of the current epoch
                list_acc[phase].append(epoch_acc)
                list_loss[phase].append(epoch_loss)
                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )

                # deep copy the model
                if (phase == "val") and epoch_loss < best_loss:
                    best_epoch = epoch + 1
                    best_acc = epoch_acc
                    best_loss = epoch_loss

                    # store model parameters only if the generalization error improved (i.e. early stopping)
                    if early_stop:
                        best_weights = copy.deepcopy(self.state_dict())

            toc = timer()
            print("Finished, elapsed time: {:.2f}min\n".format((toc - tic) / 60))

        # report training results
        print("Finish at epoch: {}".format(epoch + 1))
        print(
            "Best epoch: {} with Acc = {:4f} and loss = {:4f}".format(
                best_epoch, best_acc, best_loss
            )
        )

        # if early stopping is enabled, make sure that the parameters are used, which resulted in the best
        # generalization error
        if early_stop:
            self.load_state_dict(best_weights)

        return list_acc, list_loss

    def predict(self, data_loader, proba=False, use_cuda=False):
        r"""Prediction function of the model

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            PyTorch DataLoader that handles data
        proba : bool
            Indicates whether the network should produce probabilistic output.
        use_cuda : bool
            Specified whether all computations will be performed on the GPU
        
        Returns
        -------
        output : torch.Tensor
            The computed output for each sample in the DataLoader. 
        target_output : torch.Tensor
            The real label for each sample in the DataLoader.
        """
        # set training mode of the model to False
        # self.train(False)

        # detect the number of samples that will be classified and initialize tensor that stores the targets of each
        # sample
        n_samples = len(data_loader.dataset)

        # iterate over all samples
        batch_start = 0
        for i, (data, label, *_) in enumerate(data_loader):

            batch_size = data[0].shape[0]

            # transfer sample data to GPU if computations are performed there
            if use_cuda:
                data = [d.cuda() for d in data]

            # do not keep track of the gradients during the forward propagation
            with torch.no_grad():
                batch_out = self(data, proba).data.cpu()

            # initialize tensor that holds the results of the forward propagation for each sample
            if i == 0:
                output = torch.Tensor(n_samples, batch_out.shape[-1])
                target_output = torch.Tensor(n_samples, label.shape[-1]).type_as(label)

            # update output and target_output tensor with the current results
            output[batch_start : batch_start + batch_size] = batch_out
            target_output[batch_start : batch_start + batch_size] = label

            # continue with the next batch
            batch_start += batch_size

        # return the forward propagation results and the real targets
        output.squeeze_(-1)
        target_output.squeeze_(-1)
        return output, target_output
