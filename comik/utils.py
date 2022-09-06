r"""Module of utility functionalities used in different parts of the
package.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)


def _init_kmeans(
    x: torch.Tensor,
    n_clusters: int,
    n_local_trials: int = None,
    use_cuda: bool = False,
    distance: str = "euclidean",
):
    r"""Initialization method for K-Means (k-Means++)

    Parameters
    ----------
    x : Tensor
        Data that will be used for clustering provided as a tensor of shape
        (n_samples x n_dimensions).
    n_clusters : int
        Number of clusters that will be computed.
    n_local_trials : int
        Number of local seeding trails. Defaults to None.
    use_cuda : bool
        Flag that determines whether computations should be performed on the GPU.
        Defaults to False.
    distance : str
        Distance measure used for clustering. Defaults to 'euclidean'.
    
    Returns
    -------
    clusters : Tensor
        Initial centers for each cluster.
    """
    n_samples, n_features = x.size()

    # initialize tensor that will hold the cluster centers and send it to GPU if needed
    clusters = torch.Tensor(n_clusters, n_features)
    if use_cuda:
        clusters = clusters.cuda()

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    # pick first cluster center randomly
    clusters[0] = x[np.random.randint(n_samples)]

    # initialize list of distances to the selected centroid and calculate current potential
    if distance == "cosine":
        # calculate distance of each point to the selected centroid using the distance measure of the spherical k-Means
        closest_dist_sq = 1 - clusters[[0]].mm(x.t())
        closest_dist_sq = closest_dist_sq.view(-1)
    elif distance == "euclidean":
        # calculate distance of each point to the selected centroid using the Euclidean distance measure
        closest_dist_sq = torch.cdist(clusters[[0]], x, p=2)
        closest_dist_sq = closest_dist_sq.view(-1)
    else:
        raise ValueError("Unknown value for parameter mode: {}".format(distance))
    current_pot = closest_dist_sq.sum().item()

    # pick the remaining n_clusters-1 cluster centers
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional to the squared distance to the closest
        # existing center
        rand_vals = np.random.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(-1).cpu(), rand_vals)

        # calculate distance of each data point to the candidates
        if distance == "cosine":
            distance_to_candidates = 1 - x[candidate_ids].mm(x.t())
        elif distance == "euclidean":
            distance_to_candidates = torch.cdist(x[candidate_ids], x, p=2)
        else:
            raise ValueError("Unknown value for parameter mode: {}".format(distance))

        # iterate over the candidates for the new cluster center and select the most suitable
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = torch.min(closest_dist_sq, distance_to_candidates[trial])
            new_pot = new_dist_sq.sum().item()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        clusters[c] = x[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return clusters


def kmeans_gpu(
    x: torch.Tensor,
    n_clusters: int,
    distance: str = "euclidian",
    max_iters: int = 100,
    verbose: bool = True,
    init: str = None,
    tol: float = 1e-4,
):
    r"""Performing k-Means clustering (Lloyd's algorithm) with Tensors utilizing GPU resources.

    Parameters
    ----------
    x : Tensor
        Data that will be used for clustering provided as a tensor of shape
        (n_samples x n_dimensions).
    n_clusters : int
        Number of clusters that will be computed.
    distance : str
        Distance measure used for clustering. Defaults to 'euclidean'.
    max_iters : int
        Maximal number of iterations used in the K-Means clustering. Defaults to 100.
    verbose : bool
        Flag to activate verbose output. Defaults to True.
    init : str
        Initialization process for the K-Means algorithm. Defaults to None.
    tol : float
        Relative tolerance with regards to Frobenius norm of the difference in the cluster
        centers of two consecutive iterations to declare convergence. It's not advised to set
        `tol=0` since convergence might never be declared due to rounding errors. Use a very
        small number instead. Defaults to 1e-4.

    Returns
    -------
    clusters : Tensor
        Tensor that contains the cluster centers, i.e. result of the K-Means algorithm. The shape of
        the tensor is (n_clusters x n_dimensions).
    """
    # make sure there are more samples than requested clusters
    if x.shape[0] < n_clusters:
        raise ValueError(
            f"n_samples={x.shape[0]} should be >= n_clusters={n_clusters}."
        )

    # check whether the input tensor is on the GPU
    use_cuda = x.is_cuda

    # store number of data points and dimensionality of each data point
    n_samples, n_features = x.size()

    # determine initialization procedure for this run of the k-Means algorithm
    if init == "k-means++":
        print("        Initialization method for k-Means: k-Means++")
        clusters = _init_kmeans(x, n_clusters, use_cuda=use_cuda, distance=distance)
    elif init is None:
        print("        Initialization method for k-Means: random")
        indices = torch.randperm(n_samples)[:n_clusters]
        if use_cuda:
            indices = indices.cuda()
        clusters = x[indices]
    else:
        raise ValueError("Unknown initialization procedure: {}".format(init))

    # perform Lloyd's algorithm iteratively until convergence or the number of iterations exceeds max_iters
    prev_sim = np.inf
    for n_iter in range(max_iters):
        # calculate the distance of data points to clusters using the selected distance measure. Use the calculated
        # distances to assign each data point to a cluster
        if distance == "cosine":
            sim = x.mm(clusters.t())
            tmp, assign = sim.max(dim=-1)
        elif distance == "euclidean":
            sim = torch.cdist(x, clusters, p=2)
            tmp, assign = sim.min(dim=-1)
        else:
            raise ValueError("Unknown distance measure: {}".format(distance))

        # get the mean distance to the cluster centers
        sim_mean = tmp.mean()
        if (n_iter + 1) % 10 == 0 and verbose:
            print(
                "        k-Means iter: {}, distance: {}, objective value: {}".format(
                    n_iter + 1, distance, sim_mean
                )
            )

        # update clusters
        for j in range(n_clusters):
            # get all data points that were assigned to the current cluster
            index = assign == j

            # if no data point was assigned to the current cluster, use the data point furthest away from every cluster
            # as new cluster center
            if index.sum() == 0:
                if distance == "cosine":
                    idx = tmp.argmin()
                elif distance == "euclidean":
                    idx = tmp.argmax()
                clusters[j] = x[idx]
                tmp[idx] = 1

            # otherwise, update the center of the current cluster based on all data points assigned to this cluster
            else:
                xj = x[index]
                c = xj.mean(0)
                clusters[j] = c / c.norm()

        # stop k-Means if the difference in the cluster center is below the tolerance (i.e. the algorithm converged)
        if torch.abs(prev_sim - sim_mean) / (torch.abs(sim_mean) + 1e-20) < tol:
            break
        prev_sim = sim_mean

    return clusters


def kmeans(
    x: torch.Tensor,
    n_clusters: int,
    distance: str = "euclidian",
    max_iters: int = 100,
    verbose: bool = True,
    init: str = None,
    tol: float = 1e-4,
    use_cuda: bool = False,
):
    r"""Wrapper for the k-Means clustering algorithm to utilize either GPU or CPU resources.
    We always recommend to use the well-tested scikit-learn implementation (i.e. set
    'use_cuda' to False) unless there is an important reason to utilize GPU.

    Parameters
    ----------
    x : Tensor
        Data that will be used for clustering provided as a tensor of shape
        (n_samples x n_dimensions).
    n_clusters : int
        Number of clusters that will be computed.
    distance : str
        Distance measure used for clustering. Defaults to 'euclidean'.
    max_iters : int
        Maximal number of iterations used in the K-Means clustering. Defaults to 100.
    verbose : bool
        Flag to activate verbose output. Defaults to True.
    init : str
        Initialization process for the K-Means algorithm. Defaults to None.
    tol : float
        Relative tolerance with regards to Frobenius norm of the difference in the cluster
        centers of two consecutive iterations to declare convergence. It's not advised to set
        `tol=0` since convergence might never be declared due to rounding errors. Use a very
        small number instead. Defaults to 1e-4.
    use_cuda : bool
        Determine whether to utilize GPU resources or compute kmeans on CPU resources. If set to
        False, scikit-learn's implementation of kmeans will be used. Defaults to False.
    
    Returns
    -------
    clusters : Tensor
        Tensor that contains the cluster centers, i.e. result of the K-Means algorithm. The shape of
        the tensor is (n_clusters x n_dimensions).
    """
    # use GPU implementation if use_cuda was set to true
    if use_cuda:
        clusters = kmeans_gpu(x, n_clusters, distance, max_iters, verbose, init, tol)

    # otherwise, cast Tensors to numpy arrays and use scikit-learn's implementation of kmeans
    else:
        aux_x = x.cpu().numpy()
        sklearn_kmeans = KMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iters,
            tol=tol,
            verbose=int(verbose),
            algorithm="full",
        ).fit(aux_x)

        clusters = torch.Tensor(sklearn_kmeans.cluster_centers_)

        # make sure that the cluster centers are on the GPU if the input is on the GPU
        if x.is_cuda:
            clusters = clusters.cuda()

    return clusters


def sample_data(
    data_loader: torch.utils.data.DataLoader, n_features: int, n_samples: int = 100000
):
    r"""Utility function that returns a specified number of samples as a tensor. The samples will
    be taken from a specified PyTorch DataLoader object.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        PyTorch DataLoader object that handles access to training data. In general, other objects 
        can be used to access the data. The only prerequisite is that it is possible to retreive
        the data and label as PyTorch Tensors when iterated over the object. We strongly recommend
        to use a PyTorch DataLoader object.
    n_features : int
        Number of features in the data set. In other words, the data set consists of data points
        with n_features dimentions.
    n_samples : int
        Number of data points that will be sampled from the data.

    Returns
    -------
    samples : Tensor
        Tensor containing the sampled data points.
    """
    # initialize the Tensor that stores all sampled data points
    samples = torch.zeros(n_samples, n_features)

    # determine the number of data points sampled per batch
    #    -> we make sure that we sample at least 500 data points per batch to reduce runtime
    n_samples_per_batch = max(
        (n_samples + len(data_loader) - 1) // len(data_loader), 500
    )

    # iterate over the data set
    already_sampled = 0
    for data, _ in data_loader:
        # stop if already enough data points have been sampled
        if already_sampled >= n_samples:
            break

        # make sure to sample at most the number of data points in the current batch
        max_samples_per_batch = min(data.size(0), n_samples_per_batch)

        # sample random indices of the data Tensor (number of sampled indices is either the
        # maximum number of data points or n_samples_per_batch, whatever is smaller)
        indices = torch.randperm(data.size(0))[:max_samples_per_batch]
        current_samples = data[indices]

        # only use a subset of the sampled data points in this batch, if this batch would
        # exceed the maximum number of samples
        current_size = current_samples.size(0)
        if already_sampled + current_size > n_samples:
            current_size = n_samples - already_sampled
            data_oliogmers = data_oliogmers[:current_size]

        # update the samples Tensor with the current batch of sampled data points
        samples[already_sampled : already_sampled + current_size] = current_samples
        already_sampled += current_size

    # return the sampled data points
    print(f"sample_data routine returned {already_sampled} sampled data points")
    return samples[:already_sampled, :]


def sample_data_multiomics(
    data_loader: torch.utils.data.DataLoader, n_features: list, n_samples: int = 100000
):
    r"""Data sampling utility function that is specily designed for multi omics datasets.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        PyTorch DataLoader object that handles access to training data. In general, other objects 
        can be used to access the data. The only prerequisite is that it is possible to retreive
        the data and label as PyTorch Tensors when iterated over the object. We strongly recommend
        to use a PyTorch DataLoader object.
    n_features : int
        Number of features in the data set. In other words, the data set consists of data points
        with n_features dimentions.
    n_samples : int
        Number of data points that will be sampled from the data.

    Returns
    -------
    samples : list
        List of tensors containing the sampled data points.
    """
    # initialize the tensors that will store the samples
    samples = []
    for n_f in n_features:
        samples.append(torch.zeros(n_samples, n_f))

    # determine the number of data points sampled per batch
    #    -> we make sure that we sample at least 500 data points per batch to reduce runtime
    n_samples_per_batch = max(
        (n_samples + len(data_loader) - 1) // len(data_loader), 500
    )

    # iterate over the dataset
    already_sampled = 0
    for data, _ in data_loader:
        # stop if already enough data points have been sampled
        if already_sampled >= n_samples:
            break

        # iterate over all data types and sample independendly for all
        current_size = 0
        for i in range(len(n_features)):
            # make sure to sample at most the number of data points in the current batch
            max_samples_per_batch = min(data[i].size(0), n_samples_per_batch)

            # sample random indices of the data Tensor (number of sampled indices is either the
            # maximum number of data points or n_samples_per_batch, whatever is smaller)
            indices = torch.randperm(data[i].size(0))[:max_samples_per_batch]
            current_samples = data[i][indices]

            # only use a subset of the sampled data points in this batch, if this batch would
            # exceed the maximum number of samples
            current_size = current_samples.size(0)
            if already_sampled + current_size > n_samples:
                current_size = n_samples - already_sampled
                data_oliogmers = data_oliogmers[:current_size]

            # update the samples Tensor with the current batch of sampled data points
            samples[i][
                already_sampled : already_sampled + current_size
            ] = current_samples

        already_sampled += current_size

    # return the sampled data points
    print(f"sample_data routine returned {already_sampled} sampled data points")
    return [s[:already_sampled, :] for s in samples]


def category_from_output(output):
    r"""This auxiliary function returns the class with highest probability from
    a network's output.

    Parameters
    ----------
    output : Tensor
        Output of a PyTorch model.
    
    Returns
    -------
    category_i : int
        Index of the category with the highest probability.
    """
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i


def recall_at_fdr(y_true, y_score, fdr_cutoff=0.05):
    r"""Compute recall at certain false discovery rate cutoffs
    """
    # convert y_true and y_score into desired format
    #   -> both have to be lists of shape [nb_samples]
    if len(y_true.shape) > 1:
        y_true_new = y_true.argmax(axis=1)
    else:
        y_true_new = y_true
    if len(y_score.shape) > 1:
        y_score_new = [y_score[j][i] for j, i in enumerate(y_true_new)]
    else:
        y_score_new = y_score
    precision, recall, _ = precision_recall_curve(y_true_new, y_score_new)
    fdr = 1 - precision
    cutoff_index = next(i for i, x in enumerate(fdr) if x <= fdr_cutoff)
    return recall[cutoff_index]


def compute_metrics_classification(y_true, y_pred):
    r"""Compute standard performance metrics for predictions of a trained model.
    
    Parameters
    ----------
    y_true : Tensor
        True value for each sample provided as a tensor of shape (n_sample).
    y_pred : Tensor
        Predicted value for each sample provided as a tensor of shape (n_samples).
    
    Returns
    -------
    df_metric : pandas.DataFrame
        Different performance metrics for the provided predictions as a Pandas DataFrame.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    metric = {}
    metric["log.loss"] = log_loss(y_true, y_pred)
    metric["accuracy"] = accuracy_score(y_true, y_pred > 0.5)

    # check for multiclass classification
    if len(y_true.shape) > 1 and len(y_pred.shape) > 1:
        metric["F_score"] = f1_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))
        metric["MCC"] = matthews_corrcoef(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    else:
        metric["F_score"] = f1_score(y_true, y_pred > 0.5)
        metric["MCC"] = matthews_corrcoef(y_true, y_pred > 0.5)

    metric["auROC"] = roc_auc_score(y_true, y_pred)
    metric["auROC50"] = roc_auc_score(y_true, y_pred, max_fpr=0.5)
    metric["auPRC"] = average_precision_score(y_true, y_pred)
    metric["recall_at_10_fdr"] = recall_at_fdr(y_true, y_pred, 0.10)
    metric["recall_at_5_fdr"] = recall_at_fdr(y_true, y_pred, 0.05)
    metric["pearson.r"], metric["pearson.p"] = stats.pearsonr(
        y_true.ravel(), y_pred.ravel()
    )
    metric["spearman.r"], metric["spearman.p"] = stats.spearmanr(
        y_true, y_pred, axis=None
    )

    df_metric = pd.DataFrame.from_dict(metric, orient="index")
    df_metric.columns = ["value"]
    df_metric.sort_index(inplace=True)

    return df_metric


def compute_metrics_regression(y_true, y_pred):
    r"""Compute standard regression performance metrics for predictions of a trained model.
    
    Parameters
    ----------
    y_true : Tensor
        True value for each sample provided as a tensor of shape (n_sample).
    y_pred : Tensor
        Predicted value for each sample provided as a tensor of shape (n_samples).
    
    Returns
    -------
    df_metric : pandas.DataFrame
        Different performance metrics for the provided predictions as a Pandas DataFrame.
    """
    raise NotImplementedError("compute_metrics_regression is not implemented yet!")


class ClassBalanceLoss(torch.nn.Module):
    r"""Implementation of the Class-Balance Loss
    Reference: Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge Belongie; Proceedings of the IEEE/CVF Conference on
               Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9268-9277
    """

    def __init__(
        self, samples_per_cls, no_of_classes, loss_type, beta, gamma, reduction="mean"
    ):
        r"""Constructor of the class-balance loss class

        Parameters
        ----------
        samples_per_cls : list of int
            List containing the number of samples per class in the dataset.
        no_of_classes : int
            Number of classes in the classification problem.
        loss_type : str
            Loss function used for the class-balance loss.
        beta : float
            Hyperparameter for class-balanced loss.
        gamma : float
            Hyperparameter for Focal loss

        Raises
        ------
            ValueError: If len(samples_per_cls) != no_of_classes
        """
        # call constructor of parent class
        super(ClassBalanceLoss, self).__init__()

        # check whether the parameters are valid
        if no_of_classes == 1:
            self.binary = True
        elif len(samples_per_cls) != no_of_classes:
            raise ValueError(
                "Dimensionality of first argument expected to be {}. Found {} instead!".format(
                    no_of_classes, len(samples_per_cls)
                )
            )

        # store user-specified parameters
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.no_of_classes
        print(weights)

    def one_hot(self, labels, num_classes, device, dtype=None, eps=1e-6):
        r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor. Implementation by Kornia
        (https://github.com/kornia).

        Parameters
        ----------
        labels : torch.Tensor
            Tensor with labels of shape :math:`(N, *)`, where N is batch size. Each value 
            is an integer representing correct classification.
        num_classes : int
            Number of classes in labels.
        device : str 
            The desired device of returned tensor.
        dtype : torch.dtype
            The desired data type of returned tensor.
        
        Returns
        -------
        one_hot : torch.Tensor
            The labels in one hot tensor of shape :math:`(N, C, *)`,
        
        Examples
        --------
            >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
            >>> one_hot(labels, num_classes=3)
            tensor([[[[1.0000e+00, 1.0000e-06],
                      [1.0000e-06, 1.0000e+00]],
            <BLANKLINE>
                     [[1.0000e-06, 1.0000e+00],
                      [1.0000e-06, 1.0000e-06]],
            <BLANKLINE>
                     [[1.0000e-06, 1.0000e-06],
                      [1.0000e+00, 1.0000e-06]]]])
        """
        if not isinstance(labels, torch.Tensor):
            raise TypeError(
                f"Input labels type is not a torch.Tensor. Got {type(labels)}"
            )

        if not labels.dtype == torch.int64:
            raise ValueError(
                f"labels must be of the same dtype torch.int64. Got: {labels.dtype}"
            )

        if num_classes < 1:
            raise ValueError(
                "The number of classes must be bigger than one."
                " Got: {}".format(num_classes)
            )

        shape = labels.shape
        one_hot = torch.zeros(
            (shape[0], num_classes) + shape[1:], device=device, dtype=dtype
        )

        return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

    def focal_loss(self, input, target, alpha, gamma=2.0, reduction="none", eps=None):
        """Criterion that computes Focal loss. Implementation by Kornia (https://github.com/kornia).
        According to :cite:`lin2018focal`, the Focal loss is computed as follows:
        .. math::
            \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
        Where:
           - :math:`p_t` is the model's estimated probability for each class.
        Args:
            input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
            target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
            alpha: Weighting factor :math:`\alpha \in [0, 1]`.
            gamma: Focusing parameter :math:`\gamma >= 0`.
            reduction: Specifies the reduction to apply to the
              output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
              will be applied, ``'mean'``: the sum of the output will be divided by
              the number of elements in the output, ``'sum'``: the output will be
              summed.
            eps: Deprecated: scalar to enforce numerical stabiliy. This is no longer used.
        Return:
            the computed loss.
        Example:
            >>> N = 5  # num_classes
            >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
            >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
            >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
            >>> output.backward()
        """
        if eps is not None and not torch.jit.is_scripting():
            warnings.warn(
                "`focal_loss` has been reworked for improved numerical stability "
                "and the `eps` argument is no longer necessary",
                DeprecationWarning,
                stacklevel=2,
            )

        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

        if not len(input.shape) >= 2:
            raise ValueError(
                f"Invalid input shape, we expect BxCx*. Got: {input.shape}"
            )

        if input.size(0) != target.size(0):
            raise ValueError(
                f"Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)})."
            )

        n = input.size(0)
        out_size = (n,) + input.size()[2:]
        if target.size()[1:] != input.size()[2:]:
            raise ValueError(f"Expected target size {out_size}, got {target.size()}")

        if not input.device == target.device:
            raise ValueError(
                f"input and target must be in the same device. Got: {input.device} and {target.device}"
            )

        # compute softmax over the classes axis
        input_soft: torch.Tensor = F.softmax(input, dim=1)
        log_input_soft: torch.Tensor = F.log_softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot: torch.Tensor = self.one_hot(
            target, num_classes=input.shape[1], device=input.device, dtype=input.dtype
        )

        # compute the actual focal loss
        weight = torch.pow(-input_soft + 1.0, gamma)

        focal = -alpha * weight * log_input_soft
        loss_tmp = torch.einsum("bc...,bc...->b...", (target_one_hot, focal))

        if reduction == "none":
            loss = loss_tmp
        elif reduction == "mean":
            loss = torch.mean(loss_tmp)
        elif reduction == "sum":
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {reduction}")
        return loss

    def forward(self, logits, labels):
        r"""Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits) where Loss is one of the standard losses used
        for Neural Networks.
        
        Parameters
        ----------
        logits : torch.Tensor
            Output of the network given as a tensor of shape (batch_size x num_classes).
        labels : torch.Tensor
            True label of each sample given as a tensor of shape (batch_size x num_classes).
        
        Returns
        -------
        cb_loss : torch.Tensor
            A float tensor representing class balanced loss.
        
        Raises
        ------
            ValueError: If an unknown loss function was specified during initialization of the ClassBalanceLoss object.
        """
        if self.binary:
            effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights)

            weights_tensor = torch.tensor(weights[1])
            labels_one_hot = labels
        else:
            effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * self.no_of_classes

            labels_one_hot = F.one_hot(labels, self.no_of_classes).float()

            # we need to adapt the dimensionality of logits if the batch size is 1
            #   -> otherwise logits and labels_one_hot have mismatching dimensionality
            if labels_one_hot.shape[0] == 1:
                logits = logits.view_as(labels_one_hot)

            weights_tensor = labels_one_hot.new_tensor(weights)
            weights_tensor = weights_tensor.unsqueeze(0)
            weights_tensor = (
                weights_tensor.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
            )
            weights_tensor = weights_tensor.sum(1)
            weights_tensor = weights_tensor.unsqueeze(1)
            weights_tensor = weights_tensor.repeat(1, self.no_of_classes)

        if self.loss_type == "focal":
            cb_loss = self.focal_loss(labels_one_hot, logits, weights_tensor)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(
                input=logits,
                target=labels_one_hot,
                pos_weight=weights_tensor,
                reduction=self.reduction,
            )
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(
                input=pred,
                target=labels_one_hot,
                weight=weights_tensor,
                reduction=self.reduction,
            )
        elif self.loss_type == "cross_entropy":
            cb_loss = F.cross_entropy(
                input=logits,
                target=labels,
                weight=torch.tensor(weights).float(),
                reduction=self.reduction,
            )
        else:
            raise ValueError(
                "Undefined loss function: {}.".format(self.loss_type)
                + "\n            Valid values are 'focal', 'sigmoid', 'softmax', and 'cross_entropy'."
            )

        return cb_loss
