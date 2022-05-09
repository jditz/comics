r"""Module of utility functionalities used in different parts of the
package.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans


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
        already_samples += current_size

    # return the sampled data points
    print(f"sample_data rounting returned {already_sampled} sampled data points")
    return samples[:already_sampled, :]
