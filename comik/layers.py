r"""This module implements custom layers that are needed to create a 
comik network.
"""

import math

import numpy as np
import torch
from torch.nn.init import kaiming_uniform_

from .graph_learner import graph_learners
from .utils import kmeans


class MatrixInverseSqrt(torch.autograd.Function):
    r"""Matrix inverse square root for a symmetric definite positive matrix
    """

    @staticmethod
    def forward(ctx, input, eps=1e-2):
        dim = input.dim()
        ctx.dim = dim
        use_cuda = input.is_cuda
        if input.size(0) < 300:
            input = input.cpu()
        e, v = torch.linalg.eigh(input, UPLO="U")
        if use_cuda and input.size(0) < 300:
            e = e.cuda()
            v = v.cuda()
        e = e.clamp(min=0)
        e_sqrt = e.sqrt_().add_(eps)
        ctx.save_for_backward(e_sqrt, v)
        e_rsqrt = e_sqrt.reciprocal()

        if dim > 2:
            output = v.bmm(v.permute(0, 2, 1) * e_rsqrt.unsqueeze(-1))
        else:
            output = v.mm(v.t() * e_rsqrt.view(-1, 1))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        e_sqrt, v = ctx.saved_variables
        if ctx.dim > 2:
            ei = e_sqrt.unsqueeze(1).expand_as(v)
            ej = e_sqrt.unsqueeze(-1).expand_as(v)
        else:
            ei = e_sqrt.expand_as(v)
            ej = e_sqrt.view(-1, 1).expand_as(v)
        f = torch.reciprocal((ei + ej) * ei * ej)
        if ctx.dim > 2:
            vt = v.permute(0, 2, 1)
            grad_input = -v.bmm((f * (vt.bmm(grad_output.bmm(v)))).bmm(vt))
        else:
            grad_input = -v.mm((f * (v.t().mm(grad_output.mm(v)))).mm(v.t()))
        return grad_input, None


def matrix_inverse_sqrt(input, eps=1e-2):
    r"""Wrapper for MatrixInverseSqrt"""
    return MatrixInverseSqrt.apply(input, eps)


class ComikLayer(torch.nn.Module):
    r"""Convolutional Graph Kernel Layer

    This layer implements a convolutional graph kernel layer as described in the paper.
    The weights of ComikLayers represent the anchor points, i.e. representative real-valued
    graph signal representation, and each layer has a graph Laplacian as a parameter.
    The anchor points are optimized with back-propagation and the graph Laplacian is
    optimized in a second step by solving the optimization problem of the specified graph
    learning framework.
    """

    def __init__(
        self,
        num_nodes: int,
        num_anchors: int,
        graph_learner: str,
        graph_learner_params=None,
    ):
        r"""Constructor of the ComikLayer class

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph. Since this layer is designed to get
            graph signal reprentations as input, num_nodes is equivalent to the
            dimensionality of inputs.
        num_anchors : int
            Number of anchor points of the layer.
        graph_learner : str
            Framework that will be used to learn a graph from the optimized
            anchor points. The framework is called after every optimization step,
            i.e. everytime the anchor points changed.
        graph_learner_params : Dictionary
            Dictionary to provide parameter settings for the graph learning framework.
            Available parameters for each framework can be found in the corresponding
            doc strings. If no dictionary is provided, the default parameters will be
            used. The dictionary has to be of the form {'param_name': param_value}.
        """

        # call constructor of parent class
        super().__init__()

        # store attributes
        self.num_nodes = num_nodes
        self.num_anchors = num_anchors

        # initialize parameters and buffers
        #   -> the Laplacian and the linear transformation matrix used in the Nyström
        #      approximation will NOT be optimized using gradients, hence, we do not
        #      calculate gradients for the Laplacian or linear transformation matrix
        #
        #   -> the anchor points will be initialized with random values; to initialize
        #      anchor points with clustering one have to call the init_weights method
        #      with appropriate parameters
        self._need_lintrans_computed = True
        self.register_buffer("lintrans", torch.Tensor(num_anchors, num_anchors))
        self.register_buffer("L", torch.Tensor(num_nodes, num_nodes))
        self.register_parameter(
            "weight", torch.nn.Parameter(torch.Tensor(num_anchors, num_nodes))
        )
        self.init_weights()

        # initialize the graph learning framework
        gl_framework = graph_learners[graph_learner]
        if graph_learner_params is None:
            self.graph_learner = lambda x: gl_framework(x)
        else:
            self.graph_learner = lambda x: gl_framework(x, **graph_learner_params)

    def train(self, mode=True):
        """Toggle between training (mode = True) and evaluation mode (mode = False)
        """
        super(ComikLayer, self).train(mode)
        if self.training is True:
            self._need_lintrans_computed = True

    def eval(self):
        """Sets model in evaluation mode
        """
        self.train(False)

    def init_weights(
        self,
        random_init: bool = True,
        data: torch.Tensor = None,
        distance: str = "euclidian",
        max_iters: int = 100,
        verbose: bool = True,
        init: str = None,
        tol: float = 1e-4,
        use_cuda: bool = False,
    ):
        """Method to initialize the anchor points
        
        This function either uses the kaiming routine to initialize the anchor
        points with uniformly distributed random values or takes a set of
        real-valued graph signal representations and performs K-Means clustering
        to calculate n cluster centers, where n is equal to self.num_anchors.
        Afterwards, The anchor point will be initialized with the cluster centers.

        Parameters
        ----------
        random_init : bool
            Flag that indicates how the anchor points should be initialized.
            Defaults to True.
        data : Tensor
            Data that will be used for clustering provided as a tensor of shape
            (n_samples x n_dimensions).
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
        """

        if random_init:
            kaiming_uniform_(self.weight, a=math.sqrt(5))

        else:
            cluster_centers = kmeans(
                data,
                n_clusters=self.num_anchors,
                distance=distance,
                max_iters=max_iters,
                verbose=verbose,
                init=init,
                tol=tol,
                use_cuda=use_cuda,
            )

            self.weight.data = cluster_centers

    def forward(self, x_in):
        """Definition of the computation performed on every call

        Parameters
        ----------
        x_in : Tensor
            Tensor containing a batch of graph signal representations. This tensor has to
            be of shape (batch_size x num_nodes).
        """
        # evaluate the kernel function between the inputs and anchor points
        x_out = self._kernel_func(x_in)

        # calculate the linear transformation factor (if needed)
        lintrans = self._compute_lintrans()

        # project each input onto the RKHS' subspace
        x_out = self._project_onto_subspace(x_in, lintrans)
        return x_out

    def _kernel_func(self, x_in):
        """Evaluation of the kernel function between each input and each anchor point

        Parameters
        ----------
        x_in : Tensor
            Tensor containing a batch of graph signal representations. This tensor has to
            be of shape (batch_size x num_nodes).

        Returns
        -------
        x_out : Tensor
            Matrix containing the kernel evaluation between each input and each anchor point.
            The tensor is of shape (batch_size x num_anchors)
        """
        # evaluate the kernel function xLy for each input-anchor pair
        x_out = torch.mm(torch.mm(x_in, self.L), self.weight.t())
        return x_out

    def _compute_lintrans(self):
        """Compute the linear transformation factor K_(ZZ)^(-1/2)

        Returns
        -------
        lintrans : Tensor
            Returns the linear transformation factor needed for the Nyström method as a tensor
            of shape (num_anchor x num_anchor)
        """
        # return the current linear transformation matrix, if there is no need to calculate
        # a new matrix
        if not self._need_lintrans_computed:
            return self.lintrans

        # calculate the new linear transformation matrix using the current anchor points
        lintrans = self._kernel_func(self.weight)
        lintrans = matrix_inverse_sqrt(lintrans)

        # if model is in evaluation mode, do not recompute linear transformation matrix
        if not self.training:
            self._need_lintrans_computed = False
            self.lintrans.data = lintrans.data

        # return the linear transformation matrix
        return lintrans

    def _project_onto_subspace(self, x_in, lintrans):
        """Projection of inputs onto the RKHS subspace that is spanned by the
        anchor points

        Parameters
        ----------
        x_in : Tensor
            Result of the kernel evaluation between each input and each anchor point.
            Tensor has to be of shape (batch_size x num_anchors)
        lintrans: Tensor
            Linear transformation matrix to project each input onto the RKHS subspace
            that is spanned by the anchor points. Tensor has to be of shape
            (num_anchors x num_anchors)

        Returns
        -------
        x_out : Tensor
            Tensor containing the projections of each input onto the subspace. The shape
            of the tensor is (batch_size x num_anchors)
        """
        batch_size, _, _ = x_in.size()

        # calculate normal matrix multiplication or batch matrix multiplication depending on whether input data is
        # presented in batch mode
        if x_in.dim() == 2:
            return torch.mm(x_in, lintrans)
        return torch.bmm(
            lintrans.expand(batch_size, self.num_anchors, self.num_anchors), x_in
        )

    def learn_graph(self):
        """Function to update the graph Laplacian given the current set of anchor points
        """
        # call the chosen graph learning framework with the current set of anchor points
        W = self.graph_learner(self.weight.t().detach().cpu().numpy())

        # calculate the degree matrix D = diag(W1) with 1 = [1, ..., 1]^T
        D = np.diag(np.matmul(W, np.ones(self.num_nodes)))

        # calculate the fraph Laplacian L = D - W
        self.L.data = D - W

