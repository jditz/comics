# -*- coding: utf-8 -*-

r"""
This module implements functions used to learn graphical representations for
data given sample observations from these variables
"""

import time

import numpy as np
from pyunlocbox import acceleration, functions, solvers
from scipy import sparse, spatial
from sklearn.covariance import graphical_lasso


def weight2degmap(N, return_array=False):
    r"""
    Generate linear operator K such that W @ 1 = K @ vec(W).

    Parameters
    ----------
    N : int
        Number of nodes on the graph

    Returns
    -------
    K : function
        Operator such that K(w) is the vector of node degrees
    Kt : function
        Adjoint operator mapping from degree space to edge weight space
    array : boolean, optional
        Indicates if the maps are returned as array (True) or callable (False).

    Notes
    -----
    Used in :func:`log_degree_barrier method`.
    """
    import numpy as np

    Ne = int(N * (N - 1) / 2)  # Number of edges
    row_idx1 = np.zeros((Ne,))
    row_idx2 = np.zeros((Ne,))
    count = 0
    for i in np.arange(1, N):
        row_idx1[count : (count + (N - i))] = i - 1
        row_idx2[count : (count + (N - i))] = np.arange(i, N)
        count = count + N - i
    row_idx = np.concatenate((row_idx1, row_idx2))
    col_idx = np.concatenate((np.arange(0, Ne), np.arange(0, Ne)))
    vals = np.ones(len(row_idx))
    K = sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(N, Ne))
    if return_array:
        return K, K.transpose()
    else:
        return lambda w: K.dot(w), lambda d: K.transpose().dot(d)


def log_degree_barrier(
    X,
    dist_type="sqeuclidean",
    alpha=1,
    beta=1,
    step=0.5,
    w0=None,
    maxit=1000,
    rtol=1e-5,
    retall=False,
    verbosity="ALL",
):
    r"""
    Learn graph by imposing a log barrier on the degrees
    This is done by solving
    :math:`\tilde{W} = \underset{W \in \mathcal{W}_m}{\text{arg}\min} \,
    \|W \odot Z\|_{1,1} - \alpha 1^{T} \log{W1} + \beta \| W \|_{F}^{2}`,
    where :math:`Z` is a pairwise distance matrix, and :math:`\mathcal{W}_m`
    is the set of valid symmetric weighted adjacency matrices.

    Parameters
    ----------
    X : array_like
        An N-by-M data matrix of M variable observations in an N-dimensional
        space. The learned graph will have N nodes.
    dist_type : string
        Type of pairwise distance between variables. See
        :func:`spatial.distance.pdist` for the possible options.
    alpha : float, optional
        Regularization parameter acting on the log barrier
    beta : float, optional
        Regularization parameter controlling the density of the graph
    step : float, optional
        A number between 0 and 1 defining a stepsize value in the admissible
        stepsize interval (see [Komodakis & Pesquet, 2015], Algorithm 6)
    w0 : array_like, optional
        Initialization of the edge weights. Must be an N(N-1)/2-dimensional
        vector.
    maxit : int, optional
        Maximum number of iterations.
    rtol : float, optional
        Stopping criterion. Relative tolerance between successive updates.
    retall : boolean
        Return solution and problem details. See output of
        :func:`pyunlocbox.solvers.solve`.
    verbosity : {'NONE', 'LOW', 'HIGH', 'ALL'}, optional
        Level of verbosity of the solver. See :func:`pyunlocbox.solvers.solve`.

    Returns
    -------
    W : array_like
        Learned weighted adjacency matrix
    problem : dict, optional
        Information about the solution of the optimization. Only returned if
        retall == True.

    Notes
    -----
    This is the solver proposed in [Kalofolias, 2016].
    """

    # Parse X
    N = X.shape[0]
    z = spatial.distance.pdist(X, dist_type)  # Pairwise distances

    # Parse stepsize
    if (step <= 0) or (step > 1):
        raise ValueError("step must be a number between 0 and 1.")

    # Parse initial weights
    w0 = np.zeros(z.shape) if w0 is None else w0
    if w0.shape != z.shape:
        raise ValueError("w0 must be of dimension N(N-1)/2.")

    # Get primal-dual linear map
    K, Kt = weight2degmap(N)
    norm_K = np.sqrt(2 * (N - 1))

    # Assemble functions in the objective
    f1 = functions.func()
    f1._eval = lambda w: 2 * np.dot(w, z)
    f1._prox = lambda w, gamma: np.maximum(0, w - (2 * gamma * z))

    f2 = functions.func()
    f2._eval = lambda w: -alpha * np.sum(
        np.log(np.maximum(np.finfo(np.float64).eps, K(w)))
    )
    f2._prox = lambda d, gamma: np.maximum(
        0, 0.5 * (d + np.sqrt(d ** 2 + (4 * alpha * gamma)))
    )

    f3 = functions.func()
    f3._eval = lambda w: beta * np.sum(w ** 2)
    f3._grad = lambda w: 2 * beta * w
    lipg = 2 * beta

    # Rescale stepsize
    stepsize = step / (1 + lipg + norm_K)

    # use acceleration for large graphs
    if N > 1000:
        accel = acceleration.fista()
    else:
        accel = None

    # Solve problem
    solver = solvers.mlfbf(L=K, Lt=Kt, step=stepsize, accel=accel)
    problem = solvers.solve(
        [f1, f2, f3], x0=w0, solver=solver, maxit=maxit, rtol=rtol, verbosity=verbosity
    )

    # Transform weight matrix from vector form to matrix form
    W = spatial.distance.squareform(problem["sol"])

    if retall:
        return W, problem
    else:
        return W


def l2_degree_reg(
    X,
    dist_type="sqeuclidean",
    alpha=1,
    s=None,
    step=0.5,
    w0=None,
    maxit=1000,
    rtol=1e-5,
    retall=False,
    verbosity="ALL",
):
    r"""
    Learn graph by regularizing the l2-norm of the degrees.
    This is done by solving
    :math:`\tilde{W} = \underset{W \in \mathcal{W}_m}{\text{arg}\min} \,
    \|W \odot Z\|_{1,1} + \alpha \|W1}\|^2 + \alpha \| W \|_{F}^{2}`, subject
    to :math:`\|W\|_{1,1} = s`, where :math:`Z` is a pairwise distance matrix,
    and :math:`\mathcal{W}_m`is the set of valid symmetric weighted adjacency
    matrices.

    Parameters
    ----------
    X : array_like
        An N-by-M data matrix of M variable observations in an N-dimensional
        space. The learned graph will have N nodes.
    dist_type : string
        Type of pairwise distance between variables. See
        :func:`spatial.distance.pdist` for the possible options.
    alpha : float, optional
        Regularization parameter acting on the l2-norm.
    s : float, optional
        The "sparsity level" of the weight matrix, as measured by its l1-norm.
    step : float, optional
        A number between 0 and 1 defining a stepsize value in the admissible
        stepsize interval (see [Komodakis & Pesquet, 2015], Algorithm 6)
    w0 : array_like, optional
        Initialization of the edge weights. Must be an N(N-1)/2-dimensional
        vector.
    maxit : int, optional
        Maximum number of iterations.
    rtol : float, optional
        Stopping criterion. Relative tolerance between successive updates.
    retall : boolean
        Return solution and problem details. See output of
        :func:`pyunlocbox.solvers.solve`.
    verbosity : {'NONE', 'LOW', 'HIGH', 'ALL'}, optional
        Level of verbosity of the solver. See :func:`pyunlocbox.solvers.solve`.

    Returns
    -------
    W : array_like
        Learned weighted adjacency matrix
    problem : dict, optional
        Information about the solution of the optimization. Only returned if
        retall == True.

    Notes
    -----
    This is the problem proposed in [Dong et al., 2015].
    """

    # Parse X
    N = X.shape[0]
    E = int(N * (N - 1.0) / 2.0)
    z = spatial.distance.pdist(X, dist_type)  # Pairwise distances

    # Parse s
    s = N if s is None else s

    # Parse step
    if (step <= 0) or (step > 1):
        raise ValueError("step must be a number between 0 and 1.")

    # Parse initial weights
    w0 = np.zeros(z.shape) if w0 is None else w0
    if w0.shape != z.shape:
        raise ValueError("w0 must be of dimension N(N-1)/2.")

    # Get primal-dual linear map
    one_vec = np.ones(E)

    def K(w):
        return np.array([2 * np.dot(one_vec, w)])

    def Kt(n):
        return 2 * n * one_vec

    norm_K = 2 * np.sqrt(E)

    # Get weight-to-degree map
    S, St = weight2degmap(N)

    # Assemble functions in the objective
    f1 = functions.func()
    f1._eval = lambda w: 2 * np.dot(w, z)
    f1._prox = lambda w, gamma: np.maximum(0, w - (2 * gamma * z))

    f2 = functions.func()
    f2._eval = lambda w: 0.0
    f2._prox = lambda d, gamma: s

    f3 = functions.func()
    f3._eval = lambda w: alpha * (2 * np.sum(w ** 2) + np.sum(S(w) ** 2))
    f3._grad = lambda w: alpha * (4 * w + St(S(w)))
    lipg = 2 * alpha * (N + 1)

    # Rescale stepsize
    stepsize = step / (1 + lipg + norm_K)

    # use acceleration for large graphs
    if N > 1000:
        accel = acceleration.fista()
    else:
        accel = None

    # Solve problem
    solver = solvers.mlfbf(L=K, Lt=Kt, step=stepsize, accel=accel)
    problem = solvers.solve(
        [f1, f2, f3], x0=w0, solver=solver, maxit=maxit, rtol=rtol, verbosity=verbosity
    )

    # Transform weight matrix from vector form to matrix form
    W = spatial.distance.squareform(problem["sol"])

    if retall:
        return W, problem
    else:
        return W


def glasso(X, alpha=1, w0=None, maxit=1000, rtol=1e-5, retall=False, verbosity="ALL"):
    r"""
    Learn graph by imposing promoting sparsity in the inverse covariance.
    This is done by solving
    :math:`\tilde{W} = \underset{W \succeq 0}{\text{arg}\min} \,
    -\log \det W - \text{tr}(SW) + \alpha\|W \|_{1,1},
    where :math:`S` is the empirical (sample) covariance matrix.

    Parameters
    ----------
    X : array_like
        An N-by-M data matrix of M variable observations in an N-dimensional
        space. The learned graph will have N nodes.
    alpha : float, optional
        Regularization parameter acting on the l1-norm
    w0 : array_like, optional
        Initialization of the inverse covariance. Must be an N-by-N symmetric
        positive semi-definite matrix.
    maxit : int, optional
        Maximum number of iterations.
    rtol : float, optional
        Stopping criterion. If the dual gap goes below this value, iterations
        are stopped. See :func:`sklearn.covariance.graph_lasso`.
    retall : boolean
        Return solution and problem details.
    verbosity : {'NONE', 'ALL'}, optional
        Level of verbosity of the solver.
        See :func:`sklearn.covariance.graph_lasso`/

    Returns
    -------
    W : array_like
        Learned inverse covariance matrix
    problem : dict, optional
        Information about the solution of the optimization. Only returned if
        retall == True.

    Notes
    -----
    This function uses the solver :func:`sklearn.covariance.graphical_lasso`.
    """

    # Parse X
    S = np.cov(X)

    # Parse initial point
    w0 = np.ones(S.shape) if w0 is None else w0
    if w0.shape != S.shape:
        raise ValueError("w0 must be of dimension N-by-N.")

    # Solve problem
    tstart = time.time()
    res = graphical_lasso(
        emp_cov=S,
        alpha=alpha,
        cov_init=w0,
        mode="cd",
        tol=rtol,
        max_iter=maxit,
        verbose=(verbosity == "ALL"),
        return_costs=True,
        return_n_iter=True,
    )

    problem = {
        "sol": res[1],
        "dual_sol": res[0],
        "solver": "sklearn.covariance.graph_lasso",
        "crit": "dual_gap",
        "niter": res[3],
        "time": time.time() - tstart,
        "objective": np.array(res[2])[:, 0],
    }

    W = problem["sol"]

    if retall:
        return W, problem
    else:
        return W


# Dictionary for easy access to learning frameworks
graph_learners = {
    "glasso": glasso,
    "dong": l2_degree_reg,
    "kalofolias": log_degree_barrier,
}
