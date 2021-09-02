import warnings
import numpy as np

from sklearn.decomposition._nmf import _initialize_nmf

from .. logging import logger, warn


def opnmf(X, n_components, max_iter=50000, tol=1e-5, init='nndsvd',
          init_W=None):
    """
    Orthogonal projective non-negative matrix factorization.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        Data matrix to be decomposed
    n_components: int
        Number of components.
    max_iter: int
        Maximum number of iterations before timing out. Defaults to 200.
    tol: float, default=1e-4
        Tolerance of the stopping condition.
    init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, default=None
        Method used to initialize the procedure.
        Valid options:

        * None: 'nndsvd' if n_components < n_features, otherwise 'random'.
        * 'random': non-negative random matrices, scaled with:
          sqrt(X.mean() / n_components)
        * 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
          initialization (better for sparseness)
        * 'nndsvda': NNDSVD with zeros filled with the average of X
          (better when sparsity is not desired)
        * 'nndsvdar': NNDSVD with zeros filled with small random values
          (generally faster, less accurate alternative to NNDSVDa
          for when sparsity is not desired)
        * 'custom': use custom matrices W and H if `update_H=True`. If
          `update_H=False`, then only custom matrix H is used.

    init_W: array (n_samples, n_components)
        Fixed initial coefficient matrix.

    Returns
    -------
    W : ndarray of shape (n_samples, n_components)
        The orthogonal non-negative factorization.
    H : ndarray of shape (n_components, n_features)
        Expansion coefficients.
    """
    if init != 'custom':
        if init_W is not None:
            warn('Initialisation was not set to "custom" but an initial W '
                 'matrix was specified. This matrix will be ignored.')
        logger.info(f'Initializing using {init}')
        W, _ = _initialize_nmf(X, n_components, init=init)
        init_W = None
    else:
        W = init_W
    delta_W = np.inf
    XX = X @ X.T

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for iter in range(max_iter):
            old_W = W

            enum = XX @ W
            denom = W @ (W.T @ XX @ W)
            W = W * enum / denom

            W[W < 1e-16] = 1e-16
            W = W / np.linalg.norm(W, ord=2)

            delta_W = (np.linalg.norm(old_W - W, ord='fro') /
                       np.linalg.norm(old_W, ord='fro'))
            if (iter % 100) == 0:
                obj = np.linalg.norm(X - W @ (W.T @ X), ord='fro')
                logger.info(f'iter={iter} diff={delta_W}, obj={obj}')
            if delta_W < tol:
                logger.info(f'Converged in {iter} iterations')
                break

    if delta_W > tol:
        warn('OPNMF did not converge with '
             f'tolerance = {tol} under {max_iter} iterations')

    H = W.T @ X

    hlen = np.linalg.norm(H, ord=2, axis=1)
    n_zero = np.sum(hlen == 0)
    if n_zero > 0:
        warnings.warn(f'low rank: {n_zero} factors have norm 0')
        hlen[hlen == 0] = 1

    Wh = W * hlen
    Whlen = np.linalg.norm(Wh, ord=2, axis=0)
    idx = np.argsort(-1 * Whlen)
    W = W[:, idx]
    H = W.T @ X

    return W, H
