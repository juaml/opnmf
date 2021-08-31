import warnings
import numpy as np

from sklearn.decomposition._nmf import _initialize_nmf


def opnmf(X, n_components, alpha=1.0, max_iter=200, tol=1e-4,
          init_H=None, init_W=None):
    """
    Orthogonal projective non-negative matrix factorization.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        Data matrix to be decomposed
    n_components: int
        Number of components.
    alpha: int
        Constant that multiplies the regularization terms.
        Set it to zero to have no regularization. Defaults to 1.0.
    max_iter: int
        Maximum number of iterations before timing out. Defaults to 200.
    tol: float, default=1e-4
        Tolerance of the stopping condition.
    init_H: array-like of shape (n_components, n_features)
        Fixed initial basis matrix.
    init_W: array (n_samples, n_components)
        Fixed initial coefficient matrix.

    Returns
    W : ndarray of shape (n_samples, n_components)
        Solution to the orthogonal non-negative factorization.
    H : ndarray of shape (n_components, n_features)
        Solution to the orthogonal non-negative factorization.
    """
    W_, H_ = _initialize_nmf(X, n_components, init='nndsvd')
    W = W_ if init_W is None else init_W
    H = H_ if init_H is None else init_H
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(max_iter):
            old_W = W
            if init_W is None:
                enum = X.dot(H.T)
                denom = W.dot(H.dot(H.T))
                W = np.nan_to_num(W * enum / denom)

            if init_H is None:
                HHTH = H.dot(H.T).dot(H)
                enum = W.T.dot(X) + alpha * H
                denom = W.T.dot(W).dot(H) + 2.0 * alpha * HHTH
                H = np.nan_to_num(H * enum / denom)

            delta_W = (np.linalg.norm(old_W - W, ord='fro') /
                       np.linalg.norm(old_W, ord='fro'))
            if delta_W < tol:
                break
    # TODO: sort components (see R code)
    return W, H
