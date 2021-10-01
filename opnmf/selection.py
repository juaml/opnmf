import numpy as np
from . import model
from . logging import logger


def _shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def rank_permute(X, min_components, max_components, step=1, max_iter=50000,
                 tolerance=1e-5, init='nndsvd', init_W=None):
    """
    Orthogonal projective non-negative matrix factorization.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        Data matrix to be decomposed
    min_components: int
        Lower bound of the number of components to test.
    max_components: int
        Upper bound of the number of components to test.
    step: int
        Spacing between values in the components range.
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
        * 'custom': use custom matrix W.

    init_W: array (n_samples, n_components)
        Fixed initial coefficient matrix.

    Returns
    -------
    good_ranks: array
        Array with the number of components that were selected.
    tested_ranks: array
        Array with the number of components that were tested.
    errors: array
        Reconstruction error for each number of components tested.
    random_errors: array
        Reconstruction error for the random permutation, for each number of
        components tested.
    estimators: array
        The fitted estimators for each number of components tested.
    """
    ranks = np.arange(min_components, max_components + 1, step)

    logger.info(f'Choosing ranks between: {ranks}')

    estimators = [
        model.OPNMF(n_components=t_rank, max_iter=max_iter, tol=tolerance,
                    init=init)
        for t_rank in ranks
    ]

    # Fit permuted
    logger.info('Fitting estimators with random permutations')
    X_perm = _shuffle_along_axis(X, 0)
    random_errors = [estimator.fit(X_perm, init_W=init_W).mse()
                     for estimator in estimators]

    logger.info('Fitting estimators with original data')
    # Fit original
    errors = [estimator.fit(X, init_W=init_W).mse()
              for estimator in estimators]

    errors = errors / np.max(errors)
    random_errors = random_errors / np.max(random_errors)
    is_good_rank = np.diff(errors) > np.diff(random_errors)

    good_ranks = ranks[np.where(is_good_rank)[0]]

    return good_ranks, ranks, errors, random_errors, estimators
