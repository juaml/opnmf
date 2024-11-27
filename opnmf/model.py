import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . opnmf import opnmf
from . selection import rank_permute
from . logging import logger


class OPNMF(TransformerMixin, BaseEstimator):
    """ orthogonal projective non-negative matrix factorization

    Parameters
    ----------
    n_components: int
        Number of components.
    alpha: int
        Constant that multiplies the regularization terms.
        Set it to zero to have no regularization. Defaults to 1.0.
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

    """

    def __init__(self, n_components=10, max_iter=50000, tol=1e-5,
                 init='nndsvd'):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init = init

    def fit(self, X, init_W=None):
        """ Learn a OPNMF model for the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed
        init_W : array-like of shape (n_samples, n_components)
            If init='custom', it is used as initial guess for the solution.

        Returns
        -------
        self
        """
        self.fit_transform(X, init_W=init_W)
        return self

    def fit_transform(self, X, init_W=None):
        """ Learn a OPNMF model for the data X and returns the transformed
        data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed
        init_W : array-like of shape (n_samples, n_components)
            If init='custom', it is used as initial guess for the solution.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data
        """

        if self.n_components == 'auto' or isinstance(self.n_components, range):
            logger.info('Doing rank selection')
            if self.n_components == 'auto':
                logger.info('Determining number of components automatically')
                min_components = 1
                max_components = X.shape[0]
                step = 1
            else:
                min_components = range.start
                max_components = range.stop
                step = range.step
            out = rank_permute(
                X, min_components, max_components, step=step,
                max_iter=self.max_iter, tolerance=self.tol, init=self.init,
                init_W=init_W)
            good_ranks, ranks, errors, random_errors, estimators = out
            chosen = estimators[good_ranks[0] - 1]
            W = chosen.coef_
            H = chosen.components_
            mse = chosen.mse_
            self.ranks_ = ranks
            self.errors_ = errors
            self.random_errors_ = random_errors
            self.good_ranks_ = good_ranks
        elif not np.issubdtype(type(self.n_components), int):
            raise ValueError('Do not know how to factorize to '
                             f'{self.n_components} components')
        else:
            # Run factorization
            W, H, mse = opnmf(
                X, n_components=self.n_components, max_iter=self.max_iter,
                tol=self.tol, init=self.init, init_W=init_W)

        # Set model variables
        self.coef_ = W
        self.n_components_ = H.shape[0]
        self.components_ = H
        self.mse_ = mse

        return self.coef_

    def transform(self, X):
        """Transform the data X according to the fitted OPNMF model.
        Added functionality by Alfie Wearn 2024-01-20

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be transformed by the model.

        Returns
        -------
        H : ndarray of shape (n_components, n_features)
            Transformed data.
            As the OPNMF is: X~W*H, this calculates a new H given pre-calculated W and a new X.
        """
        # Ensure the model is fitted
        check_is_fitted(self, 'components_')

        # Apply transformation to new subjects
        # Use the fixed W (self.coef_) learned during training to transform the new data
        _, H, _ = opnmf(X, n_components=self.n_components_, W_fixed=self.coef_,
                        max_iter=self.max_iter, tol=self.tol)
                        
        # Calculate the reconstruction
        X_reconstructed = self.coef_ @ H
        
        # Calculate MSE
        mse = np.linalg.norm(X - (self.coef_ @ H), ord='fro')

        return H, mse

    def mse(self):
        check_is_fitted(self)
        return self.mse_
