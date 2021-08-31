from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . opnmf import opnmf


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

    """

    def __init__(self, n_components=10, alpha=1.0, max_iter=200, tol=1e-4):
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        """ Learn a OPNMF model for the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed

        Returns
        -------
        self
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        """ Learn a OPNMF model for the data X and returns the transformed
        data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data
        """

        # Run factorization
        W, H = opnmf(X, n_components=self.n_components, alpha=self.alpha,
                     max_iter=self.max_iter, tol=self.tol)

        # Set model variables
        self.coef_ = W
        self.n_components_ = H.shape[0]
        self.components_ = H

        return self.coef_

    def transform(self, X):
        """Transform the data X according to the fitted OPNMF model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be transformed by the model.
        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        W, _ = opnmf(X, n_components=self.n_components,  alpha=self.alpha,
                     max_iter=self.max_iter, tol=self.tol,
                     init_H=self.components_)
        return W
