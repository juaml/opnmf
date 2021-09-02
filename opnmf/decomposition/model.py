from sklearn.base import BaseEstimator, TransformerMixin

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

        # Run factorization
        W, H = opnmf(X, n_components=self.n_components, max_iter=self.max_iter,
                     tol=self.tol, init=self.init, init_W=init_W)

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
        raise NotImplementedError("Don't know how to do this!")
