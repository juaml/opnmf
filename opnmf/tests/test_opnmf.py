import numpy as np
import scipy.io as sio
import pytest
from pathlib import Path
from opnmf.decomposition import opnmf


@pytest.fixture(scope="module")
def cwd(request):
    return Path(request.fspath).parent


def test_against_matlab(cwd):
    """Test against matlab implementation"""
    mc = sio.loadmat(cwd / 'data/faces_data.mat')

    data = mc['data']
    init_W = mc['init_W']

    W, H = opnmf.opnmf(data, n_components=6, max_iter=50000, tol=1e-5,
                       init_W=init_W, init='custom')

    mc = sio.loadmat(cwd / 'data/opnmf6_matlab.mat')

    m_W = mc['W']
    m_H = mc['H']

    np.testing.assert_array_almost_equal(H, m_H, decimal=4)
    np.testing.assert_array_almost_equal(W, m_W, decimal=4)


def test_against_R(cwd):
    """Test against R implementation"""
    mc = sio.loadmat(cwd / 'data/faces_data.mat')

    data = mc['data']
    init_W = mc['init_W']

    W, H = opnmf.opnmf(data, n_components=6, max_iter=50000, tol=1e-5,
                       init_W=init_W, init='custom')

    mc_R = sio.loadmat(cwd / 'data/opnmf6_R.mat')

    r_W = mc_R['W']
    r_H = mc_R['H']

    np.testing.assert_array_almost_equal(H, r_H, decimal=5)
    np.testing.assert_array_almost_equal(W, r_W, decimal=5)
