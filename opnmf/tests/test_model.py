from numpy.testing import assert_array_almost_equal
import scipy.io as sio
import pytest
from pathlib import Path
from opnmf import opnmf, model


@pytest.fixture(scope="module")
def cwd(request):
    return Path(request.fspath).parent


def test_model(cwd):
    """Test scikit-learn style model"""
    opnmf_model = model.OPNMF(n_components=6)
    mc = sio.loadmat(cwd / 'data/faces_data.mat')

    data = mc['data']
    m_W = opnmf_model.fit_transform(data)
    m_H = opnmf_model.components_

    W, H, _ = opnmf.opnmf(data, n_components=6)

    assert_array_almost_equal(m_W, W, decimal=3)
    assert_array_almost_equal(m_H, H, decimal=3)


def test_model_rank_selection(cwd):
    """Test auto rank selection"""
    opnmf_model = model.OPNMF(n_components='auto')
    mc = sio.loadmat(cwd / 'data/faces_data.mat')

    data = mc['data']
    data = data[:4]

    m_W = opnmf_model.fit_transform(data)
    m_H = opnmf_model.components_

    # This chooses 2 components
    W, H, _ = opnmf.opnmf(data, n_components=2)

    assert_array_almost_equal(m_W, W, decimal=4)
    assert_array_almost_equal(m_H, H, decimal=4)
