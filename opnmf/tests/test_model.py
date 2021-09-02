import numpy as np
import scipy.io as sio
import pytest
from pathlib import Path
from opnmf.decomposition import opnmf, model


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

    W, H = opnmf.opnmf(data, n_components=6)

    np.testing.assert_array_almost_equal(m_W, W, decimal=4)
    np.testing.assert_array_almost_equal(m_H, H, decimal=4)
