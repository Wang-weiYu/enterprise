import numpy as np
import pytest

from enterprise.signals import gp_signals, signal_base


@pytest.mark.parametrize("marginalizing", [False, True])
def test_likelihood_resolves_small_postfit_residual(marginalizing):
    # The pre-fit quadratic is 1e32, but the post-fit quadratic is 1.
    basis = np.array([[1.0], [0.0]])
    residual = np.array([1e16, 1.0])
    noise = signal_base.ndarray_alt(np.ones(2))

    class Collection:
        _residuals = residual

        def get_detres(self, params):
            return residual

    class PTA:
        _commonsignals = False
        _signalcollections = [Collection()]

        def get_TNr(self, params):
            return [None if marginalizing else basis.T @ residual]

        def get_TNT(self, params):
            return [None if marginalizing else basis.T @ basis]

        def get_phiinv(self, params, logdet, method):
            return [None if marginalizing else (np.array([1e-40]), np.log(1e40))]

        def get_basis(self, params):
            return [None if marginalizing else basis]

        def get_ndiag(self, params):
            return [gp_signals.MarginalizingNmat(basis, noise) if marginalizing else noise]

        def get_logsignalprior(self, params):
            return [0.0]

    expected = -0.5 * (1 + np.log(1e40) + 2 * np.log(2 * np.pi))
    assert signal_base.LogLikelihood(PTA())({}) == pytest.approx(expected, abs=1e-4, rel=0)
