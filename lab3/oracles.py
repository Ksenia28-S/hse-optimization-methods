import numpy as np
import scipy
from scipy.special import expit
from scipy.sparse import issparse


class BaseSmoothOracle(object):
    """
    Base class for smooth function.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def grad(self, x):
        """
        Computes the gradient vector at point x.
        """
        raise NotImplementedError('Grad is not implemented.')


class BaseProxOracle(object):
    """
    Base class for proximal h(x)-part in a composite function f(x) + h(x).
    """

    def func(self, x):
        """
        Computes the value of h(x).
        """
        raise NotImplementedError('Func is not implemented.')

    def prox(self, x, alpha):
        """
        Implementation of proximal mapping.
        prox_{alpha}(x) := argmin_y { 1/(2*alpha) * ||y - x||_2^2 + h(y) }.
        """
        raise NotImplementedError('Prox is not implemented.')


class BaseCompositeOracle(object):
    """
    Base class for the composite function.
    phi(x) := f(x) + h(x), where f is a smooth part, h is a simple part.
    """

    def __init__(self, f, h):
        self._f = f
        self._h = h

    def func(self, x):
        """
        Computes the f(x) + h(x).
        """
        return self._f.func(x) + self._h.func(x)

    def grad(self, x):
        """
        Computes the gradient of f(x).
        """
        return self._f.grad(x)

    def prox(self, x, alpha):
        """
        Computes the proximal mapping.
        """
        return self._h.prox(x, alpha)

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        return None


class BaseNonsmoothConvexOracle(object):
    """
    Base class for implementation of oracle for nonsmooth convex function.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def subgrad(self, x):
        """
        Computes arbitrary subgradient vector at point x.
        """
        raise NotImplementedError('Subgrad is not implemented.')

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        return None


class LeastSquaresOracle(BaseSmoothOracle):
    """
    Oracle for least-squares regression.
        f(x) = 0.5 ||Ax - b||_2^2
    """
    # TODO: implement.
    
    def __init__(self, matvec_Ax, matvec_ATx, b):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = b

    def func(self, x):
        return 0.5 * np.linalg.norm(self.matvec_Ax(x) - self.b) ** 2

    def grad(self, x):
        return self.matvec_ATx(self.matvec_Ax(x) - self.b)

class L1RegOracle(BaseProxOracle):
    """
    Oracle for L1-regularizer.
        h(x) = regcoef * ||x||_1.
    """
    # TODO: implement.
    def __init__(self, regcoef):
        self.regcoef = regcoef

    def func(self, x):
        return self.regcoef * np.linalg.norm(x, 1)
    
    def subgrad(self, x):
        return self.regcoef * np.sign(x)

    def prox(self, x, alpha):
        return np.sign(x) * np.maximum(np.abs(x) - alpha * self.regcoef, 0)

class LassoNonsmoothOracle(BaseNonsmoothConvexOracle):
    """
    Oracle for nonsmooth convex function
        0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    # TODO: implement.

    def __init__(self, matvec_Ax, matvec_ATx, b, regcoef):
        self._f = LeastSquaresOracle(matvec_Ax, matvec_ATx, b)
        self._h = L1RegOracle(regcoef)

    def func(self, x):
        return self._f.func(x) + self._h.func(x)

    def subgrad(self, x):
        return self._f.grad(x) + self._h.subgrad(x)
    
    def duality_gap(self, x):
        Ax_b = self._f.matvec_Ax(x) - self._f.b
        ATAx_b = self._f.matvec_ATx(Ax_b)
        return lasso_duality_gap(x, Ax_b, ATAx_b, self._f.b, self._h.regcoef)
    
    
class LassoProxOracle(BaseCompositeOracle, LassoNonsmoothOracle):
    """
    Oracle for 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
        f(x) = 0.5 * ||Ax - b||_2^2 is a smooth part,
        h(x) = regcoef * ||x||_1 is a simple part.
    """
    # TODO: implement.
    def __init__(self, f, h):
        super().__init__(f, h)

    def duality_gap(self, x):
        Ax_b = self._f.matvec_Ax(x) - self._f.b
        ATAx_b = self._f.matvec_ATx(Ax_b)
        return lasso_duality_gap(x, Ax_b, ATAx_b, self._f.b, self._h.regcoef)

def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for 
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    # TODO: implement.
    cf = min(1.0, regcoef / scipy.linalg.norm(ATAx_b, np.inf)) if scipy.linalg.norm(ATAx_b, np.inf) else 1.0
    mu = cf * Ax_b
    return 0.5 * np.linalg.norm(Ax_b) ** 2 + regcoef * scipy.linalg.norm(x, ord=1) + 0.5 * np.linalg.norm(mu) ** 2 + np.dot(b, mu)


def create_lasso_nonsmooth_oracle(A, b, regcoef):
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)
    return LassoNonsmoothOracle(matvec_Ax, matvec_ATx, b, regcoef)


def create_lasso_prox_oracle(A, b, regcoef):
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)
    return LassoProxOracle(LeastSquaresOracle(matvec_Ax, matvec_ATx, b),
                           L1RegOracle(regcoef))
