from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve
from time import time
from datetime import datetime


def subgradient_method(oracle, x_0, tolerance=1e-2, max_iter=1000, alpha_0=1,
                       display=False, trace=False):
    """
    Subgradient descent method for nonsmooth convex optimization.

    Parameters
    ----------
    oracle : BaseNonsmoothConvexOracle-descendant object
        Oracle with .func() and .subgrad() methods implemented for computing
        function value and its one (arbitrary) subgradient respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    alpha_0 : float
        Initial value for the sequence of step-sizes.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    # TODO: implement.
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    f_k = oracle.func(x_k)
    f_x_min, x_k_min = np.copy(f_k), np.copy(x_k)
    start_time =  datetime.now()

    for k in range(max_iter):
        if k > max_iter:
            return x_k, 'iterations_exceeded', history
        if trace:
            history['func'].append(f_k)
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['duality_gap'].append(oracle.duality_gap(x_k))
            if x_k.size <= 2:
                history['x'].append(x_k)
                
        if oracle.duality_gap(x_k) < tolerance:
            return x_k_min, 'success', history
        x_k -= (oracle.subgrad(x_k) / norm(oracle.subgrad(x_k))) * (alpha_0 / (k + 1) ** (0.5))
        f_k = oracle.func(x_k)
        if f_k < f_x_min:
            f_x_min, x_k_min = np.copy(f_k), np.copy(x_k)
            
    if trace:
        history['func'].append(f_k)
        history['time'].append((datetime.now() - start_time).total_seconds())
        history['duality_gap'].append(oracle.duality_gap(x_k))
        if x_k.size <= 2:
            history['x'].append(x_k)
    if oracle.duality_gap(x_k) < tolerance:
        return x_k_min, 'success', history
    else:
        return x_k_min, 'iterations_exceeded', history



def proximal_gradient_method(oracle, x_0, L_0=1, tolerance=1e-5,
                              max_iter=1000, trace=False, display=False):
    """
    Gradient method for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented 
        for computing function value, its gradient and proximal mapping 
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of objective function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    # TODO: implement.
    history = defaultdict(list) if trace else None
    iter_search, message = 0, None
    x_k = np.copy(x_0)
    l = np.copy(L_0)
    grad_k = oracle.grad(x_k)
    start_time = datetime.now()
    for _ in range(max_iter):
        if oracle.duality_gap(x_k) < tolerance:
            break
        if trace:
            history['func'].append(oracle.func(x_k))
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['duality_gap'].append(oracle.duality_gap(x_k))
            history['iter_search'].append(iter_search)
            if x_k.size <= 2:
                history['x'].append(x_k)
        while True:
            iter_search +=1
            y = oracle.prox(x_k - grad_k / l, 1 / l)
            if oracle._f.func(y) <= oracle._f.func(x_k) + grad_k.dot(y - x_k) + l / 2 * np.linalg.norm(y - x_k) ** 2:
                x_k = np.copy(y)
                break
            else:
                l = 2 * l

        l = l / 2
        grad_k = oracle.grad(x_k)

    if trace:
        history['func'].append(oracle.func(x_k))
        history['time'].append((datetime.now() - start_time).total_seconds())
        history['duality_gap'].append(oracle.duality_gap(x_k))
        history['iter_search'].append(iter_search)
        if x_k.size <= 2:
            history['x'].append(x_k)
     
    if oracle.duality_gap(x_k) < tolerance:
        message = 'success'
    else:
        message = 'iterations_exceeded'
    
    return x_k, message, history

def proximal_fast_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                              max_iter=1000, trace=False, display=False):
    """
    Fast gradient method for composite minimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented 
        for computing function value, its gradient and proximal mapping 
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of objective function values phi(best_point) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
    """
    # TODO: Implement
    history = defaultdict(list) if trace else None
    iter_search, message = 0, None
    x_k, l, v_k, y_k = np.copy(x_0), np.copy(L_0), np.copy(x_0), np.copy(x_0)
    A_k, sumdif_k = 0, 0
    x_star, min_f = np.copy(x_k), oracle.func(x_k)
    start_time = datetime.now()
    
    for _ in range(max_iter):
        if oracle.duality_gap(x_k) < tolerance:
            break
        if trace:
            history['func'].append(oracle.func(x_k))
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['duality_gap'].append(oracle.duality_gap(x_k))
            history['iter_search'].append(iter_search)
            if x_k.size <= 2:
                history['x'].append(x_k)

        while True:
            iter_search +=1
            a_k = (1 + np.sqrt(1 + 4 * l * A_k)) / (2 * l)
            A_next = A_k + a_k
            y_k = (A_k * x_k + a_k * v_k) / A_next
            sumdif_next = sumdif_k + a_k * oracle.grad(y_k)
            v_next = oracle.prox(x_0 - sumdif_next, A_next)
            y = (A_k * x_k + a_k * v_next) / A_next
            x_f, y_f = oracle.func(y), oracle.func(y_k)
            func_min = min(min_f, x_f, y_f)
            if min_f == x_f:
                x_star = y
            elif min_f == y_f:
                x_star = y_k

            if oracle._f.func(y) <= oracle._f.func(y_k) + np.dot(oracle.grad(y_k), y - y_k) + l / 2 * np.linalg.norm(y - y_k) ** 2:
                x_k, v_k, A_k, sumdif_k = y, v_next, A_next, sumdif_next
                break
            else:
                l = 2 * l
        l = l / 2

    if trace:
        history['func'].append(oracle.func(x_k))
        history['time'].append((datetime.now() - start_time).total_seconds())
        history['duality_gap'].append(oracle.duality_gap(x_k))
        history['iter_search'].append(iter_search)
        if x_k.size <= 2:
            history['x'].append(x_k)
            
    if oracle.duality_gap(x_k) < tolerance:
        message = 'success'
    else:
        message = 'iterations_exceeded'
        
    return x_star, message, history
