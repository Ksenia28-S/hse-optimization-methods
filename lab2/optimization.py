import numpy as np
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from utils import get_line_search_tool
from datetime import datetime


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

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
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    # TODO: Implement Conjugate Gradients method.
    start_time = datetime.now()
    message = None 
    g_k = matvec(x_k) - b
    d_k = - g_k
    #max_iter = min(max_iter, 2 * len(x_k)) if max_iter else 2 * len(x_k)
    if not max_iter:
        max_iter = len(x_k)
    for _ in range(max_iter):
        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['residual_norm'].append(np.linalg.norm(g_k))
            if x_k.size <= 2:
                history['x'].append(x_k)
        A_d_k = matvec(d_k)
        coef_x = (g_k.T @ g_k) / (d_k.T @ A_d_k)
        x_k = x_k + coef_x * d_k
        g_k_prev = np.copy(g_k)
        g_k = g_k + coef_x * A_d_k
        if np.linalg.norm(g_k) <= tolerance * np.linalg.norm(b):
            message = 'success'
            break
        coef_d = (g_k.T @ g_k) / (g_k_prev.T @ g_k_prev)
        d_k = -g_k + coef_d * d_k
    if trace:
        history['time'].append((datetime.now() - start_time).total_seconds())
        history['residual_norm'].append(np.linalg.norm(g_k))
        if x_k.size <= 2:
            history['x'].append(x_k)
    if not np.linalg.norm(g_k) <= tolerance * np.linalg.norm(b):
        message = 'iterations_exceeded'
    
    return x_k, message, history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
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
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    # TODO: Implement L-BFGS method.
    # Use line_search_tool.line_search() for adaptive step size.
    start_time = datetime.now()
    message = None
    grad_k = oracle.grad(x_k)
    grad_0_norm = np.linalg.norm(grad_k)
    H = []

    def BGFS_Multiply(v, H, gamma_0):
        if len(H) == 0:
            return gamma_0 * v
        s, y = H[-1]
        H_h = H[:-1]
        v_h = v - (s @ v) / (y @ s) * y
        z = BGFS_Multiply(v_h, H_h, gamma_0)
        result = z + (s @ v - y @ z) / (y @ s) * s
        return result

    for _ in range(max_iter):
        if trace:
            history['func'].append(oracle.func(x_k))
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['grad_norm'].append(np.linalg.norm(grad_k))
            if x_k.size <= 2:
                history['x'].append(x_k)

        if len(H) == 0:
            d = -grad_k
        else:
            s, y = H[-1]
            gamma_0 = (y @ s) / (y @ y)
            d = BGFS_Multiply(-grad_k, H, gamma_0)           
        coef_alpha = line_search_tool.line_search(oracle, x_k, d)
        x_h = x_k + coef_alpha * d
        H.append((x_h - x_k, oracle.grad(x_h) - grad_k))
        x_k, grad_k = x_h, oracle.grad(x_h)
        if len(H) > memory_size:
            H = H[1:]
        if np.linalg.norm(grad_k) ** 2 < tolerance * grad_0_norm ** 2:
            message = 'success'
            break

    if trace:
        history['func'].append(oracle.func(x_k))
        history['time'].append((datetime.now() - start_time).total_seconds())
        history['grad_norm'].append(np.linalg.norm(grad_k))
        if x_k.size <= 2:
            history['x'].append(x_k)

    if not np.linalg.norm(grad_k) ** 2 < tolerance * grad_0_norm ** 2:
        message = 'iterations_exceeded'

    return x_k, message, history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500, 
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
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
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    # TODO: Implement hessian-free Newton's method.
    # Use line_search_tool.line_search() for adaptive step size.
    start_time = datetime.now()
    message = None
    grad_k = oracle.grad(x_k)
    grad_0_norm = np.linalg.norm(grad_k)
    for _ in range(max_iter):
        if trace:
            history['func'].append(oracle.func(x_k))
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['grad_norm'].append(np.linalg.norm(grad_k))
            if x_k.size <= 2:
                history['x'].append(x_k)
        eta = min(0.5, np.sqrt(np.linalg.norm(grad_k)))
        while True:
            hess_vector = lambda y: oracle.hess_vec(x_k, y)
            d_k, conj_message, conj_history = conjugate_gradients(hess_vector, -grad_k, -grad_k, eta)
            if grad_k @ d_k < 0:
                break
            else:
                eta = eta * 10
        coef_alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1)
        x_k = x_k + coef_alpha * d_k
        grad_k = oracle.grad(x_k)
        if np.linalg.norm(grad_k) ** 2 < tolerance * grad_0_norm ** 2:
            message = 'success'
            break
    if trace:
        history['func'].append(oracle.func(x_k))
        history['time'].append((datetime.now() - start_time).total_seconds())
        history['grad_norm'].append(np.linalg.norm(grad_k))
        if x_k.size <= 2:
            history['x'].append(x_k)
            
    if not np.linalg.norm(grad_k) ** 2 < tolerance * grad_0_norm ** 2:
        message = 'iterations_exceeded'

    return x_k, message, history

# Для 3го эксперимента
def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    # TODO: Implement gradient descent
    # Use line_search_tool.line_search() for adaptive step size.
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    alpha = None
    start_time = datetime.now()
    for _ in range(max_iter):
        y_k = oracle.func(x_k)
        d_k = oracle.grad(x_k)
        if np.linalg.norm(d_k)**2  <= tolerance * np.linalg.norm(oracle.grad(x_0))**2:
            return x_k, 'success', history
        if np.isnan(d_k).any() or np.isinf(d_k).any():
            return x_k,'computational_error', history
        alpha = line_search_tool.line_search(oracle, x_k, -d_k, alpha)
        x_k = x_k - alpha * d_k
        if trace:
            history['func'].append(y_k)
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['grad_norm'].append(np.linalg.norm(d_k))
            if x_k.size <= 2:
                history['x'].append(x_k)
    if np.linalg.norm(oracle.grad(oracle.func(x_k)))**2  > tolerance * np.linalg.norm(oracle.grad(x_0))**2:
        return x_k,'iterations_exceeded', history
    else:
        return x_k, 'success', history