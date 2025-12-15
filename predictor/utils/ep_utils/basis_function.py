from scipy.special import binom
import numpy as np

def M_matrix(n):
    '''
    A matrix that transfers monomial basis function to berstein basis function
    param n: the order of polynomial
    '''
    M = np.zeros([n+1, n+1], dtype = np.float32)
    for i in range(1, n+2):
        for j in range(1, i+1):
            M[i-1,j-1] = binom(n, j-1) * binom(n+1-j, i-j) * (-1)**((i+j)%2)

    return M

def M_inv_matrix(n):
    '''
    A inverse matrix of M matrix
    param n: the order of polynomial
    '''
    return np.linalg.solve(M_matrix(n), np.eye(n+1))

    
def D_matrix(n, k): 
    '''
    A derivative matrix for monomial basis function
    param n: the order of polynomial
    param k: the order of derivatives
    '''
    D=np.eye(n+1, dtype = np.float32)
    for i in range(k):
        D = D @ np.diag(np.arange(1,n+1), 1)

    return D

def phi_m(tau, n):
    '''
    create monomial basis function

    param tau: time
    param n: degree of polynomial
    '''
    return np.array([tau **k for k in range(n+1)], dtype = np.float32).T


def phi_b(tau, n):
    assert np.max(tau) <= 1.0
    
    return phi_m(tau,n) @ M_matrix(n)


def basis_function_b(tau, 
                     n, 
                     delta_t, 
                     d=2, 
                     k=0, 
                     return_kron=False,
                     dtype=np.float32): 
    '''
    params:
        tau: normalized timestamps. e.g. [0.1, 0.2, ..., 1.0]
        n: degree of polynomial
        delta_t: timescale, e.g. 5 seconds
        d: space dim
        k: the order of derivatives
        return_kron: return kronecker product
        
    output:
        k-th order of bernstein basis function "phi_b"
    '''
    assert np.max(tau) <= 1.0
    time_scale = 1/(delta_t**k)
    phi_k = phi_m(tau, n) @ D_matrix(n, k)
    phi_k = phi_k @ M_matrix(n)

    return (time_scale * np.kron(phi_k, np.eye(d)) if return_kron else time_scale * phi_k).astype(dtype)


def basis_function_m(t, 
                     n, 
                     d=2, 
                     k=0, 
                     return_kron=False, 
                     dtype=np.float32):   
    '''
    params:
        t: timestamps. e.g. [0.1, 0.2, ..., 3.0]
        n: degree of polynomial
        d: space dim
        k: the order of derivatives
        return_kron: return kronecker product
        
    output:
        k-th order of monomial basis function "phi_m"
    '''
    
    phi_k = phi_m(t, n) @ D_matrix(n, k)

    return (np.kron(phi_k, np.eye(d)) if return_kron else phi_k).astype(dtype)


def transform_m_to_b(delta_t: float,
                     n: int, 
                     d: int =2,
                     return_kron: bool =False,
                     dtype=np.float32):
    '''
    params:
        delta_t: timescale
        n: degree of polynomial
        d: space dim
        return_kron: return kronecker product
        
    output:
        a matrix "mat" that transform monomial parameters to bernstein parameters: phi_b = mat @ phi_m
    '''
    m_inv = M_inv_matrix(n)
    monomial_scale = np.diag([delta_t ** deg for deg in range(0, n + 1)])
    return (np.kron(m_inv @ monomial_scale, np.eye(d)) if return_kron else m_inv @ monomial_scale).astype(dtype)


def transform_b_to_m(delta_t: float,
                     n: int,
                     d: int =2,
                     return_kron: bool =False,
                     dtype=np.float32):
    '''
    params:
        delta_t: timescale
        n: degree of polynomial
        d: space dim
        return_kron: return kronecker product
        
    output:
        a matrix "mat" that transform bernstein parameters to monomial parameters: phi_m = mat @ phi_b
    '''
    
    m = M_matrix(n) 
    monomial_scale = np.linalg.solve(np.diag([delta_t ** deg for deg in range(0, n + 1)]), np.eye(n+1)) 

    return (np.kron(monomial_scale @ m, np.eye(d)) if return_kron else monomial_scale @ m).astype(dtype)