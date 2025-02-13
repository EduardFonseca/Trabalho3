import sympy as sp
import numpy as np

def symbolic_DH(d, theta, a, alpha):
    """Returns a symbolic DH transformation matrix"""
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta) * sp.cos(alpha),  sp.sin(theta) * sp.sin(alpha), a * sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta) * sp.cos(alpha), -sp.cos(theta) * sp.sin(alpha), a * sp.sin(theta)],
        [0,             sp.sin(alpha),                  sp.cos(alpha),                  d],
        [0,             0,                               0,                               1]
    ])  # Using optimized DH matrix function

def numeric_DH(d,theta,a,alpha):
    '''
    Create a Denavit-Hartenberg matrix using numerical values
    Parameters:
        d: float
        theta: float
        a: float
        alpha: float
    Returns:
        np.array: 4x4 matrix
    '''
    Tx = np.array([[1, 0, 0, a],
                   [0,1,0,0],
                   [0,0,1,0],
                   [0,0,0,1]])
    Rx = np.array([[1,     0        ,      0        ,0],
                   [0, np.cos(alpha), -np.sin(alpha),0],
                   [0, np.sin(alpha),  np.cos(alpha),0],
                   [0,     0        ,      0        ,1]])
    Tz = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, d],
                   [0, 0, 0, 1]])
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                   [np.sin(theta),  np.cos(theta), 0, 0],
                   [    0        ,      0        , 1, 0],
                   [    0        ,      0        , 0, 1]])
    
    return Tz @ Rz @ Tx @ Rx

__all__ = ["symbolic_DH", "numeric_DH"]



