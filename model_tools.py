import sympy as sp
import numpy as np

def symbolic_DH(d, theta, a, alpha):
    """
    Retorna uma matriz de transformação DH simbólica ou numérica, dependendo do formato dos valores de entrada.
    Se os valores de entrada forem simbólicos, a saída será uma matriz simbólica.
    Se os valores de entrada forem numéricos, a saída será uma matriz numérica.
    
    Parâmetros:
        d: float
        theta: float
        a: float
        alpha: float
    
    Retorna:
        sp.Matrix: matriz 4x4
    """
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta) * sp.cos(alpha),  sp.sin(theta) * sp.sin(alpha), a * sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta) * sp.cos(alpha), -sp.cos(theta) * sp.sin(alpha), a * sp.sin(theta)],
        [0,             sp.sin(alpha),                  sp.cos(alpha),                  d],
        [0,             0,                               0,                             1]
    ])  # Using optimized DH matrix function

def numeric_DH(d,theta,a,alpha):
    """
    Retorna uma matriz de transformação DH numérica.
    Embora a funcao symbolic_DH possa ser usada para valores numéricos, esta funcao e cerca de 10x mais rapida.
    Parâmetros:
        d: float
        theta: float
        a: float
        alpha: float
    Retorna:
        np.array: matriz 4x4
    """
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