import sys
import os
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from model_tools import *
from auxiliar_plots import * 
np.random.seed(0)

def heart_path(n_points):
    """
    Retorna um caminho em forma de coração.
    
    Parâmetros:
        n_points: int
            Número de pontos no caminho.
    
    Retorna:
        np.array: Matriz 16xN representando a trajetória do coração.

    """
    path = np.array([])
    vector_path = np.array([])
    N = 2 * np.pi * np.linspace(0, 1, n_points)
    
    for i, p in enumerate(N):
        point = np.array([
            [0, 0, 1, 200 + 100 * np.cos(p)],
            [0, 1, 0, 20 * (16 * np.sin(p)**3)],
            [-1, 0, 0, 500 + 20 * (13 * np.cos(p) - 5 * np.cos(2 * p) - 2 * np.cos(3 * p) - np.cos(4 * p))],
            [0, 0, 0, 1]
        ])
        #stack collumns in a vector

        flat_point = point[:3, :].flatten(order='F')
        if i == 0:
            vector_path = flat_point
            path = point
        else:
            vector_path = np.vstack((vector_path, flat_point))
            path = np.dstack((path, point))
    return  vector_path ,path


Caminho, Pontos = heart_path(10)
print(Pontos.shape)

plt.figure()
ax = plt.axes(projection='3d')
plot_points(ax, Pontos, plot_axes=True, trace=True, current_point=5)

plt.show()
