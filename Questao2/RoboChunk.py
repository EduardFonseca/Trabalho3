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




if __name__ == "__main__":
    # Definição das variáveis de junta simbólicas
    th = sp.symbols('th1 th2 th3 th4 th5 th6 th7', real=True)
    l = sp.symbols('l1 l2 l3 l4', real=True)

    # Parâmetros DH do robô Schunk LWA4 7DoF
    dh_params = [
        (l[0], th[0],  0, sp.rad(90)),
        (0,  th[1], 0, sp.rad(-90)),
        (l[1], th[2],  0, sp.rad(90)),
        (0,  th[3], 0, sp.rad(-90)),
        (l[2], th[4],  0, sp.rad(90)),
        (0,  th[5], 0, sp.rad(-90)),
        (l[3], th[6],  0, 0)
    ]

    # Cálculo da matriz de transformação homogênea

    F = sp.eye(4)
    for params in dh_params:
        F = F @ symbolic_DH(*params)

    # Simplificação simbólica
    F_simplified = sp.trigsimp(F)
    F_simplified = sp.nsimplify(F_simplified, tolerance=1e-10)
    # print("Matriz de Transformação Homogênea do End-Effector:")
    # print(sp.pretty(F_simplified))


    L = np.array([300, 328, 276.5, 171.7])
    TH = np.array([0, 0, 0, 0, 0, 0, 0])

    EF_pos = F_simplified.subs(list(zip(l, L)))

    Caminho, Pontos = heart_path(20)
    print(Pontos.shape)

    angles = gradient_descent_ik(EF_pos, th, TH, Caminho, max_iter=1000, tol=0.1, alpha=0.3)

    print(angles)
    print(angles.shape)

    positions_list = np.random.random_integers(0, 19, 3)
    for i,angle in enumerate(angles):
        if i in positions_list:
            print(f"Posição: {i}")
            print(f"Angulos: {angle}")
            print(f"Posição Efetuador: {Pontos[:,:,i]}")

            dh_params = [
                (L[0], angle[0],  0, np.deg2rad(90)),
                (0   , angle[1],  0, np.deg2rad(-90)),
                (L[1], angle[2],  0, np.deg2rad(90)),
                (0   , angle[3],  0, np.deg2rad(-90)),
                (L[2], angle[4],  0, np.deg2rad(90)),
                (0   , angle[5],  0, np.deg2rad(-90)),
                (L[3], angle[6],  0, 0)
            ]

            frames = []
            F = np.eye(4)
            frames.append(F)
            for params in dh_params:
                F = F @ numeric_DH(*params)
                frames.append(F)

            plt.figure()
            ax = plt.axes(projection='3d')
            plot_points(ax, Pontos, plot_axes=True, trace=True, current_point=i)
            plot_robot(ax, np.array(frames), scale=50, origin=True)

    plt.show()




