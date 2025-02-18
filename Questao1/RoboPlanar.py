import sys
import os
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_tools import *
from auxiliar_plots import *
import numpy as np
import sympy as sp

def RRR_inverse_kinematics():
    pass


if __name__ == "__main__":
    # Robô PLANAR RRR:
    # |========================================|
    # | Ai | d       | theta     | a   | alpha |
    # |----|---------|-----------|-----|-------|
    # | 1  | 0       | th1*      | a1  | 0     |
    # | 2  | 0       | th2*      | a2  | 0     |
    # | 3  | 0       | th3*      | a3  | 0     |
    # |========================================|

    # Variaveis de junta
    th1, th2, th3, a1, a2, a3 = sp.symbols('th1 th2 th3 a1 a2 a3')

    # Matrizes de transformação homogênea
    F0 = sp.eye(4)

    A1 = symbolic_DH(0, th1, a1, 0) 
    F1 = F0 @ A1

    A2 = symbolic_DH(0, th2, a2, 0)
    F2 = F1 @ A2

    A3 = symbolic_DH(0, th3, a3, 0)
    F3 = F2 @ A3

    F3 = sp.trigsimp(F3)

    # Matriz de transformação homogênea do efetuador
    print("Matriz de transformação homogênea do efetuador:")
    print(sp.pretty(F3))
    print("\n\n")
    print("Equacoes no Latex:")
    print(sp.latex(F3))

    # teste de posicao 
    # valores de theta
    theta = [0, 0, 0]
    # valores de a
    a = [1, 1, 1]

    # frames numericos
    F0 = np.eye(4)
    F1 = F0 @ numeric_DH(0, theta[0], a[0], 0)
    F2 = F1 @ numeric_DH(0, theta[1], a[1], 0)
    F3 = F2 @ numeric_DH(0, theta[2], a[2], 0)

    frames = np.array([F0, F1, F2, F3])

    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Call the function to plot the robot
    plot_planar_robot(ax, frames, origin=True, show=True)

