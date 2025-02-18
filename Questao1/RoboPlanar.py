import sys
import os
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_tools import *
from auxiliar_plots import *
import numpy as np
import sympy as sp

def RRR_inverse_kinematics(x, y, phi, l):
    """
    Calcula a cinemática inversa para um robô planar RRR.
    Parâmetros:
        x: float
        y: float
        phi: float
        l1: float
        l2: float
        l3: float
    Retorna:
        list: uma lista contendo as soluções para as juntas theta1, theta2 e theta3
    """
    l1, l2, l3 = l
    # Calculo do xp e yp
    xp = x - l3*np.cos(phi)
    yp = y - l3*np.sin(phi)

    # dem = xp**2 + yp**2

    # Calculo do angulo th2
    c2 = (xp**2 + yp**2 - l1**2 - l2**2) / (2*l1*l2)
    # c2 = (dem - l1**2 - l2**2) / (2*l1*l2)
    abs_s2 = np.sqrt(1 - c2**2)

    possible_solutions = []
    for i in range(1,3):
        s2 = abs_s2*(-1)**i
        th2 = np.arctan2(s2, c2)
        # k = (l1+l2*c2)
        # j = l2*s2

        # Calculo do angulo th1
        c1 = ((l1+l2*c2)*xp+(l2*s2)*yp)/(xp**2 + yp**2)
        s1 = ((l1+l2*c2)*yp-(l2*s2)*xp)/(xp**2 + yp**2)
        # c1 = (xp*k + j*yp) / dem
        # s1 = (k*yp - j*xp) / dem

        th1 = np.arctan2(s1, c1)

        # Calculo do angulo th3
        th3 = phi - th1 - th2
        possible_solutions.append([th1, th2, th3])

    return possible_solutions




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

    # valores de a
    l = [1, 1, 1]

    # Teste de cinemática inversa
    x = 2.205737
    y = 1.850833
    phi = np.deg2rad(60)

    solutios = RRR_inverse_kinematics(x, y, phi, l)
    robot_colors = ['m', 'g']
    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(6, 6))

    for i, theta in enumerate(solutios):
        # frames numericos
        F0 = np.eye(4)
        F1 = F0 @ numeric_DH(0, theta[0], l[0], 0)
        F2 = F1 @ numeric_DH(0, theta[1], l[1], 0)
        F3 = F2 @ numeric_DH(0, theta[2], l[2], 0)

        frames = np.array([F0, F1, F2, F3])
        

        # Call the function to plot the robot
        plot_planar_robot(ax, frames, origin=True, robot_color=robot_colors[i], scale = 0.3)

    plt.show()
