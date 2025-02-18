import numpy as np
import matplotlib.pyplot as plt

def heart_path(n_points):
    """
    Retorna um caminho em forma de coração.
    Parâmetros:
        N: int
            numero de pontos
    Retorna:
        np.array: matriz 4x4
    """
    path = np.array([])
    N = 2 * np.pi * np.arange(n_points) / 19
    for p in N:
        point = np.array([
        [0, 0, 1, 200 + 100 * np.cos(p)],
        [0, 1, 0, 20 * (16 * np.sin(p)**3)],
        [-1, 0, 0, 500 + 20 * (13 * np.cos(p) - 5 * np.cos(2 * p) - 2 * np.cos(3 * p) - np.cos(4 * p))],
        [0, 0, 0, 1]
        ])
        path = np.append(path, point)

    return path


Caminho = heart_path(100)
print(Caminho.shape)

# Plotar o caminho planejado
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(Caminho[0, 3, :], Caminho[1, 3, :], Caminho[2, 3, :], 'r-', label='Caminho desejado')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()
