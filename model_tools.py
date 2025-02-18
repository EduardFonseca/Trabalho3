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

def gradient_descent_ik(DH, symb_var, q0, target_points, max_iter=1000, tol=1e-6, alpha=0.01):
    """
    Algoritmo de otimização de descida de gradiente para resolver o problema de cinemática inversa.
    Parâmetros:
        DH: sympy.Matrix
            Lista de parâmetros DH do robo com variaveis symboolicas para juntas moveis.
        target_points: np.array
            Vetor com pontos objetivos 12xN.
        q0: np.array
            Vetor de junta inicial.
        max_iter: int
            Número máximo de iterações. (default: 1000)
        tol: float
            Tolerância de erro. (default: 1e-6)
        alpha: float
            Taxa de aprendizado. (default: 0.01)
    Retorna:
        np.array: vetor de junta otimizado Nx7.
    """
    Q = np.array([]) # Lista com vertores das posicoes de junta
    q = q0.copy()
    q = q.reshape(7,1)
    end_effector = np.array([]) # Lista com as posicoes do efetuador final
    #transformar a matriz de parametros DH em um vetor de parametros
    DH_vector = [DH[:3,i] for i in range(4)]
    DH_vector = sp.Matrix(DH_vector).reshape(12,1)
    #Calculate the jacobian of the DH vector
    J = DH_vector.jacobian(symb_var)
    for target in target_points:
        target = target.reshape(12,1)
    
        # calculating the direct cinematic numerical
        ef = DH_vector.subs(list(zip(symb_var, q.squeeze())))
        # trasformando em numpy array
        ef = np.array(ef).astype(np.float64)
        # posicionando as colunas em um vetor
        if end_effector.size == 0:  
            end_effector = ef
        else:
            end_effector = np.vstack((end_effector, ef))

        # Claculando o erro quadratico medio
        MSE = np.mean((target - ef)**2)
        print("MSE:", MSE)
        iter = 0
        #loop
        while MSE > tol:
            iter += 1
            #calculo do gradiente cartesiano atual
            dE = -(target - ef)
            #calculo do jacobiano
            j = np.array(J.subs(list(zip(symb_var, q.squeeze())))).astype(np.float64)
            #calculo do gradiente de junta
            dq = np.linalg.pinv(j) @ dE
            #atualizacao do vetor de junta
            q = q - alpha * dq

            #calculando o efetuador final
            ef = DH_vector.subs(list(zip(symb_var, q.squeeze())))
            ef = np.array(ef).astype(np.float64)

            MSE = np.mean((target - ef)**2)
            print("Iter:", iter, "MSE:", MSE)
        
        if Q.size == 0:
            Q = q.flatten()
        else:
            Q = np.vstack((Q, q.flatten()))
    
    return Q

        


__all__ = ["symbolic_DH", "numeric_DH", "gradient_descent_ik"]