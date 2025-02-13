import sympy as sp
from model_tools import symbolic_DH

# Define symbolic joint variables
th1, th2, th3, th4, th5, th6, th7, l1, l2, l3, l4 = sp.symbols('th1 th2 th3 th4 th5 th6 th7 l1 l2 l3 l4', real=True)

# Define DH parameters for Planar RRP robot
# (d, theta, a, alpha)

dh_params = [
    (l1, th1,  0, sp.rad(90)),
    ( 0, th2,  0, sp.rad(-90)),
    (l2, th3,  0, sp.rad(90)),
    ( 0, th4,  0, sp.rad(-90)),
    (l3, th5,  0, sp.rad(90)),
    ( 0, th6,  0, sp.rad(-90)),
    (l4, th7,  0, 0)
]

# Compute transformation matrices
F = sp.eye(4)
for params in dh_params:
    F = F @ symbolic_DH(*params)  # Compute each transformation iteratively

# Apply targeted simplifications
F_simplified = sp.trigsimp(F)
F_simplified = sp.nsimplify(F_simplified, tolerance=1e-10)

# # Print results
print("LaTeX Output:")
print(sp.latex(F_simplified))
print("Pretty Output:")
print(sp.pretty(F_simplified))