import numpy as np
def Vector(x, y, z):
    """represent a point in Cartesion coordinate as a (3, 1) ndarray

    Args:
        x (float): x
        y (float): y
        z (float): z

    Returns:
        ndarray: (3, 1)
    """    
    return np.array([x, y, z], dtype=float).reshape(-1, 1)

def mirror_Vector(vector: np.ndarray[float, 1]):
    return Vector(vector[0], vector[1], -vector[2])

def print_Vector(vector:np.ndarray, fmt='7.2f'):
    vector = vector.ravel()
    print(f'({vector[0]:{fmt}}, {vector[1]:{fmt}}, {vector[2]:{fmt}})')
    