import numpy as np

from OpenAEM.line import DLS

def pi_packet(l=0.4, alpha=10.0, n=7, beta=45.0):
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    heights = np.array([1.0 - i*l*np.tan(alpha) for i in range(n)])
    for i in range(n):
        p0 = np.array([-i*l, -heights[i]*0.5, 0])
        p1 = np.array([-i*l + heights[i]*np.tan(beta), -heights[i]*0.5, heights[i]])
        p2 = np.array([-i*l + heights[i]*np.tan(beta), heights[i]*0.5, heights[i]])
        p3 = np.array([-i*l, heights[i]*0.5, 0])

def lambda_packet(l=0.4, alpha=10.0, n=7):
    pass

def mirror_eddies(eddies:list[DLS]):
    mirror = []
    for line in eddies:
        mirror.append(line.mirror().reverse())

def plot_eddies():
    pass