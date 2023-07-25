import numpy as np

from OpenAEM.line import DLS

def pi_packet(l=0.4, alpha=10.0, n=7, beta=45.0):
    eddy = []
    
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    heights = np.array([1.0 - i*l*np.tan(alpha) for i in range(n)])
    for i in range(n):
        p0 = np.array([-i*l, -heights[i]*0.5, 0])
        p1 = np.array([-i*l + heights[i]*np.tan(beta), -heights[i]*0.5, heights[i]])
        p2 = np.array([-i*l + heights[i]*np.tan(beta), heights[i]*0.5, heights[i]])
        p3 = np.array([-i*l, heights[i]*0.5, 0])
        eddy.append(DLS(p0, p1))
        eddy.append(DLS(p1, p2))
        eddy.append(DLS(p2, p3))
        
    return eddy

def lambda_packet(l=0.4, alpha=10.0, n=7, beta=45.0):
    eddy = []
    
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    heights = np.array([1.0 - i*l*np.tan(alpha) for i in range(n)])
    for i in range(n):
        p0 = np.array([-i*l, -heights[i]*0.5, 0])
        p1 = np.array([-i*l + heights[i]*np.tan(beta), 0.0, heights[i]])
        p2 = np.array([-i*l, heights[i]*0.5, 0])
        eddy.append(DLS(p0, p1))
        eddy.append(DLS(p1, p2))
        
    return eddy

def mirror_eddy(eddy:list[DLS]):
    mirror = []
    for line in eddy:
        mirror.append(line.mirror().reverse())
        
    return mirror

def plot_eddy(eddy:list[DLS], ax, color='black'):
    xm = []; ym = []; zm = []
    um = []; vm = []; wm = []
    for curve in eddy:
        pts = curve.points()
        ax.plot(pts[0, :], pts[1, :], pts[2, :], color=color)
        im = np.size(pts, axis=1) // 2
        xm.append(pts[0, im]); ym.append(pts[1, im]); zm.append(pts[2, im])
        dirm = curve.dirs()
        um.append(dirm[0, im]); vm.append(dirm[1, im]); wm.append(dirm[2, im])
    
    ax.quiver(xm, ym, zm, um, vm, wm, length=0.1, color='red')