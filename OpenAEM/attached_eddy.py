import numpy as np

from OpenAEM.line import DLS
from OpenAEM.biot_savart import biot_savart

class Attached_Eddy:
    def __init__(self, curves: np.ndarray[DLS, 1]):
        """attached eddy by several curves
        ! It's the user's responsibility to make sure their eddy makes sense !
        Args:
            curves (np.ndarray[Line, 1]): curves
        """        
        self.curves = curves
    
    def create_mirror(self):
        """mirror eddy about x-y plane

        Returns:
            Attached_Eddy : mirrored eddy with reversed circulation
        """        
        mirrors = []
        for curve in self.curves[::-1]:
            mirror = curve.mirror()
            mirrors.append(mirror.reverse()) # count for reversed circulation
            
        return Attached_Eddy(np.array(mirrors))
    
    def plot(self, ax, color='black'):
        xm = []; ym = []; zm = []
        um = []; vm = []; wm = []
        for curve in self.curves:
            pts = curve.points()
            ax.plot(pts[0, :], pts[1, :], pts[2, :], color=color)
            # mid point index
            im = np.size(pts, axis=1) // 2
            xm.append(pts[0, im]); ym.append(pts[1, im]); zm.append(pts[2, im])
            dirm = curve.dirs()
            um.append(dirm[0, im]); vm.append(dirm[1, im]); wm.append(dirm[2, im])
            
        ax.quiver(xm, ym, zm, um, vm, wm, length=0.1, color='red')
    
    def induced_velocity(self, xv, r0=0.1):
        """induced velocity
        ! it's non-dimensional velocity induced by non-dimensional eddy geometry
        Args:
            xv (ndarray): request position (must be scaled by \delta before)
            r0 (float, optional): cut off radius. Defaults to 0.1.

        Returns:
            ndarray: induced velocity by the eddy (must be scaled by u0 after)
        """        
        uvv = np.zeros((3, 1))
        for curve in self.curves:
            uvv += biot_savart(xv, curve)
            
        return uvv
    
    
class Pi_Eddy(Attached_Eddy):
    def __init__(self, a = 1, b = np.sqrt(2), alpha = 45, origin=np.zeros(3)):
        """Pi eddy

        Args:
            a (float, optional): head rod length. Defaults to 1.
            b (float, optional): leg length. Defaults to np.sqrt(2).
            alpha (float, optional): inclination angle. Defaults to 45 (deg).
            origin (ndarray, optional): origin of eddy. Defaults to np.zeros((3, 1)).
        """        
        alpha = np.deg2rad(alpha)
        pts = np.array([[0, -0.5*a, 0],
                        [b*np.cos(alpha), -0.5*a, b*np.sin(alpha)],
                        [b*np.cos(alpha),  0.5*a, b*np.sin(alpha)],
                        [0, 0.5*a, 0]])
        pts += origin
        curves = []
        for i in range(3):
            p0 = pts[i]
            p1 = pts[i+1]
            curves.append(DLS(p0, p1))
        
        self.curves = np.array(curves)
        

class Lambda_Eddy(Attached_Eddy):
    pass