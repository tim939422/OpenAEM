import numpy as np
from scipy import integrate

class Line_Vortex:
    '''
    a line vortex consists of one or more curve and each curve must implement
    following API:
    1. get_t: discretized parameter t
    2. get_xv: vector x=x(t) along the curve
    3. get_xvd: direction vector x'=x'(t) along the curve
    '''

    def __init__(self):
        self.curves = []
        
    def add_components(self, curve):
        self.curves.append()
    
        
    @staticmethod
    def biot_savart(xv, curve):
        """apply biot savart law to compute induced velocity
        Compute the following integral

                        --
                     1 |   s X x'(t) dt
            u_v(x) = - | --------------
                     2 |     s^3
                     --
        Args:
            xv (float array): location request
            curve (curve): a general curve with required API implemented

        Returns:
            float array: induced velocity at xv
        """        
        xv   = xv.reshape(-1, 1)
        xpv  = curve.get_xv()
        xpvd = curve.get_xvd()
        t = curve.get_t()
        
        sv  = xv - xpv
        s   = np.linalg.norm(sv, axis=0)
        return -0.5*integrate.simpson(np.cross(sv, xpvd, axis=0)/s**3, t)
    
    
if __name__ == '__main__':
    p0 = np.array([0, 0, -50000])
    p1 = np.array([0, 0, 50000])
    
    from line import Line
    line = Line(p0, p1)
    
    xv = np.array([1, 0, 0])
    uvv = Line_Vortex.biot_savart(xv, line)
    print(uvv)