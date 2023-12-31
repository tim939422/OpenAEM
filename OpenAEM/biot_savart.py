import numpy as np
from scipy import integrate
from OpenAEM.line import DLS

def biot_savart(xv, curve: DLS, r0 = 0.1):
    """apply biot savart law to compute induced velocity
    Compute the following integral

                    --
                 1 |   s X x'(t) dt
     u_v(x) = -  - | --------------
                 2 |     s^3
                 --
    Args:
        xv (float ndarray): location request
        curve (Line): a general curve with required API implemented

    Returns:
        float array: (3,) induced velocity at xv
        
    Remarks:
        the circulation has the same direction as the curve
    """
    # if distance <= r0, the line vortex concept is invalid
    # return zero velocity
    if curve.distance2pt(xv) <= r0:
        return np.zeros(3)
    
    xpv  = curve.points()
    xpvd = curve.dirs()
    t = curve.get_t()
    
    sv  = xv.reshape(-1, 1) - xpv
    s   = np.linalg.norm(sv, axis=0)
    uvv = -0.5*integrate.simpson(np.cross(sv, xpvd, axis=0)/s**3, t)
    return uvv
    
def biot_savart_eddy(xv, eddy: list[DLS], r0 = 0.1):
    uvv = np.zeros(3)
    for curve in eddy:
        uvv += biot_savart(xv, curve, r0=r0)
        
    return uvv
    
if __name__ == '__main__':
    import OpenAEM
    xvp0 = np.array([0, 0, -50000])
    xvp1 = np.array([0, 0,  50000])
    line = OpenAEM.DLS(xvp0, xvp1)

    
    xv = np.array([1, 0, 0])
    uvv = biot_savart(xv, line)
    print(uvv)