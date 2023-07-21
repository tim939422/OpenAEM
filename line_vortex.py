import numpy as np
from scipy import integrate
from line import Line
    

def biot_savart(xv, curve: Line, r0 = 0.1):
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
        float array: induced velocity at xv
        
    Remarks:
        the circulation has the same direction as the curve
    """
    # if distance <= r0, the line vortex concept is invalid
    # return zero velocity
    if curve.get_distance(xv) <= r0:
        return np.zeros((3, 1))
    
    xpv  = curve.get_points()
    xpvd = curve.get_direction()
    t = curve.get_t()
    
    sv  = xv - xpv
    s   = np.linalg.norm(sv, axis=0)
    uvv = -0.5*integrate.simpson(np.cross(sv, xpvd, axis=0)/s**3, t)
    return uvv.reshape(-1, 1)
    
    
if __name__ == '__main__':
    from vector import Vector
    
    xvp0 = Vector(0, 0, -50000)
    xvp1 = Vector(0, 0,  50000)
    line = Line(xvp0, xvp1)

    
    xv = Vector(0.1, 0, 0)
    uvv = biot_savart(xv, line)
    print(uvv)