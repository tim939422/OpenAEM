import numpy as np
from scipy import integrate
import OpenAEM


def biot_savart(rod: OpenAEM.DLS, xv: np.ndarray, r0=0.1):
    distance = rod.distance2pts(xv)
    uv = np.zeros_like(xv)
    
    xv = xv[:, distance > r0]
    xpv = rod.points()
    sv = xv[:, :, np.newaxis] - xpv[:, np.newaxis, :]
    s  = np.linalg.norm(sv, axis=0)[np.newaxis, :, :]
    

def biot_savart_1(eddy: list[OpenAEM.DLS], xv: np.ndarray):
    """vectorized Biot Savart integral

    Args:
        curve (list[OpenAEM.DLS]): eddy geometry
        xv (np.ndarray): sample points (3, M)

    Returns:
        numpy ndarray: velocity at xv (3, M)
    """    
    uv = np.zeros_like(xv)
    
    for rod in eddy:
        xpv = rod.points()
        
        sv = xv[:, :, np.newaxis] - xpv[:, np.newaxis, :]
        s  = np.linalg.norm(sv, axis=0)[np.newaxis, :, :]
        
        xpvd = rod.dirs()[:, np.newaxis, :]
        
        # parameter of rod curve
        t = rod.get_t()
        uv += -0.5*integrate.simps(np.cross(sv, xpvd, axis=0)/s**3, t)
    
    return uv


def biot_savart_2(eddy: list[OpenAEM.DLS], xvs: np.ndarray):
    """Biot Savart integral (point by point)

    Args:
        curve (list[OpenAEM.DLS]): eddy geometry
        xv (np.ndarray): sample points (3, M)

    Returns:
        numpy ndarray: velocity at xv (3, M)
    """    
    uv = np.zeros_like(xvs)
    npts = np.size(xvs, axis=1)
    
    for i in range(npts):
        xv = xvs[:, i][:, np.newaxis]
        for rod in eddy:
            xpv = rod.points()
            sv = xv - xpv
            s  = np.linalg.norm(sv, axis=0)
            
            xpvd = rod.dirs()
            
            # parameter of rod curve
            t = rod.get_t()
            uv[:, i] += -0.5*integrate.simpson(np.cross(sv, xpvd, axis=0)/s**3, t)
            
    return uv

def biot_savart_3(eddy: list[OpenAEM.DLS], xv: np.ndarray):
    uv = np.zeros_like(xv)
    
    for rod in eddy:
        xpv = rod.points()
        
        sv = xv[:, :, np.newaxis] - xpv[:, np.newaxis, :]
        s  = np.linalg.norm(sv, axis=0)[np.newaxis, :, :]
        
        xpvd = rod.get_dir()
        
        t = rod.get_t()
        
        uv += -0.5*integrate.simps(np.cross(sv, xpvd, axis=0)/s**3, t)
        
    return uv