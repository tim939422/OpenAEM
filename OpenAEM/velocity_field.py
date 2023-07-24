import numpy as np

def get_grid(Retau, dxp, Lx, Ly, Lz, h):
    """cell centered grid within a box [0, Lx]x[0, Ly]x[0, Lz]

    Args:
        Retau (float): Retau
        dxp (float): resolution in viscous scale
        Lx (float): domain length
        Ly (float): domain length
        Lz (float): domain length
        h (float): outer scale

    Returns:
        ndarray: (nc, nc, nc) grid points
    """    
    
    lnu = compute_lnu(Retau, h)
    dx  = dxp*lnu
    x = get_grid_1d(0, Lx, dx)
    y = get_grid_1d(0, Ly, dx)
    z = get_grid_1d(0, Lz, dx)
    
    return np.meshgrid(x, y, z, indexing='ij')
    
def get_grid_1d(start: float, end: float, target_ds: float):
    """create 1d uniform grid with a target resolution

    Args:
        start (float): starting position
        end (float): ending position
        target_ds (float): target resolution

    Returns:
        ndarray: (nc, ) grid points
    """    
    length = end - start
    nc = int(np.ceil(length/target_ds))
    actual_ds = length / nc
    return 0.5*actual_ds + np.arange(nc)*actual_ds
    
def compute_lnu(Retau: float, h=1.0):
    """compute viscous length scale

    Args:
        Retau (float): Retau
        h (float, optional): outer length scale. Defaults to 1.0.

    Returns:
        float: viscous length scale
    
    Formulas:
        Retau = h/l_nu
    """    
    return h/Retau