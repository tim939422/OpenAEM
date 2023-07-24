import numpy as np

class Wall_Patch:
    def __init__(self, xmin=0.0, xmax=1.0, zmin=0.0, zmax=1.0) -> None:
        
        self.xmin = xmin; self.xmax = xmax
        self.zmin = zmin; self.zmax = zmax
        
    def __str__(self) -> str:
        return f'Domain: [{self.xmin}, {self.xmax}]x[{self.zmin}, {self.zmax}]'
    
    def place_eddies(self, lam, seed=None):
        if seed != None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random.default_rng()
            
        npts = rng.poisson()
        area = self.area()
        npts = rng.poisson(lam=lam*area)
        x = rng.uniform(self.xmin, self.xmax, size=npts)
        z = rng.uniform(self.zmin, self.zmax, size=npts)
        
        return np.vstack((x, z)) # (2, npts)
        
    def area(self):
        return (self.xmax - self.xmin)*(self.zmax - self.zmin)
            
        
    