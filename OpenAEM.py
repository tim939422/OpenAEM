import numpy as np
from scipy import integrate

class LineVortex:
    """Straight line vortex segment
    1. line points from p0 to p1
    2. induced velocity and geometry are normalized by a proper u0 and delta
    """    
    def __init__(self, p0, p1, n_element=100):
        """constructor for a line vortex

        Args:
            p0 (float array): starting point
            p1 ((float array): ending point
            n_element: number of discretized elements
        """        
        self.p0 = p0.reshape(-1, 1) # reshape to column vector
        self.p1 = p1.reshape(-1, 1)
        self.n_element = n_element
        
        # discretized parametric space
        self.t = self.__discretize_params()
        
        # discretized point along the segment
        self.xpv = self.__sample_points()
        
        # dot_xpv: x'(t)
        self.dot_xpv = self.__compute_dot_xpv()
    
    def biot_savart(self, xv):
        sv = xv.reshape(-1, 1) - self.xpv
        s  = np.linalg.norm(sv, axis=0)
        uvv = -0.5*integrate.simpson(np.cross(sv, self.dot_xpv, axis=0)/s**3, self.t)
        
        return uvv
        
        
    
    def __discretize_params(self, method='uniform'):
        """discretize parameter space

        Args:
            method (str, optional): discretization method. Defaults to 'uniform'.

        Returns:
            float array: _description_
        """        
        if method == 'uniform':
            return np.linspace(0, 1, self.n_element + 1).reshape(1, -1)
        else:
            print('Unsupported discretization type')
            
    def __sample_points(self):
        """discretized point along the line vortex

        Returns:
            float array: (3, M)
        """        
        return np.matmul(self.p0, 1 - self.t) + np.matmul(self.p1, self.t)
    
    def __compute_dot_xpv(self):
        """compute x'(t)

        Returns:
            float array: (3, M)
        """
        dot_xpv = -self.p0 + self.p1
        return np.repeat(dot_xpv, self.n_element + 1, axis=1)
    
    
if __name__ == '__main__':
    p0 = np.array([0, 0, -1000])
    p1 = np.array([0, 0,  1000])
    line_vortex =  LineVortex(p0, p1, n_element=100000)
    
    x = np.array([1, 0, 0])
    uvv = line_vortex.biot_savart(x)
    print(uvv)