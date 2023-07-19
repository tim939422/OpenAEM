import numpy as np


class Line:
    '''
    A line parameterized by a 0 <= t <= 1

        xv = (1-t)*p0 + t*p1
        
    '''
    def __init__(self, p0, p1, ds=0.01):
        self.p0 = p0.reshape(-1, 1)
        self.p1 = p1.reshape(-1, 1)

        
        self.t = self.__discretize_tspace(ds)
        self.np = np.size(self.t, axis=1) # num of points
        self.nc = self.np - 1 # num of cells
        self.ds = self.__compute_length()/len(self.t - 1)
        
    # public methods
    def get_np(self):
        return self.np
    def get_nc(self):
        return self.nc
    def get_ds(self):
        return self.ds
    def get_t(self):
        return self.t
    
    def get_length(self):
        return self.__compute_length()
    def get_xv(self):
        return self.__compute_xv()
    def get_xvd(self):
        return self.__compute_xpvd()
    def get_distance(self, xv):
        s = np.linalg.norm(xv.reshape(-1, 1) - self.get_xv(), axis=0)
        return np.min(s)
        
    
    
    # private methods
    # this is overkill for just a simple straight line. However, they are
    # provided here for further spline support
    def __discretize_tspace(self, ds):
        """discretize parametric space

        Args:
            ds (float): request spacing
            
        Returns:
            float array: discretized parameter t
        """
        length = self.__compute_length()
        n = int(np.ceil(length/ds))
        t = np.linspace(0, 1, n + 1).reshape(1, -1)
        return t
        
    def __compute_length(self):
        """analytical length of a line

        Returns:
            float: line length
        """        
        return np.linalg.norm(self.p1 - self.p0)
    
    def __compute_xv(self):
        """discretized point on the line

        Returns:
            float array: (3, M)
        """        
        return np.matmul(self.p0, 1 - self.t) + np.matmul(self.p1, self.t)
    
    def __compute_xpvd(self):
        dot_xv = -self.p0 + self.p1
        return np.repeat(dot_xv, self.np, axis=1)
    
if __name__ == '__main__':
    p0 = np.array([0, 0, 0])
    p1 = np.array([1, 0, 0])
    line_1 = Line(p0, p1)