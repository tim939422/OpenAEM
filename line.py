import numpy as np


class Line:
    '''
    A line parameterized by a 0 <= t <= 1

        xv = (1-t)*xv0 + t*xv1
        
    '''
    def __init__(self, xv0, xv1, ds=0.01):
        """construct a line object

        Args:
            xv0 ((3, 1) numpy.ndarray): starting point
            xv1 ((3, 1) numpy.ndarray): ending point
            ds (float, optional): request uniform spacing. Defaults to 0.01.
        """        
        self.xv0 = xv0
        self.xv1 = xv1
        
        # obtain discretized line
        length  = self.get_length()
        
        self.__ndivs = int(np.ceil(length/ds)) # number of divisions
        self.__npts  = self.__ndivs + 1 # number of points
        self.__ds    = length / self.__ndivs # actual spacing
        
        # parameters (1, npts)
        self.__t     = np.linspace(0.0, 1.0, self.__npts).reshape(1, -1)
        
        
    # public methods
    def get_distance(self, xv):
        """distance from a point xv to current line
                
        Args:
            xv (array (3, 1)): point
        
        Returns:
            float: distance
            
        Formulas:
                     |(x - x0) x (x - x1)|
        distance = -------------------
                        |x1 - x0|
        see: https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        """        
        linev    = self.xv1 - self.xv0
        cross_product = np.cross(xv - self.xv0, xv - self.xv1, axis=0)
        distance = np.linalg.norm(cross_product)/np.linalg.norm(linev)
        return distance
    def get_length(self):
        """length of segment from xv0 to xv1
            
        Returns:
            float: length

        Formulas:
            length = || xv1 - xv0 ||
        """        
        return np.linalg.norm(self.xv1 - self.xv0)
    
    def get_points(self):
        """sample points along the line

        Returns:
            float ndarray: (3, M) array and each column is a point
        """        
        return np.matmul(self.xv0, 1 - self.get_t()) + np.matmul(self.xv1, self.get_t())
    
    def get_direction(self):
        xvd = -self.xv0 + self.xv1
        return np.repeat(xvd, self.get_npts(), axis=1)
    
    # wrappers to fetch private attributes
    def get_ndvis(self):
        return self.__ndivs
    def get_npts(self):
        return self.__npts
    def get_ds(self):
        return self.__ds
    def get_t(self):
        return self.__t

    
if __name__ == '__main__':
    pass