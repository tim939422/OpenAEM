import numpy as np

class Line:
    def __init__(self, p0, p1) -> None:
        """analytical line by two points

        Args:
            p0 (ndarray): (3,)
            p1 (ndarray): (3,)
        """        
        self.p0 = p0
        self.p1 = p1
        self.dir = p1 - p0
        
    
    def distance2(self, p):
        """distance from a point p to current line
                
        Args:
            p (ndarray): (3,) point
        
        Returns:
            float: distance
            
        Formulas:
                     |(p - p0) x (p - p1)|
        distance = -------------------
                        |p1 - p0|
        see: https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        """        
        cross_product = np.cross(p - self.p0, p - self.p1)
        distance = np.linalg.norm(cross_product)/np.linalg.norm(self.dir)
        return distance
    
    @staticmethod
    def intersect(line_1, line_2):
        """intersection of two lines

        Args:
            line_1 (Line): line 1
            line_2 (Line): line 2

        Returns:
            ndarray: (3,) intersection point
        """        
        p0 = line_1.get_p0(); d0 = line_1.get_dir()
        p1 = line_2.get_p0(); d1 = line_2.get_dir()
        
        At = np.vstack((d0, -d1))
        A  = At.T
        b  = (p1 - p0).reshape(-1, 1)
        C = np.matmul(At, A)
        d = np.matmul(At, b)
        t, _ = np.linalg.solve(C, d)
        return p0 + t*d0
    
    @staticmethod
    def mirror_point(p, symmetry_plane='xz'):
        """mirror a point 

        Args:
            p (ndarray): (3,)
            symmetry_plane (str, optional): symmetry. Defaults to 'xz'.

        Returns:
            ndarray: (3,)
        """        
        if symmetry_plane == 'xz':
            return np.array([p[0], -p[1], p[2]])
        elif symmetry_plane == 'xy':
            return np.array([p[0], p[1], -p[2]])
        elif symmetry_plane == 'yz':
            return np.array([-p[0], p[1], p[2]])
        
    
    # getter and setter
    def get_p0(self):
        """get a point on a line

        Returns:
            ndarray: (3,) point
        """        
        return self.p0
    
    def get_p1(self):
        return self.p1
    
    def get_dir(self):
        """get direction vector of a line

        Returns:
            ndarray: (3,)
        """        
        return self.dir

class DLS(Line):
    def __init__(self, p0, p1, ds=0.01) -> None:
        """discretized directed line segment (DLS)

        Args:
            p0 (ndarray): starting point
            p1 (ndarray): ending point
            ds (float, optional): target spacing. Defaults to 0.01.
        
        Formulas:
            p = (1-t)*p0 + t*p1 (0 <= t <= 1)
        """        
        super().__init__(p0, p1)
        
        # discretize DSL
        self.length = np.linalg.norm(self.p1 - self.p0)
        self.n = int(np.ceil(self.length / ds))
        self.ds = self.length / self.n # actual spacing
        self.t = 1/self.n*0.5 + np.arange(self.n)/self.n
        
    # public methods
    def points(self):
        """sample points along the line (division center)

        Returns:
            float ndarray: (3, n) array and each column is a point
        """
        x = (1 - self.t)*self.p0[0] + self.t*self.p1[0]
        y = (1 - self.t)*self.p0[1] + self.t*self.p1[1]
        z = (1 - self.t)*self.p0[2] + self.t*self.p1[2]
        
        return np.vstack((x, y, z))
        
    
    def dirs(self):
        """direction vectors at each division center

        Returns:
            ndarray: (3, n)
        """        
        xdirs = np.ones(self.n)*self.get_dir()[0]
        ydirs = np.ones(self.n)*self.get_dir()[1]
        zdirs = np.ones(self.n)*self.get_dir()[2]
        return np.vstack((xdirs, ydirs, zdirs))
    
    def reverse(self):
        """reverse current DLS

        Returns:
            DSL: point from p1 to p0
        """        
        return DLS(self.p1, self.p0, ds=self.ds)
    def mirror(self, symmetry_plane='xy'):
        p0 = Line.mirror_point(self.p0, symmetry_plane=symmetry_plane)
        p1 = Line.mirror_point(self.p1, symmetry_plane=symmetry_plane)
        return DLS(p0, p1, ds=self.ds)
        
    
    # wrappers to fetch private attributes
    def get_n(self):
        """number of subdivisions

        Returns:
            int: n
        """        
        return self.n
    def get_ds(self):
        """actual spacing

        Returns:
            float: ds
        """        
        return self.ds
    def get_t(self):
        """discretized parameter t

        Returns:
            ndarray: (n,)
        """        
        return self.t
    def get_length(self):
        """length of DLS
            
        Returns:
            float: length

        Formulas:
            length = || p1 - p0 ||
        """        
        return self.length
    
if __name__ == '__main__':
    pass