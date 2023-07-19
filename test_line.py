from pytest import approx
import numpy as np

from line import Line
class Test_Line:
    def test_get_length(self):
        p0 = np.array([0, 0, 0])
        p1 = np.array([1, 0, 0])
        line_1 = Line(p0, p1)
        assert(line_1.get_length() == approx(1.0))
        
        p0 = np.array([0, 0, 0])
        p1 = np.array([1, 1, 1])
        line_2 = Line(p0, p1)
        assert(line_2.get_length() == approx(np.sqrt(3)))
        
    def test_dimension(self):
        p0 = np.array([0, 0, 0])
        p1 = np.array([1, 0, 0])
        line = Line(p0, p1)
        assert(line.get_nc() == 100)
        assert(line.get_np() == 101)
        
    def test_get_xv(self):
        p0 = np.array([0, 0, 0]).reshape(-1, 1)
        p1 = np.array([1, 1, 1]).reshape(-1, 1)
        line = Line(p0, p1)
        xv = line.get_xv()
        
        # test if starting and ending points are on the line
        assert(xv[:, 0].reshape(-1,1) == approx(p0))
        assert(xv[:, -1].reshape(-1,1) == approx(p1))
        
        # test unit vector along the line
        # problem with zero equality test: must be fixed in the future
        t1 = (p1 - p0)/np.linalg.norm(p1 - p0)
        pm = xv[:, np.random.choice(xv.shape[1])].reshape(-1, 1)
        t2 = (pm - p0)/np.linalg.norm(pm - p0)
        assert(approx(t1[0,0]) == approx(t2[0,0], 1e-6))
        assert(approx(t1[1,0]) == approx(t2[1,0], 1e-6))
        assert(approx(t1[2,0]) == approx(t2[2,0], 1e-6))
        
        
    def test_get_xvd(self):
        p0 = np.array([0, 0, 0]).reshape(-1, 1)
        p1 = np.array([1, 1, 1]).reshape(-1, 1)
        line = Line(p0, p1)
        xvd = line.get_xvd()
        assert(xvd[0, 0] == approx(1.0))
        
    def test_get_distance(self):
        p0 = np.array([0, 0, 0]).reshape(-1, 1)
        p1 = np.array([1, 1, 1]).reshape(-1, 1)
        line = Line(p0, p1)
        
        
        
        