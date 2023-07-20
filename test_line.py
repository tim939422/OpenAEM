from pytest import approx
import numpy as np

from line import Line
from vector import Vector

class Test_Line:
    ABS_EPS = 1e-10
    
    def test_get_length(self):
        xv0 = Vector(0, 0, 0)
        xv1 = Vector(1, 0, 0)
        line = Line(xv0, xv1)
        assert(line.get_length() == approx(1.0))
        
        xv0 = Vector(0, 0, 0)
        xv1 = Vector(1, 1, 1)
        line = Line(xv0, xv1)
        assert(line.get_length() == approx(np.sqrt(3)))

        
    def test_get_distance(self):
        xv0 = Vector(0, 0, 0)
        xv1 = Vector(1, 1, 1)
        line = Line(xv0, xv1)
        
        # test zero distance
        xv = Vector(0.2, 0.2, 0.2)
        assert(line.get_distance(xv) == approx(0.0, abs=Test_Line.ABS_EPS))
        
        # 
        xv = Vector(1.0, 1.0, 0.0)
        smin = line.get_distance(xv)
        assert( smin == approx(np.sqrt(2)/np.sqrt(3)))
        
        # horizontal line
        xv0 = Vector(1, -0.5, 1)
        xv1 = Vector(1, 0.5, 1)
        line = Line(xv0, xv1)
        xv = Vector(1, 0, 1)
        assert(line.get_distance(xv) == approx(0.0, abs= Test_Line.ABS_EPS))
        
        
    def test_discretization(self):
        xv0 = Vector(0, 0, 0)
        xv1 = Vector(1, 0, 0)
        line = Line(xv0, xv1, ds = 0.2)
        assert(line.get_ds() == approx(0.2))
        assert(line.get_ndvis() == 5)
        assert(line.get_npts() == 6)
        assert(line.get_t() == approx(np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).reshape(1, -1)))
        
        xv0 = Vector(0, 0, 0)
        xv1 = Vector(1, 0, 0)
        line = Line(xv0, xv1, ds = 0.15)
        assert(line.get_ds() == approx(1.0/7))
        assert(line.get_ndvis() == 7)
        assert(line.get_npts() == 8)
        assert(line.get_t() == approx(np.linspace(0, 1.0, 8).reshape(1, -1)))
        
    def test_get_points(self):
        xv0 = Vector(0, 0, 0)
        xv1 = Vector(1, 0, 0)
        line = Line(xv0, xv1, ds = 0.2)
        pts = np.array([[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        assert(line.get_points() == approx(pts))
        
    def test_get_direction(self):
        xv0 = Vector(0, 0, 0)
        xv1 = Vector(1, 0, 0)
        line = Line(xv0, xv1, ds = 0.2)
        directions = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        assert(line.get_direction() == approx(directions))