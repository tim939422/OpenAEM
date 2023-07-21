from pytest import approx
import numpy as np

import OpenAEM

class Test_Line:
    ABS_EPS = 1e-10
    
    def test_get_length(self):
        xv0 = OpenAEM.Vector(0, 0, 0)
        xv1 = OpenAEM.Vector(1, 0, 0)
        line = OpenAEM.Line(xv0, xv1)
        assert(line.get_length() == approx(1.0))
        
        xv0 = OpenAEM.Vector(0, 0, 0)
        xv1 = OpenAEM.Vector(1, 1, 1)
        line = OpenAEM.Line(xv0, xv1)
        assert(line.get_length() == approx(np.sqrt(3)))

        
    def test_get_distance(self):
        xv0 = OpenAEM.Vector(0, 0, 0)
        xv1 = OpenAEM.Vector(1, 1, 1)
        line = OpenAEM.Line(xv0, xv1)
        
        # test zero distance
        xv = OpenAEM.Vector(0.2, 0.2, 0.2)
        assert(line.get_distance(xv) == approx(0.0, abs=Test_Line.ABS_EPS))
        
        # 
        xv = OpenAEM.Vector(1.0, 1.0, 0.0)
        smin = line.get_distance(xv)
        assert( smin == approx(np.sqrt(2)/np.sqrt(3)))
        
        # horizontal line
        xv0 = OpenAEM.Vector(1, -0.5, 1)
        xv1 = OpenAEM.Vector(1, 0.5, 1)
        line = OpenAEM.Line(xv0, xv1)
        xv = OpenAEM.Vector(1, 0, 1)
        assert(line.get_distance(xv) == approx(0.0, abs= Test_Line.ABS_EPS))
        
        
    def test_discretization(self):
        xv0 = OpenAEM.Vector(0, 0, 0)
        xv1 = OpenAEM.Vector(1, 0, 0)
        line = OpenAEM.Line(xv0, xv1, ds = 0.2)
        assert(line.get_ds() == approx(0.2))
        assert(line.get_ndvis() == 5)
        assert(line.get_npts() == 6)
        assert(line.get_t() == approx(np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).reshape(1, -1)))
        
        xv0 = OpenAEM.Vector(0, 0, 0)
        xv1 = OpenAEM.Vector(1, 0, 0)
        line = OpenAEM.Line(xv0, xv1, ds = 0.15)
        assert(line.get_ds() == approx(1.0/7))
        assert(line.get_ndvis() == 7)
        assert(line.get_npts() == 8)
        assert(line.get_t() == approx(np.linspace(0, 1.0, 8).reshape(1, -1)))
        
    def test_get_points(self):
        xv0 = OpenAEM.Vector(0, 0, 0)
        xv1 = OpenAEM.Vector(1, 0, 0)
        line = OpenAEM.Line(xv0, xv1, ds = 0.2)
        pts = np.array([[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        assert(line.get_points() == approx(pts))
        
    def test_get_direction(self):
        xv0 = OpenAEM.Vector(0, 0, 0)
        xv1 = OpenAEM.Vector(1, 0, 0)
        line = OpenAEM.Line(xv0, xv1, ds = 0.2)
        directions = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        assert(line.get_direction() == approx(directions))
        
    def test_get_reverse_line(self):
        xv0 = OpenAEM.Vector(0, 0, 0)
        xv1 = OpenAEM.Vector(1, 0, 0)
        line = OpenAEM.Line(xv0, xv1, ds = 0.2)
        reversed_line = line.reverse_line()
        directions = np.array([[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        assert(reversed_line.get_direction() == approx(directions))
        
    def test_mirror_line(self):
        xv0 = OpenAEM.Vector(0, 0, 1)
        xv1 = OpenAEM.Vector(1, 0, 1)
        line = OpenAEM.Line(xv0, xv1, ds = 0.2)
        
        mirror = line.mirror_line()
        
        assert(mirror.xv0 == approx(OpenAEM.Vector(0, 0, -1)))
        assert(mirror.xv1 == approx(OpenAEM.Vector(1, 0, -1)))