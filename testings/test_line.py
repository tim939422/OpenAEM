import OpenAEM
from pytest import approx
import numpy as np

class Test_Line:
    ABS_EPS = 1e-10
    def test_intersect(self):
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 1.0, 0.0])
        line_1 = OpenAEM.Line(p0, p1)    
        p0 = np.array([0.0, 1.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        line_2 = OpenAEM.Line(p0, p1)
        
        cross = OpenAEM.Line.intersect(line_1, line_2)
        assert(cross == approx(np.array([0.5, 0.5, 0.0])))
        
    def test_mirror_point(self):
        p = np.array([1.0, 1.0, 1.0])
        assert(OpenAEM.Line.mirror_point(p, symmetry_plane='xz')
               == approx(np.array([1.0, -1.0, 1.0])))
        assert(OpenAEM.Line.mirror_point(p, symmetry_plane='xy')
               == approx(np.array([1.0, 1.0, -1.0])))
        assert(OpenAEM.Line.mirror_point(p, symmetry_plane='yz')
               == approx(np.array([-1.0, 1.0, 1.0])))
        
    def test_distance2(self):
        p0 = np.array([0, 0, 0])
        p1 = np.array([1, 1, 1])
        line = OpenAEM.Line(p0, p1)
        
        # test zero distance
        p = np.array((0.2, 0.2, 0.2))
        assert(line.distance2(p) == approx(0.0, abs=Test_Line.ABS_EPS))
        
        # 
        p = np.array([1.0, 1.0, 0.0])
        smin = line.distance2(p)
        assert( smin == approx(np.sqrt(2)/np.sqrt(3)))
        
        # horizontal line
        p0 = np.array([1, -0.5, 1])
        p1 = np.array((1, 0.5, 1))
        line = OpenAEM.Line(p0, p1)
        p = np.array([1, 0, 1])
        assert(line.distance2(p) == approx(0.0, abs= Test_Line.ABS_EPS))
        
    def test_get_length(self):
        p0 = np.array([0, 0, 0])
        p1 = np.array([1, 0, 0])
        line = OpenAEM.DLS(p0, p1)
        assert(line.get_length() == approx(1.0))
        
        p0 = np.array([0, 0, 0])
        p1 = np.array([1, 1, 1])
        line = OpenAEM.DLS(p0, p1)
        assert(line.get_length() == approx(np.sqrt(3)))
        
    def test_get_n(self):
        p0 = np.array([0, 0, 0])
        p1 = np.array([1, 0, 0])
        line = OpenAEM.DLS(p0, p1, ds = 0.2)
        assert(line.get_n() == 5)
        
        line = OpenAEM.DLS(p0, p1, ds = 0.15)
        assert(line.get_n() == 7)
        
    def test_get_ds(self):
        p0 = np.array([0, 0, 0])
        p1 = np.array([1, 0, 0])
        line = OpenAEM.DLS(p0, p1, ds = 0.2)
        assert(line.get_ds() == approx(0.2))
        
        line = OpenAEM.DLS(p0, p1, ds = 0.15)
        assert(line.get_ds() == approx(1/7))
        
    def test_get_t(self):
        p0 = np.array([0, 0, 0])
        p1 = np.array([1, 0, 0])
        line = OpenAEM.DLS(p0, p1, ds = 0.2)
        assert(line.get_t() == approx(np.array([0.1, 0.3, 0.5, 0.7, 0.9])))
        
        line = OpenAEM.DLS(p0, p1, ds = 0.15) # 7 divisions
        t = np.linspace(0, 1, 7, endpoint=False)
        t = t + 0.5*(t[1] - t[0])
        assert(line.get_t() == approx(t))
        
    def test_points(self):
        p0 = np.array([0, 0, 0])
        p1 = np.array([1, 0, 0])
        line = OpenAEM.DLS(p0, p1, ds = 0.2)
        pts = np.array([[0.1, 0.3, 0.5, 0.7, 0.9],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0]])
        
        assert(line.points() == approx(pts))
        
    def test_dirs(self):
        p0 = np.array([0, 0, 0])
        p1 = np.array([1, 0, 0])
        line = OpenAEM.DLS(p0, p1, ds=0.2)
        dirs = np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0]])
        assert(line.dirs() == approx(dirs))
        
    def test_reverse(self):
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        line = OpenAEM.DLS(p0, p1)
        rline = line.reverse()
        assert(rline.get_dir() == approx(np.array([-1.0, 0.0, 0.0])))
        
    def test_mirror(self):
        p0 = np.array([0, 0, 1])
        p1 = np.array([1, 0, 1])
        line = OpenAEM.DLS(p0, p1)
        
        assert(line.mirror(symmetry_plane='xy').get_p0() == approx(np.array([0, 0, -1])))
        assert(line.mirror(symmetry_plane='xy').get_p1() == approx(np.array([1, 0, -1])))
        
    