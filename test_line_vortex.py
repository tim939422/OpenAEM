
from pytest import approx

from vector import Vector
from line import Line
from line_vortex import biot_savart

class Test_Line_Vortex:
    ABS_EPS = 1e-10
    def test_biot_savart_single_pt(self):
        L = 1e5
        p0 = Vector(0, 0, -L/2)
        p1 = Vector(0, 0, L/2)
        line = Line(p0, p1)
        
        xv = Vector(1, 0, 0)
        u, v, w = biot_savart(xv, line)
        assert(u == approx(0.0, abs=Test_Line_Vortex.ABS_EPS))
        assert(v == approx(1.0))
        assert(w == approx(0.0, abs=Test_Line_Vortex.ABS_EPS))
        
        