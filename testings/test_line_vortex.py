
from pytest import approx

import OpenAEM

class Test_Line_Vortex:
    ABS_EPS = 1e-10
    def test_biot_savart_single_pt(self):
        L = 1e5
        p0 = OpenAEM.Vector(0, 0, -L/2)
        p1 = OpenAEM.Vector(0, 0, L/2)
        line = OpenAEM.DLS(p0, p1)
        
        xv = OpenAEM.Vector(1, 0, 0)
        u, v, w = OpenAEM.biot_savart(xv, line)
        assert(u == approx(0.0, abs=Test_Line_Vortex.ABS_EPS))
        assert(v == approx(1.0))
        assert(w == approx(0.0, abs=Test_Line_Vortex.ABS_EPS))
        
        