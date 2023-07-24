
from pytest import approx
import numpy as np
import OpenAEM

class Test_Biot_Savart:
    ABS_EPS = 1e-10
    def test_biot_savart_single_pt(self):
        L = 1e5
        p0 = np.array([0, 0, -L/2])
        p1 = np.array([0, 0, L/2])
        line = OpenAEM.DLS(p0, p1)
        
        xv = np.array([1, 0, 0])
        uv = OpenAEM.biot_savart(xv, line)
        assert(uv == approx(np.array([0.0, 1.0, 0.0])))
        
        