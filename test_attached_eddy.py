import numpy as np
from pytest import approx

from line import Line
from vector import Vector, mirror_Vector
from attached_eddy import Attached_Eddy

class Test_Attached_Eddy:
    def test_mirror(self):
        pts = np.array([Vector(0, -0.5, 0),
                        Vector(0, -0.5, 1),
                        Vector(0,  0.5, 1),
                        Vector(0,  0.5, 0)])
        
        curves = []
        for i in range(3):
            p0 = pts[i]
            p1 = pts[i+1]
            curves.append(Line(p0, p1))
            
        curves = np.array(curves)
        Pi_eddy = Attached_Eddy(curves)
        Pi_eddy_mirror = Pi_eddy.create_mirror()
        
        for i in range(3):
            p0 = Pi_eddy_mirror.curves[i].xv0
            p1 = Pi_eddy_mirror.curves[i].xv1
            assert(p0 == approx(mirror_Vector(pts[3 - i])))
            assert(p1 == approx(mirror_Vector(pts[2 - i])))
            