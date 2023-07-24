import numpy as np
from pytest import approx

import OpenAEM

class Test_Attached_Eddy:
    def test_mirror(self):
        pts = np.array([OpenAEM.Vector(0, -0.5, 0),
                        OpenAEM.Vector(0, -0.5, 1),
                        OpenAEM.Vector(0,  0.5, 1),
                        OpenAEM.Vector(0,  0.5, 0)])
        
        curves = []
        for i in range(3):
            p0 = pts[i]
            p1 = pts[i+1]
            curves.append(OpenAEM.DLS(p0, p1))
            
        curves = np.array(curves)
        Pi_eddy = OpenAEM.Attached_Eddy(curves)
        Pi_eddy_mirror = Pi_eddy.create_mirror()
        
        for i in range(3):
            p0 = Pi_eddy_mirror.curves[i].xv0
            p1 = Pi_eddy_mirror.curves[i].xv1
            assert(p0 == approx(OpenAEM.mirror_Vector(pts[3 - i])))
            assert(p1 == approx(OpenAEM.mirror_Vector(pts[2 - i])))
            