from pytest import approx
from vector import *

class Test_Vector:
    def test_mirror_Vector(self):
        x1  = Vector(1, 1, 1)
        x1m = mirror_Vector(x1)
        assert(x1m == approx(Vector(1, 1, -1)))