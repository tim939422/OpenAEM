from pytest import approx
import OpenAEM

class Test_Vector:
    def test_mirror_Vector(self):
        x1  = OpenAEM.Vector(1, 1, 1)
        x1m = OpenAEM.mirror_Vector(x1)
        assert(x1m == approx(OpenAEM.Vector(1, 1, -1)))