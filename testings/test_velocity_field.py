import OpenAEM.velocity_field as velocity_field
from pytest import approx
import numpy as np

class Test_Velocity_Field:
    def test_grid_1d(self):
        x = velocity_field.get_grid_1d(0.0, 1.0, 0.2)
        assert(x == approx(np.array([0.1, 0.3, 0.5, 0.7, 0.9])))