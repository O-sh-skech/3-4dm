import unittest
from dataset1 import true_function

class TestTrueFunction(unittest.TestCase):
    def test_x_zero(self):
        y = true_function(0)
        self.assertAlmostEqual(y, 0.0)

if __name__ == "__main__":
    unittest.main()