#test_utils.py

import unittest
import numpy as np
from utils import get_rotated_direction

class UtilsTestCase(unittest.TestCase):
	""" Tests for encoders.utils """
	def setUp(self):
		self.arr = np.arange(0,9).reshape([3,3])
		self.arr3d = np.arange(0,11*11*11).reshape([11,11,11])
		self.direction_values = {
			"n": 1,
			"w": 3,
			"s": 7,
			"e": 5,
			"o": 4
		}
		
	def test_one_rot(self):
		for k, v in self.direction_values.items():
			new_a = np.rot90(self.arr, k=1)
			new_direction = get_rotated_direction(k, 1)
			new_direction_value = self.direction_values[new_direction]
			new_value = new_a.flatten()[new_direction_values]
			self.assertEqual(new_value, v)
		
if __name__ == "__main__":
	unittest.main()