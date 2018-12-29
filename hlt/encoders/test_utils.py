#test_utils.py

import unittest
import numpy as np
from hlt.encoders.utils import crop, roll_and_crop, tile

class UtilsTestCase(unittest.TestCase):
	""" Tests for encoders.utils """
	def setUp(self):
		self.arr = np.arange(0,121).reshape([11,11])
		self.arr3d = np.arange(0,11*11*11).reshape([11,11,11])
		self.radius = self.arr.shape[0] // 2

	def test_tile_1(self):
		reps = 1
		tiled_arr = tile(arr=self.arr, reps = reps)
		expected = tuple(x * reps for x in self.arr.shape)
		self.assertEqual(tiled_arr.shape, expected)

	def test_tile_2(self):
		reps = 2
		tiled_arr = tile(arr=self.arr, reps = reps)
		expected = tuple(x * reps for x in self.arr.shape)
		self.assertEqual(tiled_arr.shape, expected)

	def test_tile_3(self):
		reps = 3
		tiled_arr = tile(arr=self.arr, reps = reps)
		expected = tuple(x * reps for x in self.arr.shape)
		self.assertEqual(tiled_arr.shape, expected)
	
	def test_roll_and_crop_roll(self):
		y = self.arr.shape[0] - 1
		x = self.arr.shape[1] - 1
		rolled = roll_and_crop(arr=self.arr, x=x, y=y, radius=self.radius)
		self.assertTupleEqual(rolled.shape, self.arr.shape)

	def test_roll_and_crop_full_center(self):
		y = self.arr.shape[0] // 2
		x = self.arr.shape[1] // 2
		rolled = roll_and_crop(arr=self.arr, x=x, y=y, radius=self.radius)
		self.assertListEqual(rolled.tolist(), self.arr.tolist())
	
	def test_crop(self):
		bounding = (3,3,3)
		cropped = crop(arr=self.arr3d, bounding=bounding)
		self.assertTupleEqual(cropped.shape, bounding)

if __name__ == "__main__":
	unittest.main()