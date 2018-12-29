import numpy as np
from operator import add

def roll_and_crop(arr: np.array, x: int, y: int, radius: int) -> np.array:
	""" rolls a given array so that the point is in the center and sides cropped
		input:
			arr (np.array): 		2-d or 3-d array to roll
			x (int):				center x coordinate
			y (int):				center y coordinate
			radius (int): 			radius of the array with the point as the center
		output:
			out (np.array): 		rolled and shifted array with point as its center
			input:
			arr (np.array): 		2-d array to roll
			point (tuple or None):	point to be the center
			radius (int): 			radius of the array with the point as the center
		output:
			out (np.array): 		rolled and shifted array with point as its center

		e.g.
			input: 
				arr = 
					a b c d
					e f g h
					i j k l
					m n o p
				point = (0, 2)
				radius = 1
			output:
				i j k l i
				m n o p m
				a b c d a
				e f g h e
				i j k l i			
	
	"""
	shortest_side = min(arr.shape[:2])
	min_length = radius * 2 + 1

	if min_length > shortest_side:
		reps = int(min_length / shortest_side) + 1
		arr = tile(arr, reps)
	shift_rows =  radius - y
	shift_cols =  radius - x
	shifted = np.roll(arr, [shift_rows, shift_cols], axis=[0,1])
	cropped = shifted[:min_length, :min_length]
	return cropped

def tile(arr: np.array, reps: int) -> np.array:
	tile_reps = (reps, reps) + arr.shape[2:]
	return np.tile(arr, reps=tile_reps)

