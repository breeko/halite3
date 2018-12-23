import numpy as np
from operator import add

def roll_and_crop(arr: np.array, point: tuple, radius: int) -> np.array:
	""" rolls a given array so that the point is in the center and sides cropped
		input:
			arr (np.array): 		2-d array to roll
			point (tuple or None):	point to be the center
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
	if point is None:
		point = len(arr) // 2, len(arr[0]) // 2 

	diameter = min(len(arr), len(arr[0]))
	min_diameter = radius * 2 + 1
	if min_diameter > diameter:
		reps = int(min_diameter / diameter) + 1
		arr = tile(arr, reps)
	
	shift_rows = radius - point[0]
	shift_cols = radius - point[1]
	shifted = np.roll(arr, [shift_rows, shift_cols], axis=[0,1])
	crop_shape = [radius * 2 + 1 for _ in range(len(shifted))]
	cropped = crop(shifted, crop_shape)

	return cropped

def crop(arr: np.array, bounding: tuple):
	""" crops an array based on bounding dimensions """
	start = tuple(map(lambda a, da: a//2-da//2, arr.shape, bounding))
	end = tuple(map(add, start, bounding))
	slices = tuple(map(slice, start, end))
	return arr[slices]

def tile(arr: np.array, reps: int) -> np.array:
	return np.tile(np.tile(arr.T, reps).T, reps)
