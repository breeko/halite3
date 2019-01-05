import shutil
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", action="store", dest="path", type=str, required=True, help="path of files to move")
parser.add_argument("-t", "--train", action="store", dest="train", type=str, required=True, help="path of train directory")
parser.add_argument("-v", "--val", action="store", dest="val", type=str, required=True, help="path of validation directory")
parser.add_argument("-e", "--perc-train", action="store", dest="perc", type=float, required=True, help="percentage of train")
parser.add_argument("-x", "--ext", action="store", dest="ext", type=str, required=True, help="file extension")

args = parser.parse_args()

def split_test_train(path: str, file_ext: str, perc_train: float, train_path: str, val_path: str):
	assert perc_train <= 100, "perc train must be less than or equal to 100"
	assert perc_train >= 0, "perc train must be greater than or equal to 0"
	if perc_train > 1.0:
		perc_train = perc_train / 100.0
	files = set([f for f in os.listdir(path) if f[-len(file_ext):].lower() == file_ext.lower()])
	num_files = len(files)
	num_train = int(num_files * perc_train)
	num_val = num_files - num_train
	train_files = np.random.choice(list(files), size=num_train, replace=False)
	val_files = files.difference(train_files)
	if path != train_path:
		for f in train_files:
			shutil.move("{}/{}".format(path, f), "{}".format(train_path))
	if path != val_path:
		for f in val_files:
			shutil.move("{}/{}".format(path, f), "{}".format(val_path))
	print("Done!")
	print("Train files: {}\n Validation files: {}".format(num_train, num_val))

if __name__ == "__main__":
	split_test_train(path=args.path, file_ext=args.ext, perc_train=args.perc, train_path=args.train, val_path=args.val)
	