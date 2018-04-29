from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf
from random import sample
from random import choice

from tensorflow.python.platform import app
from delf import feature_io
from os import listdir
tf.logging.set_verbosity(tf.logging.INFO)
from time import time

_DISTANCE_THRESHOLD = 0.8

def n_inliers(features_1_path, features_2_path):
	
	# Read features.
	try:
		locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(
				features_1_path)
		num_features_1 = locations_1.shape[0]
	
		locations_2, _, descriptors_2, _, _ = feature_io.ReadFromFile(
				features_2_path)
		num_features_2 = locations_2.shape[0]
	except:
		return 0

	# Find nearest-neighbor matches using a KD tree.
	t0 = time()
	d1_tree = cKDTree(descriptors_1)
	_, indices = d1_tree.query(
			descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

	# Select feature locations for putative matches.
	locations_2_to_use = np.array([
			locations_2[i,]
			for i in range(num_features_2)
			if indices[i] != num_features_1
	])
	locations_1_to_use = np.array([
			locations_1[indices[i],]
			for i in range(num_features_2)
			if indices[i] != num_features_1
	])

	# Perform geometric verification using RANSAC.
	try:
		_, inliers = ransac(
			(locations_1_to_use, locations_2_to_use),
			AffineTransform,
			min_samples=3,
			residual_threshold=20,
			max_trials=20)
		return sum(inliers)
	except:
		return 0


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	
	parser = argparse.ArgumentParser()
	parser.register('type', 'bool', lambda v: v.lower() == 'true')
	parser.add_argument(
			'--test_features_path',
			type=str,
			default='/beegfs/ss8464/code/data/query_features',
			help="""
			Path to the folder with test .delfs.
			""")
	parser.add_argument(
			'--train_features_path',
			type=str,
			default='/beegfs/ss8464/code/train_features',
			help="""
			Path to the folder with train .delfs.
			""")
	parser.add_argument(
			'--csv_path',
			type=str,
			default="recognitionDELF.csv",
			help="""
			Path for submission file to be written. 
			""")
	parser.add_argument(
			'--labels_path',
			type = str,
			default='train.csv',
			help = """,
			Path for train.csv
			""")
	cmd_args, unparsed = parser.parse_known_args()
		
	feature_dir_train = cmd_args.train_features_path
	feature_dir_test = cmd_args.test_features_path
	labels_path = cmd_args.labels_path 
	csv_path = cmd_args.csv_path
	
	NUM_CLASSES = 14951
	train = [None] * NUM_CLASSES # Number of classes 
	
	labels = {}
	
	file = open(labels_path).read().splitlines()[1:]
	
	for line in file:
		cols = line.split(',')
		img_id = cols[0].replace('"','')
		label = int(cols[2])
		
		labels[img_id] = label
		
		if train[label] == None:
			train[label] = [img_id]
		else:
			train[label].append(img_id)
			
	
	knn = open("rec_knn_output.csv").read().splitlines()[1:]
	
	file = open(csv_path, 'w')
	file.write("id,landmarks")
	
	for line in knn:
		cols = line.split(',')
		test_id = cols[0]
		train_ids1 = cols[1].split()[:4] # Getting the 1st 4 for each 
		class_ids = [labels[i] for i in train_ids1] # Getting the corresponding labels 
		train_ids2 = [choice(train[i]) for i in class_ids]
		
		total_inliers = 0
		
		preds = []
		
		feature_test_path = feature_dir_test + '/' + test_id + '.delf'
		
		for class_id, train_id1, train_id2 in zip(class_ids, train_ids1, train_ids2):
			feature_train_path1 = feature_dir_train + '/' + train_id1 + '.delf'
			feature_train_path2 = feature_dir_train + '/' + train_id2 + '.delf'
			
			inliers1 = n_inliers(feature_test_path, feature_train_path1)
			inliers2 = n_inliers(feature_test_path, feature_train_path2)
			
			preds.append((inliers1 + inliers2, class_id)) # Adding a total inliers for this class 
			total_inliers += inliers1 + inliers2
			
		pred = max(preds)
		final_label = pred[1]
		if(total_inliers==0):
			total_inliers = 0.001
		confidence = pred[0] / total_inliers
		
		line = test_id + ',' + str(final_label) + ' ' + str(confidence)
		
		file.write('\n' + line)
		file.flush()

	
	file.close()
			