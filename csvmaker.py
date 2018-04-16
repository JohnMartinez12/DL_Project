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

from tensorflow.python.platform import app
from delf import feature_io
from os import listdir
tf.logging.set_verbosity(tf.logging.INFO)
from time import time

_DISTANCE_THRESHOLD = 0.8

def n_inliers(features_1_path, features_2_path):
	tf.logging.set_verbosity(tf.logging.INFO)

	# Read features.
	locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(
			features_1_path)
	num_features_1 = locations_1.shape[0]
	
	locations_2, _, descriptors_2, _, _ = feature_io.ReadFromFile(
			features_2_path)
	num_features_2 = locations_2.shape[0]

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
	_, inliers = ransac(
			(locations_1_to_use, locations_2_to_use),
			AffineTransform,
			min_samples=3,
			residual_threshold=20,
			max_trials=1000)


	return sum(inliers)


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.register('type', 'bool', lambda v: v.lower() == 'true')
	parser.add_argument(
			'--query_features_path',
			type=str,
			default='query_features',
			help="""
			Path to the folder with query .delfs.
			""")
	parser.add_argument(
			'--database_features_path',
			type=str,
			default='database_features',
			help="""
			Path to the folder with database .delfs.
			""")
	parser.add_argument(
			'--thresh',
			type=int,
			default=10,
			help="""
			Minimum inliers for image to be included.
			""")
	parser.add_argument(
			'--nclosest',
			type=int,
			default=150,
			help="""
			Maximum row size.
			""")
	parser.add_argument(
			'--random_size',
			type=int,
			default=-1,
			help="""
			Number of query images to be sampled. Use -1 or default for no sampling. 
			""")
	parser.add_argument(
			'--csv_path',
			type=str,
			default="submission.csv",
			help="""
			Path for submission file to be written. 
			""")

	cmd_args, unparsed = parser.parse_known_args()
		
	feature_dir_query = cmd_args.query_features_path
	feature_dir_database = cmd_args.database_features_path
	inliers_thresh = cmd_args.thresh
	n_closest = cmd_args.nclosest
	random_size = cmd_args.random_size
	csv_path = cmd_args.csv_path
	
	failures = 0
	file = open(csv_path, 'w')
	file.write("id, images")
	
	query_dirs = listdir(feature_dir_query)
	database_dirs = listdir(feature_dir_database)
	
	if random_size >= 0:
		query_dirs = sample(query_dirs, random_size)
		
	for feature_query in query_dirs:
		feature_query_path = feature_dir_query+'/'+feature_query
		feature_query_id = feature_query.split('.')[0]
		# Getting the path and id 
		h = []
		for feature_database in database_dirs:
			feature_database_path = feature_dir_database + '/' + feature_database
			feature_database_id = feature_database.split('.')[0]
			
			# Calculating # of outliers between query feature and db feature
			try:
				similarity = n_inliers(feature_query_path, feature_database_path)
				if similarity >= inliers_thresh:
					h.append((similarity, feature_database_id))
				print(feature_query_id, feature_database_id, similarity)
			except:
				print(feature_query_id, feature_database_id, "FAIL")
				failures += 1		
				
		# Sorting by similarity in descending order 
		h.sort(key = lambda x: x[0],reverse = True)
		images = [i[1] for i in h[:n_closest]]
		
		line = '\n' + feature_query_id + ',' + ' '.join(images)
		
		file.write(line)
		print(line,end='')
	file.close()
	print("Failures:",failures)
			
		
		
		
			
			
	
	
	
	
	
	
	