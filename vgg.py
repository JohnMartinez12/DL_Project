import tensorflow as tf
import numpy as np
from os import listdir
import os
from scipy.misc import imread, imresize
from time import time

data_dir_query = 'Dataset_Directory/query'
data_dir_database = 'Dataset_Directory/test'

feature_dir_query = 'feature_query_np'
feature_dir_database = 'feature_test_np'

# The images 
imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])

parameters = []

# - - - - Convolutional Layers - - - - 

# zero-mean input 
with tf.name_scope('preprocess') as scope:
	mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
	images = imgs-mean

# conv1_1
with tf.name_scope('conv1_1') as scope:
	kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
											 stddev=1e-1), name='weights')
	conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
	biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
						 trainable=True, name='biases')
	out = tf.nn.bias_add(conv, biases)
	conv1_1 = tf.nn.relu(out, name=scope)
	parameters += [kernel, biases]
	
# conv1_2
with tf.name_scope('conv1_2') as scope:
	kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
											 stddev=1e-1), name='weights')
	conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
	biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
						 trainable=True, name='biases')
	out = tf.nn.bias_add(conv, biases)
	conv1_2 = tf.nn.relu(out, name=scope)
	parameters += [kernel, biases]
	
# pool1
pool1 = tf.nn.max_pool(conv1_2,
		       ksize=[1, 2, 2, 1],
		       strides=[1, 2, 2, 1],
		       padding='SAME',
		       name='pool1')
		
# conv2_1
with tf.name_scope('conv2_1') as scope:
	kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
											 stddev=1e-1), name='weights')
	conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
	biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
						 trainable=True, name='biases')
	out = tf.nn.bias_add(conv, biases)
	conv2_1 = tf.nn.relu(out, name=scope)
	parameters += [kernel, biases]
	
# conv2_2
with tf.name_scope('conv2_2') as scope:
	kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
											 stddev=1e-1), name='weights')
	conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
	biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
						 trainable=True, name='biases')
	out = tf.nn.bias_add(conv, biases)
	conv2_2 = tf.nn.relu(out, name=scope)
	parameters += [kernel, biases]
	
# pool2
pool2 = tf.nn.max_pool(conv2_2,
					   ksize=[1, 2, 2, 1],
					   strides=[1, 2, 2, 1],
					   padding='SAME',
					   name='pool2')

# conv3_1
with tf.name_scope('conv3_1') as scope:
	kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
											 stddev=1e-1), name='weights')
	conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
	biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
						 trainable=True, name='biases')
	out = tf.nn.bias_add(conv, biases)
	conv3_1 = tf.nn.relu(out, name=scope)
	parameters += [kernel, biases]

# conv3_2
with tf.name_scope('conv3_2') as scope:
	kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
											 stddev=1e-1), name='weights')
	conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
	biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
						 trainable=True, name='biases')
	out = tf.nn.bias_add(conv, biases)
	conv3_2 = tf.nn.relu(out, name=scope)
	parameters += [kernel, biases]

# conv3_3
with tf.name_scope('conv3_3') as scope:
	kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
											 stddev=1e-1), name='weights')
	conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
	biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
						 trainable=True, name='biases')
	out = tf.nn.bias_add(conv, biases)
	conv3_3 = tf.nn.relu(out, name=scope)
	parameters += [kernel, biases]

# pool3
pool3 = tf.nn.max_pool(conv3_3,
					   ksize=[1, 2, 2, 1],
					   strides=[1, 2, 2, 1],
					   padding='SAME',
					   name='pool3')
					
# conv4_1
with tf.name_scope('conv4_1') as scope:
	kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
											 stddev=1e-1), name='weights')
	conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
	biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
						 trainable=True, name='biases')
	out = tf.nn.bias_add(conv, biases)
	conv4_1 = tf.nn.relu(out, name=scope)
	parameters += [kernel, biases]

# conv4_2
with tf.name_scope('conv4_2') as scope:
	kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
											 stddev=1e-1), name='weights')
	conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
	biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
						 trainable=True, name='biases')
	out = tf.nn.bias_add(conv, biases)
	conv4_2 = tf.nn.relu(out, name=scope)
	parameters += [kernel, biases]

# conv4_3
with tf.name_scope('conv4_3') as scope:
	kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
											 stddev=1e-1), name='weights')
	conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
	biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
						 trainable=True, name='biases')
	out = tf.nn.bias_add(conv, biases)
	conv4_3 = tf.nn.relu(out, name=scope)
	parameters += [kernel, biases]

# pool4
pool4 = tf.nn.max_pool(conv4_3,
					   ksize=[1, 2, 2, 1],
					   strides=[1, 2, 2, 1],
					   padding='SAME',
					   name='pool4')
					
# conv5_1
with tf.name_scope('conv5_1') as scope:
	kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
											 stddev=1e-1), name='weights')
	conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
	biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
						 trainable=True, name='biases')
	out = tf.nn.bias_add(conv, biases)
	conv5_1 = tf.nn.relu(out, name=scope)
	parameters += [kernel, biases]

# conv5_2
with tf.name_scope('conv5_2') as scope:
	kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
											 stddev=1e-1), name='weights')
	conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
	biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
						 trainable=True, name='biases')
	out = tf.nn.bias_add(conv, biases)
	conv5_2 = tf.nn.relu(out, name=scope)
	parameters += [kernel, biases]

# conv5_3
with tf.name_scope('conv5_3') as scope:
	kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
											 stddev=1e-1), name='weights')
	conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
	biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
						 trainable=True, name='biases')
	out = tf.nn.bias_add(conv, biases)
	conv5_3 = tf.nn.relu(out, name=scope)
	parameters += [kernel, biases]

# pool5
pool5 = tf.nn.max_pool(conv5_3,
					   ksize=[1, 2, 2, 1],
					   strides=[1, 2, 2, 1],
					   padding='SAME',
					   name='pool4')

# - - - - Fully Connected Layers - - - - 
					
# fc1
with tf.name_scope('fc1') as scope:
	shape = int(np.prod(pool5.get_shape()[1:]))
	fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
												 dtype=tf.float32,
												 stddev=1e-1), name='weights')
	fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
						 trainable=True, name='biases')
	pool5_flat = tf.reshape(pool5, [-1, shape])
	fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
	fc1 = tf.nn.relu(fc1l)
	parameters += [fc1w, fc1b]


# fc2
with tf.name_scope('fc2') as scope:
	fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
													dtype=tf.float32,
													stddev=1e-1), name='weights')
	fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
									trainable=True, name='biases')
	fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
	fc2 = tf.nn.relu(fc2l)
	parameters += [fc2w, fc2b]
	
# fc3
with tf.name_scope('fc3') as scope:
	fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
													dtype=tf.float32,
													stddev=1e-1), name='weights')
	fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
									trainable=True, name='biases')
	fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
	fc3 = tf.nn.relu(fc3l)
	parameters += [fc3w, fc3b]
	
def min(s):
	return "%.2f min" % (s/60)
	
def load_save_features(sess, data_dir, dirs, feature_dir):
	# Creating the directory for query images 
	if not os.path.exists(feature_dir):
		os.mkdir(feature_dir)
	n_dir = len(dirs)
	step_dir = n_dir // 100
	for i, d in enumerate(dirs):
		path = feature_dir+'/'+d[:-4]
		if os.path.exists(path+'.npy'):
			continue
		t0 = time()
		# Reading the image 
		img = imread(data_dir+'/'+d)
		img = imresize(img, (224, 224))
		# Getting, saving the feature vector
		feature = sess.run(fc3, feed_dict={imgs: [img]})
		np.save(path, feature)	
		# Show progress
		if i % step_dir == 0:
			t1 = time()
			print("Progress: %.2f%% Time:" % (100*i/n_dir),min(t1-t0))	
		
	
with tf.Session() as sess:
	# Loading weights
	print('Load weights...')
	t0 = time()
	weight_file = 'vgg16_weights.npz'
	weights = np.load(weight_file)
	keys = sorted(weights.keys())
	for i, k in enumerate(keys):
		sess.run(parameters[i].assign(weights[k]))
	t1 = time()
	print('Load complete. Time:',min(t1-t0))
	
	# Loading images 
	dirs_query = listdir(data_dir_query)
	dirs_database = listdir(data_dir_database)
	
	n_query = len(dirs_query)
	n_database = len(dirs_database)
	
	step_query = n_query // 100
	step_database = n_database // 100
	
	print('Load/save query images...')
	load_save_features(sess, data_dir_query, dirs_query, feature_dir_query)
	print("Query images complete.")	

	print("Load/save database images...")
	load_save_features(sess, data_dir_database, dirs_database, feature_dir_database)
	print('Database images complete.')
	print("Saving complete.")