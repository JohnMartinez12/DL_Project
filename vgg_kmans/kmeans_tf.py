from os import listdir
import os
import numpy as np
import tensorflow as tf
from time import time
tf.logging.set_verbosity(tf.logging.ERROR)

print("Loading data...")
feature_dir_query = 'feature_query_np'
feature_dir_database = 'feature_test_np'

imgs_query = [i for i in listdir(feature_dir_query)]
imgs_database = [i for i in listdir(feature_dir_database)]

k = len(imgs_query)
query = np.vstack([np.load(feature_dir_query+'/'+i) for i in imgs_query])
feature_size = query.shape[-1]
query = query.reshape(k, feature_size)
database = np.vstack([np.load(feature_dir_database+'/'+i) for i in imgs_database])
n = len(imgs_database)
database = database.reshape(n, feature_size)


# Clustering
centroids = tf.placeholder(tf.float32, [None, feature_size])
points = tf.placeholder(tf.float32, [None, feature_size])

ones_like = tf.ones((tf.shape(points)[0], 1))
p1 = tf.matmul(
	tf.expand_dims(tf.reduce_sum(tf.square(points), 1), 1),
	tf.ones(shape=(1, k))
)
p2 = tf.transpose(tf.matmul(
	tf.reshape(tf.reduce_sum(tf.square(centroids), 1), shape=[-1, 1]),
	ones_like,
	transpose_b=True
))

distance = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(points, centroids, transpose_b=True))

# assign each point to a closest centroid
point_to_centroid_assignment = tf.placeholder(tf.int64, (n,))


# recalculate the centers
total = tf.unsorted_segment_sum(points, point_to_centroid_assignment, k)
count = tf.unsorted_segment_sum(ones_like, point_to_centroid_assignment, k)
means = total / count

def print_time_since(t0):
	print("Time: %.2f" % ((time()-t0)/60),"min")
	

# Preparing list of lists for csv 
clusters = [[(-1,i[:-4])] for i in imgs_query]

# Beginning clustering 
with tf.Session() as sess:
	iters = 1
	step = max(iters // 20,1)
	batch_size = n // 203
	start = 0
	stop = batch_size
	
	new_means = query
	
	for i in range(iters):
		t0 = time()
		start = 0
		stop = batch_size
		
		pred = np.array([],dtype=np.int64)
		
		# Assemling a predictions array, with batches of the database 
		while start < n:
			batch = database[start:stop, :]
			batch_idxes = list(range(start,min(stop,n)))
			feed_dict = {
				centroids:new_means,
				points:batch
			}
			# Predicting batch 	
			dist = sess.run(distance, feed_dict =feed_dict)
			pred_batch = np.argmin(dist, axis = 1)
			pred = np.concatenate([pred,pred_batch]) 
			
			# Collecting cluster data 
			if i == iters-1:
				for idx, batch_idx in enumerate(batch_idxes):
					cluster_id = pred[batch_idx] # The cluster this image was assigned to
					img_id = imgs_database[batch_idx][:-4] # The image id
				
					curr_dist = dist[idx, cluster_id]
					pair = curr_dist, img_id
					clusters[cluster_id].append(pair)
			
			start += batch_size
			stop += batch_size
		
		# Getting the new centroids from the pred
		feed_dict = {
			points:database,
			point_to_centroid_assignment:pred
		}
		new_means = sess.run(means, feed_dict=feed_dict)
		# Showing progress
		if i % step == 0:
			print("Progress: %.2f%%" % (100*i/iters), end = ' ')
			print_time_since(t0)			

print("Done. Writing csv...")
t0 = time()
for i in range(k):
	# Sorting, then keeping only image ids
	clusters[i].sort(key=lambda x: x[0])
	clusters[i] = [j[1] for j in clusters[i]]

file = open("submission.csv","w")
file.write("id,images")
clusters = [c for c in clusters if c != None]

for cluster in clusters:
	query_img = cluster[0]
	test_imgs = cluster[1:]
	line = '\n' + query_img+',' + ' '.join(test_imgs)
	file.write(line)    

file.close()
print("Done. Time: %.2f" % ((time()-t0)/60),"min")
