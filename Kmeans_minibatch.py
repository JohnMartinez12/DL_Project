# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import functools
#import matplotlib.pyplot as plt
#%matplotlib inline

def scope(function):
    name = function.__name__
    attribute = '_cache_' + name
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self,attribute):
            with tf.variable_scope(name):
                setattr(self,attribute,function(self))
        return getattr(self,attribute)
    return decorator

class KMeans:

	def __init__(self, nb_clusters, nb_tries=10, nb_iterations=10):

		self.nb_clusters = nb_clusters
		self.nb_iterations = nb_iterations
		self.nb_tries = nb_tries

	
		self.graph = tf.get_default_graph()

		with self.graph.as_default():
			# Batch of L vectors with E features
			# shape = [batch, L , E ]
			self.X_in = tf.placeholder("float", [None, None, None])
			self.b = tf.shape(self.X_in)[0]
			
			# Tile the number of tries to execute  along Batch dimension (for better accuracy)
			self.X = tf.tile(self.X_in, [self.nb_tries, 1, 1])

			self.B = tf.shape(self.X)[0]
			self.L = tf.shape(self.X)[1]
			self.E = tf.shape(self.X)[2]

			# Take randomly 'nb_clusters' vectors from X
			batch_range = tf.tile(tf.reshape(tf.range(self.B, dtype=tf.int32), shape=[self.B, 1, 1]), [1, self.nb_clusters, 1])
			random = tf.random_uniform([self.B, self.nb_clusters, 1], minval = 0, maxval = self.L - 1, dtype = tf.int32)
			indices = tf.concat([batch_range, random], axis = 2)
			self.centroids = tf.gather_nd(self.X, indices)

			self.reshaped_X = tf.reshape(self.X, [self.B*self.L, self.E])
			self.network

			# Create a session for this model based on the constructed graph
			self.sess = tf.Session(graph = self.graph)


	def init(self):
		with self.graph.as_default():
			self.sess.run(tf.global_variables_initializer())


	@scope
	def network(self):

		i = tf.constant(0)
		cond = lambda i, m: tf.less(i, self.nb_iterations)
		_ , self.centroids = tf.while_loop(cond, self.body,[i, self.centroids], shape_invariants=[i.get_shape(), tf.TensorShape([None, None, None])])

		# Compute the inertia of each try and take the best for each batch
		centroids = tf.expand_dims(self.centroids, 1)
		X = tf.expand_dims(self.X, 2)
		inertia = tf.reduce_sum(tf.norm(X - centroids, axis=3), axis=[1,2])
		inertia = tf.reshape(inertia, [self.b, self.nb_tries])
		bests = tf.argmin(inertia, 1, output_type=tf.int32)
		index = bests + tf.range(self.b)*self.nb_tries
		self.centroids = tf.reshape(self.centroids, [self.b*self.nb_tries, self.nb_clusters, self.E])
		self.centroids = tf.gather(self.centroids, index)

		return self.centroids, self.get_labels(self.centroids, self.X_in)


	def body(self ,i, centroids):
		with tf.name_scope('iteration'):
				# Checking the closest clusters
				# [B, L]
				labels = self.get_labels(centroids, self.X)

				elems_tot = (self.X, labels)
				elems_count = (tf.ones_like(self.X), labels)

				total = tf.map_fn(lambda x: tf.unsorted_segment_sum(x[0], x[1], self.nb_clusters), elems_tot, dtype=tf.float32)
				count = tf.map_fn(lambda x: tf.unsorted_segment_sum(x[0], x[1], self.nb_clusters), elems_count, dtype=tf.float32)				
				
				new_centroids = total/count	
				return [i+1, new_centroids]


	def get_labels(self, centroids, X):
		centroids_ = tf.expand_dims(centroids, 1)
		X_ = tf.expand_dims(X, 2)
		return tf.argmin(tf.norm(X_ - centroids_, axis=3), axis=2, output_type=tf.int32)

	def fit(self, X_train):
		return self.sess.run(self.network, {self.X_in: X_train})

if __name__ == "__main__":

	#imgs_query = np.load('imgs_query.npy')
	#imgs_database = np.load('imgs_database.npy')
     
    imgs_query = np.array([i for i in listdir(img_matrix_dir_query)]) 
    imgs_database = np.array([i for i in listdir(img_matrix_dir_database)]) 
    #images = np.vstack([np.load(img_matrix_dir_query+'/'+i) for i in ]) 

	images = np.vstack((imgs_database,imgs_query)) 

	test = np.load('feature_test_np_small.npy')
	query = np.load('feature_query_np_small.npy')
	test = np.vstack((test,query))
	test = np.reshape(test,(1,1250,1000))
	
	
	nb_clusters = len(query)
	
	error = 0
	kmeans = KMeans(nb_clusters, nb_tries=3, nb_iterations=20)

	#X, y = make_blobs(n_samples=nb_samples, centers=nb_clusters, n_features=E, cluster_std=2.0)
	#X_ = X[np.newaxis,:]
	#y = y[np.newaxis,:]

	kmeans.init()
	centroids, labels = kmeans.fit(test)
	

ids_query = listdir('Dataset_Directory/query')
ids_database = listdir('Dataset_Directory/test')
n = len(labels)
labels = labels.reshape(n,) # DB, query
all_ids = ids_database + ids_query

start = len(ids_database)

file = open("submission.csv", "w")
file.write("id,images")

for i in range(start,n):
    id_query = all_ids[i]
    query_label = labels[i]
    
    db_labels = labels[:start-1]
    

    
    indexes = np.where(db_labels == query_label)[0]

    ids_cluster = [ids_database[j][:-4] for j in indexes.astype(int)]
    
    line = '\n'+id_query[:-4] + ',' + ' '.join(ids_cluster)
    
    file.write(line)
    

file.close()
    
    
'''
	#labels = K
	pred = kmeans.predict(query) 
	print(pred)
	

	for i in range(len(pred)):
		plt.figure(figsize=(20,10))
		cluster = np.where(pred[i] == labels)
		c = cluster[0]

		for j in range(len(idx)):
			plt.subplot(2,len(idx),len(c)+j)
			plt.imshow(imgs_database[images[j]])
			#title = 'test image {}'.format(c[j]),',for query imag {}'.format(i)
			plt.title(title)
    
'''






		#plt.figure(figsize=(20,10))
	    
      	
    	


        	
        	
        	
        	
        	
        	
        	
        	
