# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf
from PIL import Image
import scipy.misc
from scipy.misc import imsave
import cPickle as pickle

def save_images(samples, save_path):
    # [0, 1] -> [0,255]
    samples = ((samples+1.)*(255./2)).astype('int32')
    X=samples
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[2]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples/rows
    h, w = X[:,:,0].shape[:]
    img = np.zeros((h*nh, w*nw))

    for n in range(n_samples):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = X[:,:,n]

    imsave(save_path, img)

def trans_to_vector(samples):
	n_samples = samples.shape[2]
	X = samples[:,:,0:n_samples/2]
	h, w = X[:,:,0].shape[:]
	vector = []
	for i in range(h):
		for j in range(w):
			vector.append(np.mean(X[i,j,:]))
	#print("befor:")
	#print(vector)
	# for i in range(len(vector)):
	# 	if vector[i] > 0.5:
	# 		vector[i]=1
	# 	else:
	# 		vector[i]=0
	index = vector.index(max(vector))
	vector = np.array(vector)
	if index == 0:
		vec2 = vector
	else:
		vec2 = np.concatenate((vector[index:], vector[0:index]), axis=0)
	#print("after:")
	#print(vec2)

	return vec2



def change_shape(data, n_classes, index):
	Y = []
	for i in range(len(data)):
		y = np.zeros([n_classes])
		y[index]=1
		Y.append(y)
	Y = np.reshape(Y,[-1, n_classes])
	return Y

def load_data(ngdata_dir, okdata_dir):
	train_NG=[]
	filepath = ngdata_dir
	for img_name in os.listdir(filepath):
		img = Image.open(filepath+img_name)
		image = np.asarray(img)
		image=np.ndarray.flatten(image)
		train_NG.append(image)
	train_NG = np.reshape(train_NG, [len(train_NG), 64*64])
	train_NG = (train_NG.astype(np.float32) - 127.5)/127.5
	labels_NG = change_shape(train_NG, 2, 1)

	train_OK=[]
	filepath = okdata_dir
	for img_name in os.listdir(filepath):
		img = Image.open(filepath+img_name)
		image = np.asarray(img)
		image=np.ndarray.flatten(image)
		train_OK.append(image)
	train_OK = np.reshape(train_OK, [len(train_OK), 64*64])
	train_OK = (train_OK.astype(np.float32) - 127.5)/127.5
	labels_OK = change_shape(train_OK, 2, 0)
	
	train_X = []
	train_Y = []
	number = 4000
	for i in range(number):
		train_X.append(train_OK[i])
		train_X.append(train_NG[i])
		train_Y.append(labels_OK[i])
		train_Y.append(labels_NG[i])
	train_X = np.reshape(train_X, [-1, 64*64])
	train_Y = np.reshape(train_Y, [-1,2])

	return train_X, train_Y

def load_ng_data(ngdata_dir):
	train_NG=[]
	filepath = ngdata_dir
	filenames = []
	#with open('ng_filename.txt','w') as f:
	for img_name in os.listdir(filepath):
		#f.write(img_name)
		#f.write('\r\n')
		filenames.append(img_name)
		img = Image.open(filepath+img_name)
		image = np.asarray(img)
		image=np.ndarray.flatten(image)
		train_NG.append(image)
	train_NG = np.reshape(train_NG, [len(train_NG), 64*64])
	train_NG = (train_NG.astype(np.float32) - 127.5)/127.5

	return train_NG, filenames

# Conv2D Layer
def conv2d(name, inputs, in_features, out_features, kernel_size, strides=[1,2,2,1], padding='SAME', with_biases = False):
	with tf.variable_scope(name):
		filters_values = tf.get_variable('weights', [kernel_size, kernel_size, in_features, out_features], 
			dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
		conv = tf.nn.conv2d(inputs, filters_values, strides = strides, padding = padding)
		if with_biases:
			biases = tf.get_variable('biases', [out_features], dtype=tf.float32, initializer=tf.constant_initializer(0.01))
			return tf.nn.bias_add(conv, biases)
		else:
			return conv

# Fully_Connected Layer
def fullyconnected(name, inputs, in_features, out_features, with_biases=True):
	with tf.variable_scope(name):
		matrix = tf.get_variable('Matrix', [in_features, out_features], dtype=tf.float32,
			initializer=tf.random_normal_initializer(stddev=0.02))
		if with_biases:
			biases = tf.get_variable('bias', [out_features], dtype=tf.float32, initializer=tf.constant_initializer(0.01))
			return tf.matmul(inputs, matrix)+ biases
		else:
			return tf.matmul(inputs, matrix)

# Bottle_Neck Block
def residual_bottleneck(name, inputs, in_filter, out_filter, is_training=True, strides=[1,1,1,1], padding='SAME', acitvate_before_residual=False):
	orig_input = inputs
	inputs = tf.contrib.layers.batch_norm(inputs, scale=True, is_training=is_training, updates_collections=None)
	inputs = tf.nn.relu(inputs)
	
	if acitvate_before_residual: # shared_activation
		orig_input = inputs

	with tf.variable_scope(name):
		# step 1
		conv = conv2d(name+'_conv1', inputs, in_filter, out_filter/4, 1, strides=strides, padding=padding)
		conv = tf.contrib.layers.batch_norm(conv, scale=True, is_training=is_training, updates_collections=None)
		conv = tf.nn.relu(conv)
		# step 2
		conv = conv2d(name+'_conv2', conv, out_filter/4, out_filter/4, 3, strides=[1,1,1,1])
		conv = tf.contrib.layers.batch_norm(conv, scale=True, is_training=is_training, updates_collections=None)
		conv = tf.nn.relu(conv)
		# step 3
		conv = conv2d(name+'_conv3', conv, out_filter/4, out_filter, 1, strides=[1,1,1,1])

		if in_filter != out_filter:
			orig_input = conv2d(name+'_project', orig_input, in_filter, out_filter, 1, strides=strides, padding=padding)
		
		conv += orig_input

		return conv

# Block
def residual_block(name, inputs, in_filter, out_filter, is_training=True, strides=[1,1,1,1], padding='SAME', acitvate_before_residual=False):
	orig_input = inputs
	inputs = tf.contrib.layers.batch_norm(inputs, scale=True, is_training=is_training, updates_collections=None)
	inputs = tf.nn.relu(inputs)
	
	if acitvate_before_residual: # shared_activation
		orig_input = inputs

	with tf.variable_scope(name):
		# step 1
		conv = conv2d(name+'_conv1', inputs, in_filter, out_filter, 3, strides=strides, padding=padding)
		conv = tf.contrib.layers.batch_norm(conv, scale=True, is_training=is_training, updates_collections=None)
		conv = tf.nn.relu(conv)
		# step 2
		conv = conv2d(name+'_conv2', conv, out_filter, out_filter, 3, strides=[1,1,1,1])

		if in_filter != out_filter:
			orig_input = tf.nn.avg_pool(orig_input, strides, strides,'VALID')
			orig_input = tf.pad(orig_input, [[0, 0], [0, 0], [0, 0], [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
		
		conv += orig_input

		return conv


def batch_active_conv(name, current, in_features, out_features, kernel_size, is_training, keep_prob):
	with tf.variable_scope(name):
		current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
		current = tf.nn.relu(current)
		current = conv2d(name+'conv', current, in_features, out_features, kernel_size, strides=[1,1,1,1])
		current = tf.nn.dropout(current, keep_prob)
		return current

def ResNet(inputs, image_size, channel, is_training):
	current = tf.reshape(inputs, [-1, image_size, image_size, channel])
	
	current = conv2d('pre_conv', current, 1, 16, 3, strides=[1,2,2,1])

	current = residual_bottleneck('neckbottle1', current, 16, 16, strides=[1,1,1,1], is_training=is_training, acitvate_before_residual=True)
	current = residual_bottleneck('neckbottle2.1', current, 16, 128, strides=[1,1,1,1], is_training=is_training)
	current = tf.nn.avg_pool(current,[1,2,2,1],[1,2,2,1], 'SAME')
	current = residual_bottleneck('neckbottle2.2', current, 128, 128, strides = [1,1,1,1], is_training=is_training)
	current = residual_bottleneck('neckbottle3.1', current, 128, 256, strides = [1,1,1,1], is_training=is_training)
	current = tf.nn.avg_pool(current,[1,2,2,1],[1,2,2,1], 'SAME')
	current = residual_bottleneck('neckbottle3.2', current, 256, 256, strides = [1,1,1,1], is_training=is_training)

	current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
	feature1 = tf.nn.relu(current)
	feature2 = tf.nn.avg_pool(feature1, [1,8,8,1], [1,8,8,1], 'VALID')
	feature3 = tf.reshape(feature2, [-1, 256])

	final_result = fullyconnected('fc1', feature3, 256, 2)

	return final_result

if __name__=='__main__':
	ngdata_dir = './data/ngdata64/'
	okdata_dir = './data/okdata64/'
	LODE_MODEL = False
	batch_size = 64
	learning_rate = 0.1


	xs = tf.placeholder('float32', shape=[None, 64*64])
	ys = tf.placeholder('float32', shape=[None, 2])
	lr = tf.placeholder('float32', shape=[]) # learning rate
	keep_prob = tf.placeholder('float32')
	is_training = tf.placeholder('bool',shape=[])
	logits = ResNet(xs, 64, 1, is_training)
	prediction = tf.nn.softmax(logits)

	# loss
	# loss
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = ys))
	l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
	train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(cross_entropy )
	#train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

	saver = tf.train.Saver()
	os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.5
	session = tf.InteractiveSession(config=config)

	session.run(tf.global_variables_initializer())


	train_data, train_labels = load_data(ngdata_dir, okdata_dir)
	batch_count = len(train_data) / batch_size
	batches_data = np.split(train_data[:batch_size*batch_count], batch_count)
	batches_labels = np.split(train_labels[:batch_size*batch_count], batch_count)

	if LODE_MODEL:
		print("[*] Reading checkpoints...")
		ckpt = tf.train.get_checkpoint_state('./model/')
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			saver.restore(session, os.path.join('./model/', ckpt_name))
		train_data, filenames = load_ng_data('./data/centerng/')
		batch_count = len(train_data) / batch_size
		batches_data = np.split(train_data[:batch_size*batch_count], batch_count)
		fulldata = []
		for batch_idx in xrange(batch_count):
				_xs= batches_data[batch_idx]
				output_feature = session.run(feature3, feed_dict={xs:_xs, is_training: False, keep_prob: 1.0})
				#output_feature2 = tf.nn.max_pool(output_feature, [1,2,2,1], [1,2,2,1], 'SAME')
				for idx in range(batch_size):
					#save_images(output_feature[idx], './features3/'+filenames[batch_idx*batch_size+idx][:-4]+'.png')
					#vec2 = trans_to_vector(output_feature[idx])
					fulldata.append(output_feature[idx])

		write_file = open('./data/screen_feature3.pkl','wb')
		pickle.dump(fulldata, write_file)
		pickle.dump(filenames, write_file)
		write_file.close()


	else:
		epoches=300
		for epoch in xrange(1, 1+epoches):
			if epoch == 100: learning_rate=0.01
			if epoch == 200: learning_rate=0.001
			for batch_idx in xrange(batch_count):
				xs_, ys_ = batches_data[batch_idx], batches_labels[batch_idx]
				batch_res = session.run([ train_step, cross_entropy, accuracy ], 
					feed_dict = { xs: xs_, ys: ys_, lr: learning_rate, is_training: True})
				if batch_idx % 100 == epoch: print epoch, batch_idx, batch_res[1:]

			checkpoint_path = os.path.join('./resnet_model/','resnet.ckpt')
			saver.save(session, checkpoint_path, global_step = epoch)
			#print("***********    model saved   **********")
				# test_results = run_in_batch_avg(session, [ cross_entropy, accuracy ], [ xs, ys ], 
				# 	feed_dict = { xs: data['test_data'], ys: data['test_labels'], is_training: False, keep_prob: 1. })
				# print epoch, batch_res[1:], test_results











