import tensorflow as tf
import numpy as np
import math

def batch_norm(x, phase_train, scope):
	n_out = x.get_shape().as_list()[-1]
	with tf.variable_scope(scope):
		beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
									 name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
									  name='gamma', trainable=True)

		batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments_2d')
		ema = tf.train.ExponentialMovingAverage(decay=0.5)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(phase_train, mean_var_with_update,
							lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return normed


def leaky_relu(x, leak=0.01, name='leaky_relu'):
		return tf.maximum(x, x * leak, name=name)


class build_network():
	def __init__(self, config_dict):
		num_classes = config_dict["num_classes"]
		learning_rate = config_dict["learning_rate"]

		self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
		self.labels = tf.placeholder(tf.int64, [None], name='y')

		self.dropout = tf.placeholder(tf.float32, name='dropout')
		self.is_train = tf.placeholder(tf.bool, name='is_train')
		self.version = tf.constant('v1.3.0', name='version')

		with tf.name_scope("cal_loss") as scope:
				self.logits = self.network(self.x, self.dropout, self.is_train,  num_classes)

				opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
				#self.grads = opt.compute_gradients(self.loss, tf.global_variables())
				#self.optimizer = opt.apply_gradients(self.grads)

				self.loss = tf.losses.softmax_cross_entropy(onehot_labels = tf.one_hot(self.labels, num_classes), logits=self.logits)
				self.correct_pred = tf.equal(tf.argmax(self.logits, 1), self.labels)
				self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

				self.optimizer = opt.minimize(self.loss)
		
	def conv_block(self, net, filters,  dropout, is_train, i):
				std = 0.001
				conv1 = tf.layers.conv2d(inputs=net, filters=filters[i], 
											kernel_size=(3,3), padding="same", strides=1, 
											activation=leaky_relu, kernel_initializer= tf.truncated_normal_initializer(stddev=std, dtype=tf.float32), 
											name="conv2d_a_%d" % i)

				conv2 = tf.layers.conv2d(inputs=net, filters=filters[i], 
											kernel_size=(3,3), padding="same", strides=1, 
											activation=tf.nn.relu, kernel_initializer= tf.truncated_normal_initializer(stddev=std, dtype=tf.float32),
											name="conv2d_b_%d" % i)

				conv3 = tf.layers.conv2d(inputs=net, filters=filters[i], 
											kernel_size=(1,1), padding="same", strides=1, 
											activation=tf.nn.relu, kernel_initializer= tf.truncated_normal_initializer(stddev=std, dtype=tf.float32),
											name="conv2d_c_%d" % i)

				conv = tf.concat(axis=3, values=[conv1, conv2, conv3])
				#conv = conv1 + conv2 + conv3

				#conv = tf.layers.conv2d(inputs=conv, filters=filters[i], 
				#							kernel_size=kernel_size[i], padding="same", dilation_rate=4, 
				#							activation=leaky_relu, kernel_initializer= tf.truncated_normal_initializer(stddev=std, dtype=tf.float32))
			
				conv = batch_norm(conv, is_train, scope='bn_2d_%d' % i)
				conv = tf.layers.dropout(inputs=conv, rate= dropout)

				#conv += tf.layers.dense(net, 3*filters[i])
				#conv += tf.layers.conv2d(net, filters=3*filters[i], kernel_size=(1,1), padding="same")
				if i % 3 == 1:
					conv += net

				if i % 4 == 1 and i <=12:
					#net = tf.layers.max_pooling2d(inputs=conv, pool_size=(2,2), strides=(2,2))
					conv = tf.layers.max_pooling2d(inputs=conv, pool_size=(2,2), strides=(2,2))

				return conv

	def network(self, x, dropout, is_train, num_classes):
		filters = [32, 32] + [64] * 32
		#filters = [32] * 12
		block_num = 10

		#net = x
		net = tf.map_fn(lambda im: tf.image.random_flip_left_right(im), x)
		#net = tf.contrib.image.rotate(x, 15 * math.pi / 180)

		for i in range(block_num):
			net = self.conv_block(net, filters, dropout, is_train, i)

		net = tf.reshape(net, [-1, filters[i]*4*4*3])
		#net = tf.layers.dense(net, 32)
		logits = tf.layers.dropout(inputs=net, rate= dropout)
		logits = tf.layers.dense(logits, num_classes)

		return logits
