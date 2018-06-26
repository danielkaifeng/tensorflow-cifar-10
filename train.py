
from sys import argv
from model import *
import cPickle
import random

from tensorflow.python.framework import graph_util
import json


batch_size = 256
dropout = 0.3
num_classes = 10
config_dict = {
		"num_classes": num_classes,
		"learning_rate": 0.0001
}

with tf.device("/gpu:1"):
	model = build_network(config_dict)
	logits = tf.multiply(model.logits, 1, name='logit')



def unpickle(filename):
	with open(filename, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict


def shuffle_batch(trX,trY,batch_size):
		rg_x = range(trY.shape[0])
		random.shuffle(rg_x)
		x_collection = [np.array(trX[rg_x[x:x+batch_size]]) for x in range(0,len(rg_x),batch_size)] 
		y_collection = [np.array(trY[rg_x[x:x+batch_size]]) for x in range(0,len(rg_x),batch_size)]
		
		return x_collection,y_collection

def get_batch_data(filename):
	res = unpickle(filename)

	im = res['data']
	u = np.mean(im)
	var = np.var(im)
	#im = im/255.

	im = np.reshape(im, (10000, 3, 32, 32))
	im = np.transpose(im, (0,2,3,1))


	return im, np.array(res['labels'])

trX = []
trY = []
for i in range(5):
	tmpX, tmpY = get_batch_data("data_batch_%d" % (i+1))
	trX += list(tmpX)
	trY += list(tmpY)
trY = np.array(trY)
trX = np.array(trX)

test_x, test_y = get_batch_data("test_batch")
#print trX.shape
#print trY.shape


epochs = 90000

saver=tf.train.Saver()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4

read_log = argv[1]

prefix = 'cifar-10'



init_global = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()
with tf.Session(config=config) as sess:
	tf_writer = tf.summary.FileWriter('graph_log', sess.graph)
	sess.run(init_global)
	sess.run(init_local)


	if read_log == "log":			
		with open("log/checkpoint",'r') as f1:
				txt = f1.readline()
				point = txt.strip().replace('model_checkpoint_path: ','').replace("\"",'')
				print point
				saver.restore(sess,"log/%s"%point)

	print 'training start...'

	for step in range(epochs):
		x_collection, y_collection  = shuffle_batch(trX, trY, batch_size)

		for x,y in zip(x_collection, y_collection)[:-1]:
			sess.run(model.optimizer, feed_dict={model.x: x, model.labels: y, model.dropout: dropout, model.is_train: True})
	
		#if step % 5 == 0:
		feed = {model.x: x, model.labels: y, model.dropout: dropout, model.is_train: True}
		loss, acc = sess.run([model.loss, model.acc], feed_dict=feed)

		feed = {model.x: test_x[:1000], model.labels: test_y[:1000], model.dropout: 0, model.is_train: False}
		val_acc = sess.run(model.acc, feed_dict=feed)
		print "Epoch %d/%d - loss: %f - acc: %f\tval_acc: %f" % (step+1, epochs, loss, acc, val_acc)
		
		if step % 100 == 20:	
			checkpoint_filepath='log/step-%d.ckpt' % step
			saver.save(sess,checkpoint_filepath)
			print 'checkpoint saved!'


		if step % 100 == 99 and False: 
			output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, 
											output_node_names=["x", "y", 'dropout', 'logit','is_train','version'])
			with tf.gfile.FastGFile('./load_pb/%s_%d_TP%f_FP%f.pb' %(prefix,step,TP_mean_acc,FP_mean_acc), mode='wb') as f:
				f.write(output_graph_def.SerializeToString())

			print '\nmodel protobuf saved!\n'
			



