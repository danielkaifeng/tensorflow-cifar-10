from sys import argv
import numpy as np

import cPickle

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict

res = unpickle(argv[1])
print res['data'].shape

