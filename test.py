import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/curr/xuechao/prog/caffe/python')

import caffe

input_image_file = sys.argv[1]

output_file = sys.argv[2]

model_file = 'alexnet.caffemodel'

deploy_prototxt = '/curr/xuechao/prog/caffe/models/bvlc_alexnet/deploy.prototxt'

net = caffe.Net(deploy_prototxt, model_file, caffe.TEST)

#layer = 'fc7'
layer = 'conv1'
if layer not in net.blobs:
	raise TypeError("Invalid layer name: " + layer)

imagemean_file = '/curr/xuechao/prog/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(imagemean_file).mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255.0)

#net.blobs['data'].reshape(1,3,227,227)

img = caffe.io.load_image(input_image_file)
net.blobs['data'].data[...] = transformer.preprocess('data', img)

#output = net.forward()

for layer_name, param in net.params.iteritems():
	print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

#with open(output_file, 'w') as f:
#	np.savetxt(f, net.blobs[layer].data[0].reshape(1, -1), fmt='%.4f', delimiter='\n')

#print net.params[layer][0].data.shape
#print net.params[layer][1].data.shape

data2 = net.params[layer][0].data
data3 = net.params[layer][1].data

print data2.shape
print data3.shape

print data2

data = net.blobs[layer].data
print data[0].shape
print data[0][20,5,5]

