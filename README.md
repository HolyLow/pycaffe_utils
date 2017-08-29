
# Usage:

    export CAFFE_ROOT=$your caffe root$

    python decode.py bvlc_alexnet_deploy.prototxt AlexNet_compressed.net $CAFFE_ROOT/alexnet.caffemodel 

    cd $CAFFE_ROOT

    ./build/tools/caffe test --model=models/bvlc_alexnet/train_val.prototxt --weights=alexnet.caffemodel --iterations=1000 --gpu 0


# Test Result:
	I1022 20:18:58.336736 13182 caffe.cpp:198] accuracy_top1 = 0.57074
	I1022 20:18:58.336745 13182 caffe.cpp:198] accuracy_top5 = 0.80254

Reference

https://prateekvjoshi.com/2016/04/26/how-to-extract-feature-vectors-from-deep-neural-networks-in-python-caffe/
http://caffe.berkeleyvision.org/gathered/examples/feature_extraction.html
https://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
https://github.com/BVLC/caffe/issues/1146
https://groups.google.com/forum/#!topic/caffe-users/vn-TJUtif4Y
https://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html
