import sys
caffe_root = '/home/epaii/caffe-deeplab/python'
sys.path.insert(0, caffe_root)
import os
import caffe
from caffe import layers as L, params as P

def conv_bn_act(inputs, nout, scope, kernel_size=3, stride=1, pad=1, activation='relu', fix_bn=False):

    out = L.Convolution(inputs, kernel_size=kernel_size, stride=stride, pad=pad, 
                        num_output=nout, weight_filler=dict(type='msra'), bias_term=False,
                        param=dict(lr_mult=1.0, decay_mult=1.0), name=scope)
    out = L.BatchNorm(out, name=scope+'/bn', use_global_stats=fix_bn, in_place=True,
                      moving_average_fraction=0.997,
                      param=[dict(lr_mult=0, decay_mult=0),
                      dict(lr_mult=0, decay_mult=0),
                      dict(lr_mult=0, decay_mult=0)])
    out = L.Scale(out, name=scope+'/sc', bias_term=True, in_place=True,
                  param=[dict(lr_mult=1.0, decay_mult=0),
                  dict(lr_mult=1.0, decay_mult=0)],
                  filler=dict(value=1), bias_filler=dict(value=0))
    if activation == 'relu':
        out = L.ReLu(out, name=scope+'/relu', in_place=True)
    return out


def densenet_seg(batch_size, lmdb):

    n = caffe.NetSpec()
    
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    net = conv_bn_act(n.data, 32, 'conv1_1')
    net = conv_bn_act(net, 32, 'cnv1_2', 3, 2)
    n.conv2 = conv_bn_act(net, 32, 'conv2_1')
    with open('try.prototxt', 'w') as f:
        f.write(str(n.to_proto()))

densenet_seg(64, '/mnt/data/lmdb')