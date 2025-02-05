from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import cv2
import scipy.io
import tensorflow as tf
import tf_slim as slim
import scipy.misc as sic
import utils.network as network
import subprocess
import numpy as np

def lrelu(x):
    return tf.maximum(x*0.2,x)

def bilinear_up_and_concat(x1, x2, output_channels, in_channels, scope):
    with tf.compat.v1.variable_scope(scope):
        upconv = tf.compat.v1.image.resize_images(x1, [tf.shape(x1)[1]*2, tf.shape(x1)[2]*2])
        upconv.set_shape([None, None, None, in_channels])
        upconv = slim.conv2d(upconv,output_channels,[3,3], rate=1, activation_fn=None, weights_initializer=tf.initializers.GlorotUniform(),scope='up_conv1')
        upconv_output =  tf.concat([upconv, x2], axis=3)
        upconv_output.set_shape([None, None, None, output_channels*2])
    return upconv_output

def DialUNet(input, channel=32, output_channel=5,reuse=False,ext=""):
    if reuse:
        tf.compat.v1.get_variable_scope().reuse_variables()
    conv1=slim.conv2d(input,channel,[1,1], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_1')
    conv1=slim.conv2d(conv1,channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='same' )
    conv2=slim.conv2d(pool1,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_1')
    conv2=slim.conv2d(conv2,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='same' )
    conv3=slim.conv2d(pool2,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_1')
    conv3=slim.conv2d(conv3,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='same' )
    conv4=slim.conv2d(pool3,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_1')
    conv4=slim.conv2d(conv4,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='same' )
    conv5=slim.conv2d(pool4,channel*16,[3,3], rate=2, activation_fn=lrelu, scope=ext+'g_conv5_1')
    conv5=slim.conv2d(conv5,channel*16,[3,3], rate=4, activation_fn=lrelu, scope=ext+'g_conv5_2')

    up6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"g_up_1" )
    conv6=slim.conv2d(up6,  channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_1')
    conv6=slim.conv2d(conv6,channel*8,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv6_2')
    up7 =  bilinear_up_and_concat( conv6, conv3, channel*4, channel*8, scope=ext+"g_up_2" )
    conv7=slim.conv2d(up7,  channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv7_1')
    conv7=slim.conv2d(conv7,channel*4,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv7_2')
    up8 =  bilinear_up_and_concat( conv7, conv2, channel*2, channel*4, scope=ext+"g_up_3" )
    conv8=slim.conv2d(up8,  channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv8_1')
    conv8=slim.conv2d(conv8,channel*2,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv8_2')
    up9 =  bilinear_up_and_concat( conv8, conv1, channel, channel*2, scope=ext+"g_up_4" )
    conv9=slim.conv2d(up9,  channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv9_1')
    conv9=slim.conv2d(conv9, output_channel,[3,3], rate=1, activation_fn=None,  scope=ext+'g_conv9_2')
    return conv9
