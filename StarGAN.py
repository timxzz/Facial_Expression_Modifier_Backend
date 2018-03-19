import tensorflow as tf
import numpy as np

from op_StarGAN import Operator
from nn_module import conv, deconv, instance_norm, lrelu
from util import get_shape_c

class StarGAN(Operator):
    def __init__(self, sess, project_name):
        Operator.__init__(self, sess, project_name)


    def generator(self, x, c, reuse=None):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()

            f = 64
            image_size = self.data_size
            c_num = self.data_label_vector_size
            p = "SAME"

            x = tf.concat([x, tf.tile(tf.reshape(c, [-1, 1, 1, get_shape_c(c)[-1]]),\
                                      [1, x.get_shape().as_list()[1], x.get_shape().as_list()[2], 1])],\
                          axis=3)
            
            # Down-sampling
            x = conv(x, [7, 7, 3+c_num, f], stride=1, padding=p, name='ds_1', reuse=reuse)
            x = instance_norm(x, 'in_ds_1')
            x = tf.nn.relu(x)
            x = conv(x, [4, 4, f, f*2], stride=2, padding=p, name='ds_2', reuse=reuse)
            x = instance_norm(x, 'in_ds_2')
            x = tf.nn.relu(x)
            x = conv(x, [4, 4, f*2, f*4], stride=2, padding=p, name='ds_3', reuse=reuse)
            x = instance_norm(x, 'in_ds_3')
            x = tf.nn.relu(x)
            
            # Bottleneck
            x_r = conv(x, [3, 3, f*4, f*4], stride=1, padding=p, name='bneck_1a', reuse=reuse)
            x_r = instance_norm(x_r, 'in_bneck_1a')
            x_r = tf.nn.relu(x_r)
            x_r = conv(x_r, [3, 3, f*4, f*4], stride=1, padding=p, name='bneck_1b', reuse=reuse)
            x_r = instance_norm(x_r, 'in_bneck_1b')
            x = x + x_r
            x = tf.nn.relu(x)
            
            x_r = conv(x, [3, 3, f*4, f*4], stride=1, padding=p, name='bneck_2a', reuse=reuse)
            x_r = instance_norm(x_r, 'in_bneck_2a')
            x_r = tf.nn.relu(x_r)
            x_r = conv(x_r, [3, 3, f*4, f*4], stride=1, padding=p, name='bneck_2b', reuse=reuse)
            x_r = instance_norm(x_r, 'in_bneck_2b')
            x = x + x_r
            x = tf.nn.relu(x)
            
            x_r = conv(x, [3, 3, f*4, f*4], stride=1, padding=p, name='bneck_3a', reuse=reuse)
            x_r = instance_norm(x_r, 'in_bneck_3a')
            x_r = tf.nn.relu(x_r)
            x_r = conv(x_r, [3, 3, f*4, f*4], stride=1, padding=p, name='bneck_3b', reuse=reuse)
            x_r = instance_norm(x_r, 'in_bneck_3b')
            x = x + x_r
            x = tf.nn.relu(x)
            
            x_r = conv(x, [3, 3, f*4, f*4], stride=1, padding=p, name='bneck_4a', reuse=reuse)
            x_r = instance_norm(x_r, 'in_bneck_4a')
            x_r = tf.nn.relu(x_r)
            x_r = conv(x_r, [3, 3, f*4, f*4], stride=1, padding=p, name='bneck_4b', reuse=reuse)
            x_r = instance_norm(x_r, 'in_bneck_4b')
            x = x + x_r
            x = tf.nn.relu(x)
            
            x_r = conv(x, [3, 3, f*4, f*4], stride=1, padding=p, name='bneck_5a', reuse=reuse)
            x_r = instance_norm(x_r, 'in_bneck_5a')
            x_r = tf.nn.relu(x_r)
            x_r = conv(x_r, [3, 3, f*4, f*4], stride=1, padding=p, name='bneck_5b', reuse=reuse)
            x_r = instance_norm(x_r, 'in_bneck_5b')
            x = x + x_r
            x = tf.nn.relu(x)
            
            x_r = conv(x, [3, 3, f*4, f*4], stride=1, padding=p, name='bneck_6a', reuse=reuse)
            x_r = instance_norm(x_r, 'in_bneck_6a')
            x_r = tf.nn.relu(x_r)
            x_r = conv(x_r, [3, 3, f*4, f*4], stride=1, padding=p, name='bneck_6b', reuse=reuse)
            x_r = instance_norm(x_r, 'in_bneck_6b')
            x = x + x_r
            x = tf.nn.relu(x)
            
            # Up-sampling
            x = deconv(x, [4, 4, f*4, f*2], stride=2, padding=p, name='us_1', reuse=reuse)
            x = instance_norm(x, 'in_us_1')
            x = tf.nn.relu(x)
            x = deconv(x, [4, 4, f*2, f], stride=2, padding=p, name='us_2', reuse=reuse)
            x = instance_norm(x, 'in_us_2')
            x = tf.nn.relu(x)
            x = conv(x, [7, 7, f, 3], stride=1, padding=p, name='us_3', reuse=reuse)

            x = tf.nn.tanh(x)
            
        return x
    
    def discriminator(self, x, reuse=None):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            f = 64
            f_max = f*8
            image_size = self.data_size
            k_size = int(image_size / np.power(2, 5))            
            c_num = self.data_label_vector_size
            p = "SAME"
            
            x = conv(x, [4, 4, 3, f], stride=2, padding=p, name='conv_1', reuse=reuse)
            x = lrelu(x)
            x = conv(x, [4, 4, f, f*2], stride=2, padding=p, name='conv_2', reuse=reuse)
            x = lrelu(x)
            x = conv(x, [4, 4, f*2, f*4], stride=2, padding=p, name='conv_3', reuse=reuse)
            x = lrelu(x)
            x = conv(x, [4, 4, f*4, f*8], stride=2, padding=p, name='conv_4', reuse=reuse)
            x = lrelu(x)
            x = conv(x, [4, 4, f*8, f*16], stride=2, padding=p, name='conv_5', reuse=reuse)
            x = lrelu(x)
            
            if image_size == 128:
                x = conv(x, [4, 4, f*16, f*32], stride=2, padding=p, name='conv_6', reuse=reuse)
                x = lrelu(x)
                f_max = f_max * 2
                k_size = int(k_size / 2)
                
            out_src = conv(x, [3, 3, f_max, 1], stride=1, padding=p, name='conv_out_src', reuse=reuse)
            out_cls = conv(x, [k_size, k_size, f_max, c_num], stride=1, name='conv_out_cls', reuse=reuse)
        
        return out_src, tf.squeeze(out_cls)
