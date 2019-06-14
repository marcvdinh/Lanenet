#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-18 下午3:59
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : cnn_basenet.py
# @IDE: PyCharm Community Edition
"""
The base convolution neural networks mainly implement some useful cnn functions
"""
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

class CNN_blocks(object):
    """
    Base model for other specific cnn ctpn_models
    """

    def __init__(self):
        pass

    @staticmethod
    def spatial_dropout(input_tensor, keep_prob,  name, training=True,seed=1234):

        def f1():
            input_shape = input_tensor.get_shape().as_list()
            noise_shape = tf.constant(value=[input_shape[0], 1, 1, input_shape[3]])
            return tf.nn.dropout(input_tensor, keep_prob, noise_shape, seed=seed, name="spatial_dropout")

        def f2():
            return input_tensor

        with tf.variable_scope(name):

            if training:
                output = f1()
            else:
                output = f2()

            return output

    @staticmethod
    @slim.add_arg_scope
    def prelu(inputdata, name, decoder=False):

        if decoder:
            return tf.nn.relu(features=inputdata, name=name)

        alpha = tf.get_variable(name + '_alpha', shape=inputdata.get_shape()[-1],
                                initializer=tf.constant_initializer(0.0),
                                dtype = tf.float32)

        pos = tf.nn.relu(inputdata)
        neg = alpha * (inputdata - abs(inputdata)) * 0.5
        return pos + neg

    @staticmethod
    def unpool(updates, mask, k_size=[1, 2, 2, 1], output_shape=None, name=''):

        with tf.variable_scope(name):
            mask = tf.cast(mask, tf.int32)
            input_shape = tf.shape(updates, out_type=tf.int32)
            #  calculation new shape
            if output_shape is None:
                output_shape = (input_shape[0], input_shape[1] * k_size[1], input_shape[2] * k_size[2], input_shape[3])

            # calculation indices for batch, height, width and feature maps
            one_like_mask = tf.ones_like(mask, dtype=tf.int32)
            batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
            batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3]) % output_shape[1]
            x = (mask // output_shape[3]) % output_shape[
                2]  # mask % (output_shape[2] * output_shape[3]) // output_shape[3]
            feature_range = tf.range(output_shape[3], dtype=tf.int32)
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = tf.size(updates)
            indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
            values = tf.reshape(updates, [updates_size])
            ret = tf.scatter_nd(indices, values, output_shape)

        return ret

    @staticmethod
    @slim.add_arg_scope
    def initial_block(self,inputdata, training=True, name='initial_block'):

        # Convolutional branch
        net_conv = tf.layers.conv2d(inputdata, 13, kernel_size=[3,3], strides=2, padding='SAME', activation=None, name=name+'_conv')
        net_conv = tf.contrib.layers.batch_norm(net_conv, is_training=training, fused=True, scope=name+'_batchnorm')
        net_conv = self.prelu(net_conv, name+'_prelu')

        #max pool
        net_pool = tf.layers.max_pooling2d(inputdata, [2,2], strides=2, name=name+'_maxpool')

        #concatenated ouput
        net_concat = tf.concat([net_conv, net_pool], axis=3, name=name+'_concat')
        return net_concat

    @staticmethod
    @slim.add_arg_scope
    def bottleneck(self,inputs,
                   output_depth,
                   filter_size,
                   regularizer_prob,
                   projection_ratio=4,
                   seed=0,
                   training=True,
                   downsampling=False,
                   upsampling=False,
                   pooling_indices=None,
                   output_shape=None,
                   dilated=False,
                   dilation_rate=None,
                   asymmetric=False,
                   decoder=False,
                   name='bottleneck'):

        reduced_depth = int(inputs.get_shape().as_list()[3] / projection_ratio)

        with tf.contrib.framework.arg_scope([self.prelu], decoder=decoder):

            if downsampling:

                net_main, pooling_indices = tf.nn.max_pool_with_argmax(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                                       padding='SAME', name=name + '_main_max_pool')

                # pad
                inputshape = inputs.get_shape().as_list()
                depth2pad = abs(inputshape[3] - output_depth)
                paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, depth2pad]])
                net_main = tf.pad(net_main, paddings=paddings, name=name + '_main_pad')

                # =============SUB BRANCH==============
                # First projection that has a 2x2 kernel and strides 2
                net = tf.layers.conv2d(inputs, reduced_depth, [2, 2], strides=2, padding='SAME', name=name + '_conv1')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm1')
                net = self.prelu(net, name=name + '_prelu1')

                # Second conv block
                net = tf.layers.conv2d(net, reduced_depth, [filter_size, filter_size], padding='SAME',name=name + '_conv2')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm2')
                net = self.prelu(net, name=name + '_prelu2')

                # Final projection with 1x1 kernel
                net = tf.layers.conv2d(net, output_depth, [1, 1], padding='SAME',name=name + '_conv3')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm3')
                net = self.prelu(net, name=name + '_prelu3')

                # Regularizer
                net = self.spatial_dropout(net, keep_prob=1-regularizer_prob, seed=seed, name=name + '_spatial_dropout')

                # Finally, combine the two branches together via an element-wise addition
                net = tf.add(net, net_main, name=name + '_add')
                net = self.prelu(net, name=name + '_last_prelu')

                # also return inputs shape for convenience later
                return net, pooling_indices, inputshape

                # ============DILATION CONVOLUTION BOTTLENECK====================
                # Everything is the same as a regular bottleneck except for the dilation rate argument
            elif dilated:
                # Check if dilation rate is given
                if not dilation_rate:
                    raise ValueError('Dilation rate is not given.')

                # Save the main branch for addition later
                net_main = inputs

                # First projection with 1x1 kernel (dimensionality reduction)
                net = tf.layers.conv2d(inputs, reduced_depth, [1, 1], padding='SAME',name=name + '_conv1')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm1')
                net = self.prelu(net, name=name + '_prelu1')

                # Second conv block --- apply dilated convolution here
                net = tf.layers.conv2d(net, reduced_depth, [filter_size, filter_size], dilation_rate=dilation_rate,padding='SAME',
                                  name=name + '_dilated_conv2')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm2')
                net = self.prelu(net, name=name + '_prelu2')

                # Final projection with 1x1 kernel (Expansion)
                net = tf.layers.conv2d(net, output_depth, [1, 1], padding='SAME',name=name + '_conv3')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm3')
                net = self.prelu(net, name=name + '_prelu3')

                # Regularizer
                net = self.spatial_dropout(net, keep_prob=1-regularizer_prob, seed=seed, name=name + '_spatial_dropout')
                net = self.prelu(net, name=name + '_prelu4')

                # Add the main branch
                net = tf.add(net_main, net, name=name + '_add_dilated')
                net = self.prelu(net, name=name + '_last_prelu')

                return net

                # ===========ASYMMETRIC CONVOLUTION BOTTLENECK==============
                # Everything is the same as a regular bottleneck except for a [5,5] kernel decomposed into two [5,1] then [1,5]
            elif asymmetric:
                # Save the main branch for addition later
                net_main = inputs

                # First projection with 1x1 kernel (dimensionality reduction)
                net = tf.layers.conv2d(inputs, reduced_depth, [1, 1], padding='SAME',name=name + '_conv1')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm1')
                net = self.prelu(net, name=name + '_prelu1')

                # Second conv block --- apply asymmetric conv here
                net = tf.layers.conv2d(net, reduced_depth, [filter_size, 1],padding='SAME', name=name + '_asymmetric_conv2a')
                net = tf.layers.conv2d(net, reduced_depth, [1, filter_size], padding='SAME',name=name + '_asymmetric_conv2b')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm2')
                net = self.prelu(net, name=name + '_prelu2')

                # Final projection with 1x1 kernel
                net = tf.layers.conv2d(net, output_depth, [1, 1], padding='SAME',name=name + '_conv3')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm3')
                net = self.prelu(net, name=name + '_prelu3')

                # Regularizer
                net = self.spatial_dropout(net, keep_prob=1-regularizer_prob, seed=seed, name=name + '_spatial_dropout')
                net = self.prelu(net, name=name + '_prelu4')

                # Add the main branch
                net = tf.add(net_main, net, name=name + '_add_asymmetric')
                net = self.prelu(net, name=name + '_last_prelu')

                return net

                # ============UPSAMPLING BOTTLENECK================
                # Everything is the same as a regular one, except convolution becomes transposed.
            elif upsampling:
                # Check if pooling indices is given
                if pooling_indices == None:
                    raise ValueError('Pooling indices are not given.')

                # Check output_shape given or not
                if output_shape == None:
                    raise ValueError('Output depth is not given')

                # =======MAIN BRANCH=======
                # Main branch to upsample. output shape must match with the shape of the layer that was pooled initially, in order
                # for the pooling indices to work correctly. However, the initial pooled layer was padded, so need to reduce dimension
                # before unpooling. In the paper, padding is replaced with convolution for this purpose of reducing the depth!
                net_unpool = tf.layers.conv2d(inputs, output_depth, [1, 1], padding='SAME',name=name + '_main_conv1')
                net_unpool = tf.contrib.layers.batch_norm(net_unpool, is_training=training, scope=name + 'batch_norm1')
                net_unpool = self.unpool(net_unpool, pooling_indices, output_shape=output_shape, name='unpool')

                # ======SUB BRANCH=======
                # First 1x1 projection to reduce depth
                net = tf.layers.conv2d(inputs, reduced_depth, [1, 1], padding='SAME',name=name + '_conv1')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm2')
                net = self.prelu(net, name=name + '_prelu1')

                # Second conv block -----------------------------> NOTE: using tf.nn.conv2d_transpose for variable input shape.
                net_unpool_shape = net_unpool.get_shape().as_list()
                output_shape = [net_unpool_shape[0], net_unpool_shape[1], net_unpool_shape[2], reduced_depth]
                output_shape = tf.convert_to_tensor(output_shape)
                filter_size = [filter_size, filter_size, reduced_depth, reduced_depth]
                filters = tf.get_variable(shape=filter_size, initializer=tf.contrib.layers.xavier_initializer(),
                                          dtype=tf.float32, name=name + '_transposed_conv2_filters')

                # net = tf.layers.conv2d_transpose(net, reduced_depth, [filter_size, filter_size], strides=2, name=name+'_transposed_conv2')
                net = tf.nn.conv2d_transpose(net, filter=filters, strides=[1, 2, 2, 1], output_shape=output_shape,
                                             name=name + '_transposed_conv2')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm3')
                net = self.prelu(net, name=name + '_prelu2')

                # Final projection with 1x1 kernel
                net = tf.layers.conv2d(net, output_depth, [1, 1], padding='SAME',name=name + '_conv3')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm4')
                net = self.prelu(net, name=name + '_relu3')

                # Regularizer
                net = self.spatial_dropout(net, keep_prob=1 - regularizer_prob, seed=seed, name=name + '_spatial_dropout')
                net = self.prelu(net, name=name + '_prelu4')

                # Finally, add the unpooling layer and the sub branch together
                net = tf.add(net, net_unpool, name=name + '_add_upsample')
                net = self.prelu(net, name=name + '_last_prelu')

                return net

                # OTHERWISE, just perform a regular bottleneck!
                # ==============REGULAR BOTTLENECK==================
                # Save the main branch for addition later
            net_main = inputs

            # First projection with 1x1 kernel
            net = tf.layers.conv2d(inputs, reduced_depth, [1, 1],padding='SAME', name=name + '_conv1')
            net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm1')
            net = self.prelu(net, name=name + '_prelu1')

            # Second conv block
            net = tf.layers.conv2d(net, reduced_depth, [filter_size, filter_size],padding='SAME', name=name + '_conv2')
            net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm2')
            net = self.prelu(net, name=name + '_prelu2')

            # Final projection with 1x1 kernel
            net = tf.layers.conv2d(net, output_depth, [1, 1], padding='SAME',name=name + '_conv3')
            net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm3')
            net = self.prelu(net, name=name + '_prelu3')

            # Regularizer
            net = self.spatial_dropout(net, keep_prob=1-regularizer_prob, seed=seed, name=name + '_spatial_dropout')
            net = self.prelu(net, name=name + '_prelu4')

            # Add the main branch
            net = tf.add(net_main, net, name=name + '_add_regular')
            net = self.prelu(net, name=name + '_last_prelu')

        return net