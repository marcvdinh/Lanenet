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

class CNNBaseModel(object):
    """
    Base model for other specific cnn ctpn_models
    """

    def __init__(self):
        pass

    @staticmethod
    def conv2d(inputdata, out_channel, kernel_size, padding='SAME',
               stride=1, w_init=None, b_init=None,
               split=1, use_bias=True, data_format='NHWC', name=None):
        """
        Packing the tensorflow conv2d function.
        :param name: op name
        :param inputdata: A 4D tensorflow tensor which ust have known number of channels, but can have other
        unknown dimensions.
        :param out_channel: number of output channel.
        :param kernel_size: int so only support square kernel convolution
        :param padding: 'VALID' or 'SAME'
        :param stride: int so only support square stride
        :param w_init: initializer for convolution weights
        :param b_init: initializer for bias
        :param split: split channels as used in Alexnet mainly group for GPU memory save.
        :param use_bias:  whether to use bias.
        :param data_format: default set to NHWC according tensorflow
        :return: tf.Tensor named ``output``
        """
        with tf.variable_scope(name):
            in_shape = inputdata.get_shape().as_list()
            channel_axis = 3 if data_format == 'NHWC' else 1
            in_channel = in_shape[channel_axis]
            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
            assert in_channel % split == 0
            assert out_channel % split == 0

            padding = padding.upper()

            if isinstance(kernel_size, list):
                filter_shape = [kernel_size[0], kernel_size[1]] + [in_channel / split, out_channel]
            else:
                filter_shape = [kernel_size, kernel_size] + [in_channel / split, out_channel]

            if isinstance(stride, list):
                strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' \
                    else [1, 1, stride[0], stride[1]]
            else:
                strides = [1, stride, stride, 1] if data_format == 'NHWC' \
                    else [1, 1, stride, stride]

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            w = tf.get_variable('W', filter_shape, initializer=w_init)
            b = None

            if use_bias:
                b = tf.get_variable('b', [out_channel], initializer=b_init)

            if split == 1:
                conv = tf.nn.conv2d(inputdata, w, strides, padding, data_format=data_format)
            else:
                inputs = tf.split(inputdata, split, channel_axis)
                kernels = tf.split(w, split, 3)
                outputs = [tf.nn.conv2d(i, k, strides, padding, data_format=data_format)
                           for i, k in zip(inputs, kernels)]
                conv = tf.concat(outputs, channel_axis)

            ret = tf.identity(tf.nn.bias_add(conv, b, data_format=data_format)
                              if use_bias else conv, name=name)

        return ret

    @staticmethod
    def relu(inputdata, name=None):
        """

        :param name:
        :param inputdata:
        :return:
        """
        return tf.nn.relu(features=inputdata, name=name)

    @staticmethod
    def sigmoid(inputdata, name=None):
        """

        :param name:
        :param inputdata:
        :return:
        """
        return tf.nn.sigmoid(x=inputdata, name=name)

    @staticmethod
    def maxpooling(inputdata, kernel_size, stride=None, padding='VALID',
                   data_format='NHWC', name=None):
        """

        :param name:
        :param inputdata:
        :param kernel_size:
        :param stride:
        :param padding:
        :param data_format:
        :return:
        """
        padding = padding.upper()

        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, list):
            kernel = [1, kernel_size[0], kernel_size[1], 1] if data_format == 'NHWC' else \
                [1, 1, kernel_size[0], kernel_size[1]]
        else:
            kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' \
                else [1, 1, kernel_size, kernel_size]

        if isinstance(stride, list):
            strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' \
                else [1, 1, stride[0], stride[1]]
        else:
            strides = [1, stride, stride, 1] if data_format == 'NHWC' \
                else [1, 1, stride, stride]

        return tf.nn.max_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    @staticmethod
    def avgpooling(inputdata, kernel_size, stride=None, padding='VALID',
                   data_format='NHWC', name=None):
        """

        :param name:
        :param inputdata:
        :param kernel_size:
        :param stride:
        :param padding:
        :param data_format:
        :return:
        """
        if stride is None:
            stride = kernel_size

        kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' \
            else [1, 1, kernel_size, kernel_size]

        strides = [1, stride, stride, 1] if data_format == 'NHWC' else [1, 1, stride, stride]

        return tf.nn.avg_pool(value=inputdata, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    @staticmethod
    def globalavgpooling(inputdata, data_format='NHWC', name=None):
        """

        :param name:
        :param inputdata:
        :param data_format:
        :return:
        """
        assert inputdata.shape.ndims == 4
        assert data_format in ['NHWC', 'NCHW']

        axis = [1, 2] if data_format == 'NHWC' else [2, 3]

        return tf.reduce_mean(input_tensor=inputdata, axis=axis, name=name)

    @staticmethod
    def layernorm(inputdata, epsilon=1e-5, use_bias=True, use_scale=True,
                  data_format='NHWC', name=None):
        """
        :param name:
        :param inputdata:
        :param epsilon: epsilon to avoid divide-by-zero.
        :param use_bias: whether to use the extra affine transformation or not.
        :param use_scale: whether to use the extra affine transformation or not.
        :param data_format:
        :return:
        """
        shape = inputdata.get_shape().as_list()
        ndims = len(shape)
        assert ndims in [2, 4]

        mean, var = tf.nn.moments(inputdata, list(range(1, len(shape))), keep_dims=True)

        if data_format == 'NCHW':
            channnel = shape[1]
            new_shape = [1, channnel, 1, 1]
        else:
            channnel = shape[-1]
            new_shape = [1, 1, 1, channnel]
        if ndims == 2:
            new_shape = [1, channnel]

        if use_bias:
            beta = tf.get_variable('beta', [channnel], initializer=tf.constant_initializer())
            beta = tf.reshape(beta, new_shape)
        else:
            beta = tf.zeros([1] * ndims, name='beta')
        if use_scale:
            gamma = tf.get_variable('gamma', [channnel], initializer=tf.constant_initializer(1.0))
            gamma = tf.reshape(gamma, new_shape)
        else:
            gamma = tf.ones([1] * ndims, name='gamma')

        return tf.nn.batch_normalization(inputdata, mean, var, beta, gamma, epsilon, name=name)

    @staticmethod
    def instancenorm(inputdata, epsilon=1e-5, data_format='NHWC', use_affine=True, name=None):
        """

        :param name:
        :param inputdata:
        :param epsilon:
        :param data_format:
        :param use_affine:
        :return:
        """
        shape = inputdata.get_shape().as_list()
        if len(shape) != 4:
            raise ValueError("Input data of instancebn layer has to be 4D tensor")

        if data_format == 'NHWC':
            axis = [1, 2]
            ch = shape[3]
            new_shape = [1, 1, 1, ch]
        else:
            axis = [2, 3]
            ch = shape[1]
            new_shape = [1, ch, 1, 1]
        if ch is None:
            raise ValueError("Input of instancebn require known channel!")

        mean, var = tf.nn.moments(inputdata, axis, keep_dims=True)

        if not use_affine:
            return tf.divide(inputdata - mean, tf.sqrt(var + epsilon), name='output')

        beta = tf.get_variable('beta', [ch], initializer=tf.constant_initializer())
        beta = tf.reshape(beta, new_shape)
        gamma = tf.get_variable('gamma', [ch], initializer=tf.constant_initializer(1.0))
        gamma = tf.reshape(gamma, new_shape)
        return tf.nn.batch_normalization(inputdata, mean, var, beta, gamma, epsilon, name=name)

    @staticmethod
    def dropout(inputdata, rate, noise_shape=None, name=None):
        """

        :param name:
        :param inputdata:
        :param rate:
        :param noise_shape:
        :return:
        """
        return tf.nn.dropout(inputdata, rate=rate, noise_shape=noise_shape, name=name)

    @staticmethod
    def fullyconnect(inputdata, out_dim, w_init=None, b_init=None,
                     use_bias=True, name=None):
        """
        Fully-Connected layer, takes a N>1D tensor and returns a 2D tensor.
        It is an equivalent of `tf.layers.dense` except for naming conventions.

        :param inputdata:  a tensor to be flattened except for the first dimension.
        :param out_dim: output dimension
        :param w_init: initializer for w. Defaults to `variance_scaling_initializer`.
        :param b_init: initializer for b. Defaults to zero
        :param use_bias: whether to use bias.
        :param name:
        :return: tf.Tensor: a NC tensor named ``output`` with attribute `variables`.
        """
        shape = inputdata.get_shape().as_list()[1:]
        if None not in shape:
            inputdata = tf.reshape(inputdata, [-1, int(np.prod(shape))])
        else:
            inputdata = tf.reshape(inputdata, tf.stack([tf.shape(inputdata)[0], -1]))

        if w_init is None:
            w_init = tf.contrib.layers.variance_scaling_initializer()
        if b_init is None:
            b_init = tf.constant_initializer()

        ret = tf.layers.dense(inputs=inputdata, activation=lambda x: tf.identity(x, name='output'),
                              use_bias=use_bias, name=name,
                              kernel_initializer=w_init, bias_initializer=b_init,
                              trainable=True, units=out_dim)
        return ret

    @staticmethod
    def layerbn(inputdata, is_training, name):
        """

        :param inputdata:
        :param is_training:
        :param name:
        :return:
        """

        return tf.layers.batch_normalization(inputs=inputdata, training=is_training, name=name)

    @staticmethod
    def layergn(inputdata, name, group_size=32, esp=1e-5):
        """

        :param inputdata:
        :param name:
        :param group_size:
        :param esp:
        :return:
        """
        with tf.variable_scope(name):
            inputdata = tf.transpose(inputdata, [0, 3, 1, 2])
            n, c, h, w = inputdata.get_shape().as_list()
            group_size = min(group_size, c)
            inputdata = tf.reshape(inputdata, [-1, group_size, c // group_size, h, w])
            mean, var = tf.nn.moments(inputdata, [2, 3, 4], keep_dims=True)
            inputdata = (inputdata - mean) / tf.sqrt(var + esp)

            # 每个通道的gamma和beta
            gamma = tf.Variable(tf.constant(1.0, shape=[c]), dtype=tf.float32, name='gamma')
            beta = tf.Variable(tf.constant(0.0, shape=[c]), dtype=tf.float32, name='beta')
            gamma = tf.reshape(gamma, [1, c, 1, 1])
            beta = tf.reshape(beta, [1, c, 1, 1])

            # 根据论文进行转换 [n, c, h, w, c] 到 [n, h, w, c]
            output = tf.reshape(inputdata, [-1, c, h, w])
            output = output * gamma + beta
            output = tf.transpose(output, [0, 2, 3, 1])

        return output

    @staticmethod
    def squeeze(inputdata, axis=None, name=None):
        """

        :param inputdata:
        :param axis:
        :param name:
        :return:
        """
        return tf.squeeze(input=inputdata, axis=axis, name=name)

    @staticmethod
    def deconv2d(inputdata, out_channel, kernel_size, padding='SAME',
                 stride=1, w_init=None, b_init=None,
                 use_bias=True, activation=None, data_format='channels_last',
                 trainable=True, name=None):
        """
        Packing the tensorflow conv2d function.
        :param name: op name
        :param inputdata: A 4D tensorflow tensor which ust have known number of channels, but can have other
        unknown dimensions.
        :param out_channel: number of output channel.
        :param kernel_size: int so only support square kernel convolution
        :param padding: 'VALID' or 'SAME'
        :param stride: int so only support square stride
        :param w_init: initializer for convolution weights
        :param b_init: initializer for bias
        :param activation: whether to apply a activation func to deconv result
        :param use_bias:  whether to use bias.
        :param data_format: default set to NHWC according tensorflow
        :return: tf.Tensor named ``output``
        """
        with tf.variable_scope(name):
            in_shape = inputdata.get_shape().as_list()
            channel_axis = 3 if data_format == 'channels_last' else 1
            in_channel = in_shape[channel_axis]
            assert in_channel is not None, "[Deconv2D] Input cannot have unknown channel!"

            padding = padding.upper()

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            ret = tf.layers.conv2d_transpose(inputs=inputdata, filters=out_channel,
                                             kernel_size=kernel_size,
                                             strides=stride, padding=padding,
                                             data_format=data_format,
                                             activation=activation, use_bias=use_bias,
                                             kernel_initializer=w_init,
                                             bias_initializer=b_init, trainable=trainable,
                                             name=name)
        return ret

    @staticmethod
    def dilation_conv(input_tensor, k_size, out_dims, rate, padding='SAME',
                      w_init=None, b_init=None, use_bias=False, name=None):
        """

        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param rate:
        :param padding:
        :param w_init:
        :param b_init:
        :param use_bias:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            in_shape = input_tensor.get_shape().as_list()
            in_channel = in_shape[3]
            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"

            padding = padding.upper()

            if isinstance(k_size, list):
                filter_shape = [k_size[0], k_size[1]] + [in_channel, out_dims]
            else:
                filter_shape = [k_size, k_size] + [in_channel, out_dims]

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            w = tf.get_variable('W', filter_shape, initializer=w_init)
            b = None

            if use_bias:
                b = tf.get_variable('b', [out_dims], initializer=b_init)

            conv = tf.nn.atrous_conv2d(value=input_tensor, filters=w, rate=rate,
                                       padding=padding, name='dilation_conv')

            if use_bias:
                ret = tf.add(conv, b)
            else:
                ret = conv

        return ret

    @staticmethod
    def spatial_dropout(input_tensor, rate, is_training, name, seed=1234):
        """
        空间dropout实现
        :param input_tensor:
        :param rate:
        :param is_training:
        :param name:
        :param seed:
        :return:
        """

        def f1():
            input_shape = input_tensor.get_shape().as_list()
            noise_shape = tf.constant(value=[input_shape[0], 1, 1, input_shape[3]])
            return tf.nn.dropout(input_tensor, rate, noise_shape, seed=seed, name="spatial_dropout")

        def f2():
            return input_tensor

        with tf.variable_scope(name_or_scope=name):

            #output = tf.cond(is_training, f1, f2)
            if is_training is True:
                output = f1()
            else:
                output = f2()

            return output

    @staticmethod
    def lrelu(inputdata, name, alpha=0.2):
        """

        :param inputdata:
        :param alpha:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            return tf.nn.relu(inputdata) - alpha * tf.nn.relu(-inputdata)

    @staticmethod
    @slim.add_arg_scope
    def prelu(inputdata, name, decoder=False):

        if decoder:
            return tf.nn.relu(features=inputdata, name=name)

        alpha = tf.get_variable(name + '_alpha', shape=inputdata.get_shape()[-1],
                                initializer=tf.constant_initializer(0.0),
                                dtype=tf.float32)

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
    def initial_block(self, inputdata, training=True, name='initial_block'):

        # Convolutional branch
        net_conv = tf.layers.conv2d(inputdata, 13, kernel_size=[3, 3], strides=2, padding='SAME', activation=None,
                                    name=name + '_conv')
        net_conv = tf.contrib.layers.batch_norm(net_conv, is_training=training, fused=True, scope=name + '_batchnorm')
        net_conv = self.prelu(net_conv, name + '_prelu')

        # max pool
        net_pool = tf.layers.max_pooling2d(inputdata, [2, 2], strides=2, name=name + '_maxpool')

        # concatenated ouput
        net_concat = tf.concat([net_conv, net_pool], axis=3, name=name + '_concat')
        return net_concat

    @staticmethod
    @slim.add_arg_scope
    def bottleneck(self, inputs,
                   output_depth,
                   filter_size,
                   regularizer_prob,
                   training,
                   projection_ratio=4,
                   seed=0,
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
                net = tf.layers.conv2d(net, reduced_depth, [filter_size, filter_size], padding='SAME',
                                       name=name + '_conv2')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm2')
                net = self.prelu(net, name=name + '_prelu2')

                # Final projection with 1x1 kernel
                net = tf.layers.conv2d(net, output_depth, [1, 1], padding='SAME', name=name + '_conv3')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm3')
                net = self.prelu(net, name=name + '_prelu3')

                # Regularizer
                net = self.spatial_dropout(net, rate=1 - regularizer_prob, seed=seed, is_training=training,
                                           name=name + '_spatial_dropout')

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
                net = tf.layers.conv2d(inputs, reduced_depth, [1, 1], padding='SAME', name=name + '_conv1')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm1')
                net = self.prelu(net, name=name + '_prelu1')

                # Second conv block --- apply dilated convolution here
                net = tf.layers.conv2d(net, reduced_depth, [filter_size, filter_size], dilation_rate=dilation_rate,
                                       padding='SAME',
                                       name=name + '_dilated_conv2')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm2')
                net = self.prelu(net, name=name + '_prelu2')

                # Final projection with 1x1 kernel (Expansion)
                net = tf.layers.conv2d(net, output_depth, [1, 1], padding='SAME', name=name + '_conv3')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm3')
                net = self.prelu(net, name=name + '_prelu3')

                # Regularizer
                net = self.spatial_dropout(net, rate=1 - regularizer_prob, seed=seed, is_training=training,
                                           name=name + '_spatial_dropout')
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
                net = tf.layers.conv2d(inputs, reduced_depth, [1, 1], padding='SAME', name=name + '_conv1')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm1')
                net = self.prelu(net, name=name + '_prelu1')

                # Second conv block --- apply asymmetric conv here
                net = tf.layers.conv2d(net, reduced_depth, [filter_size, 1], padding='SAME',
                                       name=name + '_asymmetric_conv2a')
                net = tf.layers.conv2d(net, reduced_depth, [1, filter_size], padding='SAME',
                                       name=name + '_asymmetric_conv2b')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm2')
                net = self.prelu(net, name=name + '_prelu2')

                # Final projection with 1x1 kernel
                net = tf.layers.conv2d(net, output_depth, [1, 1], padding='SAME', name=name + '_conv3')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm3')
                net = self.prelu(net, name=name + '_prelu3')

                # Regularizer
                net = self.spatial_dropout(net, rate=1 - regularizer_prob, seed=seed, is_training=training,
                                           name=name + '_spatial_dropout')
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
                net_unpool = tf.layers.conv2d(inputs, output_depth, [1, 1], padding='SAME', name=name + '_main_conv1')
                net_unpool = tf.contrib.layers.batch_norm(net_unpool, is_training=training, scope=name + 'batch_norm1')
                net_unpool = self.unpool(net_unpool, pooling_indices, output_shape=output_shape, name='unpool')

                # ======SUB BRANCH=======
                # First 1x1 projection to reduce depth
                net = tf.layers.conv2d(inputs, reduced_depth, [1, 1], padding='SAME', name=name + '_conv1')
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
                net = tf.layers.conv2d(net, output_depth, [1, 1], padding='SAME', name=name + '_conv3')
                net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm4')
                net = self.prelu(net, name=name + '_relu3')

                # Regularizer
                net = self.spatial_dropout(net, rate=1 - regularizer_prob, seed=seed, is_training=training,
                                           name=name + '_spatial_dropout')
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
            net = tf.layers.conv2d(inputs, reduced_depth, [1, 1], padding='SAME', name=name + '_conv1')
            net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm1')
            net = self.prelu(net, name=name + '_prelu1')

            # Second conv block
            net = tf.layers.conv2d(net, reduced_depth, [filter_size, filter_size], padding='SAME', name=name + '_conv2')
            net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm2')
            net = self.prelu(net, name=name + '_prelu2')

            # Final projection with 1x1 kernel
            net = tf.layers.conv2d(net, output_depth, [1, 1], padding='SAME', name=name + '_conv3')
            net = tf.contrib.layers.batch_norm(net, is_training=training, scope=name + '_batch_norm3')
            net = self.prelu(net, name=name + '_prelu3')

            # Regularizer
            net = self.spatial_dropout(net, rate=1 - regularizer_prob, seed=seed, is_training=training, name=name + '_spatial_dropout')
            net = self.prelu(net, name=name + '_prelu4')

            # Add the main branch
            net = tf.add(net_main, net, name=name + '_add_regular')
            net = self.prelu(net, name=name + '_last_prelu')

        return net