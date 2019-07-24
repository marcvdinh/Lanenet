#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午3:54
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_back_end.py
# @IDE: PyCharm
"""
LaneNet backend branch which is mainly used for binary and instance segmentation loss calculation
"""
import tensorflow as tf

from config import global_config
from lanenet_model import lanenet_discriminative_loss
from semantic_segmentation_zoo import cnn_basenet

CFG = global_config.cfg


class LaneNetBackEnd(cnn_basenet.CNNBaseModel):
    """
    LaneNet backend branch which is mainly used for binary and instance segmentation loss calculation
    """
    def __init__(self, phase):
        """
        init lanenet backend
        :param phase: train or test
        """
        super(LaneNetBackEnd, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)

        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    @classmethod
    def _compute_class_weighted_cross_entropy_loss(cls, onehot_labels, logits, classes_weights):
        """

        :param onehot_labels:
        :param logits:
        :param classes_weights:
        :return:
        """
        loss_weights = tf.reduce_sum(tf.multiply(onehot_labels, classes_weights), axis=3)
        #loss_weights = tf.expand_dims(loss_weights,axis=1)
        print("cross entropy loss shapes:")
        print(onehot_labels.shape)
        print(logits.shape)
        print(loss_weights.shape)
        #print(classes_weights.shape)

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=logits,
            weights=loss_weights
        )

        return loss

    def compute_loss(self, lane_seg_logits, lane_label,
                     drive_seg_logits, drive_label,
                     name, reuse):
        """
        compute lanenet loss
        :param binary_seg_logits:
        :param binary_label:
        :param instance_seg_logits:
        :param instance_label:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # calculate class weighted binary seg loss
# calculate class weighted instance seg loss
            with tf.variable_scope(name_or_scope='lane_seg'):

                pix_bn = tf.layers.batch_normalization(
                    inputs=lane_seg_logits, training=self._is_training, name='pix_bn', axis=1)
                pix_relu = tf.nn.relu(pix_bn, name='pix_relu')
                pix_embedding = tf.layers.conv2d(inputs=pix_relu, 
                                                           filters=CFG.TRAIN.EMBEDDING_FEATS_DIMS, 
                                                            kernel_size=[1, 1], 
                                                            data_format = "channels_first",
                                                            activation=None, 
                                                            use_bias=False,
                                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),
                                                            padding="SAME",
                                                            name = 'pix_embedding_conv')
                lane_embedding = tf.transpose(pix_embedding,[0,2,3,1])
                lane_image_shape = (lane_embedding.get_shape().as_list()[1], lane_embedding.get_shape().as_list()[2])
                lane_segmentation_loss, l_var, l_dist, l_reg = \
                    lanenet_discriminative_loss.discriminative_loss(
                        lane_embedding, lane_label, CFG.TRAIN.EMBEDDING_FEATS_DIMS,
                        lane_image_shape, 0.5, 3.0, 1.0, 1.0, 0.001
                    )

            # calculate class weighted instance seg loss
            with tf.variable_scope(name_or_scope='drive_seg'):

                pix_bn = tf.layers.batch_normalization(
                    inputs=drive_seg_logits, training=self._is_training, name='pix_bn', axis=1)
                pix_relu = tf.nn.relu(pix_bn, name='pix_relu')
                pix_embedding = tf.layers.conv2d(inputs=pix_relu, 
                                                           filters=CFG.TRAIN.EMBEDDING_FEATS_DIMS, 
                                                            kernel_size=[1, 1], 
                                                            data_format = "channels_first",
                                                            activation=None, 
                                                            use_bias=False,
                                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),
                                                            padding="SAME",
                                                            name = 'pix_embedding_conv')
                drive_embedding = tf.transpose(pix_embedding,[0,2,3,1])
                drive_image_shape = (drive_embedding.get_shape().as_list()[1], drive_embedding.get_shape().as_list()[2])
                drive_segmentation_loss, l_var, l_dist, l_reg = \
                    lanenet_discriminative_loss.discriminative_loss(
                        drive_embedding, drive_label, CFG.TRAIN.EMBEDDING_FEATS_DIMS,
                        drive_image_shape, 0.5, 3.0, 1.0, 1.0, 0.001
                    )
            
            l2_reg_loss = tf.constant(0.0, tf.float32)
            for vv in tf.trainable_variables():
                if 'bn' in vv.name or 'gn' in vv.name:
                    continue
                else:
                    l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
            l2_reg_loss *= 0.001
            total_loss = lane_segmentation_loss + drive_segmentation_loss + l2_reg_loss

            ret = {
                'total_loss': total_loss,
                'lane_seg_logits': lane_embedding,
                'drive_seg_logits': drive_embedding,
                'binary_seg_loss': lane_segmentation_loss,
                'discriminative_loss': drive_segmentation_loss
            }

        return ret

    def inference(self, lane_seg_logits, drive_seg_logits, name, reuse):
        """

        :param binary_seg_logits:
        :param instance_seg_logits:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):

            with tf.variable_scope(name_or_scope='lane_seg'):
                pix_bn = tf.layers.batch_normalization(
                    inputs=lane_seg_logits, training=self._is_training, name='pix_bn', axis=1)
                pix_relu = tf.nn.relu6(pix_bn, name='pix_relu')
                lane_seg_prediction = tf.layers.conv2d(inputs=pix_relu, 
                                                           filters=CFG.TRAIN.EMBEDDING_FEATS_DIMS, 
                                                            kernel_size=[1, 1], 
                                                            data_format = "channels_first",
                                                            activation=None, 
                                                            use_bias=False,
                                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),
                                                            padding="SAME",
                                                            name = 'pix_embedding_conv')
                lane_seg_prediction = tf.transpose(instance_seg_prediction, [0,2,3,1])

            with tf.variable_scope(name_or_scope='drive_seg'):

                pix_bn = tf.layers.batch_normalization(
                    inputs=drive_seg_logits, training=self._is_training, name='pix_bn', axis=1)
                pix_relu = tf.nn.relu6(pix_bn, name='pix_relu')
                drive_seg_prediction = tf.layers.conv2d(inputs=pix_relu, 
                                                           filters=CFG.TRAIN.EMBEDDING_FEATS_DIMS, 
                                                            kernel_size=[1, 1], 
                                                            data_format = "channels_first",
                                                            activation=None, 
                                                            use_bias=False,
                                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),
                                                            padding="SAME",
                                                            name = 'pix_embedding_conv')
                drive_seg_prediction = tf.transpose(instance_seg_prediction, [0,2,3,1])

        return lane_seg_prediction, drive_seg_prediction
