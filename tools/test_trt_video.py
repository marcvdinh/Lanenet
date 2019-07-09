#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image
"""
import argparse
import os.path as ops
import time

import cv2
import glog as log
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

from config import global_config
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='data/test.mp4',help='The image path or the src image save dir')
    parser.add_argument('--frozen_model', type=str, default= 'model/tusimple_lanenet_enet/enet4/frozen/trt_model.pb',  help='The model weights path')
    parser.add_argument('--net', type=str, default='enet',
                        help='The net flag which determins the net\'s architecture')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph



def test_lanenet(video_path, frozen_model, net_flag='enet'):
    """

    :param video_path:
    :param weights_path:
    :return:
    """
    assert ops.exists(video_path), '{:s} not exist'.format(video_path)

    cap = cv2.VideoCapture(video_path)
    log.info('Start reading video and preprocessing')

    
    #postprocessor = lanenet_postprocess.LaneNetPostProcessor()

    graph = load_graph(frozen_model)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)

    input_tensor = graph.get_tensor_by_name('prefix/input_tensor:0')
    
    binary_seg_ret = graph.get_tensor_by_name('prefix/lanenet_model/'+net_flag+'_backend/binary_seg/ArgMax:0')
    instance_seg_ret = graph.get_tensor_by_name('prefix/lanenet_model/'+ net_flag +'_backend/instance_seg/pix_embedding_conv/Conv2D:0')

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config, graph=graph)
    
    
    with sess.as_default():

        while(cap.isOpened()):
            ret, frame = cap.read()
            t_start = time.time()
            image_vis = frame
            image = cv2.resize(frame, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0
            log.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))


            t_start = time.time()
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )
            t_cost = time.time() - t_start
            log.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))

            
            for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
                instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
            embedding_image = np.array(instance_seg_image[0], np.uint8)

            
            cv2.imshow('instance_mask_image', embedding_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    sess.close()

    return


if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    test_lanenet(args.video_path, args.frozen_model, args.net)
