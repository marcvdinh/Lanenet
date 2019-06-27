#this script accelerates a TensorFlow graph with TensorRT using the TF-TRT optimizer. 
# Must be run on the target platform as optimizations are GPU specific
import os
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.framework import graph_io

_input = 'input_1'
_output = 'output_node0'
outputs = ['lanenet_model_1/enet_backend/instance_seg/pix_embedding_conv/Conv2D','lanenet_model_1/enet_frontend/enet_decode_module/fullconv/Relu']


def get_frozen_graph():
  with tf.gfile.FastGFile('/home/marc/PycharmProjects/LaneNet/model/tusimple_lanenet_enet/enet4/saved_model.pb', "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


frozen_graph_def = get_frozen_graph()
trt_graph_def = trt.create_inference_graph(frozen_graph_def,
					outputs,
					max_batch_size=4,
					max_workspace_size_bytes=1 << 30,
					precision_mode='FP16')
tf.reset_default_graph()
g = tf.Graph()
with tf.Session(graph=g) as sess:
	with g.as_default():
		tf.import_graph_def(
  		graph_def=trt_graph_def,
  		name='')
	graph_io.write_graph(g, '.', 'trt_frozen.pb', as_text=False)