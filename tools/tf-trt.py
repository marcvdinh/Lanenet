#this script accelerates a TensorFlow graph with TensorRT using the TF-TRT optimizer. 
# Must be run on the target platform as optimizations are GPU specific
import os, argparse
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.framework import graph_io


_input = 'input_tensor'
_output = 'output_node0'
#outputs = ['lanenet_model/'+ net +'_backend/instance_seg/pix_embedding_conv/Conv2D','lanenet_model/'+net+'_backend/binary_seg/ArgMax']


def get_frozen_graph(frozen_path):
  with tf.gfile.FastGFile(frozen_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


def optimize_graph(frozen_path, net_flag):
	frozen_graph_def = get_frozen_graph(frozen_path)
	# We precise the file fullname of our freezed graph
	absolute_model_dir = "/".join(frozen_path.split('/')[:-1])
	output_graph = "trt_model.pb"

	outputs = ['lanenet_model/'+ net +'_backend/instance_seg/pix_embedding_conv/Conv2D','lanenet_model/'+net+'_backend/binary_seg/ArgMax']

	trt_graph_def = trt.create_inference_graph(frozen_graph_def,
					outputs,
					max_batch_size=2,
					minimum_segment_size=5,
					max_workspace_size_bytes=1 << 30,
					precision_mode='FP16')
	tf.reset_default_graph()
	g = tf.Graph()
	with tf.Session(graph=g) as sess:
		with g.as_default():
			tf.import_graph_def(
  			graph_def=trt_graph_def,
  			name='')
		graph_io.write_graph(g, absolute_model_dir, output_graph, as_text=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", help="frozen model to export")
	 parser.add_argument("--net", type=str, default="", help="backbone architecture")
    args = parser.parse_args()

optimize_graph(args.model_path, args.net)
