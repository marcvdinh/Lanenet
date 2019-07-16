#this script optimizes a Tensorflow/UFF graph to the trt engine. Must be run on the target machine

import tensorflow as tf
import tensorrt as trt
import uff
import os


_input = 'input_1'
_output = 'output_node0'
outputs = ['lanenet_model/mobilenet_backend/instance_seg/pix_embedding_conv/Conv2D','lanenet_model/mobilenet_backend/binary_seg/ArgMax']




def build_engine(model, outputs):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        # Configure the builder here.
        builder.max_workspace_size = 2**30
        builder.max_batch_size = 1
        parser.register_input("Placeholder", (1,256,512,3))
        parser.register_output(outputs[0])
        parser.register_output(outputs[1])
        # Parse the model to create a network.
        parser.parse(model, network)
            # Build and return the engine. Note that the builder, network and parser are destroyed when this function returns.
        return builder.build_cuda_engine(network)


TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
#frozen_graph_def = get_frozen_graph()
UFF_MODEL = uff.from_tensorflow_frozen_model('./saved_model.pb',
					outputs,
               output_filename='uff_model.uff',
               text = True)
model_file = 'uff_model.uff'
engine = build_engine(model_file, outputs)

