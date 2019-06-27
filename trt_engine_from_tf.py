#this script optimizes a Tensorflow/UFF graph to the trt engine. Must be run on the target machine

import tensorflow as tf
import tensorrt as trt
import uff

_input = 'input_1'
_output = 'output_node0'
outputs = ['lanenet_model_1/enet_backend/instance_seg/pix_embedding_conv/Conv2D','lanenet_model_1/enet_frontend/enet_decode_module/fullconv/Relu']




def build_engine():
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        # Configure the builder here.
        builder.max_workspace_size = 2**30
        builder.max_batch_size = 1
        parser.register_input("Placeholder", (1,256,512,3))
        parser.register_output(outputs[0])
        parser.register_output(outputs[1])
        # Parse the model to create a network.
        with UFF_MODEL as model:
            parser.parse(model, network)
            # Build and return the engine. Note that the builder, network and parser are destroyed when this function returns.
            with builder.build_cuda_engine(network) as engine:
               #write engine to file
               with open("lanenet_enet.engine", "wb") as f:
                  f.write(engine.serialize())



TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
#frozen_graph_def = get_frozen_graph()
UFF_MODEL = uff.from_tensorflow_frozen_model('./model/tusimple_lanenet_enet/enet4/saved_model.pb',
					outputs,
               output_file_name='./model/tusimple_lanenet_enet/enet4/saved.model.uff')
build_engine()

