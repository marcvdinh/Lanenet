import os, argparse
from config import global_config
from lanenet_model import lanenet
import tensorflow as tf
import numpy
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

CFG = global_config.cfg
dir = os.path.dirname(os.path.realpath(__file__))
net_flag = "enet"
model_dir = "/home/marcdinh/LaneNet/model/tusimple_lanenet_enet/enet4"



def freeze_graph(model_dir, output_node_names, optimize=False):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])

    if not os.path.exists(absolute_model_dir + "/inference"):
        os.mkdir(absolute_model_dir + "/inference")
    if not os.path.exists(absolute_model_dir+"/frozen"):
        os.mkdir(absolute_model_dir+"/frozen")


    inference_checkpoint = absolute_model_dir + "/inference/inference_graph"
    output_graph = absolute_model_dir + "/frozen/saved_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True
    
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test', net_flag=net_flag)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    saver = tf.train.Saver()


    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)
    #we load the weights into the inference graph
    with sess.as_default():
        saver.restore(sess=sess, save_path=input_checkpoint)
        saver.save(sess,inference_checkpoint)
        tf.train.write_graph(sess.graph.as_graph_def(), absolute_model_dir + "/inference", 'inference_graph.pb', as_text=True)
    sess.close()

    
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(inference_checkpoint + '.meta', clear_devices=clear_devices)

        

        # We restore the weights
        saver.restore(sess, inference_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the useful nodes
        )
        if optimize == True:
            output_graph_def = tf.compat.v1.graph_util.remove_training_nodes(output_graph_def)
        
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
    sess.close()
    return output_graph_def

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/home/marcdinh/LaneNet/model/tusimple_lanenet_enet/enet4", help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, default="lanenet_model/enet_backend/instance_seg/pix_embedding_conv/Conv2D,lanenet_model/enet_backend/binary_seg/ArgMax",
                        help="The name of the output nodes, comma separated.")

    parser.add_argument("--optimize", type=bool, default=False,
                        help="use the TF optimizer to prune unused nodes and fold layers")
    args = parser.parse_args()

freeze_graph(args.model_dir, args.output_node_names, args.optimize)
