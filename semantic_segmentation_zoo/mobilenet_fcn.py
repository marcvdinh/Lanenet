import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy
from semantic_segmentation_zoo import cnn_basenet
from config import global_config

CFG = global_config.cfg

class MOBILENETFCN(cnn_basenet.CNNBaseModel):
    def __init__(self, phase):
        """

        """
        super(MOBILENETFCN, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._net_intermediate_results = collections.OrderedDict()
        self._skip_connections = True
    
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
    
    def block(self, net, input_filters, output_filters, expansion, stride, name):
        res_block = net
        res_block = slim.conv2d(inputs=net, num_outputs=input_filters * expansion, kernel_size=[1, 1], scope = name + "_conv11")
        res_block = slim.separable_conv2d(inputs=res_block, num_outputs=None, kernel_size=[3, 3], stride=stride, scope = name + "_conv33")
        res_block = slim.conv2d(inputs=res_block, num_outputs=output_filters, kernel_size=[1, 1], activation_fn=None, scope = name + "_2conv11")
        if stride == 2:
            return res_block
        else:
            if input_filters != output_filters:
                net = slim.conv2d(inputs=net, num_outputs=output_filters, kernel_size=[1, 1], activation_fn=None, scope = name + "_3conv11")
            return tf.add(res_block, net)


    def blocks(self, net, expansion, output_filters, repeat, stride, name):
        input_filters = net.shape[3].value

        # first layer should take stride into account
        net = self.block( net=net, input_filters=input_filters, output_filters=output_filters, expansion=expansion, stride=stride, name=name)

        for i in range(1, repeat):
            net = self.block( net=net, input_filters=input_filters, output_filters=output_filters, expansion=expansion, stride=1, name = name + str(i))
        return net

    def mobilenet_v2_arg_scope(self,weight_decay, depth_multiplier=1.0, regularize_depthwise=False,
                           dropout_keep_prob=1.0):

        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        if regularize_depthwise:
            depthwise_regularizer = regularizer
        else:
            depthwise_regularizer = None

        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': self._is_training, 'center': True, 'scale': True }):

            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):

                with slim.arg_scope([slim.separable_conv2d],
                    weights_regularizer=depthwise_regularizer, depth_multiplier=depth_multiplier):

                    with slim.arg_scope([slim.dropout], is_training=self._is_training, keep_prob=dropout_keep_prob) as sc:

                        return sc

    def _mobilenet_fcn_encode(self,inputs,
                 dropout_keep_prob=0.999,
                 depth_multiplier=1.0,
                 spatial_squeeze=True,
                 name='encode'):

        expansion = 6

        with tf.variable_scope(name):
            regularizer = tf.contrib.layers.l2_regularizer(0.0004)
            with slim.arg_scope([slim.conv2d ],activation_fn=tf.nn.relu, weights_regularizer=regularizer),\
             slim.arg_scope([slim.separable_conv2d],weights_regularizer=None, depth_multiplier=1.0),\
             slim.arg_scope([slim.dropout], is_training=self._is_training, keep_prob=0.99),\
             slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': self._is_training, 'center': True, 'scale': True }):

                net = self.conv2d(inputs, 32, 3, name='conv11', stride=2)

                net = self.blocks(net=net, expansion=1, output_filters=16, repeat=1, stride=1, name = "bottleneck1")
                self._net_intermediate_results['residual_1'] = { 'data' : net, 'shape': net.get_shape().as_list()}
                net = self.blocks(net=net, expansion=expansion, output_filters=24, repeat=2, stride=2,name = "bottleneck2")
                self._net_intermediate_results['residual_2'] = { 'data' : net,'shape': net.get_shape().as_list()}
                net = self.blocks(net=net, expansion=expansion, output_filters=32, repeat=3, stride=2, name = "bottleneck3")
                self._net_intermediate_results['residual_3'] = { 'data' : net,'shape': net.get_shape().as_list()}
                net = self.blocks(net=net, expansion=expansion, output_filters=64, repeat=4, stride=2, name = "bottleneck4")
                self._net_intermediate_results['residual_4'] = { 'data' : net,'shape': net.get_shape().as_list()}
                net = self.blocks(net=net, expansion=expansion, output_filters=96, repeat=3, stride=1, name = "bottleneck5")
                self._net_intermediate_results['residual_5'] = { 'data' : net,'shape': net.get_shape().as_list()}
                net = self.blocks(net=net, expansion=expansion, output_filters=160, repeat=3, stride=2, name = "bottleneck6")
                self._net_intermediate_results['residual_6'] = { 'data' : net,'shape': net.get_shape().as_list()}
                net = self.blocks(net=net, expansion=expansion, output_filters=320, repeat=1, stride=1, name = "bottleneck7")
       

                self._net_intermediate_results['shared_encoding'] = {
                        'net': net,
                        'shape': net.get_shape().as_list()}
            
    def _mobilenet_fcn_decode(self, name="decode"):
        
        shape = self._net_intermediate_results['shared_encoding']['shape']
        res1 = self._net_intermediate_results['residual_1']['data']
        res2 = self._net_intermediate_results['residual_2']['data']
        res3 = self._net_intermediate_results['residual_3']['data']
        res4 = self._net_intermediate_results['residual_4']['data']
        res5 = self._net_intermediate_results['residual_5']['data']
        res6 = self._net_intermediate_results['residual_6']['data']

        regularizer = tf.contrib.layers.l2_regularizer(0.0004)
        
        with slim.arg_scope([slim.conv2d ],activation_fn=tf.nn.relu, weights_regularizer=regularizer),\
             slim.arg_scope([slim.separable_conv2d],weights_regularizer=None, depth_multiplier=1.0),\
             slim.arg_scope([slim.dropout], is_training=self._is_training, keep_prob=0.99),\
             slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': self._is_training, 'center': True, 'scale': True }):

            with tf.variable_scope('binary_seg'):
                net = self._net_intermediate_results['shared_encoding']['net']
                
                res6 = slim.conv2d(inputs=res6, num_outputs=320, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_6']['shape'][1],self._net_intermediate_results['residual_6']['shape'][2]])
                net = tf.add(net,res6)
                net = self.blocks(net=net, expansion=1, output_filters=64, repeat=1, stride=1, name = "bottleneck1")

                res5 = slim.conv2d(inputs=res5, num_outputs=64, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_5']['shape'][1],self._net_intermediate_results['residual_5']['shape'][2]])
                net = tf.add(net,res5)
                net = self.blocks(net=net, expansion=1, output_filters=64, repeat=1, stride=1, name = "bottleneck2")

                res4 = slim.conv2d(inputs=res4, num_outputs=64, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_4']['shape'][1],self._net_intermediate_results['residual_4']['shape'][2]])
                net = tf.add(net,res4)
                net = self.blocks(net=net, expansion=1, output_filters=64, repeat=1, stride=1, name = "bottleneck3")

                res3 = slim.conv2d(inputs=res3, num_outputs=64, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_3']['shape'][1],self._net_intermediate_results['residual_3']['shape'][2]])
                net = tf.add(net,res3)
                net = self.blocks(net=net, expansion=1, output_filters=64, repeat=1, stride=1,name = "bottleneck4")
                
                res2 = slim.conv2d(inputs=res2, num_outputs=64, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_2']['shape'][1],self._net_intermediate_results['residual_2']['shape'][2]])
                net = tf.add(net,res2)
                net = self.blocks(net=net, expansion=1, output_filters=64, repeat=1, stride=1,name = "bottleneck5")
                
                res1 = slim.conv2d(inputs=res1, num_outputs=64, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_1']['shape'][1],self._net_intermediate_results['residual_1']['shape'][2]])
                net = tf.add(net,res1)
                net = self.blocks(net=net, expansion=1, output_filters=64, repeat=1, stride=1, name = "bottleneck6")
                
                net = tf.image.resize_images(net, [CFG.TRAIN.IMG_HEIGHT,CFG.TRAIN.IMG_WIDTH])
                net = self.blocks(net=net, expansion=1, output_filters=64, repeat=1, stride=1, name = "bottleneck7")
                
                net =slim.conv2d(inputs=net, num_outputs=2, kernel_size=[1, 1], activation_fn=tf.nn.softmax, scope = "final")
                self._net_intermediate_results['binary_segment_logits'] = {
                        'data': net,
                        'shape': net.get_shape().as_list()}
            with tf.variable_scope('instance_seg'):
                net = self._net_intermediate_results['shared_encoding']['net']
                
                res6 = slim.conv2d(inputs=res6, num_outputs=320, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_6']['shape'][1],self._net_intermediate_results['residual_6']['shape'][2]])
                net = tf.add(net,res6)
                net = self.blocks(net=net, expansion=1, output_filters=64, repeat=1, stride=1, name = "bottleneck1")

                res5 = slim.conv2d(inputs=res5, num_outputs=64, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_5']['shape'][1],self._net_intermediate_results['residual_5']['shape'][2]])
                net = tf.add(net,res5)
                net = self.blocks(net=net, expansion=1, output_filters=64, repeat=1, stride=1, name = "bottleneck2")

                res4 = slim.conv2d(inputs=res4, num_outputs=64, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_4']['shape'][1],self._net_intermediate_results['residual_4']['shape'][2]])
                net = tf.add(net,res4)
                net = self.blocks(net=net, expansion=1, output_filters=64, repeat=1, stride=1, name = "bottleneck3")

                res3 = slim.conv2d(inputs=res3, num_outputs=64, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_3']['shape'][1],self._net_intermediate_results['residual_3']['shape'][2]])
                net = tf.add(net,res3)
                net = self.blocks(net=net, expansion=1, output_filters=64, repeat=1, stride=1,name = "bottleneck4")
                
                res2 = slim.conv2d(inputs=res2, num_outputs=64, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_2']['shape'][1],self._net_intermediate_results['residual_2']['shape'][2]])
                net = tf.add(net,res2)
                net = self.blocks(net=net, expansion=1, output_filters=64, repeat=1, stride=1,name = "bottleneck5")
                
                res1 = slim.conv2d(inputs=res1, num_outputs=64, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_1']['shape'][1],self._net_intermediate_results['residual_1']['shape'][2]])
                net = tf.add(net,res1)
                net = self.blocks(net=net, expansion=1, output_filters=64, repeat=1, stride=1, name = "bottleneck6")

                net = tf.image.resize_images(net, [CFG.TRAIN.IMG_HEIGHT,CFG.TRAIN.IMG_WIDTH])
                net = self.blocks(net=net, expansion=1, output_filters=64, repeat=1, stride=1, name = "bottleneck7")

                self._net_intermediate_results['instance_segment_logits'] = {
                        'data': net,
                        'shape': net.get_shape().as_list()}
    def build_model(self, input_tensor, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            # ENET fcn encode part
            self._mobilenet_fcn_encode(inputs=input_tensor, name='mobilenet_encode_module')
            # vgg16 fcn decode part
            self._mobilenet_fcn_decode(name='mobilenet_decode_module')

        return self._net_intermediate_results

if __name__ == '__main__':
    """
    test code
    """
    test_in_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
    model = MOBILENETFCN(phase='train')
    ret = model.build_model(input_tensor=test_in_tensor, name='mobilenetfcn')
    for layer_name, layer_info in ret.items():
        print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))
