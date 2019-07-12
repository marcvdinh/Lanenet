import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy
from semantic_segmentation_zoo import cnn_basenet
from config import global_config

CFG = global_config.cfg

class MOBILENETV1FCN(cnn_basenet.CNNBaseModel):
    def __init__(self, phase):
        """

        """
        super(MOBILENETV1FCN, self).__init__()
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
    
    def block(self, net, kernel, stride, depth, name):
        
        net = slim.separable_conv2d(inputs=net, kernel_size=kernel,num_outputs=None, activation_fn =tf.nn.relu6, stride=stride, scope = name + "_depthwise")
        net = slim.conv2d(inputs=net, num_outputs=depth, kernel_size=[1, 1], activation_fn=tf.nn.relu6,stride=1, scope = name + "_pointwise")
        return net
    
    def _mobilenet_fcn_encode(self,inputs,
                 dropout_keep_prob=0.999,
                 depth_multiplier=1.0,
                 spatial_squeeze=True,
                 name='encode'):

        

        with tf.variable_scope(name):
            regularizer = tf.contrib.layers.l2_regularizer(0.0004)
            with slim.arg_scope([slim.conv2d ], weights_regularizer=regularizer),\
             slim.arg_scope([slim.separable_conv2d],weights_regularizer=None, depth_multiplier=1.0),\
             slim.arg_scope([slim.dropout], is_training=self._is_training, keep_prob=0.99),\
             slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': self._is_training, 'center': True, 'scale': True }):

                net = self.conv2d(inputs, 32, 3, name='conv11', stride=2)

                net = self.block(net, kernel=[3,3], stride=1, depth =64, name ="depthsepconv1")
                self._net_intermediate_results['residual_1'] = { 'data' : net, 'shape': net.get_shape().as_list()}

                net = self.block(net, kernel=[3,3], stride=2, depth =128,name ="depthsepconv2_1")
                net = self.block(net, kernel=[3,3], stride=1, depth =128, name ="depthsepconv2_2")
                self._net_intermediate_results['residual_2'] = { 'data' : net, 'shape': net.get_shape().as_list()}

                net = self.block(net, kernel=[3,3], stride=2, depth =256, name ="depthsepconv3_1")
                net = self.block(net, kernel=[3,3], stride=1, depth =256, name ="depthsepconv3_2")
                self._net_intermediate_results['residual_3'] = { 'data' : net, 'shape': net.get_shape().as_list()}

                net = self.block(net, kernel=[3,3], stride=2, depth =512, name ="depthsepconv4_1")
                net = self.block(net, kernel=[3,3], stride=1, depth =512, name ="depthsepconv4_2")
                net = self.block(net, kernel=[3,3], stride=1, depth =512, name ="depthsepconv4_3")
                net = self.block(net, kernel=[3,3], stride=1, depth =512, name ="depthsepconv4_4")
                net = self.block(net, kernel=[3,3], stride=1, depth =512, name ="depthsepconv4_5")
                self._net_intermediate_results['residual_4'] = { 'data' : net, 'shape': net.get_shape().as_list()}

                net = self.block(net, kernel=[3,3], stride=2, depth =1024, name ="depthsepconv5_1")
                net = self.block(net, kernel=[3,3], stride=1, depth =1024, name ="depthsepconv5_2")


                self._net_intermediate_results['shared_encoding'] = {
                        'net': net,
                        'shape': net.get_shape().as_list()}
            
    def _mobilenet_fcn_decode(self, name="decode"):
        
        shape = self._net_intermediate_results['shared_encoding']['shape']
        res1 = self._net_intermediate_results['residual_1']['data']
        res2 = self._net_intermediate_results['residual_2']['data']
        res3 = self._net_intermediate_results['residual_3']['data']
        res4 = self._net_intermediate_results['residual_4']['data']
        

        regularizer = tf.contrib.layers.l2_regularizer(0.0004)
        
        with slim.arg_scope([slim.conv2d ],activation_fn=tf.nn.relu, weights_regularizer=regularizer),\
             slim.arg_scope([slim.separable_conv2d],weights_regularizer=None, depth_multiplier=1.0),\
             slim.arg_scope([slim.dropout], is_training=self._is_training, keep_prob=0.99),\
             slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': self._is_training, 'center': True, 'scale': True }):

            with tf.variable_scope('binary_seg'):
                net = self._net_intermediate_results['shared_encoding']['net']
                
                res4 = slim.conv2d(inputs=res4, num_outputs=1024, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_4']['shape'][1],self._net_intermediate_results['residual_4']['shape'][2]])
                net = tf.add(net,res4)
                net = self.block(net=net,kernel=[3,3], depth=64,  stride=1, name = "depthsepconv1")

                res3 = slim.conv2d(inputs=res3, num_outputs=64, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_3']['shape'][1],self._net_intermediate_results['residual_3']['shape'][2]])
                net = tf.add(net,res3)
                net = self.block(net=net,kernel=[3,3], depth=64,  stride=1, name = "depthsepconv2")

                res2 = slim.conv2d(inputs=res2, num_outputs=64, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_2']['shape'][1],self._net_intermediate_results['residual_2']['shape'][2]])
                net = tf.add(net,res2)
                net = self.block(net=net,kernel=[3,3], depth=64,  stride=1, name = "depthsepconv3")

                res1 = slim.conv2d(inputs=res1, num_outputs=64, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_1']['shape'][1],self._net_intermediate_results['residual_1']['shape'][2]])
                net = tf.add(net,res1)
                net = self.block(net=net,kernel=[3,3], depth=64,  stride=1, name = "depthsepconv4")
                
                net = tf.image.resize_images(net, [CFG.TRAIN.IMG_HEIGHT,CFG.TRAIN.IMG_WIDTH])
                net =slim.conv2d(inputs=net, num_outputs=2, kernel_size=[1, 1], activation_fn=tf.nn.softmax, scope = "final")
                self._net_intermediate_results['binary_segment_logits'] = {
                        'data': net,
                        'shape': net.get_shape().as_list()}
            with tf.variable_scope('instance_seg'):
                net = self._net_intermediate_results['shared_encoding']['net']
                
                res4 = slim.conv2d(inputs=res4, num_outputs=1024, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_4']['shape'][1],self._net_intermediate_results['residual_4']['shape'][2]])
                net = tf.add(net,res4)
                net = self.block(net=net,kernel=[3,3], depth=64,  stride=1, name = "depthsepconv1")

                res3 = slim.conv2d(inputs=res3, num_outputs=64, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_3']['shape'][1],self._net_intermediate_results['residual_3']['shape'][2]])
                net = tf.add(net,res3)
                net = self.block(net=net,kernel=[3,3], depth=64, stride=1, name = "depthsepconv2")

                res2 = slim.conv2d(inputs=res2, num_outputs=64, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_2']['shape'][1],self._net_intermediate_results['residual_2']['shape'][2]])
                net = tf.add(net,res2)
                net = self.block(net=net,kernel=[3,3], depth=64,  stride=1, name = "depthsepconv3")

                res1 = slim.conv2d(inputs=res1, num_outputs=64, kernel_size=[1, 1], activation_fn=None)
                net = tf.image.resize_images(net, [self._net_intermediate_results['residual_1']['shape'][1],self._net_intermediate_results['residual_1']['shape'][2]])
                net = tf.add(net,res1)
                net = self.block(net=net,kernel=[3,3], depth=64,  stride=1, name = "depthsepconv4")

                net = tf.image.resize_images(net, [CFG.TRAIN.IMG_HEIGHT,CFG.TRAIN.IMG_WIDTH])
                net = self.block(net=net, kernel=[3,3],depth=64, stride = 1, name = "bottleneck7")
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
