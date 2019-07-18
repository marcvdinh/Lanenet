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
    def block_upsample(self, net, output_filters, stride, name):
        
        block = tf.contrib.layers.separable_conv2d(inputs=net, 
        num_outputs=None, 
        kernel_size=[3, 3],
        weights_regularizer=tf.contrib.layers.l2_regularizer(0.0004), 
        weights_initializer=tf.keras.initializers.truncated_normal(),
        padding ="SAME",
        activation_fn = None,
        #bias_initializer=tf.keras.initializers.constant(),
        data_format = "NCHW",
        stride=[stride,stride],
        scope = name + "_conv33")
        
        block= tf.layers.batch_normalization(block, axis=1, training = self._is_training)
        block = tf.nn.relu6(block)
        
        block = tf.layers.conv2d(inputs=block, 
        filters=output_filters, 
        kernel_size=[1, 1], 
        data_format = "channels_first",
        activation=None, 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),
        padding="SAME",
        name = name + "_projection")

        block= tf.layers.batch_normalization(block, axis=1, training = self._is_training)
        block = tf.nn.relu6(block)
        return block

    def block(self, net, input_filters, output_filters, expansion, stride, name):
        res_block = net
        res_block = tf.layers.conv2d(inputs=net, 
        filters=input_filters * expansion, 
        kernel_size=[1, 1], 
        data_format = "channels_first",
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),
        padding="SAME",
         name = name + "_expansion")
        res_block = tf.layers.batch_normalization(res_block, axis=1, training = self._is_training)
        res_block = tf.nn.relu6(res_block)
        
        res_block = tf.contrib.layers.separable_conv2d(inputs=res_block, 
        num_outputs=None, 
        kernel_size=[3, 3],
        activation_fn = None,
        weights_regularizer=tf.contrib.layers.l2_regularizer(0.0004), 
        weights_initializer=tf.keras.initializers.truncated_normal(),
        #pointwise_initializer=None,
        padding ="SAME",
        #bias_initializer=tf.keras.initializers.constant(),
        data_format = "NCHW",stride=[stride,stride], 
        scope = name + "_conv33")
        res_block = tf.layers.batch_normalization(res_block, axis=1, training = self._is_training)
        res_block = tf.nn.relu6(res_block)
        
        res_block = tf.layers.conv2d(inputs=res_block, 
        filters=output_filters, 
        kernel_size=[1, 1], 
        data_format = "channels_first",
        activation=None, 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),
        padding="SAME",
        name = name + "_projection")
        res_block = tf.layers.batch_normalization(res_block, axis=1, training = self._is_training)
        if stride == 2:
            return res_block
        else:
            if input_filters != output_filters:
                net = tf.layers.conv2d(inputs=net, filters=output_filters, kernel_size=[1, 1], data_format ="channels_first", padding="SAME",kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),activation=None, name = name + "_conv11")
                res_block = tf.layers.batch_normalization(res_block, axis=1, training = self._is_training)
                res_block = tf.nn.relu6(res_block)
            return tf.add(res_block, net)


    def blocks(self, net, expansion, output_filters, repeat, stride, name):
        input_filters = net.shape[1].value

        # first layer should take stride into account
        net = self.block( net=net, input_filters=input_filters, output_filters=output_filters, expansion=expansion, stride=stride, name=name)

        for i in range(1, repeat):
            net = self.block( net=net, input_filters=input_filters, output_filters=output_filters, expansion=expansion, stride=1, name = name + str(i))
        return net

    def _mobilenet_fcn_encode(self,inputs,
                 dropout_keep_prob=0.999,
                 depth_multiplier=1.0,
                 name='encode'):

        expansion = 6

        with tf.variable_scope(name):
           
            with slim.arg_scope([slim.dropout], is_training=self._is_training, keep_prob=0.99):


                assert inputs.get_shape().as_list()[1] == 3
                net = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[3, 3], data_format ="channels_first", padding="SAME",kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),activation=None, name = name + "_3conv11")

                net = self.blocks(net=net, expansion=1, output_filters=64, repeat=1, stride=2, name = "bottleneck1")
                self._net_intermediate_results['residual_1'] = { 'data' : net, 'shape': net.get_shape().as_list()}
                net = self.blocks(net=net, expansion=expansion, output_filters=128, repeat=2, stride=2,name = "bottleneck2")
                self._net_intermediate_results['residual_2'] = { 'data' : net,'shape': net.get_shape().as_list()}
                net = self.blocks(net=net, expansion=expansion, output_filters=256, repeat=2, stride=2, name = "bottleneck3")
                self._net_intermediate_results['residual_3'] = { 'data' : net,'shape': net.get_shape().as_list()}
                net = self.blocks(net=net, expansion=expansion, output_filters=512, repeat=6, stride=2, name = "bottleneck4")
                self._net_intermediate_results['residual_4'] = { 'data' : net,'shape': net.get_shape().as_list()}
                net = self.blocks(net=net, expansion=expansion, output_filters=1024, repeat=2, stride=2, name = "bottleneck5")


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
        
        with  slim.arg_scope([slim.dropout], is_training=self._is_training, keep_prob=0.99):
             

            with tf.variable_scope('binary_seg'):
                net = self._net_intermediate_results['shared_encoding']['net']

                res4 = tf.layers.conv2d(inputs=res4,
                                        filters=1024,
                                        kernel_size=[1, 1],
                                        data_format = "channels_first",
                                        activation=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),
                                        padding="SAME",
                                        )
                net = tf.layers.conv2d_transpose(inputs = net, filters = 1024, kernel_size = [2,2], padding = "SAME", data_format="channels_first",strides =2)
                net = tf.add(net,res4)
                net = self.block_upsample(net=net,  output_filters=512,  stride=1,name = "up1")

                res3 = tf.layers.conv2d(inputs=res3, 
                                        filters=512,
                                        kernel_size=[1, 1], 
                                        data_format = "channels_first",
                                        activation=None, 
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),
                                        padding="SAME",
                                        )
                net = tf.layers.conv2d_transpose(inputs = net, filters = 512, kernel_size = [2,2], padding = "SAME", data_format="channels_first",strides =2)
                net = tf.add(net,res3)
                net = self.block_upsample(net=net,  output_filters=256,  stride=1,name = "up2")
                
                res2 = tf.layers.conv2d(inputs=res2, 
                                        filters=256,
                                        kernel_size=[1, 1], 
                                        data_format = "channels_first",
                                        activation=None, 
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),
                                        padding="SAME",
                                        )
                net = tf.layers.conv2d_transpose(inputs = net, filters = 256, kernel_size = [2,2], padding = "SAME", data_format="channels_first",strides =2)
                net = tf.add(net,res2)
                net = self.block_upsample(net=net,  output_filters=128,  stride=1,name = "up3")
                
                res1 = tf.layers.conv2d(inputs=res1, 
                                        filters=128,
                                        kernel_size=[1, 1], 
                                        data_format = "channels_first",
                                        activation=None, 
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),
                                        padding="SAME",
                                        )
                net = tf.layers.conv2d_transpose(inputs = net, filters = 128, kernel_size = [2,2], padding = "SAME", data_format="channels_first",strides =2)
                net = tf.add(net,res1)
                net = self.block_upsample(net=net,  output_filters=64,  stride=1,name = "up4")
                
                net = tf.layers.conv2d_transpose(inputs = net, filters = 64, kernel_size = [2,2], padding = "SAME", data_format="channels_first",strides =2)
                net = self.block_upsample(net=net,  output_filters=64,  stride=1,name = "up5")
                net = tf.layers.conv2d(inputs=net, 
                                        filters=2, 
                                        kernel_size=[1, 1], 
                                        data_format = "channels_first",
                                        activation=None, 
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),
                                        padding="SAME",
                                        name = name + "_projection")
                self._net_intermediate_results['binary_segment_logits'] = {
                        'data': net,
                        'shape': net.get_shape().as_list()}
            
            
            with tf.variable_scope('instance_seg'):
                net = self._net_intermediate_results['shared_encoding']['net']

                res4 = tf.layers.conv2d(inputs=res4,
                                        filters=1024,
                                        kernel_size=[1, 1],
                                        data_format = "channels_first",
                                        activation=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),
                                        padding="SAME",
                                        )
                net = tf.layers.conv2d_transpose(inputs = net, filters = 1024, kernel_size = [2,2], padding = "SAME", data_format="channels_first",strides =2)
                net = tf.add(net,res4)
                net = self.block_upsample(net=net,  output_filters=512,  stride=1,name = "up1")
                

                res3 = tf.layers.conv2d(inputs=res3, 
                                        filters=512,
                                        kernel_size=[1, 1], 
                                        data_format = "channels_first",
                                        activation=None, 
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),
                                        padding="SAME",
                                        )
                net = tf.layers.conv2d_transpose(inputs = net, filters = 512, kernel_size = [2,2], padding = "SAME", data_format="channels_first",strides =2)
                net = tf.add(net,res3)
                net = self.block_upsample(net=net,  output_filters=256,  stride=1,name = "up2")
                
                
                res2 = tf.layers.conv2d(inputs=res2, 
                                        filters=256,
                                        kernel_size=[1, 1], 
                                        data_format = "channels_first",
                                        activation=None, 
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),
                                        padding="SAME",
                                        )
                net = tf.layers.conv2d_transpose(inputs = net, filters = 256, kernel_size = [2,2], padding = "SAME", data_format="channels_first",strides =2)
                net = tf.add(net,res2)
                net = self.block_upsample(net=net,  output_filters=128,  stride=1,name = "up3")
                
                
                res1 = tf.layers.conv2d(inputs=res1, 
                                        filters=128,
                                        kernel_size=[1, 1], 
                                        data_format = "channels_first",
                                        activation=None, 
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004),
                                        padding="SAME",
                                        )
                net = tf.layers.conv2d_transpose(inputs = net, filters = 128, kernel_size = [2,2], padding = "SAME", data_format="channels_first",strides =2)
                net = tf.add(net,res1)
                net = self.block_upsample(net=net,  output_filters=64,  stride=1,name = "up4")
                
                
                net = tf.layers.conv2d_transpose(inputs = net, filters = 64, kernel_size = [2,2], padding = "SAME", data_format="channels_first",strides =2)
                net = self.block_upsample(net=net,  output_filters=64,  stride=1, name = "up5")
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
    test_in_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 3, 256, 512], name='input')
    model = MOBILENETFCN(phase='train')
    ret = model.build_model(input_tensor=test_in_tensor, name='mobilenetfcn')
    for layer_name, layer_info in ret.items():
        print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))
