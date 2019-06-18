import collections
import tensorflow as tf
import numpy
from semantic_segmentation_zoo import cnn_basenet
from config import global_config

CFG = global_config.cfg


class ENETFCN(cnn_basenet.CNNBaseModel):
    def __init__(self, phase):
        """

        """
        super(ENETFCN, self).__init__()
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

    def _enet_fcn_encode(self,inputs,
             name='shared_encoder', reuse=False):

        with tf.variable_scope(name, reuse=reuse):
            # Set the primary arg names. Fused batch_norm is faster than normal batch norm.
            with tf.contrib.framework.arg_scope([self.initial_block, self.bottleneck], training=self._is_training), \
                 tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm], fused=True), \
                 tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.conv2d_transpose], activation_fn=None):
                # =================INITIAL BLOCK=================
                net = self.initial_block(self, inputdata=inputs, name='initial_block_1')
                #net = self.initial_block(self, inputdata=net, name='initial_block_' )

                # Save for skip connection later
                if self._skip_connections:
                    net_one = net

                # ===================STAGE ONE=======================
                net, pooling_indices_1, inputs_shape_1 = self.bottleneck(self, net, output_depth=64, filter_size=3,
                                                                    regularizer_prob=0.01, downsampling=True,
                                                                    name='bottleneck1_0')
                net = self.bottleneck(self,net, output_depth=64, filter_size=3, regularizer_prob=0.01, name='bottleneck1_1')
                net = self.bottleneck(self,net, output_depth=64, filter_size=3, regularizer_prob=0.01, name='bottleneck1_2')
                net = self.bottleneck(self,net, output_depth=64, filter_size=3, regularizer_prob=0.01, name='bottleneck1_3')
                net = self.bottleneck(self,net, output_depth=64, filter_size=3, regularizer_prob=0.01, name='bottleneck1_4')

                # Save for skip connection later
                if self._skip_connections:
                    net_two = net

                # regularization prob is 0.1 from bottleneck 2.0 onwards
                with tf.contrib.framework.arg_scope([self.bottleneck], regularizer_prob=0.1):
                    # ===================STAGE TWO========================

                    net, pooling_indices_2, inputs_shape_2 = self.bottleneck(self,net, output_depth=128, filter_size=3,
                                                                        downsampling=True, name='bottleneck2_0')
                    net = self.bottleneck(self,net, output_depth=128, filter_size=3, name='bottleneck' + '2_1')
                    net = self.bottleneck(self,net, output_depth=128, filter_size=3, dilated=True, dilation_rate=2,
                                         name='bottleneck' + '2_2')
                    net = self.bottleneck(self,net, output_depth=128, filter_size=5, asymmetric=True,
                                         name='bottleneck' + '2_3')
                    net = self.bottleneck(self,net, output_depth=128, filter_size=3, dilated=True, dilation_rate=4,
                                         name='bottleneck' + '2_4')
                    net = self.bottleneck(self,net, output_depth=128, filter_size=3, name='bottleneck' + '2_5')
                    net = self.bottleneck(self,net, output_depth=128, filter_size=3, dilated=True, dilation_rate=8,
                                         name='bottleneck' +  '2_6')
                    net = self.bottleneck(self,net, output_depth=128, filter_size=5, asymmetric=True,
                                         name='bottleneck'  + '2_7')
                    net = self.bottleneck(self,net, output_depth=128, filter_size=3, dilated=True, dilation_rate=16,
                                         name='bottleneck'  + '2_8')

                    self._net_intermediate_results['shared_encoding'] = {
                        'net': net,
                        'net1': net_one,
                        'net2': net_two,
                        'pooling_indices_1': pooling_indices_1,
                        'pooling_indices_2': pooling_indices_2,
                        'inputs_shape_1': inputs_shape_1,
                        'inputs_shape_2': inputs_shape_2,
                        'shape': net.get_shape().as_list()

                    }
                    return

    def _enet_fcn_decode(self, name='decode', skip_connections=True):

        with tf.variable_scope(name):
            with tf.contrib.framework.arg_scope([self.bottleneck], training=self._is_training):
                net = self._net_intermediate_results['shared_encoding']['net']
                net_one = self._net_intermediate_results['shared_encoding']['net1']
                net_two = self._net_intermediate_results['shared_encoding']['net2']
                pooling_indices_1 = self._net_intermediate_results['shared_encoding']['pooling_indices_1']
                pooling_indices_2 = self._net_intermediate_results['shared_encoding']['pooling_indices_2']
                inputs_shape_1 = self._net_intermediate_results['shared_encoding']['inputs_shape_1']
                inputs_shape_2 = self._net_intermediate_results['shared_encoding']['inputs_shape_2']

                with tf.variable_scope('binary_seg'):
                    with tf.contrib.framework.arg_scope([self.bottleneck], regularizer_prob=0.1):
                    # ===================STAGE THREE========================
                        binary_net = self.bottleneck(self,net, output_depth=128, filter_size=3, name='bottleneck'  + '3_1')
                        binary_net = self.bottleneck(self,binary_net, output_depth=128, filter_size=3, dilated=True, dilation_rate=2,
                                          name='bottleneck'  + '3_2')
                        binary_net = self.bottleneck(self,binary_net, output_depth=128, filter_size=5, asymmetric=True,
                                          name='bottleneck'  + '3_3')
                        binary_net = self.bottleneck(self,binary_net, output_depth=128, filter_size=3, dilated=True, dilation_rate=4,
                                          name='bottleneck'  + '3_4')
                        binary_net = self.bottleneck(self,binary_net, output_depth=128, filter_size=3, name='bottleneck'  + '3_5')
                        binary_net = self.bottleneck(self,binary_net, output_depth=128, filter_size=3, dilated=True, dilation_rate=8,
                                          name='bottleneck'  + '3_6')
                        binary_net = self.bottleneck(self,binary_net, output_depth=128, filter_size=5, asymmetric=True,
                                          name='bottleneck'  + '3_7')
                        binary_net = self.bottleneck(self,binary_net, output_depth=128, filter_size=3, dilated=True, dilation_rate=16,
                                          name='bottleneck'  + '_38')

                    #=====================DECODER==========================

                    with tf.contrib.framework.arg_scope([self.bottleneck], regularizer_prob=0.1, decoder=True):
                        # ===================STAGE FOUR========================
                        bottleneck_name_name = "bottleneck4"

                        # The decoder section, so start to upsample.
                        binary_net = self.bottleneck(self,binary_net, output_depth=64, filter_size=3, upsampling=True,
                                     pooling_indices=pooling_indices_2, output_shape=inputs_shape_2,
                                     name=bottleneck_name_name + '_0')

                        # Perform skip connections here
                        if skip_connections:
                            binary_net = tf.add(binary_net, net_two, name=bottleneck_name_name + '_skip_connection')

                        binary_net = self.bottleneck(self,binary_net, output_depth=64, filter_size=3, name=bottleneck_name_name + '_1')
                        binary_net = self.bottleneck(self,binary_net, output_depth=64, filter_size=3, name=bottleneck_name_name + '_2')

                        # ===================STAGE FIVE========================
                        bottleneck_name_name = "bottleneck5"

                        binary_net = self.bottleneck(self,binary_net, output_depth=16, filter_size=3, upsampling=True,
                                     pooling_indices=pooling_indices_1, output_shape=inputs_shape_1,
                                     name=bottleneck_name_name + '_0')

                        #    perform skip connections here
                        if skip_connections:
                            binary_net = tf.add(binary_net, net_one, name=bottleneck_name_name + '_skip_connection')

                        binary_net = self.bottleneck(self,binary_net, output_depth=16, filter_size=3, name=bottleneck_name_name + '_1')

                # =============FINAL CONVOLUTION=============
            logits = tf.contrib.layers.conv2d_transpose(binary_net, CFG.TRAIN.CLASSES_NUMS, [2, 2], stride=2, scope='fullconv')
            self._net_intermediate_results['binary_segment_logits'] = {
                'data': logits,
                'shape': logits.get_shape().as_list()
            }

            with tf.variable_scope('instance_seg'):
                with tf.contrib.framework.arg_scope([self.bottleneck], training=self._is_training):
                    with tf.contrib.framework.arg_scope([self.bottleneck], regularizer_prob=0.1):
                        # ===================STAGE THREE========================
                        instance_net = self.bottleneck(self,net, output_depth=128, filter_size=3, name='bottleneck' + '3_1')
                        instance_net = self.bottleneck(self,instance_net, output_depth=128, filter_size=3, dilated=True, dilation_rate=2,
                          name='bottleneck' + '3_2')
                        instance_net = self.bottleneck(self,instance_net, output_depth=128, filter_size=5, asymmetric=True,
                          name='bottleneck' + '3_3')
                        instance_net = self.bottleneck(self,instance_net, output_depth=128, filter_size=3, dilated=True, dilation_rate=4,
                          name='bottleneck' + '3_4')
                        instance_net = self.bottleneck(self,instance_net, output_depth=128, filter_size=3, name='bottleneck' + '3_5')
                        instance_net = self.bottleneck(self,instance_net, output_depth=128, filter_size=3, dilated=True, dilation_rate=8,
                          name='bottleneck' + '3_6')
                        instance_net = self.bottleneck(self,instance_net, output_depth=128, filter_size=5, asymmetric=True,
                          name='bottleneck' + '3_7')
                        instance_net = self.bottleneck(self,instance_net, output_depth=128, filter_size=3, dilated=True, dilation_rate=16,
                          name='bottleneck' + '_38')

                    # =====================DECODER==========================

                    with tf.contrib.framework.arg_scope([self.bottleneck], regularizer_prob=0.1, decoder=True):
                        # ===================STAGE FOUR========================
                        bottleneck_name_name = "bottleneck4"

                        # The decoder section, so start to upsample.
                        instance_net = self.bottleneck(self, instance_net, output_depth=64, filter_size=3, upsampling=True,
                          pooling_indices=pooling_indices_2, output_shape=inputs_shape_2,
                          name=bottleneck_name_name + '_0')

                        # Perform skip connections here
                        if skip_connections:
                            instance_net = tf.add(instance_net, net_two, name=bottleneck_name_name + '_skip_connection')

                        instance_net = self.bottleneck(self,instance_net, output_depth=64, filter_size=3, name=bottleneck_name_name + '_1')
                        instance_net = self.bottleneck(self,instance_net, output_depth=64, filter_size=3, name=bottleneck_name_name + '_2')

                        # ===================STAGE FIVE========================
                        bottleneck_name_name = "bottleneck5"

                        instance_net = self.bottleneck(self,instance_net, output_depth=16, filter_size=3, upsampling=True,
                          pooling_indices=pooling_indices_1, output_shape=inputs_shape_1,
                          name=bottleneck_name_name + '_0')

                        # perform skip connections here
                        if skip_connections:
                            instance_net = tf.add(instance_net, net_one, name=bottleneck_name_name + '_skip_connection')

                        instance_net = self.bottleneck(self,instance_net, output_depth=16, filter_size=3, name=bottleneck_name_name + '_1')
                        # =============FINAL CONVOLUTION=============
                        instance_logits = tf.contrib.layers.conv2d_transpose(instance_net, 4, [2, 2],
                                                                    stride=2, scope='fullconv')  #nb of filters tested [4,64]
                self._net_intermediate_results['instance_segment_logits'] = {
                    'data': instance_logits,
                    'shape': instance_logits.get_shape().as_list()
                }

    def build_model(self, input_tensor, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            # ENET fcn encode part
            self._enet_fcn_encode(inputs=input_tensor, name='enet_encode_module')
            # vgg16 fcn decode part
            self._enet_fcn_decode(name='enet_decode_module')

        return self._net_intermediate_results

if __name__ == '__main__':
    """
    test code
    """
    test_in_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 512, 512, 3], name='input')
    model = ENETFCN(phase='train')
    ret = model.build_model(test_in_tensor, name='enetfcn')
    for layer_name, layer_info in ret.items():
        print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))