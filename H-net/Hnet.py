import collections
import tensorflow as tf
import numpy as np




class HNET():
    def __init__(self, phase):
        """

        """
        super(HNET, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._net_intermediate_results = collections.OrderedDict()

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

    def convblock(self, inputs, filters, size, name, stride=1):
        net = tf.layers.conv2d(inputs, filters, [size,size], strides=stride, padding='SAME', name=name + '_conv1')
        net = tf.contrib.layers.batch_norm(net, is_training= self._is_training, scope=name + '_batch_norm1')
        net = tf.nn.relu(net, name= name + '_relu1')
        self._net_intermediate_results[name + '_linear+BN+ReLU'] = {
            'shape': net.get_shape().as_list()}


        net = tf.layers.conv2d(net, filters, [size, size], strides=stride, padding='SAME', name=name + '_conv2')
        net = tf.contrib.layers.batch_norm(net, is_training=self._is_training, scope=name + '_batch_norm2')
        net = tf.nn.relu(net, name=name + '_relu2')
        self._net_intermediate_results[name + '_linear+BN+ReLU_2'] = {
            'shape': net.get_shape().as_list()}

        net = tf.layers.max_pooling2d(net, [2,2], 2, padding='SAME', name=name + '_maxpool')
        self._net_intermediate_results[name + 'maxpool'] = {
            'shape': net.get_shape().as_list()}


        return net

    def hnet(self, inputs, name='hnet'):

        with tf.variable_scope(name):
            hnet = self.convblock(inputs, 16, 3, name ='block1')
            hnet = self.convblock(hnet, 32, 3, name ='block2')
            hnet = self.convblock(hnet, 64, 3, name='block3')

            hnet = tf.layers.flatten(hnet)
            hnet = tf.layers.dense(hnet, 1024, name='linear1')
            hnet = tf.contrib.layers.batch_norm(hnet, is_training=self._is_training, scope=name + '_batch_norm1')
            hnet = tf.nn.relu(hnet, name=name + '_relu1')
            self._net_intermediate_results['linear+BN+ReLU'] = {
                'shape': hnet.get_shape().as_list()}

            hnet = tf.layers.dense(hnet, 6, name='final')
            self._net_intermediate_results['linear'] = {
                'shape': hnet.get_shape().as_list()}

        return self._net_intermediate_results

if __name__ == '__main__':
    """
    test code
    """
    test_in_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 128, 64, 3], name='input')
    model = HNET(phase='train')
    ret = model.hnet(test_in_tensor, name='h-net')
    for layer_name, layer_info in ret.items():
        print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))