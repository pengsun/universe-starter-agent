import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="VALID", dtype=tf.float32, collections=None):
    """ conv layer, valid padding, init like Torch 7"""
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        bound = 1.0 / np.sqrt(fan_in)

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-bound, bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], dtype, tf.random_uniform_initializer(-bound, bound),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def linear(x, size, name):
    """ linear layer, init like Torch 7"""
    fan_in = int(x.get_shape()[1])
    bound = 1.0 / np.sqrt(fan_in)
    w_shape = [x.get_shape()[1], size]
    w = tf.get_variable(name + "/w", w_shape, initializer=tf.random_uniform_initializer(-bound, bound))
    b = tf.get_variable(name + "/b", [size], initializer=tf.random_uniform_initializer(-bound, bound))
    return tf.matmul(x, w) + b


def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


class Convx2LSTMActorCritic(object):
    def __init__(self, ob_space, ac_space):
        # screen input
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        # conv block I
        x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))

        # conv block II
        x = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))

        #
        x = flatten(x)

        # linear layer
        x = linear(x, 256, 'fc')

        # add singleton batch dim for LSTM time axis
        x = tf.expand_dims(x, [0])

        # LSTM layer
        size = 256
        lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])

        # output: policy
        self.logits = linear(x, ac_space, "action")
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        # output: value
        self.vf = tf.reshape(linear(x, 1, "value"), [-1])
        # output: LSTM states
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

        # collect all parameters
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]
