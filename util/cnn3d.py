from __future__ import division
import tensorflow as tf

def conv3d_layer(name,bottom,output_dim, kernel_size=3,depth=3,stride=1,temporal_stride=1,padding='SAME',bias_term=True,weights_initializer=None,biases_initializer=None):
    print name+" input shape: "+str(bottom.get_shape().as_list())
    input_dim = bottom.get_shape().as_list()[-1]

    with tf.variable_scope(name):
        if weights_initializer is None and biases_initializer is None:
            if weights_initializer is None:
                weights_initializer=tf.random_normal_initializer()
            if bias_term and biases_initializer is None:
                biases_initializer = tf. constant_initializer(0.)

            weights = tf.get_variable("weights",
                [depth,kernel_size, kernel_size, input_dim, output_dim],
                initializer=weights_initializer)
            if bias_term:
                biases = tf.get_variable("biases", output_dim,
                    initializer=biases_initializer)

            print str(weights.name)+" initialized as random or retrieved from graph"
            if bias_term:
                print biases.name+" initialized as random or retrieved from graph"

        else:
            weights=tf.get_variable("weights",shape=None,initializer=weights_initializer)
            if bias_term:
                biases = tf.get_variable("biases", shape=None,
                    initializer=biases_initializer)

            print weights.name+" initialized from pre-trained parameters or retrieved from graph"
            if bias_term:
                print biases.name+" initialized from pre-trained parameters or retrieved from graph"


    conv = tf.nn.conv3d(bottom, filter=weights,
        strides=[1, temporal_stride,stride, stride, 1], padding=padding)
    if bias_term:
        conv = tf.nn.bias_add(conv, biases)
    return conv

def conv3d_relu_layer(name, bottom,output_dim,kernel_size=3,depth=3,stride=1,temporal_stride=1,padding='SAME',bias_term=True,weights_initializer=None,biases_initializer=None):
    conv = conv3d_layer(name, bottom, output_dim,kernel_size,depth, stride,temporal_stride, padding,bias_term, weights_initializer, biases_initializer)
    relu = tf.nn.relu(conv)
    return relu

def pooling3d_layer(name,bottom,kernel_size,depth,stride,temporal_stride,padding='SAME'):
    print name+" input shape: "+str(bottom.get_shape().as_list())
    pool = tf.nn.max_pool3d(bottom, ksize=[1, depth,kernel_size, kernel_size, 1],
        strides=[1, temporal_stride,stride, stride, 1], padding=padding, name=name)
    return pool

def fc_layer(name, bottom, output_dim, bias_term=True, weights_initializer=None,
             biases_initializer=None):
    # flatten bottom input
    # input has shape [batch, in_height, in_width, in_channels]
    shape = bottom.get_shape().as_list()
    input_dim = 1
    for d in shape[1:]:
        input_dim *= d
    flat_bottom = tf.reshape(bottom, [-1, input_dim])

    # weights and biases variables
    with tf.variable_scope(name):
        if weights_initializer is None and biases_initializer is None:
            # initialize the variables
            if weights_initializer is None:
                weights_initializer = tf.random_normal_initializer()
            if bias_term and biases_initializer is None:
                biases_initializer = tf.constant_initializer(0.)

            # weights has shape [input_dim, output_dim]
            weights = tf.get_variable("weights", [input_dim, output_dim],
                initializer=weights_initializer)
            if bias_term:
                biases = tf.get_variable("biases", output_dim,
                    initializer=biases_initializer)

            print weights.name+" initialized as random or retrieved from graph"
            if bias_term:
                print biases.name+" initialized as random or retrieved from graph"
        else:
            weights = tf.get_variable("weights", shape=None,
                initializer=weights_initializer)
            if bias_term:
                biases = tf.get_variable("biases", shape=None,
                    initializer=biases_initializer)

            print weights.name+" initialized from pre-trained parameters or retrieved from graph"
            if bias_term:
                print biases.name+" initialized from pre-trained parameters or retrieved from graph"

    if bias_term:
        fc = tf.nn.xw_plus_b(flat_bottom, weights, biases)
    else:
        fc = tf.matmul(flat_bottom, weights)
    return fc

def fc_relu_layer(name, bottom, output_dim, bias_term=True,
                  weights_initializer=None, biases_initializer=None):
    fc = fc_layer(name, bottom, output_dim, bias_term, weights_initializer,
                  biases_initializer)
    relu = tf.nn.relu(fc)
    return relu




