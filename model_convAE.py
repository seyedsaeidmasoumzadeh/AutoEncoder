from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np


class AutoEncoder(object):
    def __init__(self):
        # Training Parameters
        self.learning_rate = 0.001

        # Network Parameters
        self.num_input = 30000 # data input (img shape: 100*100*3)

        # tf Graph input
        self.X = tf.placeholder("float", [None, self.num_input])

        # Building the encoder
        def encoder(x):
            # ------------ Convolution Layer -------------
            W_c1 = tf.get_variable(name='W_c1', shape=[5, 5, 3, 25],
                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b_c1 = tf.get_variable(name='b_c1', shape=[25],
                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            out_c1 = tf.nn.conv2d(x, W_c1, strides=[1, 1, 1, 1], padding='SAME')
            out_c1 = tf.nn.bias_add(out_c1, b_c1)
            # Max Pooling Layer -------------
            out_p1 = tf.nn.max_pool(out_c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Dropout Layer
            out_do1 = tf.nn.dropout(out_p1, keep_prob=0.75)
            out_do1_reshaped = tf.reshape(out_do1, shape=[-1, 50 * 50 * 25])
            # ------------ Fully Connected Layer #1 -------------
            input_size = out_do1_reshaped.shape[1:]
            input_size = int(np.prod(input_size))
            W_fc1 = tf.get_variable(name='w_fc1', shape=[input_size, 50 * 50 * 5],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b_fc1 = tf.get_variable(name='b_fc1', shape=[50 * 50 * 5],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            input_fc1 = tf.reshape(out_do1_reshaped, [-1, input_size])
            out_fc1 = tf.nn.relu(tf.add(tf.matmul(input_fc1, W_fc1), b_fc1))
            # ------------ Dropout Layer -------------
            out_do2 = tf.nn.dropout(out_fc1, keep_prob=0.75)
            # ------------ Fully Connected Layer #2 -------------
            input_size = out_do2.shape[1:]
            input_size = int(np.prod(input_size))
            W_fc2 = tf.get_variable(name='W_fc2', shape=[input_size, 50 * 50 ],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b_fc2 = tf.get_variable(name='b_fc2', shape=[50 * 50],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            input_fc2 = tf.reshape(out_do1_reshaped, [-1, input_size])
            out_fc2 = tf.nn.relu(tf.add(tf.matmul(input_fc2, W_fc2), b_fc2))

            return out_fc2

        # Building the decoder
        def decoder(x):
            # ------------ Fully Connected Layer #3 -------------
            input_size = x.shape[1:]
            input_size = int(np.prod(input_size))
            W_fc3 = tf.get_variable(name='W_fc3', shape=[input_size, 50 * 50 * 5],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b_fc3 = tf.get_variable(name='b_fc3', shape=[50 * 50 * 5],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            input_fc3 = tf.reshape(x, [-1, input_size])
            out_fc3 = tf.nn.relu(tf.add(tf.matmul(input_fc3, W_fc3), b_fc3))
            # ------------ Dropout Layer -------------
            out_do3 = tf.nn.dropout(out_fc3, keep_prob=0.75)
            # ------------ Fully Connected Layer #4 -------------
            input_size = out_do3.shape[1:]
            input_size = int(np.prod(input_size))
            W_fc4 = tf.get_variable(name='W_fc4', shape=[input_size, 50 * 50 * 25],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b_fc4 = tf.get_variable(name='b_fc4', shape=[50 * 50 * 25],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            input_fc4 = tf.reshape(x, [-1, input_size])
            out_fc4 = tf.nn.relu(tf.add(tf.matmul(input_fc4, W_fc4), b_fc4))
            # ------------ Dropout Layer -------------
            out_do4 = tf.nn.dropout(out_fc4, keep_prob=0.75)
            out_do4 = tf.reshape(out_do4, shape=[-1, 50, 50, 25])
            # ------------ De-Convolution Layer -------------
            out_dc1 = tf.contrib.layers.conv2d_transpose(out_do4, num_outputs=25, kernel_size=[5, 5],
                                                         stride=[1, 1], padding='SAME',
                                                         weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(
                                                             uniform=False), biases_initializer=None,
                                                         activation_fn=tf.nn.relu)
            # ------------ Up-sampling Layer -------------
            upsampling = tf.keras.layers.UpSampling2D([2, 2])
            out_up1 = upsampling.apply(out_dc1)
            # ------------ Fully Connected Layer #5 -------------
            input_size = out_up1.shape[1:]
            input_size = int(np.prod(input_size))
            W_fc5 = tf.get_variable(name='W_fc5', shape=[input_size, 100 * 100 * 3],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b_fc5 = tf.get_variable(name='b_fc5', shape=[100 * 100 * 3],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            input_fc5 = tf.reshape(out_up1, [-1, input_size])
            out_fc5 = tf.nn.relu(tf.add(tf.matmul(input_fc5, W_fc5), b_fc5))

            return out_fc5

        # Construct model
        input = tf.reshape(self.X, shape=[-1, 100, 100, 3])
        self.encoder_op = encoder(input)
        self.decoder_op = decoder(self.encoder_op)

        # Prediction
        self.y_pred = self.decoder_op
        # True output is the input data in auto-encoder
        self.y_true = self.X

        # Define loss and optimizer
        self.loss = tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()
