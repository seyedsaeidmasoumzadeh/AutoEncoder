from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class AutoEncoder(object):
    def __init__(self):
        # Training Parameters
        self.learning_rate = 0.001

        # Network Parameters
        self.num_hidden_1 = 128  # 1st layer num features
        self.num_hidden_2 = 96  # 2st layer num features
        self.num_input = 30000 # data input (img shape: 3*28*28)

        # tf Graph input (only pictures)
        self.X = tf.placeholder("float", [None, self.num_input])

        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.num_input, self.num_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_input]))
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([self.num_input]))

        }


        # Building the encoder
        def encoder(x):
            # Encoder Hidden layer with sigmoid activation #1
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,self.weights['encoder_h1']),
                                           self.biases['encoder_b1']))
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,self.weights['encoder_h2']),
                                           self.biases['encoder_b2']))
            return layer_2

        # Building the decoder
        def decoder(x):
            # Decoder Hidden layer with sigmoid activation #1
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                           self.biases['decoder_b1']))
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                           self.biases['decoder_b2']))
            return layer_2

        # Construct model
        self.encoder_op = encoder(self.X)
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
