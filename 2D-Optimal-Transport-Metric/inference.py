from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sklearn

from tensorflow.contrib.layers import flatten
import sys
import numpy as np


slim = tf.contrib.slim


class siamese:

    # Create model
    def __init__(self, sess, margin=10., batch_size=30, lamb=10.0, lossType='wassersteinContrastiveLoss'):
        # lossType: 1 for my loss, 2 for standard loss
        self.margin = margin
        self.batch_size = batch_size
        self.lamb = lamb
        self.lossType = lossType

        with tf.variable_scope("siamese") as scope:
            self.o1,  _ = self.lenet(self.x1)
            #self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2,  _ = self.lenet(self.x2)
            #self.o2 = self.network(self.x2)
        

        self.label1 = tf.placeholder(tf.float32, [None])
        self.label2 = tf.placeholder(tf.float32, [None])
        self.simLabelGeneration(self.label1, self.label2)
        self.build_model() #
    
    
    def network(self, x):
        weights = []
        fc1 = self.fc_layer(x, 1024, "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1, 1024, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.fc_layer(ac2, 2, "fc3")
        return fc3

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc
    
    def lenet(self, images, num_classes=256, is_training=False,
          dropout_keep_prob=0.5,
          prediction_fn=slim.softmax,
          scope='LeNet'):
      
        end_points = {}

        with tf.variable_scope(scope, 'LeNet', [images]):
            net = end_points['conv1'] = slim.conv2d(images, 32, [5, 5], scope='conv1')
            net = end_points['pool1'] = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
            net = end_points['conv2'] = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = end_points['pool2'] = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
            net = slim.flatten(net)
            end_points['Flatten'] = net

            net = end_points['fc3'] = slim.fully_connected(net, 1024, scope='fc3')
            if not num_classes:
                return net, end_points
            net = end_points['dropout3'] = slim.dropout(
                net, dropout_keep_prob, is_training=is_training, scope='dropout3')
            logits = end_points['Logits'] = slim.fully_connected(
                net, num_classes, activation_fn=None, scope='fc4')

        #end_points['Predictions'] = prediction_fn(logits, scope='Predictions')        
        #lenet.default_image_size = 28

        return logits, end_points
        
        
    def lenet_arg_scope(self, weight_decay=0.0):
      """Defines the default lenet argument scope.
      Args:
        weight_decay: The weight decay to use for regularizing the model.
      Returns:
        An `arg_scope` to use for the inception v3 model.
      """
      with slim.arg_scope(
          [slim.conv2d, slim.fully_connected],
          weights_regularizer=slim.l2_regularizer(weight_decay),
          weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
          activation_fn=tf.nn.relu) as sc:
        return sc

    

    def calculateGroundMetric(self, groupFea1, groupFea2):
        """
        calculate the ground metric between sketch and shape
        """
        square_groupFea1 = tf.reduce_sum(tf.square(groupFea1), axis=1)
        square_groupFea1 = tf.expand_dims(square_groupFea1, axis=1)

        square_groupFea2 = tf.reduce_sum(tf.square(groupFea2), axis=1)
        square_groupFea2 = tf.expand_dims(square_groupFea2, axis=0)

        correlationTerm = tf.matmul(groupFea1, tf.transpose(groupFea2, perm=[1, 0]))

        groundMetricRaw = tf.add(tf.subtract(square_groupFea1, tf.multiply(2., correlationTerm)), square_groupFea2)

        # Flatten the groundMetric as a vector
        groundMetricRawFlatten = tf.reshape(groundMetricRaw, [-1])
        return groundMetricRawFlatten, groundMetricRaw
    
    
    
    
    def calculateGroundMetricContrastive(self, groupFea1, groupFea2):
        """
        calculate the ground metric between sketch and shape
        """
        square_groupFea1 = tf.reduce_sum(tf.square(groupFea1), axis=1)
        square_groupFea1 = tf.expand_dims(square_groupFea1, axis=1)

        square_groupFea2 = tf.reduce_sum(tf.square(groupFea2), axis=1)
        square_groupFea2 = tf.expand_dims(square_groupFea2, axis=0)

        correlationTerm = tf.matmul(groupFea1, tf.transpose(groupFea2, perm=[1, 0]))

        groundMetric = tf.add(tf.subtract(square_groupFea1, tf.multiply(2., correlationTerm)), square_groupFea2)

        # Get ground metric cost for negative pair 
        hinge_groundMetric = tf.maximum(0., self.margin -  groundMetric)

        GM_positivePair = tf.multiply(self.simLabelMatrix, groundMetric)
        GM_negativePair = tf.multiply(1 - self.simLabelMatrix, hinge_groundMetric) 

        GM = tf.add(GM_positivePair, GM_negativePair)

        expGM = tf.exp(tf.multiply(-10., GM))    # This is for optimizing "T"

        # Flatten the groundMetric as a vector
        GMFlatten = tf.reshape(GM, [-1])

        return GMFlatten, expGM
    
    
     
    def sinkhornIter(self, groundMetric):
        with tf.variable_scope('sinkhorn') as scope:
            print(groundMetric.get_shape())
            u0 = tf.constant(1./self.batch_size, shape=[self.batch_size, 1], name='u0')
    
            groundMetric_ = tf.transpose(groundMetric, perm=[1, 0])      # transpose the ground metric
            u = tf.get_variable(name='u', shape=[self.batch_size, 1], initializer=tf.constant_initializer(1.0))
            v = tf.get_variable(name='v', shape=[self.batch_size, 1], initializer=tf.constant_initializer(1.0))
            

            v_assign_op = v.assign(tf.div(u0, tf.matmul(tf.exp(tf.multiply(-self.lamb, groundMetric_)), u) + 1e-10))
            u_assign_op = u.assign(tf.div(u0, tf.matmul(tf.exp(tf.multiply(-self.lamb, groundMetric)), v) + 1e-10))
            u_reset = u.assign(tf.constant(1.0, shape=[self.batch_size, 1], name='reset_u'))
            T = tf.matmul(tf.matmul(tf.diag(tf.reshape(u, [-1])), tf.exp(tf.multiply(-self.lamb, groundMetric))), tf.diag(tf.reshape(v, [-1])), name='T')
            T_flatten = tf.reshape(T, [-1])
            #T_flatten = tf.constant(1.0/(self.batch_size*self.batch_size), shape=[self.batch_size*self.batch_size]) 


            # This is simple for debugging
            debug1 = tf.reduce_mean(tf.matmul(tf.exp(tf.multiply(-self.lamb, groundMetric_)), u))
            debug2 = tf.reduce_sum(T)

            return v_assign_op, u_assign_op, u, v, T, T_flatten, u_reset, debug1, debug2
        

    def simLabelGeneration(self, label1, label2):
        label1 = tf.tile(tf.expand_dims(label1, 1), [1, self.batch_size])
        label2 = tf.tile(tf.expand_dims(label2, 1), [self.batch_size, 1])

        self.simLabel = tf.cast(tf.equal(tf.reshape(label1, [-1]), tf.reshape(label2, [-1])), tf.float32) 
        self.simLabelMatrix = tf.reshape(self.simLabel, [self.batch_size, self.batch_size])

    def contrastiveLoss(self):
        #### for the siamese network 1      #############
        #### Check data type tensorflow #################      
        # 1 for similar pair, 0 for non-similar
        
        #label_p = self.simLabel
        label_p = tf.diag_part(self.simLabelMatrix)
        label_n = tf.subtract(1.0, self.simLabelMatrix)
        #o1_tile = tf.reshape(tf.tile(self.o1, [self.batch_size, 1]), [-1, 1])
        #o2_tile = tf.reshape(tf.tile(self.o2, [1, self.batch_size]), [-1, 1])
        
        euclidean_p = tf.reduce_sum(tf.square(tf.subtract(self.o1, self.o2)), axis=1)               # |f(x1) - f(x2)|^2
        euclidean_n = self.margin - tf.sqrt((euclidean_p+1e-10))
        loss_p = tf.reduce_mean(tf.multiply(label_p, euclidean_p), name='loss_p') 
        loss_n = tf.reduce_mean(tf.multiply(label_n, tf.square(tf.maximum(0., euclidean_n)), name='loss_n'))
        loss = loss_p + loss_n
        
        return loss_p, loss_n, loss
        

    def wassersteinContrastiveLoss(self): # used loss
        #### for the siamese network 1      #############

        loss = tf.multiply(self.T_flatten, tf.pow(self.groundMetricFlatten, 2))# the final loss (sumation)
        loss_p = tf.multiply(self.T_flatten, tf.multiply(self.simLabel, tf.pow(self.groundMetricFlatten, 2)))
        loss_n = tf.multiply(self.T_flatten, tf.multiply(1-self.simLabel, tf.pow(self.groundMetricFlatten, 2)))

        return tf.reduce_sum(loss_p), tf.reduce_sum(loss_n), tf.reduce_sum(loss)


    def build_model(self):
        if self.lossType == "contrastiveLoss_gxdai" or self.lossType == "contrastiveLoss_standard":

            self.loss_p, self.loss_n, self.loss  = self.contrastiveLoss()
            self.loss_sum = tf.summary.scalar('loss', self.loss)

        elif self.lossType == "inverseWassersteinCL": #

            # calculate ground metric network
            self.groundMetricFlatten, self.groundMetric  = self.calculateGroundMetricContrastive(self.o1, self.o2)  #

            # sinkhorn iterations
            self.v_assign_op, self.u_assign_op, self.u, self.v, self.T, self.T_flatten, self.u_reset, self.debug1, self.debug2 = self.sinkhornIter(self.groundMetric)
    
            var_list = tf.trainable_variables()
            #for var in var_list:
                #print(var.name)
            print(">>>>>>>>>")
            self.var_list = [var for var in var_list if "siamese" in var.name]
            #for var in self.var_list:
                #print(var.name)
            print(">>>>>>>>>>>")
            self.loss_p, self.loss_n, self.loss  = self.wassersteinContrastiveLoss()
            self.standard_loss_p, self.standard_loss_n, self.standard_loss  = self.contrastiveLoss()
            self.loss_sum = tf.summary.scalar('loss', self.loss)



