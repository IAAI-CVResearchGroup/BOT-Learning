# ======================================
# Filename: model.py
# Description: Build computational graph and train.
# The loss types in this file include:
# (1) Batch-wise optimal-transport distance loss
# (2) Contrastive loss
# (3) Lifted Loss
# (4) n-pair Loss
# Author: Han Sun, Zhiyuan Chen, Lin Xu
#
# Project: Optimal Transport for Cross-modality Retrieval
# Github: https://github.com/IAAI-CVResearchGroup/Batch-wise-Optimal-Transport-Metric/tree/master/shrec14
# Copyright (C): IAAI
# Code:
import tensorflow as tf
import numpy as np
import numpy.matlib
from scipy.spatial import distance
import os
import sys
from data import Dataset
import time
from RetrievalEvaluation import RetrievalEvaluation
import pandas as pd
import pdb
from tensorflow.contrib.losses.python.metric_learning import metric_loss_ops

# tf.set_random_seed(222)
# np.random.seed(222)

class model(Dataset):
    """ Create the model for weighted contrastive loss"""

    def __init__(self, lamb=20., ckpt_dir='./checkpoint', ckpt_name='model',
                 batch_size=30, margin=10., learning_rate=0.0001, momentum=0.9, sketch_train_list=None,
                 sketch_test_list=None, shape_list=None, num_views_shape=12, class_num=90, normFlag=0,
                 logdir=None, lossType='contrastiveLoss', activationType='relu', phase='train', inputFeaSize=4096,
                 outputFeaSize=100, maxiter=100000):

        """
        lamb:           The parameter for sinkhorn iteration
        ckpt_dir:       The directory for saving checkpoint file
        ckpt_name:      The name of checkpoint file
        batch_size:     The training batch_size
        margin:         The margin for contrastive loss
        learning_rate:  The learning rate
        momentum:       The momentum
        sketch_train_list:  The list file of training sketch
        sketch_test_list:   The list file of testing sketch
        shape_list:         The list file of shape
        num_views_shape:    The total number of shape views
        class_num:          The total nubmer of classes
        normFlag:           1 for normalizing input features, 0 for not normalizing input features
        logdir:             The log directory
        lossType:           choosing the loss function
        activationType:     The activation function
        phase:              choosing training or testing
        inputFeaSize:       The dimension of input features
        outputFeaSize:      The dimension of output features
        maxiter:            The maximum number of iterations
        """

        self.lamb = lamb
        self.ckpt_dir = ckpt_dir
        self.ckpt_name = ckpt_name
        self.batch_size = batch_size
        self.logdir = logdir
        self.class_num = class_num
        self.num_views_shape = num_views_shape
        self.maxiter = maxiter
        self.inputFeaSize = inputFeaSize
        self.outputFeaSize = outputFeaSize
        self.margin = margin
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.lossType = lossType
        self.phase = phase
        self.normFlag = normFlag
        self.activationType = activationType

        print("self.lamb               =       {:2.5f}".format(self.lamb))
        print("self.ckpt_dir           =       {:10}".format(self.ckpt_dir))
        print("self.ckpt_name          =       {:10}".format(self.ckpt_name))
        print("self.batch_size         =       {:5d}".format(self.batch_size))
        print("self.logdir             =       {:10}".format(self.logdir))
        print("self.class_num          =       {:5d}".format(self.class_num))
        print("self.num_views_shape    =       {:5d}".format(self.num_views_shape))
        print("self.maxiter            =       {:5d}".format(self.maxiter))
        print("self.inputFeaSize       =       {:5d}".format(self.inputFeaSize))
        print("self.outputFeaSize      =       {:5d}".format(self.outputFeaSize))
        print("self.margin             =       {:2.5f}".format(self.margin))
        print("self.learning_rate      =       {:2.5f}".format(self.learning_rate))
        print("self.momentum           =       {:2.5f}".format(self.momentum))
        print("self.lossType           =       {:10}".format(self.lossType))
        print("self.phase              =       {:10}".format(self.phase))
        print("self.normFlag           =       {:10d}".format(self.normFlag))
        print("self.activationType     =       {:10}".format(self.activationType))

        # class inheritance from Dataset class
        Dataset.__init__(self, sketch_train_list=sketch_train_list, sketch_test_list=sketch_test_list,
                         shape_list=shape_list, num_views_shape=num_views_shape, feaSize=inputFeaSize,
                         class_num=class_num, phase=phase, normFlag=normFlag)

        self.build_model()

    def sketchNetwork(self, x):
        """
        network for sketch domain (metric network)

        note: We tried two different activation functions, i.e. sigmoid, relu
        """
        if self.activationType == 'relu':
            stddev = 0.01
            fc1 = self.fc_layer(x, 2000, "fc1", stddev)
            ac1 = tf.nn.relu(fc1)
            fc2 = self.fc_layer(ac1, 1000, "fc2", stddev)
            ac2 = tf.nn.relu(fc2)
            fc3 = self.fc_layer(ac2, 500, "fc3", 0.1)
            fc4 = self.fc_layer(fc3, self.outputFeaSize, "fc4", 0.1)

            return fc4, fc3
        elif self.activationType == 'sigmoid':
            stddev = 0.1
            fc1 = self.fc_layer(x, 2000, "fc1", stddev)
            ac1 = tf.nn.sigmoid(fc1)
            fc2 = self.fc_layer(ac1, 1000, "fc2", stddev)
            ac2 = tf.nn.sigmoid(fc2)
            fc3 = self.fc_layer(ac2, self.outputFeaSize, "fc3", stddev)
            ac3 = tf.nn.sigmoid(fc3)

            return ac3, ac2

    def shapeNetwork(self, x):
        """
        network for shape domain

        note: We tried two different activation functions, i.e. simgmoid, relu
        """
        if self.activationType == 'relu':
            stddev = 0.01
            fc1 = self.fc_layer(x, 2000, "fc1", stddev)
            ac1 = tf.nn.relu(fc1)
            fc2 = self.fc_layer(ac1, 2000, "fc2", stddev)
            ac2 = tf.nn.relu(fc2)
            fc3 = self.fc_layer(ac2, 1000, "fc3", stddev)
            ac3 = tf.nn.relu(fc3)
            fc4 = self.fc_layer(ac3, 500, "fc4", 0.1)
            fc5 = self.fc_layer(fc4, self.outputFeaSize, "fc5", 0.1)

            return fc5, fc4
        elif self.activationType == 'sigmoid':
            stddev = 0.1
            fc1 = self.fc_layer(x, 2000, "fc1", stddev)
            ac1 = tf.nn.sigmoid(fc1)
            fc2 = self.fc_layer(ac1, 1000, "fc2", stddev)
            ac2 = tf.nn.relu(fc2)
            fc3 = self.fc_layer(ac2, 500, "fc3", stddev)
            ac3 = tf.nn.sigmoid(fc3)
            fc4 = self.fc_layer(ac3, self.outputFeaSize, "fc4", stddev)
            ac4 = tf.nn.sigmoid(fc4)

            return ac4, ac3

    def fc_layer(self, bottom, n_weight, name, stddev):
        # assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[-1]
        initer = tf.truncated_normal_initializer(stddev=stddev)
        W = tf.get_variable(name + 'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        ## rescale weight by 0.1
        #    W = tf.mul(.1, W)
        b = tf.get_variable(name + 'b', dtype=tf.float32,
                            initializer=tf.constant(0.0, shape=[n_weight], dtype=tf.float32))
        ## rescale biase by 0.1
        #    b = tf.mul(.1, b)
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)

        return fc

    def calculateGroundMetric(self, sketch_fea, shape_fea):
        """
        calculate the ground metric between sketch and shape
        """
        square_sketch_fea = tf.reduce_sum(tf.square(sketch_fea), axis=2)
        square_sketch_fea = tf.expand_dims(square_sketch_fea, axis=2)

        square_shape_fea = tf.reduce_sum(tf.square(shape_fea), axis=2)

        square_shape_fea = tf.expand_dims(square_shape_fea, axis=1)

        correlationTerm = tf.matmul(sketch_fea, tf.transpose(shape_fea, perm=[0, 2, 1]))

        groundMetricRaw = tf.add(tf.subtract(square_sketch_fea, tf.multiply(2., correlationTerm)), square_shape_fea)

        # Flatten the groundMetric as a vector
        groundMetricRaw = tf.reshape(groundMetricRaw, [self.batch_size, -1])

        return groundMetricRaw

    def calculateGroundMetricContrastive(self, batchFea1, batchFea2, labelMatrix):
        """
        calculate the ground metric between two batch of features
        """

        # Print the tensor shape for debug

        # print("Calculate the ground metrics between two batches of features")
        # print("batchFea1.get_shape().as_list() = {}".format(batchFea1.get_shape().as_list()))
        # print("batchFea2.get_shape().as_list() = {}".format(batchFea2.get_shape().as_list()))
        # print("labelMatrix.get_shape().as_list() = {}".format(labelMatrix.get_shape().as_list()))
        # ============== Euclidean Distance ==============
        squareBatchFea1 = tf.reduce_sum(tf.square(batchFea1), axis=1)
        squareBatchFea1 = tf.expand_dims(squareBatchFea1, axis=1)

        squareBatchFea2 = tf.reduce_sum(tf.square(batchFea2), axis=1)
        squareBatchFea2 = tf.expand_dims(squareBatchFea2, axis=0)

        correlationTerm = tf.matmul(batchFea1, tf.transpose(batchFea2, perm=[1, 0]))
        groundMetric = tf.add(tf.subtract(squareBatchFea1, tf.multiply(2., correlationTerm)), squareBatchFea2)
        # =================================================

        # Get ground metric cost for negative pair
        hinge_groundMetric = tf.maximum(0., self.margin - groundMetric)

        GM_positivePair = tf.multiply(labelMatrix, groundMetric)
        GM_negativePair = tf.multiply(1 - labelMatrix, hinge_groundMetric)
        GM = tf.add(GM_positivePair, GM_negativePair)
        # TODO Modified from -1 to -10.
        # It can be set as you want. We think $-10$ is OK for most of works.
        expGM = tf.exp(tf.multiply(-10., GM))  # This is for optimizing "T"
        # Flatten the groundMetric as a vector
        GMFlatten = tf.reshape(GM, [-1])
        # print("expGM.get_shape().as_list() = {}".format(expGM.get_shape().as_list()))

        return GMFlatten, expGM

    def simLabelGeneration(self, label1, label2):
        """
        The similarity between a pair of labels.
        """
        # batch_size * batch_size
        label1 = tf.tile(label1, [1, self.batch_size])
        # (batch_size * batch_size) * 1
        label2 = tf.tile(label2, [self.batch_size, 1])
        simLabel = tf.cast(tf.equal(tf.reshape(label1, [-1]), tf.reshape(label2, [-1])), tf.float32)
        simLabelMatrix = tf.reshape(simLabel, [self.batch_size, self.batch_size])

        return simLabelMatrix

    def sinkhornIter(self, groundMetric, name='sinkhorn'):
        with tf.variable_scope(name) as scope:
            # u0 = tf.constant(1./self.gmWidth, shape=[self.gmWidth, 1], name='u0')
            u0 = tf.constant(1. / self.batch_size, shape=[self.batch_size, 1], name='u0')
            print("groundMetric.get_shape().as_list() = {}".format(groundMetric.get_shape().as_list()))
            groundMetric_ = tf.transpose(groundMetric, perm=[1, 0])  # transpose the ground metric

            epsilon = 1e-12

            u = tf.get_variable(name='u', shape=[self.batch_size, 1], initializer=tf.constant_initializer(1.0))
            v = tf.get_variable(name='v', shape=[self.batch_size, 1], initializer=tf.constant_initializer(1.0))

            v_assign_op = v.assign(tf.div(u0, tf.matmul(tf.exp(tf.multiply(-self.lamb, groundMetric_)), u) + epsilon))
            u_assign_op = u.assign(tf.div(u0, tf.matmul(tf.exp(tf.multiply(-self.lamb, groundMetric)), v) + epsilon))
            u_reset = u.assign(tf.constant(1.0, shape=[self.batch_size, 1], name='reset_u'))

            T = tf.matmul(tf.matmul(tf.diag(tf.reshape(u, [-1])), tf.exp(tf.multiply(-self.lamb, groundMetric))),
                          tf.diag(tf.reshape(v, [-1])), name='T')
            T_flatten = tf.reshape(T, [-1])

            debug = tf.reduce_sum(T)

            return v_assign_op, u_assign_op, u, v, T, T_flatten, u_reset, debug

    def contrastiveLoss(self, input_sketch, input_shape):
        # Original contrastive loss.
        pairwiseDistance = tf.reduce_sum(tf.square(input_sketch - input_shape), axis=2, name='pairwiseDistance')
        hinge_distance = tf.maximum(0., tf.subtract(self.margin, pairwiseDistance))

        print(pairwiseDistance.get_shape())
        print(hinge_distance.get_shape())
        print(self.simLabel.get_shape())
        self.positivePairDistance = tf.multiply(self.simLabel, pairwiseDistance)
        self.negativePairDistance = tf.multiply(1 - self.simLabel, hinge_distance)

        self.contrastiveDistance = self.positivePairDistance + self.negativePairDistance

        self.loss = tf.reduce_mean(tf.reduce_sum(self.contrastiveDistance, axis=1), axis=0)
        self.loss_summary = tf.summary.scalar('loss', self.loss)

    def wassersteinLoss(self, input_sketch, input_shape):
        # Batch-wise optimal transport learning loss
        self.groundMetricFlatten, self.groundMetric = self.calculateGroundMetricContrastive(input_sketch, input_shape)

        # sinkhorn iter
        self.v_assign_op, self.u_assign_op, self.u, self.v, self.T, self.T_flatten, self.u_reset, self.debug = self.sinkhornIter(
            self.groundMetric)

        # variable list for ground metric network
        self.loss = tf.reduce_sum(tf.multiply(self.groundMetricFlatten, self.T_flatten), name='loss')
        self.loss_summary = tf.summary.scalar('loss', self.loss)

    def build_network(self):
        if self.lossType == 'npairLoss':
            # Network for npair loss
            with tf.variable_scope('sketch') as scope:
                self.output_sketch_fea, self.output_sketch_debug = self.sketchNetwork(self.input_sketch_fea)
                scope.reuse_variables()

            with tf.variable_scope('shape') as scope:
                self.output_shape_fea, self.output_shape_debug = self.shapeNetwork(self.input_shape_fea)
                scope.reuse_variables()
        else:
            # Network for other loss objectives
            with tf.variable_scope('sketch') as scope:
                self.output_sketch_fea_1, self.output_sketch_debug_1 = self.sketchNetwork(self.input_sketch_fea_1)
                scope.reuse_variables()
                self.output_sketch_fea_2, self.output_sketch_debug_2 = self.sketchNetwork(self.input_sketch_fea_2)

            with tf.variable_scope('shape') as scope:
                self.output_shape_fea_1, self.output_shape_debug_1 = self.shapeNetwork(self.input_shape_fea_1)
                scope.reuse_variables()
                self.output_shape_fea_2, self.output_shape_debug_2 = self.shapeNetwork(self.input_shape_fea_2)

    def build_model(self):
        if self.lossType == 'LiftedLoss':
            # input sketch placeholder
            self.input_sketch_fea_1 = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputFeaSize],
                                                     name='input_sketch_fea_1')
            self.input_sketch_label_1 = tf.placeholder(tf.int32, shape=[self.batch_size], name='input_sketch_label_1')
            self.input_sketch_fea_2 = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputFeaSize],
                                                     name='input_sketch_fea_2')
            self.input_sketch_label_2 = tf.placeholder(tf.int32, shape=[self.batch_size], name='input_sketch_label_2')

            # input shape placeholder
            self.input_shape_fea_1 = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputFeaSize],
                                                    name='input_shape_fea_1')
            self.input_shape_label_1 = tf.placeholder(tf.int32, shape=[self.batch_size], name='input_shape_label_1')
            self.input_shape_fea_2 = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputFeaSize],
                                                    name='input_shape_fea_2')
            self.input_shape_label_2 = tf.placeholder(tf.int32, shape=[self.batch_size], name='input_shape_label_2')
        elif self.lossType == 'npairLoss':
            self.input_sketch_fea = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputFeaSize],
                                                   name='input_sketch_fea')
            self.input_sketch_label = tf.placeholder(tf.int32, shape=[self.batch_size], name='input_sketch_label')

            self.input_shape_fea = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputFeaSize],
                                                  name='input_shape_fea')
            self.input_shape_label = tf.placeholder(tf.int32, shape=[self.batch_size], name='input_shape_label')
        else:
            self.input_sketch_fea_1 = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputFeaSize],
                                                     name='input_sketch_fea_1')
            self.input_sketch_label_1 = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name='input_sketch_label_1')
            self.input_sketch_fea_2 = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputFeaSize],
                                                     name='input_sketch_fea_2')
            self.input_sketch_label_2 = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name='input_sketch_label_2')

            # input shape placeholder
            self.input_shape_fea_1 = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputFeaSize],
                                                    name='input_shape_fea_1')
            self.input_shape_label_1 = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name='input_shape_label_1')
            self.input_shape_fea_2 = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputFeaSize],
                                                    name='input_shape_fea_2')
            self.input_shape_label_2 = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name='input_shape_label_2')

            # similarity matrix for sketch
            self.simLabel_sketch = self.simLabelGeneration(self.input_sketch_label_1, self.input_sketch_label_2)

            # similarity matrix for shape
            self.simLabel_shape = self.simLabelGeneration(self.input_shape_label_1, self.input_shape_label_2)

            # similarity matrix for cross 1
            self.simLabel_cross_1 = self.simLabelGeneration(self.input_sketch_label_1, self.input_shape_label_1)

            # similarity matrix for cross 2
            self.simLabel_cross_2 = self.simLabelGeneration(self.input_sketch_label_2, self.input_shape_label_2)

        self.build_network()

        print(self.lossType)

        # Loss Type 1: Batch-wise Optimal Transport Learning Loss
        if self.lossType == 'weightedContrastiveLoss':

            print("Caculating the ground matrix")
            self.GM_sketch, self.expGM_sketch = self.calculateGroundMetricContrastive(self.output_sketch_fea_1,
                                                                                      self.output_sketch_fea_2,
                                                                                      self.simLabel_sketch)
            self.GM_shape, self.expGM_shape = self.calculateGroundMetricContrastive(self.output_shape_fea_1,
                                                                                    self.output_shape_fea_2,
                                                                                    self.simLabel_shape)
            self.GM_cross_1, self.expGM_cross_1 = self.calculateGroundMetricContrastive(self.output_sketch_fea_1,
                                                                                        self.output_shape_fea_1,
                                                                                        self.simLabel_cross_1)
            self.GM_cross_2, self.expGM_cross_2 = self.calculateGroundMetricContrastive(self.output_sketch_fea_2,
                                                                                        self.output_shape_fea_2,
                                                                                        self.simLabel_cross_2)

            print("Sinkhorn iteration construction")

            self.sketch_v_assign_op, self.sketch_u_assign_op, self.sketch_u, self.sketch_v, self.sketch_T, self.sketch_T_flatten, self.sketch_u_reset, self.sketch_debug = self.sinkhornIter(
                self.expGM_sketch, name='sinkhorn_sketch')
            self.shape_v_assign_op, self.shape_u_assign_op, self.shape_u, self.shape_v, self.shape_T, self.shape_T_flatten, self.shape_u_reset, self.shape_debug = self.sinkhornIter(
                self.expGM_shape, name='sinkhorn_shape')
            self.cross_1_v_assign_op, self.cross_1_u_assign_op, self.cross_1_u, self.cross_1_v, self.cross_1_T, self.cross_1_T_flatten, self.cross_1_u_reset, self.cross_1_debug = self.sinkhornIter(
                self.expGM_cross_1, name='sinkhorn_cross_1')
            self.cross_2_v_assign_op, self.cross_2_u_assign_op, self.cross_2_u, self.cross_2_v, self.cross_2_T, self.cross_2_T_flatten, self.cross_2_u_reset, self.cross_2_debug = self.sinkhornIter(
                self.expGM_cross_2, name='sinkhorn_cross_2')

            # create loss

            self.loss_sketch = tf.reduce_sum(tf.multiply(self.GM_sketch, self.sketch_T_flatten), name='sketch_loss')
            self.loss_shape = tf.reduce_sum(tf.multiply(self.GM_shape, self.shape_T_flatten), name='shape_loss')
            self.loss_cross_1 = tf.reduce_sum(tf.multiply(self.GM_cross_1, self.cross_1_T_flatten), name='cross_1_loss')
            self.loss_cross_2 = tf.reduce_sum(tf.multiply(self.GM_cross_2, self.cross_2_T_flatten), name='cross_2_loss')
            self.loss = tf.add_n([self.loss_sketch, self.loss_shape, self.loss_cross_1, self.loss_cross_2], name='loss')
            # distance used for retrieval

            self.loss_summary_sketch = tf.summary.scalar('sketch_loss', self.loss_sketch)
            self.loss_summary_shape = tf.summary.scalar('shape_loss', self.loss_shape)
            self.loss_summary_cross_1 = tf.summary.scalar('cross_loss_1', self.loss_cross_1)
            self.loss_summary_cross_2 = tf.summary.scalar('cross_loss_2', self.loss_cross_2)
            self.loss_summary = tf.summary.scalar('loss', self.loss)

        # Loss Type 2: Original Contrastive Loss
        elif self.lossType == 'contrastiveLoss':

            print("Caculating the ground matrix")
            self.GM_sketch, self.expGM_sketch = self.calculateGroundMetricContrastive(self.output_sketch_fea_1,
                                                                                      self.output_sketch_fea_2,
                                                                                      self.simLabel_sketch)
            self.GM_shape, self.expGM_shape = self.calculateGroundMetricContrastive(self.output_shape_fea_1,
                                                                                    self.output_shape_fea_2,
                                                                                    self.simLabel_shape)
            self.GM_cross_1, self.expGM_cross_1 = self.calculateGroundMetricContrastive(self.output_sketch_fea_1,
                                                                                        self.output_shape_fea_1,
                                                                                        self.simLabel_cross_1)
            self.GM_cross_2, self.expGM_cross_2 = self.calculateGroundMetricContrastive(self.output_sketch_fea_2,
                                                                                        self.output_shape_fea_2,
                                                                                        self.simLabel_cross_2)

            # create loss
            self.loss = tf.reduce_mean(tf.add_n([self.GM_sketch, self.GM_shape, self.GM_cross_1, self.GM_cross_2]),
                                       name='loss')
            # distance used for retrieval

            # self.loss_summary_sketch = tf.summary.scalar('sketch_loss', self.loss_sketch)
            # self.loss_summary_shape = tf.summary.scalar('shape_loss', self.loss_shape)
            # self.loss_summary_cross_1 = tf.summary.scalar('cross_loss_1', self.loss_cross_1)
            # self.loss_summary_cross_2 = tf.summary.scalar('cross_loss_2', self.loss_cross_2)
            self.loss_summary = tf.summary.scalar('loss', self.loss)

        # Loss Type 3: Lifted Loss
        elif self.lossType == 'LiftedLoss':
            self.sketch_embedding_1 = tf.concat([self.output_sketch_fea_1, self.output_sketch_fea_2], 0)
            self.sketch_label_1 = tf.concat([self.input_sketch_label_1, self.input_sketch_label_2], 0)

            self.shape_embedding_1 = tf.concat([self.output_shape_fea_1, self.output_shape_fea_2], 0)
            self.shape_label_1 = tf.concat([self.input_shape_label_1, self.input_shape_label_2], 0)

            self.cross_embedding_1 = tf.concat([self.output_sketch_fea_1, self.output_shape_fea_1], 0)
            self.cross_label_1 = tf.concat([self.input_sketch_label_1, self.input_shape_label_1], 0)

            self.cross_embedding_2 = tf.concat([self.output_sketch_fea_2, self.output_shape_fea_2], 0)
            self.cross_label_2 = tf.concat([self.input_sketch_label_2, self.input_shape_label_2], 0)

            self.loss_sketch = metric_loss_ops.lifted_struct_loss(self.sketch_label_1,
                                                                  self.sketch_embedding_1, margin=self.margin)
            self.loss_shape = metric_loss_ops.lifted_struct_loss(self.shape_label_1,
                                                                 self.shape_embedding_1, margin=self.margin)
            self.loss_cross_1 = metric_loss_ops.lifted_struct_loss(self.cross_label_1,
                                                                   self.cross_embedding_1, margin=self.margin)
            self.loss_cross_2 = metric_loss_ops.lifted_struct_loss(self.cross_label_2,
                                                                   self.cross_embedding_2, margin=self.margin)
            self.loss = tf.add_n([self.loss_sketch, self.loss_shape, self.loss_cross_1, self.loss_cross_2], name='loss')

            self.loss_summary_sketch = tf.summary.scalar('sketch_loss', self.loss_sketch)
            self.loss_summary_shape = tf.summary.scalar('shape_loss', self.loss_shape)
            self.loss_summary_cross_1 = tf.summary.scalar('cross_loss_1', self.loss_cross_1)
            self.loss_summary_cross_2 = tf.summary.scalar('cross_loss_2', self.loss_cross_2)
            self.loss_summary = tf.summary.scalar('loss', self.loss)
        # ===============================

        # Loss Type 4: n-pair loss
        elif self.lossType == 'npairLoss':
            self.loss = metric_loss_ops.npairs_loss(self.input_sketch_label, self.output_sketch_fea,
                                                    self.output_shape_fea, reg_lambda=0.0002)
            self.loss_summary = tf.summary.scalar('loss', self.loss)
        # ===============================

        var_list = tf.trainable_variables()
        self.gm_var_list = [var for var in var_list if ('sketch' in var.name or 'shape' in var.name)]
        # for var in self.gm_var_list:
        #    print(var.name)

    def ckpt_status(self):
        print("[*] Reading checkpoint ...")
        ckpt = tf.train.get_checkpoint_state(os.path.join(self.ckpt_dir, self.lossType))
        if ckpt and ckpt.model_checkpoint_path:
            self.model_checkpoint_path = ckpt.model_checkpoint_path
            return True
        else:
            return None

    def train(self):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 60000, 0.9)
        self.gm_optim = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=self.momentum).minimize(self.loss, var_list=self.gm_var_list)
        # self.shape_optim = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum).minimize(self.loss_shape, var_list=self.gm_var_list)
        # self.sketch_optim = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum).minimize(self.loss_sketch, var_list=self.gm_var_list)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        start_time = time.time()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            writer = tf.summary.FileWriter(self.logdir, sess.graph)
            sess.run(init)

            if self.ckpt_status():
                print("[*] Load SUCCESS")
                print(self.model_checkpoint_path)
                saver.restore(sess, self.model_checkpoint_path)
            else:
                print("[*] Load failed")

            mAP = []
            loss = []
            cost = []
            df = pd.DataFrame(columns=['mAP', 'Loss'])
            row_num = 0
            for iter in range(self.maxiter):

                # Load data
                if self.lossType == 'npairLoss':
                    sketch_label, shape_label, sketch_fea, shape_fea = self.nextBatch_npair(self.batch_size, 'sketch_train')
                else:
                    sketch_fea_1, sketch_label_1 = self.nextBatch(self.batch_size, 'sketch_train')
                    shape_fea_1, shape_label_1 = self.nextBatch(self.batch_size, 'shape')

                    sketch_fea_2, sketch_label_2 = self.nextBatch(self.batch_size, 'sketch_train')
                    shape_fea_2, shape_label_2 = self.nextBatch(self.batch_size, 'shape')

                # start training
                if self.lossType == 'contrastiveLoss':
                    _, loss_, loss_sum_, sketch_fea, shape_fea = sess.run(
                        [self.gm_optim, self.loss, self.loss_summary, self.output_sketch_fea_1,
                         self.output_shape_fea_1], feed_dict={
                            self.input_sketch_fea_1: sketch_fea_1,
                            self.input_sketch_label_1: sketch_label_1,
                            self.input_sketch_fea_2: sketch_fea_2,
                            self.input_sketch_label_2: sketch_label_2,
                            self.input_shape_fea_1: shape_fea_1,
                            self.input_shape_label_1: shape_label_1,
                            self.input_shape_fea_2: shape_fea_2,
                            self.input_shape_label_2: shape_label_2
                        })

                # This is the mode for weighted contrastive loss
                elif self.lossType == 'weightedContrastiveLoss':
                    M_sketch, M_shape, M_cross_1, M_cross_2 = sess.run(
                        [self.expGM_sketch, self.expGM_shape, self.expGM_cross_1, self.expGM_cross_2], feed_dict={
                            self.input_sketch_fea_1: sketch_fea_1,
                            self.input_sketch_label_1: sketch_label_1,
                            self.input_sketch_fea_2: sketch_fea_2,
                            self.input_sketch_label_2: sketch_label_2,
                            self.input_shape_fea_1: shape_fea_1,
                            self.input_shape_label_1: shape_label_1,
                            self.input_shape_fea_2: shape_fea_2,
                            self.input_shape_label_2: shape_label_2
                        })

                    # sinkhorn iteration
                    self.shape_u_reset.eval()
                    self.sketch_u_reset.eval()
                    self.cross_1_u_reset.eval()
                    self.cross_2_u_reset.eval()

                    for sinhorn_ter in range(20):
                        self.sketch_v_assign_op.eval(feed_dict={self.expGM_sketch: M_sketch})
                        self.sketch_u_assign_op.eval(feed_dict={self.expGM_sketch: M_sketch})

                        self.shape_v_assign_op.eval(feed_dict={self.expGM_shape: M_shape})
                        self.shape_u_assign_op.eval(feed_dict={self.expGM_shape: M_shape})

                        self.cross_1_v_assign_op.eval(feed_dict={self.expGM_cross_1: M_cross_1})
                        self.cross_1_u_assign_op.eval(feed_dict={self.expGM_cross_1: M_cross_1})

                        self.cross_2_v_assign_op.eval(feed_dict={self.expGM_cross_2: M_cross_2})
                        self.cross_2_u_assign_op.eval(feed_dict={self.expGM_cross_2: M_cross_2})

                    _, loss_, loss_sum_, sketch_fea, shape_fea, lr = sess.run(
                        [self.gm_optim, self.loss, self.loss_summary, self.output_sketch_fea_1,
                         self.output_shape_fea_1, learning_rate], feed_dict={
                            self.input_sketch_fea_1: sketch_fea_1,
                            self.input_sketch_label_1: sketch_label_1,
                            self.input_sketch_fea_2: sketch_fea_2,
                            self.input_sketch_label_2: sketch_label_2,
                            self.input_shape_fea_1: shape_fea_1,
                            self.input_shape_label_1: shape_label_1,
                            self.input_shape_fea_2: shape_fea_2,
                            self.input_shape_label_2: shape_label_2,
                            global_step: iter
                        })

                elif self.lossType == 'LiftedLoss':
                    _, loss_, loss_sum_, sketch_fea, shape_fea, lr = sess.run(
                        [self.gm_optim, self.loss, self.loss_summary, self.output_sketch_fea_1,
                         self.output_shape_fea_1, learning_rate], feed_dict={
                            self.input_sketch_fea_1: sketch_fea_1,
                            self.input_sketch_label_1: np.squeeze(sketch_label_1),
                            self.input_sketch_fea_2: sketch_fea_2,
                            self.input_sketch_label_2: np.squeeze(sketch_label_2),
                            self.input_shape_fea_1: shape_fea_1,
                            self.input_shape_label_1: np.squeeze(shape_label_1),
                            self.input_shape_fea_2: shape_fea_2,
                            self.input_shape_label_2: np.squeeze(shape_label_2),
                            global_step: iter
                        })

                elif self.lossType == 'npairLoss':
                    _, loss_, loss_sum_, sketch_fea_npair, shape_fea_npair, lr = sess.run(
                        [self.gm_optim, self.loss, self.loss_summary, self.output_sketch_fea,
                         self.output_shape_fea, learning_rate], feed_dict={
                            self.input_sketch_fea: sketch_fea,
                            self.input_sketch_label: np.squeeze(sketch_label),
                            self.input_shape_fea: shape_fea,
                            self.input_shape_label: np.squeeze(shape_label),
                            global_step: iter
                        })

                writer.add_summary(loss_sum_, iter)
                cost.append(loss_)
                # ========================================
                # TODO Each Epoch output Loss and mAP (One epoch = 2850 steps)
                # We show the first epoch, then every next 5 epochs
                if iter == 2849:
                    saver.save(sess, os.path.join(self.ckpt_dir, self.lossType, self.ckpt_name), global_step=iter)
                    mAP_ = self.evaluation_test(sess)
                    mAP.append(mAP_)
                    loss_epoch = np.array(cost).mean()
                    loss.append(loss_epoch)
                    print('Epoch:{}, mAP: {:.4f}, Loss: {:.4f}, LR: {:.7f}'.format(row_num, mAP_, loss_epoch, lr))
                    sys.stdout.flush()
                    df.loc[row_num] = [mAP_, loss_epoch]
                    row_num += 1
                    cost = []
                if (iter + 1) % (2850 * 5) == 0:
                    saver.save(sess, os.path.join(self.ckpt_dir, self.lossType, self.ckpt_name), global_step=iter)
                    mAP_ = self.evaluation_test(sess)
                    mAP.append(mAP_)
                    loss_epoch = np.array(cost).mean()
                    loss.append(loss_epoch)
                    print('Epoch:{}, mAP: {:.4f}, Loss: {:.4f}, LR: {:.7f}'.format(row_num, mAP_, loss_epoch, lr))
                    sys.stdout.flush()
                    df.loc[row_num] = [mAP_, loss_epoch]
                    row_num += 1
                    cost = []
                # ========================================
                if iter % 500 == 0:
                    print("Iteration: [%5d] [total number of examples: %5d] time: %4.4f, loss: %.8f" % (
                    iter, self.shape_num, time.time() - start_time, loss_))
                    sys.stdout.flush()

            with open('/home/sri-dev01/CVRP_New/CVPR_2019_Results_Sun.csv', 'w') as f:
                df.to_csv(f, index=False)

    def evaluation_test(self, session):
        sketchMatrix = np.zeros((self.sketch_test_num, self.outputFeaSize))
        # TODO Modified by Sun
        # sketchMatrix = np.zeros((self.batch_size*int(self.sketch_test_num/self.batch_size), self.outputFeaSize))
        shapeMatrix = np.zeros((self.shape_num, self.outputFeaSize))

        start_time = time.time()
        # print(self.sketch_test_num)
        # print(self.sketchTestFeaset.shape)
        # print(self.batch_size)

        # The test sketch data
        for i in range(0, self.sketch_test_num, self.batch_size):

            if i + self.batch_size <= self.sketch_test_num:
                tmp_in = self.sketchTestFeaset[i:i + self.batch_size]
            else:
                # ===========================
                # TODO Modified by Sun
                # break
                batch_num = self.sketch_test_num % self.batch_size
                # ===========================
                tmp_in = np.zeros((batch_num, self.inputFeaSize))  # empty matrix
                tmp_in[:self.sketch_test_num - i] = self.sketchTestFeaset[i:]  # Get the last batch (# is less than batch size)

            # print('shape:',tmp_in.shape)
            if self.lossType == 'npairLoss':
                tmp_out = session.run(self.output_sketch_fea, feed_dict={self.input_sketch_fea: tmp_in})
            else:
                tmp_out = session.run(self.output_sketch_fea_1, feed_dict={self.input_sketch_fea_1: tmp_in})

            if i + self.batch_size <= self.sketch_test_num:
                sketchMatrix[i:i + self.batch_size] = tmp_out
            else:
                sketchMatrix[i:] = tmp_out[:self.sketch_test_num - i]  # Get the remaining data of the last batch

        #print("Time for Loading sketch is {}\n".format(time.time() - start_time))

        # The test shape data
        for i in range(0, self.shape_num, self.batch_size):
            if i + self.batch_size <= self.shape_num:
                tmp_in = self.shapeFeaset[i:i + self.batch_size]
            else:
                tmp_in = np.zeros((self.batch_size, self.inputFeaSize))  # empty matrix
                tmp_in[:self.shape_num - i] = self.shapeFeaset[i:]  # Get the last batch (# is less than batch size)

            if self.lossType == 'npairLoss':
                tmp_out = session.run(self.output_shape_fea, feed_dict={self.input_shape_fea: tmp_in})
            else:
                tmp_out = session.run(self.output_shape_fea_1, feed_dict={self.input_shape_fea_1: tmp_in})

            if i + self.batch_size <= self.shape_num:
                shapeMatrix[i:i + self.batch_size] = tmp_out
            else:
                shapeMatrix[i:] = tmp_out[:self.shape_num - i]  # Get the remaining data of the last batc
        #print('shape Matrix:', shapeMatrix.shape)

        distanceMatrix = distance.cdist(sketchMatrix, shapeMatrix)
        model_label = (self.shapeLabelset).astype(int)
        test_label = (self.sketchTestLabelset).astype(int)
        C_depths, C_individual = self.retrievalParamSP()
        C_depths = C_depths.astype(int)

        print("Retrieval evaluation")
        nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec = RetrievalEvaluation(C_depths, distanceMatrix, model_label, test_label, testMode=1)

        return map_ # For clarity, we just return map.

    def evaluation(self):
        init = tf.global_variables_initializer()  # init all variables
        saver = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            sess.run(init)
            if self.ckpt_status():
                print("[*] Load SUCCESS")
                print(self.model_checkpoint_path)
                # This checkpoint is also included in Github.
                self.model_checkpoint_path = '/data/checkpoint/shrec14/Modified/PAMI/weightedContrastiveLoss/model-484499'
                saver.restore(sess, self.model_checkpoint_path)
            else:
                print("[*] Load failed")

            distanceMatrix = np.zeros((self.sketch_test_num, self.shape_num))
            sketchMatrix = np.zeros((self.sketch_test_num, self.outputFeaSize))
            # TODO Modified by Sun
            # sketchMatrix = np.zeros((self.batch_size*int(self.sketch_test_num/self.batch_size), self.outputFeaSize))
            shapeMatrix = np.zeros((self.shape_num, self.outputFeaSize))

            start_time = time.time()
            print(self.sketch_test_num)
            print(self.sketchTestFeaset.shape)
            print(self.batch_size)
            for i in range(0, self.sketch_test_num, self.batch_size):

                if i + self.batch_size <= self.sketch_test_num:
                    tmp_in = self.sketchTestFeaset[i:i + self.batch_size]
                else:
                    # ===========================
                    # TODO Modified by Sun
                    # break
                    batch_num = self.sketch_test_num % self.batch_size
                    # ===========================
                    tmp_in = np.zeros((batch_num, self.inputFeaSize))  # empty matrix
                    tmp_in[:self.sketch_test_num - i] = self.sketchTestFeaset[
                                                        i:]  # Get the last batch (# is less than batch size)
                tmp_out = sess.run(self.output_sketch_fea_1, feed_dict={self.input_sketch_fea_1: tmp_in})

                if i + self.batch_size <= self.sketch_test_num:
                    sketchMatrix[i:i + self.batch_size] = tmp_out
                else:
                    sketchMatrix[i:] = tmp_out[:self.sketch_test_num - i]  # Get the remaining data of the last batch

            print("Time for Loading sketch is {}\n".format(time.time() - start_time))

            for i in range(0, self.shape_num, self.batch_size):
                if i + self.batch_size <= self.shape_num:
                    tmp_in = self.shapeFeaset[i:i + self.batch_size]
                else:
                    tmp_in = np.zeros((self.batch_size, self.inputFeaSize))  # empty matrix
                    tmp_in[:self.shape_num - i] = self.shapeFeaset[i:]  # Get the last batch (# is less than batch size)

                tmp_out = sess.run(self.output_shape_fea_1, feed_dict={self.input_shape_fea_1: tmp_in})

                if i + self.batch_size <= self.shape_num:
                    shapeMatrix[i:i + self.batch_size] = tmp_out
                else:
                    shapeMatrix[i:] = tmp_out[:self.shape_num - i]  # Get the remaining data of the last batc

            distanceMatrix = distance.cdist(sketchMatrix, shapeMatrix)
            model_label = (self.shapeLabelset).astype(int)
            test_label = (self.sketchTestLabelset).astype(int)
            C_depths, C_individual = self.retrievalParamSP()
            C_depths = C_depths.astype(int)

            print("Retrieval evaluation")
            nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec = RetrievalEvaluation(C_depths, distanceMatrix,
                                                                                              model_label, test_label,
                                                                                              testMode=1)

            print('The NN is %5f' % (nn_av))
            print('The FT is %5f' % (ft_av))
            print('The ST is %5f' % (st_av))
            print('The DCG is %5f' % (dcg_av))
            print('The E is %5f' % (e_av))
            print('The MAP is %5f' % (map_))
# model.py ends here.
# ======================================