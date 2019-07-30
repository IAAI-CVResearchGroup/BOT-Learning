
""" Siamese implementation using Tensorflow with MNIST example.
This siamese network embeds a 28x28 image (a point in 784D)
into a point in 2D.

By Youngwook Paul Kwon (young at berkeley.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input
from normData import normData

from datetime import datetime
from scipy.spatial import distance
from RetrievalEvaluation import RetrievalEvaluation
from sklearn import svm
import sklearn
import pickle
import random
import inference

#import system things
from tensorflow.examples.tutorials.mnist import input_data # for data
import tensorflow as tf
import numpy as np
import os
import csv
import cv2
import math

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# setup siamese network
batch_size = 64
display_interval = 500
learning_rate = 0.2
learning_rate_decay = 0.95
momentum = 0.9

# mnist data
#mnist = tf.keras.datasets.mnist
#(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

#create session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession()

# setup siamese network
##################################################################################
lossType='inverseWassersteinCL'
margin = 5.0
siamese = inference.siamese(sess=sess, margin=margin, batch_size=batch_size, lamb=0.01, lossType=lossType);
#normLabel = 1

learning_rate = tf.placeholder(tf.float32, [1])
train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate[0], momentum=momentum).minimize(siamese.loss,var_list=siamese.var_list) #
##############################################################################################################

saver = tf.train.Saver()  # save results ?
init = tf.global_variables_initializer()  # initialization
sess.run(init)
#writer = tf.summary.FileWriter('./sinkhorn_board', sess.graph)

# if you just want to load a previously trainmodel?
load = False
model_ckpt = './ckpt/model.meta'

if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = input("We found model files. Do you want to load it and continue training [yes/no]?")
    if input_var == 'yes':
        load = True
        
if load: saver.restore(sess, './ckpt/model')
    

def retrievalParamPP(test_label1, test_label2):
    shapeLabels = test_label1            ### cast all the labels as array
    sketchTestLabel = test_label2  ### cast sketch test label as array
    C_depths = np.zeros(sketchTestLabel.shape)
    unique_labels = np.unique(sketchTestLabel)
    for i in range(unique_labels.shape[0]):             ### find the numbers
        tmp_index_sketch = np.where(sketchTestLabel == unique_labels[i])[0] ## for sketch index
        tmp_index_shape = np.where(shapeLabels == unique_labels[i])[0]      ## for shape index
        C_depths[tmp_index_sketch] = tmp_index_shape.shape[0]
    return C_depths



#functions for loading CIFAR10 and CIFAR100
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def loadCIFAR10():    
    imgs = []
    labels = [] 

    a = unpickle('./cifar-10-batches-py/data_batch_1')
    imgs.extend(a[b'data'])
    labels.extend(a[b'labels'])
    b = unpickle('./cifar-10-batches-py/data_batch_2')
    imgs.extend(b[b'data'])
    labels.extend(b[b'labels'])
    c = unpickle('./cifar-10-batches-py/data_batch_3')
    imgs.extend(c[b'data'])
    labels.extend(c[b'labels'])
    d = unpickle('./cifar-10-batches-py/data_batch_4')
    imgs.extend(d[b'data'])
    labels.extend(d[b'labels'])
    e = unpickle('./cifar-10-batches-py/data_batch_5')
    imgs.extend(e[b'data'])
    labels.extend(e[b'labels'])
    
    test = unpickle('./cifar-10-batches-py/test_batch')
    test_imgs = test[b'data']
    test_labels = test[b'labels']
    test_img_list = list(np.array(test_imgs).reshape(10000, 3, 32, 32).transpose(0,2,3,1))  
    img_list = list(np.array(imgs).reshape(50000, 3, 32, 32).transpose(0,2,3,1))

    return img_list, labels, test_img_list, test_labels


def loadCIFAR100():    
    imgs = []
    labels = [] 

    a = unpickle('./cifar-100-python/train')
    imgs.extend(a[b'data'])
    labels.extend(a[b'fine_labels'])    
    
    img_list = list(np.array(imgs).reshape(50000, 3, 32, 32).transpose(0,2,3,1))
    
    test = unpickle('./cifar-100-python/test')
    test_imgs = test[b'data']
    test_labels = test[b'fine_labels']
    test_img_list = list(np.array(test_imgs).reshape(10000, 3, 32, 32).transpose(0,2,3,1))
    
    return img_list, labels, test_img_list, test_labels

    

#cifar_train_x, cifar_train_y, cifar_test_x, cifar_test_y = loadCIFAR10() 
#cifar_train_x, cifar_train_y, cifar_test_x, cifar_test_y = loadCIFAR100() 

for epoch in range(100):
    print("current learning rate is %f" %(initial_learning_rate))
    
    # mnist steps
    steps = int(math.floor(len(mnist.train.images)/(batch_size*2)))
    
    #cifar steps
    """
    steps = int(math.floor(len(cifar_train_x)/(batch_size*2)))
    if not epoch == 0:
        combined = list(zip(cifar_train_x,cifar_train_y))
        random.shuffle(combined)
        cifar10_train_x[:], cifar_train_y[:] = zip(*combined)
    """
    
    for step in range(steps):
        
        #mnist training batch
        batch_x1, batch_y1 = mnist.train.next_batch(batch_size)
        batch_x2, batch_y2 = mnist.train.next_batch(batch_size)

        batch_x1 = np.reshape(batch_x1, [batch_size, 28, 28, 1])
        batch_x2 = np.reshape(batch_x2, [batch_size, 28, 28, 1])
        batch_y1 = np.reshape(batch_y1, [batch_size])
        batch_y2 = np.reshape(batch_y2, [batch_size])

       
        #if normLabel:
        #    batch_x1 = normData(batch_x1, 0.0, 1.0)
        #    batch_x2 = normData(batch_x2, 0.0, 1.0)
        
        
        # cifar training batch
        """
        start_ind = step*batch_size*2
        end_ind = min(batch_size*2*(step+1), len(cifar_train_x)) 
        batch_x = cifar_train_x[start_ind:end_ind]
        batch_y = cifar_train_y[start_ind:end_ind]
        
        if not len(batch_x)%2==0:
            batch_x = batch_x[:-1]
            batch_y = batch_y[:-1]
            
        mid_ind = int(math.ceil(len(batch_x)/2.0))
        batch_x1 = batch_x[:mid_ind]
        batch_x2 = batch_x[mid_ind:]
        batch_y1 = batch_y[:mid_ind]
        batch_y2 = batch_y[mid_ind:]
        """

       
    
        # sinkhorn iteration
        M_eval = sess.run(siamese.groundMetric, feed_dict={
                           siamese.x1: batch_x1,
                           siamese.x2: batch_x2,
                           siamese.label1: batch_y1, 
                           siamese.label2: batch_y2})


        for sinhorn_ter in range(30):
            siamese.v_assign_op.eval(feed_dict={siamese.groundMetric:M_eval})
            siamese.u_assign_op.eval(feed_dict={siamese.groundMetric:M_eval})


        # Train model
        _, loss_all, loss_p, loss_n = sess.run([train_step, siamese.loss,
                                                siamese.loss_p, siamese.loss_n], 
                                                feed_dict={siamese.x1: batch_x1, 
                                                           siamese.x2: batch_x2, 
                                                           siamese.label1: batch_y1,
                                                           siamese.label2: batch_y2,
                                                           learning_rate:[learning_rate]})
        
        

        #writer.add_summary(loss_sum_, (epoch*steps+step))

        if np.sum(np.isnan(loss_all)):
            print('Model diverged with loss = NaN')
            sys.exit()


        if step % 10 == 0:
            print ('step %d: loss %f p_loss %f n_loss %f' % (step, loss_all, loss_p, loss_n))
           
    learning_rate = learning_rate*learning_rate_decay
    
    # minist test classification and retrieval 
    test_x1 = np.reshape(mnist.test.images, [10000, 28, 28, 1])
    embed = siamese.o1.eval({siamese.x1: test_x1})
    train_imgs = np.reshape(mnist.train.images, [55000, 28, 28, 1])
    train_labels = mnist.train.labels
    train_feas = []
    for i in range(1000):
        train_feas.extend(list(siamese.o1.eval({siamese.x1: train_imgs[i*55:(i+1)*55]})))
    
    clf = svm.SVC(decision_function_shape='ovr')
    clf.fit(train_feas, train_labels) 
    
    predictions = clf.predict(list(embed))
    acc_ = sklearn.metrics.accuracy_score(predictions, mnist.test.labels)
    acc_sum = tf.summary.scalar('acc', acc_)
    print('The acc is %f' % (acc_))
    
    #embed.tofile('embed.txt')
    srcLabel = mnist.test.labels
    dstLabel = mnist.test.labels 
    
    
    
    # cifar test classification and retrival 
    """
    embed = []
    for i in range(500):
        embed.extend(list(siamese.o1.eval({siamese.x1: cifar_test_x[i*20:(i+1)*20]})))
    train_feas = []
    for i in range(2000):
        train_feas.extend(list(siamese.o1.eval({siamese.x1: cifar_train_x[i*25:(i+1)*25]})))

    clf = svm.SVC(decision_function_shape='ovr')
    clf.fit(train_feas, cifar_train_y) 
    predictions = clf.predict(list(embed))
    acc_ = sklearn.metrics.accuracy_score(predictions, cifar_test_y)
    acc_sum = tf.summary.scalar('acc', acc_)
    print('The acc is %f' % (acc_))
      
    srcLabel = cifar_test_y[:]
    dstLabel = cifar_test_y[:]  
    srcLabel = np.array(srcLabel)
    dstLabel = np.array(dstLabel)
    """
    
    C_depths = retrievalParamPP(srcLabel, dstLabel).astype(int)        ### for retrieval evaluation
    #test_feaset_deepmetriclearning = np.zeros((len(test_x1), 100))
    ## restore pointer
    distM = distance.cdist(embed, embed)

    nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec = RetrievalEvaluation(C_depths, distM, dstLabel, srcLabel, testMode=2)
    print (('The NN is %f') % (nn_av))
    print (('The FT is %f') % (ft_av))
    print (('The ST is %f') % (st_av))
    print (('The DCG is %f') % (dcg_av))
    print (('The E is %f') % (e_av))
    print (('The MAP is %f') % (map_))
    
    mAP_sum = tf.summary.scalar('mAP', map_)

    
    saver.save(sess, './ckpt/model')        


