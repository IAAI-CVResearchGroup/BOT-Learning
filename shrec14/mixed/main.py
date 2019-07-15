# ======================================
# Filename: main.py
# Description: Main function
# Author: Han Sun, Zhiyuan Chen, Lin Xu
#
# Project: Optimal Transport for Cross-modality Retrieval
# Github: https://github.com/IAAI-CVResearchGroup/Batch-wise-Optimal-Transport-Metric/tree/master/shrec14
# Copyright (C): IAAI
# Code:
import argparse
import os
import pandas as pd
import scipy.misc 
import numpy as np 
import os 
from model import model
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

parser = argparse.ArgumentParser(description='This is the model for training weighted contrastive loss')
parser.add_argument('--lamb', dest='lamb', type=float, default=5., help='parameter for sinkhorn iteration')
parser.add_argument('--ckpt_dir', dest='ckpt_dir', type=str, default='/data/checkpoint/shrec14/Modified/PAMI/', help='directory for saving checkpoint')#TODO /Modified
parser.add_argument('--batch_size', dest='batch_size', type=int, default=30, help='# images in batch')
parser.add_argument('--margin', type=float, default=1, help='The margin of loss')
parser.add_argument('--sketch_train_list', type=str, default='/home/sri-dev01/CVRP_New/shrec14/sketch/sketch_train_sh.txt',
                    help='The training list file')
parser.add_argument('--sketch_test_list', type=str, default='/home/sri-dev01/CVRP_New/shrec14/sketch/sketch_test_sh.txt',
                    help='The testing list file')
parser.add_argument('--shape_list', type=str, default='/home/sri-dev01/CVRP_New/shrec14/shape/shape_sh.txt',
                    help='The shape list file')
parser.add_argument('--num_views_shape', type=int, default=20, help='The total number of views for shape')
parser.add_argument('--class_num', type=int, default=171, help='the total number of class')
parser.add_argument('--phase', dest='phase', default='train', help='train, test, evaluation')
parser.add_argument('--logdir', dest='logdir', default='./logs', help='name of the dataset')
parser.add_argument('--maxiter', dest='maxiter', type=int, default=1453500, help='maximum number of iterations')
parser.add_argument('--inputFeaSize', dest='inputFeaSize', type=int, default=4096, help='The dimensions of input features') 
parser.add_argument('--outputFeaSize', dest='outputFeaSize', type=int, default=100, help='The dimensions of input features') 
parser.add_argument('--lossType', dest='lossType', type=str, default='weightedContrastiveLoss', help='name of the dataset')
parser.add_argument('--activationType', dest='activationType', type=str, default='relu', help='name of the dataset')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default='0.0005', help='learning rate')
parser.add_argument('--normFlag', dest='normFlag', type=int, default=0, help='whether to normalize the input feature')
parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='momentum term of Gradient')
args = parser.parse_args()
#args.phase = 'train'
args.phase = 'evaluation'

def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
        print(args.ckpt_dir)
    wasserteinModel = model(lamb=args.lamb, ckpt_dir=args.ckpt_dir, batch_size=args.batch_size, margin=args.margin,
            sketch_train_list=args.sketch_train_list, sketch_test_list=args.sketch_test_list, shape_list=args.shape_list, 
            num_views_shape=args.num_views_shape, learning_rate=args.learning_rate, momentum=args.momentum,
            class_num=args.class_num, normFlag=args.normFlag, logdir=args.logdir, lossType=args.lossType, activationType=args.activationType, 
            phase=args.phase, inputFeaSize=args.inputFeaSize, outputFeaSize=args.outputFeaSize, maxiter=args.maxiter)

    if args.phase == 'train':
        wasserteinModel.train()
    elif args.phase == 'test':
        print('Start Evaluation')
        wasserteinModel.test(args)
    elif args.phase == 'evaluation':
        wasserteinModel.evaluation()

if __name__ == '__main__':
    tf.app.run()
# main.py ends here
# ======================================