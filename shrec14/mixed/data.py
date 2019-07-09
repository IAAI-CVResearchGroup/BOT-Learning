import numpy as np
from random import shuffle
import random
import scipy.io as sio
import argparse
import time
import sys
from tqdm import tqdm
import pdb

class Dataset:
    def __init__(self, sketch_train_list, sketch_test_list, shape_list, num_views_shape=12, feaSize=4096, class_num=90,
                 phase='train', normFlag=0):
        # Load training images (path) and labels
        """
        class_num:      The total number of class
        sketch_train_list:  The list file of training sketch
        sketch_test_list:   The list file of testing sketch
        shape_list:         The list file of shape
        num_views_shape:    The total number of shape views
        class_num:          The total nubmer of classes
        normFlag:           1 for normalizing input features, 0 for not normalizing input features
        phase:              choosing training or testing
        inputFeaSize:       The dimension of input features
        """

        self.sketch_train_list = sketch_train_list
        self.sketch_test_list = sketch_test_list
        self.shape_list = shape_list
        self.num_views_shape = num_views_shape
        self.class_num = class_num
        self.feaSize = feaSize
        self.phase = phase
        self.normFlag = normFlag

        # Sketch training data
        with open(sketch_train_list) as f:
            lines = f.readlines()
        self.sketch_train_data = [line.rstrip('\n') for line in lines]
        self.sketch_train_num = len(self.sketch_train_data)
        # Shuffle sketch train data
        shuffle(self.sketch_train_data)

        # Sketch testing data
        with open(sketch_test_list) as f:
            lines = f.readlines()
        lines = [line.rstrip('\n') for line in lines]
        self.sketch_test_data = lines
        self.sketch_test_num = len(self.sketch_test_data)

        # Shape data
        with open(shape_list) as f:
            lines = f.readlines()
        lines = [line.rstrip('\n') for line in lines]
        self.shape_data = [lines[i:i + self.num_views_shape] for i in range(0, len(lines), self.num_views_shape)]
        # import pdb; pdb.set_trace()
        shuffle(self.shape_data)
        self.shape_num = len(self.shape_data)

        # all the pointer 
        self.sketch_train_ptr = 0
        self.sketch_test_ptr = 0
        self.shape_ptr = 0

        # Load all the data
        self.loadAllData(phase)
        # self.loadAllData('train')
        # self.loadAllData('evaluation')

        if self.normFlag:
            self.normalizeData(phase)
            # self.normalizeData('evaluation')

        # create random index to shuffle data for training
        self.sketch_test_randIndex = np.random.permutation(self.sketch_test_num)  # random index for testing sketch
        self.sketch_train_randIndex = np.random.permutation(self.sketch_train_num)  # random index for training sketch
        self.shape_randIndex = np.random.permutation(self.shape_num)  # random index for shape
        self.class_randIndex = np.random.permutation(self.class_num)  # random class for shape and sketch

    def loadAllData(self, phase):

        def loadShapeData(pathSet, num_views_shape, feaSize):
            sampleNum = len(pathSet)
            # import pdb; pdb.set_trace()
            feaSet = np.zeros((sampleNum, num_views_shape, feaSize))
            labelSet = np.zeros((sampleNum, 1))
            # ==================================
            # TODO For Test Need to be modified
            #sampleNum = 50
            # ==================================
            for i in tqdm(range(sampleNum)):
                for k in range(num_views_shape):
                    filePath = pathSet[i][k].split(' ')
                    feaSet[i, k] = self.loaddata(filePath[0])
                    labelSet[i] = int(filePath[1])
                # import pdb; pdb.set_trace()
            return np.amax(feaSet, axis=1), labelSet

        def loadSketchData(pathSet, feaSize):
            sampleNum = len(pathSet)
            # import pdb; pdb.set_trace()
            feaSet = np.zeros((sampleNum, feaSize))
            labelSet = np.zeros((sampleNum, 1))
            # ==================================
            # TODO For Test Need to be modified
            #sampleNum = 50
            # ==================================
            for i in tqdm(range(sampleNum)):
                # ============ TODO FOR TEST =====
                # for i in range(sampleNum):
                # i = 43583
                #print(pathSet[i])
                # import pdb; pdb.set_trace()
                # ===============================
                filePath = pathSet[i].split(' ')
                feaSet[i] = self.loaddata(filePath[0])
                labelSet[i] = int(filePath[1])

            return feaSet, labelSet

        if phase == 'evaluation':
            print("Load sketch testing features")
            start_time = time.time()
            self.sketchTestFeaset, self.sketchTestLabelset = loadSketchData(self.sketch_test_data, self.feaSize)
            print("Loading time: {}".format(time.time() - start_time))

        elif phase == 'train':
            print("Load sketch training features")
            start_time = time.time()
            self.sketchTrainFeaset, self.sketchTrainLabelset = loadSketchData(self.sketch_train_data, self.feaSize)
            self.sketchlabel_loc = []
            for ind2 in range(self.class_num):
                sketchlabel_ = np.where(self.sketchTrainLabelset == ind2)[0]
                # ===== Test whether the dataset is right =====
                # It should be modified for different dataset
                if len(sketchlabel_) != 1000:
                    pdb.set_trace()
                self.sketchlabel_loc.append(sketchlabel_)
            print('Load sketch testing features')
            self.sketchTestFeaset, self.sketchTestLabelset = loadSketchData(self.sketch_test_data, self.feaSize)
            print("Loading time: {}".format(time.time() - start_time))

        print("Load shape features")
        start_time = time.time()
        # import pdb; pdb.set_trace()
        self.shapeFeaset, self.shapeLabelset = loadShapeData(self.shape_data, self.num_views_shape, self.feaSize)
        self.shapelabel_loc = []
        for ind3 in range(self.class_num):
            shapelabel_ = np.where(self.shapeLabelset == ind3)[0]
            # ===== Test whether the dataset is right =====
            # It should be modified for different dataset
            if len(shapelabel_) == 0:
                pdb.set_trace()
            self.shapelabel_loc.append(shapelabel_)
        print("Loading time: {}".format(time.time() - start_time))
        print("Finish Loading")
        #import pdb; pdb.set_trace()

    def nextBatch_npair(self, batch_size, phase):
        if self.sketch_train_ptr + batch_size <= self.class_num:
            sketch_fea = np.zeros((batch_size, self.feaSize))
            shape_fea = np.zeros((batch_size, self.feaSize))
            sketch_label = np.zeros((batch_size, 1))
            shape_label = np.zeros((batch_size, 1))
            batchIndex = self.class_randIndex[self.sketch_train_ptr:self.sketch_train_ptr + batch_size]

            for ind in range(len(batchIndex)):
                sketch_random_index = random.choice(self.sketchlabel_loc[batchIndex[ind]])
                sketch_fea[ind, :] = self.sketchTrainFeaset[sketch_random_index, :]
                sketch_label[ind, :] = self.sketchTrainLabelset[sketch_random_index, :]

                shape_random_index = random.choice(self.shapelabel_loc[batchIndex[ind]])
                shape_fea[ind, :] = self.shapeFeaset[shape_random_index, :]
                shape_label[ind, :] = self.shapeLabelset[shape_random_index, :]
            #pdb.set_trace()
            self.sketch_train_ptr += batch_size
        else:
            self.class_randIndex = np.random.permutation(self.class_num)
            self.sketch_train_ptr = 0
            sketch_fea = np.zeros((batch_size, self.feaSize))
            shape_fea = np.zeros((batch_size, self.feaSize))
            sketch_label = np.zeros((batch_size, 1))
            shape_label = np.zeros((batch_size, 1))
            batchIndex = self.class_randIndex[self.sketch_train_ptr:self.sketch_train_ptr + batch_size]

            for ind in range(len(batchIndex)):
                sketch_random_index = random.choice(self.sketchlabel_loc[batchIndex[ind]])
                sketch_fea[ind, :] = self.sketchTrainFeaset[sketch_random_index, :]
                sketch_label[ind, :] = self.sketchTrainLabelset[sketch_random_index, :]

                shape_random_index = random.choice(self.shapelabel_loc[batchIndex[ind]])
                shape_fea[ind, :] = self.shapeFeaset[shape_random_index, :]
                shape_label[ind, :] = self.shapeLabelset[shape_random_index, :]
            self.sketch_train_ptr += batch_size
        return sketch_label, shape_label, sketch_fea, shape_fea

    def nextBatch(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'sketch_train':
            # Load training sketch
            #import pdb; pdb.set_trace()
            if self.sketch_train_ptr + batch_size <= self.sketch_train_num:
                batchIndex = self.sketch_train_randIndex[self.sketch_train_ptr:self.sketch_train_ptr + batch_size]
                sketch_fea = self.sketchTrainFeaset[batchIndex]
                sketch_label = self.sketchTrainLabelset[batchIndex]
                self.sketch_train_ptr += batch_size
            else:
                # shuffle the list
                self.sketch_train_randIndex = np.random.permutation(self.sketch_train_num)
                self.sketch_train_ptr = 0
                batchIndex = self.sketch_train_randIndex[self.sketch_train_ptr:self.sketch_train_ptr + batch_size]
                sketch_fea = self.sketchTrainFeaset[batchIndex]
                sketch_label = self.sketchTrainLabelset[batchIndex]
                self.sketch_train_ptr += batch_size

            return sketch_fea, sketch_label
        elif phase == 'shape':
            # Loading training shapes
            if self.shape_ptr + batch_size <= self.shape_num:
                batchIndex = self.shape_randIndex[self.shape_ptr:self.shape_ptr + batch_size]
                shape_fea = self.shapeFeaset[batchIndex]
                shape_label = self.shapeLabelset[batchIndex]
                self.shape_ptr += batch_size
            else:
                # shuffle the list
                self.shape_randIndex = np.random.permutation(self.shape_num)
                self.shape_ptr = 0
                batchIndex = self.shape_randIndex[self.shape_ptr:self.shape_ptr + batch_size]
                shape_fea = self.shapeFeaset[batchIndex]
                shape_label = self.shapeLabelset[batchIndex]
                self.shape_ptr += batch_size

            # shape_fea = np.amax(shape_fea, axis=1)

            return shape_fea, shape_label

    def loaddata(self, filepath):
        fid = open(filepath, 'r')
        lines = fid.readlines()
        # import pdb; pdb.set_trace()
        return np.array(lines).astype(float)

    def getLabel(self):
        # Get sketch test label
        self.sketch_test_label = []
        for tmp_sketch in self.sketch_test_data:
            self.sketch_test_label.append(int(tmp_sketch[0].split(' ')[-1]))

        # Get shape label
        self.shape_label = []
        for tmp_shape in self.shape_data:
            self.shape_label.append(int(tmp_shape[0].split(' ')[-1]))

    def retrievalParamSP(self):
        shapeLabels = self.shapeLabelset  ### cast all the labels as array
        sketchTestLabel = self.sketchTestLabelset  ### cast sketch test label as array
        C_depths = np.zeros(sketchTestLabel.shape)
        unique_labels = np.unique(sketchTestLabel)
        C_individual = np.zeros((unique_labels.shape[0], 1))
        for i in range(unique_labels.shape[0]):  ### find the numbers
            tmp_index_sketch = np.where(sketchTestLabel == unique_labels[i])[0]  ## for sketch index
            tmp_index_shape = np.where(shapeLabels == unique_labels[i])[0]  ## for shape index
            C_depths[tmp_index_sketch] = tmp_index_shape.shape[0]
            # TODO Modified by Sun for validate sum
            C_individual[i, 0] = tmp_index_shape.shape[0]
            # import pdb; pdb.set_trace()
        return C_depths, C_individual

    def retrievalParamSS(self):
        shapeLabels = np.array(self.sketch_train_label)  ### cast all the labels as array
        sketchTestLabel = np.array(self.sketch_test_label)  ### cast sketch test label as array
        C_depths = np.zeros(sketchTestLabel.shape)
        unique_labels = np.unique(sketchTestLabel)
        for i in range(unique_labels.shape[0]):  ### find the numbers
            tmp_index_sketch = np.where(sketchTestLabel == unique_labels[i])[0]  ## for sketch index
            tmp_index_shape = np.where(shapeLabels == unique_labels[i])[0]  ## for shape index
            C_depths[tmp_index_sketch] = tmp_index_shape.shape[0]
        return C_depths

    def retrievalParamPP(self):
        shapeLabels = np.array(self.shape_label)  ### cast all the labels as array
        sketchTestLabel = np.array(self.shape_label)  ### cast sketch test label as array
        C_depths = np.zeros(sketchTestLabel.shape)
        unique_labels = np.unique(sketchTestLabel)
        for i in range(unique_labels.shape[0]):  ### find the numbers
            tmp_index_sketch = np.where(sketchTestLabel == unique_labels[i])[0]  ## for sketch index
            tmp_index_shape = np.where(shapeLabels == unique_labels[i])[0]  ## for shape index
            C_depths[tmp_index_sketch] = tmp_index_shape.shape[0]
        return C_depths

    def normalizeData(self, phase):
        ########### normalize sketch test feature ######################
        # print('Processing testing sketch\n')
        print("Normalizing shape features")

        shape_mean = np.mean(self.shapeFeaset, axis=0)
        shape_std = np.std(self.shapeFeaset, axis=0)
        self.shapeFeaset = (self.shapeFeaset - shape_mean) / shape_std
        ###### get rid of nan Dataset################
        self.shapeFeaset[np.where(np.isnan(self.shapeFeaset))] = 0
        # print(np.where(np.isnan(shape_feaset_norm)))

        if self.phase == 'train':
            print("Normalizing sketch train features")
            sketch_train_mean = np.mean(self.sketchTrainFeaset, axis=0)
            sketch_train_std = np.std(self.sketchTrainFeaset, axis=0)
            self.sketchTrainFeaset = (self.sketchTrainFeaset - sketch_train_mean) / sketch_train_std
            sketch_test_mean = np.mean(self.sketchTestFeaset, axis=0)
            sketch_test_std = np.std(self.sketchTestFeaset, axis=0)
            self.sketchTestFeaset = (self.sketchTestFeaset - sketch_test_mean) / sketch_test_std
        elif self.phase == 'evaluation':
            print("Normalizing sketch test features")
            sketch_test_mean = np.mean(self.sketchTestFeaset, axis=0)
            sketch_test_std = np.std(self.sketchTestFeaset, axis=0)
            self.sketchTestFeaset = (self.sketchTestFeaset - sketch_test_mean) / sketch_test_std


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This is for loading data')
    parser.add_argument('--sketch_train_list', type=str, default='./sketch_train.txt', help='The training list file')
    parser.add_argument('--sketch_test_list', type=str, default='./sketch_test.txt', help='The testing list file')
    parser.add_argument('--shape_list', type=str, default='./shape.txt', help='The root direcoty of input data')
    parser.add_argument('--num_views_shape', type=int, default=12, help='The total number of views')
    parser.add_argument('--class_num', type=int, default=90, help='the total number of class')
    args = parser.parse_args()
    data = Dataset(sketch_train_list=args.sketch_train_list, sketch_test_list=args.sketch_test_list,
                   shape_list=args.shape_list,
                   num_views_shape=args.num_views_shape, class_num=args.class_num)
    print('\n\n\n\n\n\n\n')
    for _ in range(1500):
        start_time = time.time()
        sketch_fea, sketch_label = data.nextBatch(5, 'sketch_train')
        shape_fea, shape_label = data.nextBatch(5, 'shape')
        print("time cost: {}".format(time.time() - start_time))
        print(sketch_fea.shape)
        print(sketch_label.shape)
        print("#############################\n")
        print(shape_fea.shape)
        print(shape_label.shape)
        print("#############################\n")
