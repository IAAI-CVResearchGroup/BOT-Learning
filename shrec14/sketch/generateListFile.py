# ======================================
# Filename: generateListFile.py
# Description: Generate location lists of sketch features in training and test set
# Author: Han Sun, Zhiyuan Chen, Lin Xu
#
# Project: Optimal Transport for Cross-modality Retrieval
# Github: https://github.com/IAAI-CVResearchGroup/Batch-wise-Optimal-Transport-Metric/tree/master/shrec14
# Copyright (C): IAAI
# Code:
import glob
import pdb

flds = glob.glob('/data/shrec14/Sketch_Features_20Angles/*')
flds.sort()
label = 0
mode = 'test'
if mode == 'train':
	train_file = open('/home/sri-dev01/CVRP_New/shrec14/sketch/sketch_train_sh.txt', 'w')
else:
	test_file = open('/home/sri-dev01/CVRP_New/shrec14/sketch/sketch_test_sh.txt', 'w')
for fld in flds:
	if mode == 'train':
		features = glob.glob(fld+'/train/*.txt')
		features.sort()
		for ind in range(len(features)):
			train_file.write("{} {}\n".format(features[ind], label))
		label += 1
	else:
		features = glob.glob(fld+'/test/*.txt')
		features.sort()
		for ind in range(len(features)):
			test_file.write("{} {}\n".format(features[ind], label))
		label += 1
# generateListFile.py ends here
# ======================================