import glob
import pdb

flds = glob.glob('/data/shrec14/Sketch_Features_20Angles/*')
flds.sort()
#pdb.set_trace()
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
		#pdb.set_trace()
	else:
		features = glob.glob(fld+'/test/*.txt')
		features.sort()
		for ind in range(len(features)):
			test_file.write("{} {}\n".format(features[ind], label))
		label += 1