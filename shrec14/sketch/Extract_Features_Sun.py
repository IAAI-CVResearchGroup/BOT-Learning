import sys
import os
import numpy as np
import caffe
import glob

caffe.set_mode_gpu()
caffe.set_device(0)
############################################
model_file = '/data/sketchShapeWeightedCL/shrec13/sketch/model/weight_iter_15000.caffemodel'
deploy_file ='./deploy.prototxt'

net = caffe.Net(deploy_file, model_file, caffe.TEST)
feaLayer = 'fc7'
if feaLayer not in net.blobs:
    print(TypeError('Invalid layer name: ' + layer1))

""" The sketch directory """
sketchDir = '/data/shrec14/SHREC14LSSTB_SKETCHES/SHREC14LSSTB_SKETCHES'
outFeaDir = '/data/shrec14/Sketch_Features'
## resized image ##
resize_H = 256
resize_W = 256
crop_H = 227
crop_W = 227
###################

##set transformer #####################

transformer = caffe.io.Transformer({'data': (1, 3, resize_H, resize_W)})
transformer.set_mean('data', np.load('./ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2, 0, 1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

net.blobs['data'].reshape(1, 3, 227, 227)
########################################
print('The mean matrix shape is', np.load('./ilsvrc_2012_mean.npy').shape)
for dirname in os.listdir(sketchDir):
    fullDir = os.path.join(sketchDir, dirname)
    print('The current directory is %s' % (fullDir))
    outDir = os.path.join(outFeaDir, dirname)
    imgDir = os.path.join(sketchDir, dirname)
    if os.path.isdir(fullDir):
        if not os.path.isdir(outDir):
            os.makedirs(outDir)
        for f in os.listdir(fullDir):
            inFile = os.path.join(fullDir, f)
            filename, _ = os.path.splitext(f)               # get the filename
            current_dir = os.path.join(imgDir,filename)
            outFile = os.path.join(outDir, filename)
            if os.path.exists(outFile):
                pass
            else:
                os.makedirs(outFile)
            features_list = glob.glob(current_dir+'/*')
            img_Num = len(features_list)
            img_Name = os.listdir(current_dir)
            for ind in range(img_Num):
                img = caffe.io.load_image(features_list[ind])
                img = transformer.preprocess('data', img)
                #print('The image shape is ', img.shape)
                crop_img = img[:, int(resize_H / 2) - int(crop_H / 2): int(resize_H / 2) + int(crop_H / 2) + 1,
                           int(resize_W / 2) - int(crop_W / 2): int(resize_W / 2) + int(crop_W / 2) + 1]
                #print('The cropped image shape is ', crop_img.shape)
                ### crop image ##########
                net.blobs['data'].data[...] = crop_img
                output = net.forward()
                #print('The feature size is ', net.blobs[feaLayer].data[0].shape)
                OutPath = outFile + '/' + img_Name[ind][:-4] + '.txt'
                with open(OutPath, 'w') as newf:
                    np.savetxt(newf, net.blobs[feaLayer].data[0], fmt='%.4f', delimiter='\n')
                #import pdb;pdb.set_trace()