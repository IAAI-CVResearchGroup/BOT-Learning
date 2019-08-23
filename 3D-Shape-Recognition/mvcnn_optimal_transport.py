# ======================================
# Filename: mvcnn_optimal_transport.py
# Description: Extract sketch features
# Author: Han Sun, Zhiyuan Chen, Lin Xu
#
# Project: Optimal Transport for Multi-modality Recognition
# Github: https://github.com/IAAI-CVResearchGroup/Batch-wise-Optimal-Transport-Metric/tree/master/3D-Shape-Recognition
# Copyright (C): IAAI
# Code:
import argparse
import os
import os.path as osp
import sys
import shutil
import time
from sklearn import svm
from sklearn.metrics import accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from RetrievalEvaluation import RetrievalEvaluation

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import pdb
import ot
import math

model_names = sorted(name for name in models.__dict__
					 if name.islower() and not name.startswith("__")
					 and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
					help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
					choices=model_names,
					help='model architecture: ' +
						 ' | '.join(model_names) +
						 ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
					metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
					help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
					help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
					help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
					help='distributed backend')
parser.add_argument('--margin', default=1.0, type=float,
					help='margin for optimal transport')
parser.add_argument('--loss-type', default='lifted', type=str,
					help='lifted or optimal transport')
parser.add_argument('--lamb', default=5.0, type=float,
					help='Regularization coefficients for sinkhorn')
parser.add_argument('--pos-margin', default=5.0, type=float,
					help='Margin for Positive pairs')
parser.add_argument('--with-softmax', default=1, type=int,
					help='Use softmax or not')

best_prec1 = 0
nview = 12


class global_args(object):
	distributed = 0
	batch_size = 240
	pretrained = True
	arch = 'alexnet'
	data = '/data/han01.sun/ModelNet40_20_srcsubset/'  # '/data/han.sun/ModelNet10_Rebuttal'
	lr = 0.001
	# lr = 0.01
	momentum = 0.9
	weight_decay = 1e-4
	# resume = '/data/checkpoint/ModelNet40/Lifted_CameraReady_Margin5/checkpoint_npairloss50.pth.tar'
	resume = None
	workers = 2
	epochs = 300
	#evaluate = True
	evaluate = False
	print_freq = 10
	start_epoch = 0
	save_dir = ' '
	margin = 12
	pos_margin = 8.0
	lamb = 0.01
	loss_type = 'optimal_transport'
	with_softmax = 1

args = global_args()
print('Save Path:', args.save_dir)
print('Margin is:', args.margin)


def optimal_transport(feature, score, target, pos_margin = 5, margin=5, lamb=5):
	loss = 0
	# Softmax Loss
	loss_softmax = torch.nn.functional.cross_entropy(score, target)
	# OPT Loss
	target = target.float()
	target = target.view(target.size(0), 1)
	target_train = (target == torch.transpose(target, 0, 1)).float()
	bsz = feature.size(0)
	mag = (feature ** 2).sum(1).expand(bsz, bsz)
	sim = feature.mm(feature.transpose(0, 1))

	dist = (mag + mag.transpose(0, 1) - 2 * sim)
	dist = torch.nn.functional.relu(dist).sqrt().cuda()

	hinge_groundMetric = torch.nn.functional.relu(margin - dist) ** 2
	Pos_groundMetric = torch.nn.functional.relu(dist - pos_margin) ** 2
	GM_PositivePair = target_train.mul(Pos_groundMetric)

	#GM_PositivePair = target_train.mul(dist ** 2)
	GM_NegativePair = (1 - target_train).mul(hinge_groundMetric)
	GM = GM_PositivePair + GM_NegativePair
	GMF = GM.view(-1)
	#Pos_ = torch.sum(GM_PositivePair) / torch.sum(target_train)
	#Neg_ = torch.sum(GM_NegativePair) / torch.sum(1 - target_train)
	# print(Pos_, Neg_)

	expGM = torch.exp(-10. * GM)

	GMFlatten = expGM.view(-1)

	uuu = np.ones([bsz]) / bsz
	vvv = np.ones([bsz]) / bsz
	reg = (-1) / lamb
	expGM_numpy = expGM.cpu().detach().numpy()
	# print('I am busy at computing sinkhorn')
	# sys.stdout.flush()

	T = ot.sinkhorn(uuu, vvv, expGM_numpy, reg, numItermax=50)
	# print('Finally done')
	# sys.stdout.flush()
	T_Flatten = torch.autograd.Variable(torch.from_numpy(T.reshape([-1]))).float().cuda()
	loss_opt = torch.sum(GMF.mul(T_Flatten))
	loss = (1.0 / 1.0) * loss_opt + (1.0 / 5.0) * loss_softmax
	# loss = torch.sum(GM/bsz)
	#import pdb; pdb.set_trace()
	return loss

class FineTuneModel(nn.Module):
	def __init__(self, original_model, arch, num_classes, softmax):
		super(FineTuneModel, self).__init__()
		self.softmax = softmax
		if arch.startswith('alexnet'):
			self.features = original_model.features
			#self.features_bn = nn.BatchNorm1d(256*6*6)
			self.classifier = nn.Sequential(
				nn.Dropout(),
				nn.Linear(256 * 6 * 6, 4096),
				nn.ReLU(inplace=True),
				nn.Dropout(),
				nn.Linear(4096, 256),
				#nn.BatchNorm1d(256)
				# Comment by add BN
				nn.ReLU(inplace=True)
			)
			self.pooling_output = nn.Linear(256, num_classes)
			self.modelName = 'alexnet'
		elif arch.startswith('resnet'):
			# Everything except the last linear layer
			self.features = nn.Sequential(*list(original_model.children())[:-1])
			self.classifier = nn.Sequential(
				nn.Linear(512, num_classes)
			)
			self.modelName = 'resnet'
		elif arch.startswith('vgg16'):
			self.features = original_model.features
			self.classifier = nn.Sequential(
				nn.Dropout(),
				nn.Linear(25088, 4096),
				nn.ReLU(inplace=True),
				nn.Dropout(),
				nn.Linear(4096, 4096),
				nn.ReLU(inplace=True),
			)
			self.pooling_output = nn.Linear(4096, num_classes)
			self.modelName = 'vgg16'

		else:
			raise ("Finetuning not supported on this architecture yet")

		# # Freeze those weights
		# for p in self.features.parameters():
		#     p.requires_grad = False

	def forward(self, x):
		f = self.features(x)
		if self.modelName == 'alexnet':
			f = f.view(f.size(0), 256 * 6 * 6)
			#f_bn = self.features_bn(f)
			f = self.classifier(f)
			#f_bn = self.features_bn(256)
			f = f.view(-1, nview, 256)
			#f_max = torch.max(f_bn, 1)[0]
			#return f_max
		elif self.modelName == 'vgg16':
			f = f.view(f.size(0), -1)
		elif self.modelName == 'resnet':
			f = f.view(f.size(0), -1)
			f = f.view(-1, nview, 512)

		#f = self.classifier(f)
		#import pdb; pdb.set_trace()
		#f = f.view(-1, nview, 256)
		f = torch.max(f, 1)[0]
		if self.softmax == 1:
			# todo get classification results
			y = self.pooling_output(f)
			return f, y
		else:
			return f

def save_checkpoint(state, is_best, filename='rotationnet_checkpoint.pth.tar'):
	torch.save(state, filename)
	# if is_best:
	#    shutil.copyfile(filename, filename2)


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch=0):
	"""Sets the learning rate to the initial LR decayed by 10 every 200 epochs"""
	# lr = args.lr * (0.8 ** (epoch // 15))
	if args.lr * (0.8 ** ((epoch - 1) // 2)) > 1e-7:
		lr = args.lr * (0.8 ** (epoch // 2))
	else:
		lr = 1e-7
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
		print('Learning Rate: {lr:.6f}'.format(lr=param_group['lr']))


def my_accuracy(output_, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	batch_size = target.size(0)
	maxk = max(topk)
	_, pred = output_.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


def train(train_loader, model, criterion, optimizer, epoch):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()
	for i, (input, target) in enumerate(train_loader):
		# print(input.shape) # (k * 20, 3, 224, 224)
		# print(target.shape)  # ([k * 20]) = batch_size
		nsamp = int(input.size(0) / nview)  # k

		# measure data loading time
		data_time.update(time.time() - end)

		input_var = torch.autograd.Variable(input)
		target = target.cuda(async=True)
		target = target[0:-1:nview]
		target_var = torch.autograd.Variable(target)  # (k)

		# compute output
		output, score = model(input_var)  # (k, 40)
		semi_batch = int(output.size(0) / 2)

		num_classes = int(output.size(1))

		# compute scores and decide target labels
		output_ = torch.nn.functional.log_softmax(output)
		# compute loss
		if args.loss_type == 'optimal_transport':
			loss = criterion(output, score, target_var, args.pos_margin, args.margin, args.lamb)
		# loss = LiftedStructureLoss()(output, target_var)
		#  print('loss is ', loss)
		if type(loss) != float:
			# import pdb; pdb.set_trace()
			losses.update(loss.item(), input.size(0))
			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
				epoch, i, len(train_loader), batch_time=batch_time,
				data_time=data_time, loss=losses))
			sys.stdout.flush()


def validate(val_loader, model, criterion):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	for i, (input, target) in enumerate(val_loader):
		target = target.cuda(async=True)
		target = target[0:-1:nview]  # k
		with torch.no_grad():
			input_var = torch.autograd.Variable(input)
			target_var = torch.autograd.Variable(target)

		# compute output
		output, score = model(input_var)  # (k, 40)
		num_classes = int(output.size(1))
		output = torch.nn.functional.log_softmax(output)

		loss = criterion(output, target_var.to("cuda"), dtype=torch.int64)

		# measure accuracy and record loss
		prec1, prec5 = my_accuracy(output.data, target, topk=(1, 5))

		losses.update(loss.item(), input.size(0))
		top1.update(prec1.item(), input.size(0) / nview)
		top5.update(prec5.item(), input.size(0) / nview)

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Test: [{0}/{1}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				i, len(val_loader), batch_time=batch_time, loss=losses,
				top1=top1, top5=top5))

	print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
		  .format(top1=top1, top5=top5))

	return top1.avg


def test(model, train_loader, val_loader):
	model.eval()

	# with torch.no_grad():
	train_feature, train_label = [], []
	for batch_idx, (input, labels) in enumerate(train_loader):
		features_batch, score_batch = model(input)
		features = features_batch.data.cpu()
		train_feature.append(features)
		train_label.extend(labels[0::nview])
	train_feature = torch.cat(train_feature, 0).numpy()
	train_feature_list = train_feature.tolist()
	train_label_list = []
	for ind in range(len(train_label)):
		label_ = train_label[ind].item()
		train_label_list.append(label_)

	val_feature, val_label = [], []
	for batch_idx, (input, labels) in enumerate(val_loader):
		features_batch, score_batch = model(input)
		features = features_batch.data.cpu()
		val_feature.append(features)
		val_label.extend(labels[0::nview])
	val_feature = torch.cat(val_feature, 0).numpy()
	val_feature_list = val_feature.tolist()
	val_label_list = []
	for ind in range(len(val_label)):
		label_ = val_label[ind].item()
		val_label_list.append(label_)

	clf = svm.SVC(decision_function_shape='ovr')
	clf.fit(train_feature_list, train_label_list)

	predictions = clf.predict(val_feature_list)

	acc_ = accuracy_score(predictions, val_label_list)

	return acc_


def retrievalParamPP(test_label1, test_label2):
	shapeLabels = test_label1  ### cast all the labels as array
	sketchTestLabel = test_label2  ### cast sketch test label as array
	C_depths = np.zeros(sketchTestLabel.shape)
	unique_labels = np.unique(sketchTestLabel)
	for i in range(unique_labels.shape[0]):  ### find the numbers
		tmp_index_sketch = np.where(sketchTestLabel == unique_labels[i])[0]  ## for sketch index
		tmp_index_shape = np.where(shapeLabels == unique_labels[i])[0]  ## for shape index
		C_depths[tmp_index_sketch] = tmp_index_shape.shape[0]
	return C_depths

# global args, best_prec1, nview, vcand

if args.batch_size % nview != 0:
	print('Error: batch size should be multiplication of the number of views,', nview)
	exit()

if args.distributed:
	dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
							world_size=args.world_size)

traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')
# Get number of classes from train directory
num_classes = len([name for name in os.listdir(traindir)])
print("num_classes = '{}'".format(num_classes))

# create model
if args.pretrained:
	print("=> using pre-trained model '{}'".format(args.arch))
	model = models.__dict__[args.arch](pretrained=True)
else:
	print("=> creating model '{}'".format(args.arch))
	model = models.__dict__[args.arch]()

model = FineTuneModel(model, args.arch, num_classes, args.with_softmax)

if not args.distributed:
	if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
		model.features = torch.nn.DataParallel(model.features)
		model.cuda()
	else:
		model = torch.nn.DataParallel(model).cuda()
else:
	model.cuda()
	model = torch.nn.parallel.DistributedDataParallel(model)

# define loss function (criterion) and optimizer
if args.loss_type == 'optimal_transport':
	print('Loss is ', args.loss_type)
	criterion = optimal_transport  # nn.CrossEntropyLoss().cuda()

##optimizer = torch.optim.SGD(model.parameters(), args.lr,
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),  # Only finetunable params
							args.lr,
							momentum=args.momentum,
							weight_decay=args.weight_decay)

# Data loading code
traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])

print('traindir is', traindir)
train_dataset = datasets.ImageFolder(
	traindir,
	transforms.Compose([
		#            transforms.CenterCrop(224),
		transforms.ToTensor(),
		normalize,
	]))

if args.distributed:
	train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
else:
	train_sampler = None

train_loader = torch.utils.data.DataLoader(
	train_dataset, batch_size=args.batch_size, shuffle=False,
	num_workers=args.workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
	datasets.ImageFolder(valdir, transforms.Compose([
		#            transforms.Scale(256),
		#            transforms.CenterCrop(224),
		transforms.ToTensor(),
		normalize,
	])),
	batch_size=args.batch_size, shuffle=False,
	num_workers=args.workers, pin_memory=True)
val_loader.dataset.imgs = sorted(val_loader.dataset.imgs)

# optionally resume from a checkpoint
if args.evaluate:
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			mAP = checkpoint['best_mAP']
			print('best_mAP:', mAP)
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
			sys.stdout.flush()
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))
	# pdb.set_trace()
	accuracy = test(model, train_loader, val_loader)
	print('Accuracy is:', accuracy)
	sys.stdout.flush()

cudnn.benchmark = True
map_list = []
count = 0

for epoch in range(args.start_epoch, args.epochs):
	if args.distributed:
		train_sampler.set_epoch(epoch)

	# adjust_learning_rate(optimizer, epoch)
	sorted_imgs = sorted(train_loader.dataset.imgs)
	train_nsamp = int(len(sorted_imgs) / nview)
	# random permutation
	inds = np.zeros((nview, train_nsamp)).astype('int')
	inds[0] = np.random.permutation(range(train_nsamp)) * nview
	for i in range(1, nview):
		inds[i] = inds[0] + i
	inds = inds.T
	# import pdb; pdb.set_trace()
	# todo just for optimal transport loss
	# inds = inds[:-3, :]
	for i in range(len(inds)):
		np.random.shuffle(inds[i])
	inds = inds.reshape(nview * (train_nsamp))
	train_loader.dataset.imgs = [sorted_imgs[i] for i in inds]

	train_loader.dataset.samples = train_loader.dataset.imgs

	# train for one epoch
	train(train_loader, model, criterion, optimizer, epoch)

	# evaluate on validation set
	sorted_imgs = sorted(val_loader.dataset.imgs)
	valid_nsamp = int(len(sorted_imgs) / nview)
	inds = np.zeros((nview, valid_nsamp)).astype('int')
	inds[0] = np.random.permutation(range(valid_nsamp)) * nview
	for i in range(1, nview):
		inds[i] = inds[0] + i
	inds = inds.T
	for i in range(len(inds)):
		np.random.shuffle(inds[i])
	inds = inds.reshape(nview * valid_nsamp)
	val_loader.dataset.imgs = [sorted_imgs[i] for i in inds]
	val_loader.dataset.samples = val_loader.dataset.imgs
	# prec1 = validate(val_loader, model, criterion)

	torch.cuda.empty_cache()
	if epoch % 1 == 0:
		model.eval()
		end = time.time()
		embed = []
		label = []
		for i, (input, target) in enumerate(val_loader):
			target = target.cuda(async=True)
			target = target[0:-1:nview]  # k
			with torch.no_grad():
				input_var = torch.autograd.Variable(input)
				target_var = torch.autograd.Variable(target)

			# compute output
			output_embed, output_score = model(input_var)
			embed.append(output_embed.data.cpu())
			label.append(target.data.cpu())

		embed = torch.cat(embed, 0)
		label = torch.cat(label, 0)
		print(embed.shape)
		m, n = embed.size(0), embed.size(0)
		distmat = torch.pow(embed, 2).sum(dim=1, keepdim=True).expand(m, n) + \
				  torch.pow(embed, 2).sum(dim=1, keepdim=True).expand(n, m).t()

		distmat.addmm_(1, -2, embed, embed.t())
		distmat = distmat.numpy()

		label = label.numpy()

		C_depths = retrievalParamPP(label, label).astype(int)  ### for retrieval evaluation

		nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec = RetrievalEvaluation(C_depths, distmat, label,
																						  label, testMode=2)
		print(('The NN is %f') % (nn_av))
		sys.stdout.flush()
		print(('The FT is %f') % (ft_av))
		sys.stdout.flush()
		print(('The ST is %f') % (st_av))
		sys.stdout.flush()
		print(('The DCG is %f') % (dcg_av))
		sys.stdout.flush()
		print(('The E is %f') % (e_av))
		sys.stdout.flush()
		print(('The MAP is %f') % (map_))
		sys.stdout.flush()

		accuracy = test(model, train_loader, val_loader)
		print(('The ACC is %f') % (accuracy))
		sys.stdout.flush()

		map_list.append(map_)
		tolerance = 3
		if len(map_list) > tolerance and max(map_list[-tolerance:]) < max(map_list[:-tolerance]):
			# import pdb; pdb.set_trace()
			count += 1
			adjust_learning_rate(optimizer, count)

		# remember best prec@1 and save checkpoint
		'''
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': args.arch,
			'state_dict': model.state_dict(),
			'best_mAP': map_,
			'optimizer': optimizer.state_dict(),
		}, is_best=False, filename=osp.join(args.save_dir, 'checkpoint_npairloss' + str(epoch + 1) + '.pth.tar'))
		'''