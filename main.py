import logging
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from sksurv.metrics import concordance_index_censored
from data import DataBowl3Classifier
import model
import argparse

import random
import numpy as np
import os

import datetime
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from model import get_inplanes, BasicBlock, ResNet, Bottleneck

from lifelines.utils import concordance_index
import wandb

# print(concordance_index_censored([True,True,False,False,True],[1,2,4,5,3],[1,2,4,3,5]))
# print(concordance_index([1,2,4,5,3],[1,2,4,3,5],[True,True,False,False,True]))

wandb.init(project='jiabaobao-dualSpatialSE')

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic = True

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
best_con = 0.
best_acc = 1e10
l1_loss = torch.nn.SmoothL1Loss()

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

def parse_opts():
	parser = argparse.ArgumentParser()
	parser.add_argument('--root_path',
						default=None,
						type=str,
						help='Root directory path')
	parser.add_argument('--annotation_path',
						default=None,
						type=str,
						help='Annotation file path')
	parser.add_argument('--result_path',
						default=None,
						type=str,
						help='Result directory path')
	parser.add_argument('--suffix',
						default=None,
						type=str,
						help='Result directory path')


def train_one_epoch(net, data_loader, entropy_loss, optimizer, exp_lr_scheduler):
	#####################  train model	##########################
	# one epoch
	global best_con, best_acc
	train_loss = 0
	count = 0
	net.train()
	# print(f"data_loader:{data_loader}")
	correct = 0.0
	event_c=torch.tensor([],dtype=torch.bool)
	label_c=torch.tensor([]).to("cuda:0")
	outputs_c=torch.tensor([]).to("cuda:0")
	for i, data in enumerate(data_loader):
		# print(f"i:{1},data:{data}")
		inputs, clinical, label, event = data
		# print(clinical.shape,'This is clinical shape')
		# print(clinical,'This is clinical data')
		# input the data into this model
		inputs = inputs.unsqueeze(1)
		label = label.unsqueeze(1)

		inputs = inputs.repeat(1, 3, 1, 1, 1)
		inputs = inputs.float().to("cuda:0")
		label = label.float().to("cuda:0")
		clinical = clinical.float().to("cuda:0")


		outputs, z1, z2 = net(inputs,clinical)
		# calculate the cross entropy loss
		loss = entropy_loss(outputs, label) + l1_loss(outputs, label)
		loss_1 = entropy_loss(z1, label)
		loss_2 = l1_loss(z2, label)
		loss_ = loss + 0.5*loss_1 + 0.5*loss_2
		testlabel = label.cpu()
	
		# calculate the gradient
		loss_.backward()
		# update the weights
		optimizer.step()

		exp_lr_scheduler.step()

		train_loss = train_loss + loss_.item()
		count = count + 1
		# correct += ((outputs.max(1)[1] == label).sum()).double()
		# print(outputs.shape, label.shape)
		# correct += sum(abs(outputs - label))
		# correct += (abs(outputs - label).sum()).double()
		correct += torch.sum(torch.abs(outputs - label)).data
		# print(event,label,outputs)
		event_c=torch.cat((event_c,event))
		label_c=torch.cat((label_c,label))
		outputs_c=torch.cat((outputs_c,outputs))

		# print(concordance_index_censored(event.view(-1).cpu().detach().numpy(),label.view(-1).cpu().detach().numpy(),outputs.view(-1).cpu().detach().numpy())
# )
		# print(outputs.max(1)[1])
	  # print(train_loss/count)
	# print('training acc is {:.3f}'.format(correct /((i+1)*inputs.shape[0])))
	# print('training acc is {:.3f}'.format(correct / ((i + 1)*inputs.shape[0])))
	# print("train epoch %f is finished" % count)
	#  get the average training loss
	# accracy = correct/436.0
	# print(concordance_index_censored(event_c.view(-1).cpu().detach().numpy(), label_c.view(-1).cpu().detach().numpy(),
	#								   outputs_c.view(-1).cpu().detach().numpy()))
	# gradurity
	# label_c = torch.round(torch.div(label_c, 4454/4454)).type(torch.int)
	# outputs_c = torch.round(torch.div(outputs_c, 4454/4454)).type(torch.int)
	# print(label_c,outputs_c)
	ctd = concordance_index_censored(event_c.view(-1).cpu().detach().numpy(), label_c.view(-1).cpu().detach().numpy(),
									 outputs_c.view(-1).cpu().detach().numpy())
	# ctd1 = concordance_index(label_c.view(-1).cpu().detach().numpy(),
	#								   outputs_c.view(-1).cpu().detach().numpy(),event_c.view(-1).cpu().detach().numpy())
	# accracy = correct/436.0
	accracy = correct/244.0
	train_loss = train_loss / count
	# print('concordance: ', ctd)
	if (1-ctd[0]) > best_con:
		best_con = 1 - ctd[0]
	if accracy < best_acc:
		best_acc = accracy
	wandb.log({'Best concordance': best_con})
	wandb.log({'Current concordance': 1-ctd[0]})
	wandb.log({'Training loss':train_loss})
	wandb.log({'Lowest MAE':best_acc})
	print('training loss: %.4f, accuracy= %.4f, concordance = %.4f, current best concordance = %.4f, current lowest MAE = %.4f' % (train_loss, accracy, 1-ctd[0], best_con, best_acc))
	# print(ctd1)
	# LOGGER.info('concordance: ', ctd)
	LOGGER.info('training loss: %.4f, accuracy= %.4f, concordance = %.4f' % (train_loss, accracy, 1-ctd[0]))
	train_acc = accracy

	return train_loss, train_acc


def validate_one_epoch(net, test_loader, entropy_loss):
	# test the mode
	# one epoch
	net.eval()
	test_loss = 0
	count = 0
	correct = 0
	for i, data in enumerate(test_loader):
		inputs, clinical, label, event = data
		# input the data into this model
		inputs = inputs.unsqueeze(0)
		label = label.unsqueeze(0)
		inputs = inputs.repeat(1, 3, 1, 1, 1)
		inputs = inputs.float().to("cuda:0")
		label = label.float().to("cuda:0")
		clinical = clinical.float().to("cuda:0")

		outputs, _, _ = net(inputs,clinical)
		# calculate the cross entropy loss
		loss = entropy_loss(outputs, label)

		test_loss = test_loss + loss.item()
		count = count + 1
		# print(outputs.max(1)[1] )
		#correct += ((outputs.max(1)[1] == label).sum()).double()
		#correct += ((outputs.max(1)[1] == label).sum()).float()
		#correct += torch.sum(torch.abs(outputs - label)).double()
		correct += torch.sum(torch.abs(outputs - label)).data
		#print(correct)
		# correct = 0
	# print('validate acc is {:.3f}'.format(correct / ((i + 1) * inputs.shape[0])))
	# get the average testing loss

	val_loss = test_loss / count
	# val_acc=correct / 124.0
	val_acc=correct / 77.0

	print('validate loss: %.4f, accuracy= %.4f' % (val_loss, val_acc))
	LOGGER.info('validate loss: %.4f, accuracy= %.4f' % (val_loss, val_acc))

	# print("validate loss= %f" % test_loss)
	return val_loss, val_acc


def load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes):
	if pretrain_path:
		print('loading pretrained model {}'.format(pretrain_path))
		pretrain=torch.load(pretrain_path, map_location='cpu')

		model.load_state_dict(pretrain['state_dict'])
		# model.load_state_dict(pretrain)
		tmp_model=model
		if model_name == 'densenet':
			tmp_model.classifier=nn.Linear(tmp_model.classifier.in_features,
											 n_finetune_classes)
		else:
			tmp_model.fc=nn.Linear(tmp_model.fc.in_features,
									 n_finetune_classes)

	return model

def train():
	################  load data ##########################
	batch_size=1
	workers=2
	train_path="./data/train"
	val_path="./data/val"
	# path = "/data/yujwu/NSCLC/survival_estimate/survival_est_xh/data/evaluat"
	#
	# if phase == "train":
	#	  path = "/data/yujwu/NSCLC/survival_estimate/survival_est_xh/data/train"

	logging.basicConfig(
		level=logging.DEBUG,
		format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
		filename="training.log")



	# path = "/data/yujwu/NSCLC/survival_estimate/tumor_segment/seg_output"
	dataset_train=DataBowl3Classifier(
		train_path, phase='train', isAugment=True)
	dataset_val=DataBowl3Classifier(val_path, phase='val', isAugment=False)

	# if phase == "evaluate":
	#	  path = "/data/yujwu/NSCLC/survival_estimate/survival_est_xh/data/evaluat"
	#
	#	  dataset = DataBowl3Classifier(path, phase = 'evaluate')
	#
	#
	train_loader_case=DataLoader(
		dataset_train, batch_size=batch_size*64, shuffle=True)
	val_loader_case=DataLoader(
		dataset_val, batch_size=batch_size, shuffle=True)

	################  define model, loss and optimizer ##########################
	# net = model.DPN92_3D()
	# r3d18_K_200ep.pth: --model resnet --model_depth 18 --n_pretrain_classes 700
	# output_class=1
	# # net=torchvision.models.video.r3d_18(pretrained=True)
	# net=torchvision.models.video.r3d_18(pretrained=True)
	# for param in net.parameters():
	#	  param.requires_grad=True
	#
	# num_featdim=net.fc.in_features
	# net.fc=nn.Linear(num_featdim, output_class)
	# net.fc=nn.Linear(num_featdim, 50)

	net = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes())
	wandb.watch(net)
	# print(num_featdim)
	net.to("cuda:0")

	# define loss
	# entropy_loss = torch.nn.CrossEntropyLoss()
	mse_loss=torch.nn.MSELoss()
	# mse_loss = nn.BCELoss()
	# define optimizer
	learnable_params=filter(lambda p: p.requires_grad, net.parameters())
	optimizer=torch.optim.Adam(learnable_params, lr = 0.0005, weight_decay=0.001)
	# optimizer = torch.optim.SGD(learnable_params, lr=0.0001)
	# print('learning rate', optimizer.state_dict()['param_groups'])

	# Decay LR by a factor of 0.1 every 7 epochs
	#exp_lr_scheduler=lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
	exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

	################  train  model ##########################
	num_epoch=800
	best_loss=9999.9
	best_acc=0
	train_loss_list, train_acc_list = [], []
	val_loss_list, val_acc_list = [], []

	writer = SummaryWriter()

	for i in range(num_epoch):
		print("----epoch %d:" % i)
		LOGGER.info("----epoch %d:" % i)
		train_loss, train_acc=train_one_epoch(
			net, train_loader_case, mse_loss, optimizer, exp_lr_scheduler)

		val_loss, val_acc=validate_one_epoch(net, val_loader_case, mse_loss)
		val_acc1=val_acc.item()

		print(len(train_loader_case.dataset),len(val_loader_case.dataset))

		# save the best model
		if val_acc1 < best_loss:
			best_loss=val_acc1
			torch.save(net.state_dict(), "best.pth")

		# save the last trained model
		torch.save(net.state_dict(), "checkpoint.pth")

		writer.add_scalar('Loss/train', train_loss, i)
		writer.add_scalar('Loss/test', val_loss, i)
		writer.add_scalar('Acc/train', train_acc, i)
		writer.add_scalar('Acc/test', val_acc, i)

		train_loss_list.append(train_loss)
		train_acc_list.append(train_acc)
		val_loss_list.append(val_loss)
		val_acc_list.append(val_acc)

	print("test complete")
	LOGGER.info("test complete")

	x = np.arange(num_epoch)
	#plt.subplot(211)
	l1 = plt.plot(x, train_loss_list, 'r--', label='training_loss')
	l2 = plt.plot(x, val_loss_list, 'b--', label='testing_loss')
	plt.title('Loss')
	plt.xlabel('Number of epochs')
	plt.ylabel('Loss values')
	plt.grid()
	plt.legend()
	#plt.show()
	plt.savefig('train_.png')

	#plt.subplot(212)
	l3 = plt.plot(x, train_acc_list, 'g--', label='training_acc')
	l4 = plt.plot(x, val_acc_list, 'y--', label='testing_acc')
	plt.title('Accuracy')
	plt.xlabel('Number of epochs')
	plt.ylabel('Accuracy')
	plt.grid()
	plt.legend()
	#plt.show()
	plt.savefig('test_.png')

	writer.close()

if __name__ == '__main__':
	train()
	print('5_2_hiddenx4layers')
