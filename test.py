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
import pdb
from tqdm import tqdm

import random
import numpy as np
import os

import datetime
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from model import get_inplanes, BasicBlock, ResNet, Bottleneck

from lifelines.utils import concordance_index

l1_loss = torch.nn.SmoothL1Loss()

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',
					default='./data/train/',
					type=str,
					help='Root directory path')
parser.add_argument('--annotation_path',
					default=None,
					type=str,
					help='Annotation file path')
parser.add_argument('--ckpt_path',
					default='./best.pth',
					type=str,
					help='checkpoint directory')
parser.add_argument('--suffix',
					default=None,
					type=str,
					help='Result directory path')

args = parser.parse_args()


def evaluate():
	#define the network
	net = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes())
	net.load_state_dict(torch.load(args.ckpt_path))    
	net = net.cuda()
	net.eval()

	# define the dataset
	dataset_test=DataBowl3Classifier(args.data_path, phase='test', isAugment=False)
	data_loader=DataLoader(dataset_test, batch_size=64, shuffle=False)

	count = 0
	correct = 0.0
	event_c=torch.tensor([],dtype=torch.bool)
	label_c=torch.tensor([]).to("cuda:0")
	outputs_c=torch.tensor([]).to("cuda:0")

	#start evaluation
	with torch.no_grad():
		for i, data in tqdm(enumerate(data_loader)):
			inputs, clinical, label, event = data
			inputs = inputs.unsqueeze(1)
			label = label.unsqueeze(1)

			inputs = inputs.repeat(1, 3, 1, 1, 1)
			inputs = inputs.float().to("cuda:0")
			label = label.float().to("cuda:0")
			clinical = clinical.float().to("cuda:0")

			outputs, z1, z2 = net(inputs,clinical)
			event_c=torch.cat((event_c,event))
			label_c=torch.cat((label_c,label))
			outputs_c=torch.cat((outputs_c,outputs))

			correct += torch.sum(torch.abs(outputs - label)).data
			ctd = concordance_index_censored(event_c.view(-1).cpu().detach().numpy(), label_c.view(-1).cpu().detach().numpy(),outputs_c.view(-1).cpu().detach().numpy())

	accuracy = correct/len(data_loader.dataset)
	con = 1 - ctd[0]
	print('concordance = %.4f, MAE = %.4f' % (con, accuracy))

evaluate()
