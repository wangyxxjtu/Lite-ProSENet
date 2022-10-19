import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import pdb

def get_inplanes():
	return [64, 128, 256, 512]

def conv3x3x3(in_planes, out_planes, stride=1):
	return nn.Conv3d(in_planes,
					 out_planes,
					 kernel_size=3,
					 stride=stride,
					 padding=1,
					 bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
	return nn.Conv3d(in_planes,
					 out_planes,
					 kernel_size=1,
					 stride=stride,
					 bias=False)

hidden_size=24
class Attention(nn.Module):
	def __init__(self):
		super(Attention, self).__init__()
		self.query_fc = nn.Linear(hidden_size, hidden_size)
		self.key_fc = nn.Linear(hidden_size, hidden_size)
		self.value_fc = nn.Linear(hidden_size, hidden_size)
		self.posi_emb = nn.Parameter(torch.zeros(1,1,hidden_size))

	def forward(self, clinical_emb):
		clinical_emb = clinical_emb + self.posi_emb
		mat_query, mat_key, mat_value = self.query_fc(clinical_emb), self.key_fc(clinical_emb), self.value_fc(clinical_emb) 
		simi = F.softmax(mat_query @ mat_key.permute(0,2,1) / math.sqrt(hidden_size), dim=-1)
		attn_out = simi @ mat_value
		
		#attn_out = self.clinical_attn_fusion(attn_out.view(clinical_emb.shape[0], -1))
		
		#return attn_out[:,:6], attn_out[:,6:]
		return attn_out + clinical_emb

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, downsample=None, frames=8):
		super().__init__()

		self.conv1 = conv3x3x3(in_planes, planes, stride)
		self.bn1 = nn.BatchNorm3d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3x3(planes, planes)
		self.bn2 = nn.BatchNorm3d(planes)
		self.downsample = downsample
		self.stride = stride
		self.frames = frames

		self.se_linear = nn.Sequential(
				nn.Linear(planes, planes//2),
				nn.ReLU(),
				nn.Linear(planes//2, planes),
				nn.Sigmoid()
				)

		if self.frames:
			self.tmp_linear = nn.Sequential(
					nn.Linear(frames, 2),
					nn.ReLU(),
					nn.Linear(2, frames),
					nn.Sigmoid()
					)

	def spatial_SE_single(self,fea):
		#fea: B F, C, H, W
		b,c,f,h,w = fea.shape
		#pdb.set_trace()
		fea_ = fea.view(b,c,f,-1)
		#fea_ = fea_.permute(0,2,1)
		pool = torch.mean(fea_, dim=-1)
		pool = pool.permute(0,2,1)
		#try:
		scale = self.se_linear(pool)
		scale = scale.permute(0,2,1)
		#except:
		#	pdb.set_trace()
		
		#pdb.set_trace()
		#fea = fea*scale[:,:,:,None,None]

		return scale[:,:,:,None,None]		

	def spatial_SE(self,fea):
		#fea: B F, C, H, W
		b,c,f,h,w = fea.shape
		#pdb.set_trace()
		fea_ = fea.view(b,c,-1)
		pool = torch.mean(fea_, dim=-1)
		#try:
		scale = self.se_linear(pool)
		#except:
		#	pdb.set_trace()
		
		#pdb.set_trace()
		single_scale = self.spatial_SE_single(fea)
		fea = fea*(scale[:,:,None,None,None]*single_scale)

		return fea		

	def temporal_SE(self, fea):
            #fea: B F, C, H, W
            b,c,f,h,w = fea.shape
            #pdb.set_trace()
            fea_ = fea.permute(0,2,1,3,4)
            fea_ = fea_.reshape(b,self.frames,-1)
            pool = torch.mean(fea_, dim=-1)
            #pdb.set_trace()
            #try:
            scale = self.tmp_linear(pool)
            #except:
            #	pdb.set_trace()
            scale_single = self.temporal_SE_single(fea)
            scale = scale[:,None]
            #pdb.set_trace()
            fea = fea*(scale[:,:,:, None,None] * scale_single)

            return fea

	def temporal_SE_single(self, fea):
		#fea: B F, C, H, W
		b,c,f,h,w = fea.shape
		#pdb.set_trace()
		fea_ = fea.reshape(b,c, self.frames,-1)
		pool = torch.mean(fea_, dim=-1)
		#pdb.set_trace()
		#try:
		scale = self.tmp_linear(pool)
		#except:
		#	pdb.set_trace()
		#fea = fea*scale[:,:,:, None,None]
		return scale[:,:,:, None,None]

		#return fea

	def forward(self, x):
		#pdb.set_trace()
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out = self.spatial_SE(out)
		if self.frames:
			out = self.temporal_SE(out)

		out += residual
		out = self.relu(out)

		return out

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1, downsample=None):
		super().__init__()

		self.conv1 = conv1x1x1(in_planes, planes)
		self.bn1 = nn.BatchNorm3d(planes)
		self.conv2 = conv3x3x3(planes, planes, stride)
		self.bn2 = nn.BatchNorm3d(planes)
		self.conv3 = conv1x1x1(planes, planes * self.expansion)
		self.bn3 = nn.BatchNorm3d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		out += residual
		if self.downsample is not None:
			residual = self.downsample(x)


		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self,
				 block,
				 layers,
				 block_inplanes,
				 n_input_channels=3,
				 conv1_t_size=7,
				 conv1_t_stride=1,
				 no_max_pool=False,
				 shortcut_type='B',
				 widen_factor=1.0,
				 n_classes=400):
		super().__init__()

		block_inplanes = [int(x * widen_factor) for x in block_inplanes]

		self.in_planes = block_inplanes[0]
		self.no_max_pool = no_max_pool

		self.conv1 = nn.Conv3d(n_input_channels,
							   self.in_planes,
							   kernel_size=(conv1_t_size, 7, 7),
							   stride=(conv1_t_stride, 2, 2),
							   padding=(conv1_t_size // 2, 3, 3),
							   bias=False)
		self.bn1 = nn.BatchNorm3d(self.in_planes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
									   shortcut_type, frames=6)
		self.layer2 = self._make_layer(block,
									   block_inplanes[1],
									   layers[1],
									   shortcut_type,
									   stride=2, frames=3)
		self.layer3 = self._make_layer(block,
									   block_inplanes[2],
									   layers[2],
									   shortcut_type,
									   stride=2, frames=None)
		self.layer4 = self._make_layer(block,
									   block_inplanes[3],
									   layers[3],
									   shortcut_type,
									   stride=2, frames=None)

		self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
		self.feat_dim = block_inplanes[3] * block.expansion
		self.drop = nn.Dropout3d(p=0.3)
		self.drop1d = nn.Dropout(p=0.3)
		self.hidden = nn.Linear(block_inplanes[3] * block.expansion +27, 50)
		self.fc = nn.Linear(50, 1)
		self.fc1 = nn.Linear(block_inplanes[3] * block.expansion, 25)

		#self.fc3 = nn.Linear(25 + hidden_size*4,1)
		self.fc3 = nn.Sequential(
				nn.Linear(25 + hidden_size*4,hidden_size),
				nn.BatchNorm1d(hidden_size),
				nn.ReLU(),
				nn.Linear(hidden_size, 1))
		self.fc4 = nn.Sequential(
				nn.Linear(25 + hidden_size*4,hidden_size),
				nn.BatchNorm1d(hidden_size),
				nn.ReLU(),
				nn.Linear(hidden_size, 1))
		# self.relu = nn.ReLU()
		self.clinical_emb_layer = nn.Linear(27,hidden_size)

		self.attn_layer1 = Attention()
		self.attn_layer2 = Attention()
		#self.attn_layer3=	Attention()
		#self.attn_layer4=	Attention()
		self.clinical_token = nn.Parameter(torch.zeros(1, hidden_size))

		self.clinical_fc=nn.Sequential(
				nn.Linear(hidden_size*8, hidden_size*4),
				nn.BatchNorm1d(hidden_size*4),
				nn.ReLU()
				)

		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				nn.init.kaiming_normal_(m.weight,
										mode='fan_out',
										nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm3d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _downsample_basic_block(self, x, planes, stride):
		out = F.avg_pool3d(x, kernel_size=1, stride=stride)
		zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
								out.size(3), out.size(4))
		if isinstance(out.data, torch.cuda.FloatTensor):
			zero_pads = zero_pads.cuda()

		out = torch.cat([out.data, zero_pads], dim=1)

		return out

	def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, frames=None):
		downsample = None
		if stride != 1 or self.in_planes != planes * block.expansion:
			if shortcut_type == 'A':
				downsample = partial(self._downsample_basic_block,
									 planes=planes * block.expansion,
									 stride=stride)
			else:
				downsample = nn.Sequential(
					conv1x1x1(self.in_planes, planes * block.expansion, stride),
					nn.BatchNorm3d(planes * block.expansion))

		layers = []
		layers.append(
			block(in_planes=self.in_planes,
				  planes=planes,
				  stride=stride,
				  downsample=downsample, frames=frames))
		self.in_planes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.in_planes, planes))

		return nn.Sequential(*layers)

	def attn_op(self, clinical_data):
		clinical_emb = self.clinical_emb_layer(clinical_data)
		A_ = clinical_emb[:,0]
		T_ = torch.sum(clinical_emb[:,1:7], dim=1)
		N_ = torch.sum(clinical_emb[:,7:12], dim=1)
		M_ = torch.sum(clinical_emb[:,12:15], dim=1)
		O_ = torch.sum(clinical_emb[:,15:20], dim=1)
		H_ = torch.sum(clinical_emb[:, 20:25], dim=1)
		G_ = torch.sum(clinical_emb[:, 25:27],dim=1)
		emb = torch.stack([self.clinical_token.expand(A_.shape[0], -1), A_, T_,N_,M_,O_,H_,G_], dim=1)

		attn = self.attn_layer1(emb)
		attn = self.attn_layer2(attn)
		#attn = self.attn_layer4(self.attn_layer3(attn))

		return attn.view(A_.shape[0],-1)

	def forward_(self, x, y):
		#pdb.set_trace()
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		if not self.no_max_pool:
			x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		# print(x.shape)
		x = self.avgpool(x)

		x = x.view(x.size(0), -1)
		x = self.fc1(x)

		x = self.relu(x)

		# print(x.shape)
		# x = self.drop(x)
		out =self.attn_op(y)
		out = self.clinical_fc(out)
		z = torch.cat([x, out], dim=-1)
		# z = self.fc2(z)
		#gate = self.gate_fc(z)
		#z = torch.cat((x, gate*y), 1)
		z1 = self.fc3(z)
		z2 = self.fc4(z)
		# z = self.relu(z)

		z1 = torch.sigmoid(z1)
		z2 = torch.sigmoid(z2)

		return z2*z1, z1, z2

	def forward(self, x, y, w=0.4):
		p12, p1, p2 = self.forward_(x,y)
		
		diff = x[:,:,1:] - x[:,:,:-1]
		diff = F.pad(diff, (0,0,0,0,1,0))
		q12,q1,q2 = self.forward_(diff,y)

		diff = x[:,:,:-1] - x[:,:,1:]
		diff = F.pad(diff, (0,0,0,0,0,1))

		o12,o1,o2 = self.forward_(diff,y)
	

		return w*p12+(1-w)*(q12+o12)/2., w*p1+(1-w)*(q1+o1)/2., w*p2+(1-w)*(q2+o2)/2.


def generate_model(model_depth, **kwargs):
	assert model_depth in [10, 18, 34, 50, 101, 152, 200]

	if model_depth == 10:
		model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
	elif model_depth == 18:
		model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
	elif model_depth == 34:
		model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
	elif model_depth == 50:
		model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
	elif model_depth == 101:
		model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
	elif model_depth == 152:
		model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
	elif model_depth == 200:
		model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

	return model

if __name__ == '__main__':
	net =ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes())
	data_dict = net.state_dict()
	x = torch.rand(8, 3, 8, 64, 64)
	pdb.set_trace()
	y = torch.rand(8, 27, 27)
	out = net(x, y)
	print(out[0].shape)
