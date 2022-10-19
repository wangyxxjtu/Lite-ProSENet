import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from data import DataBowl3Classifier
import model
import argparse

import random
import numpy as np


random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic=True

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

def train_one_epoch(net, data_loader, entropy_loss, optimizer):
    ################  train model ##########################
    # one epoch
    train_loss = 0
    count = 0
    net.train()
    # print(f"data_loader:{data_loader}")
    correct = 0.0
    for i, data in enumerate(data_loader):
        # print(f"i:{1},data:{data}")
        inputs, label = data
        # input the data into this model
        inputs = inputs.unsqueeze(1)
        print(inputs.shape)
        inputs = inputs.repeat(1, 3, 1, 1, 1)
        inputs = inputs.float()

        outputs = net(inputs)
        # calculate the cross entropy loss
        loss = entropy_loss(outputs, label)
        # calculate the gradient
        loss.backward()
        # update the weights
        optimizer.step()
        train_loss = train_loss + loss.item()
        count = count + 1
        correct += ((outputs.max(1)[1] == label).sum()).double()
        # print(outputs.max(1)[1])

        print(train_loss/count)
    # print('training acc is {:.3f}'.format(correct /((i+1)*inputs.shape[0])))
    # print('training acc is {:.3f}'.format(correct / ((i + 1)*inputs.shape[0])))
    print('training acc is {:.3f}'.format(correct / 436))
    # get the average training loss
    train_loss = train_loss / count
    print("train loss= %f"%train_loss)
    return train_loss




def validate_one_epoch(net, test_loader, entropy_loss):
    # test the mode
    # one epoch
    net.eval()
    test_loss = 0
    count = 0
    correct = 0
    for i, data in enumerate(test_loader):
        inputs, label = data
        # input the data into this model
        inputs = inputs.unsqueeze(0)
        inputs = inputs.repeat(1, 3, 1, 1, 1)
        inputs = inputs.float()
        outputs = net(inputs)
        # calculate the cross entropy loss
        loss = entropy_loss(outputs, label)

        test_loss = test_loss + loss.item()
        count = count + 1
        correct += ((outputs.max(1)[1] == label).sum()).double()
    print('validate acc is {:.3f}'.format(correct / ((i + 1) * inputs.shape[0])))
    # get the average testing loss
    test_loss = test_loss / count
    print("validate loss= %f" % test_loss)
    return test_loss


def load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')

        model.load_state_dict(pretrain['state_dict'])
        # model.load_state_dict(pretrain)
        tmp_model = model
        if model_name == 'densenet':
            tmp_model.classifier = nn.Linear(tmp_model.classifier.in_features,
                                             n_finetune_classes)
        else:
            tmp_model.fc = nn.Linear(tmp_model.fc.in_features,
                                     n_finetune_classes)

    return model

def train():
    ################  load data ##########################
    batch_size = 1
    workers =2
    train_path = "/data/yujwu/NSCLC/survival_estimate/survival_est_xh/data/train"
    val_path = "/data/yujwu/NSCLC/survival_estimate/survival_est_xh/data/val"
    # path = "/data/yujwu/NSCLC/survival_estimate/survival_est_xh/data/evaluat"
    #
    # if phase == "train":
    #     path = "/data/yujwu/NSCLC/survival_estimate/survival_est_xh/data/train"


    #path = "/data/yujwu/NSCLC/survival_estimate/tumor_segment/seg_output"
    dataset_train = DataBowl3Classifier(train_path, phase = 'train')
    dataset_val = DataBowl3Classifier(val_path, phase='val')

    # if phase == "evaluate":
    #     path = "/data/yujwu/NSCLC/survival_estimate/survival_est_xh/data/evaluat"
    #
    #     dataset = DataBowl3Classifier(path, phase = 'evaluate')
    #
    #
    train_loader_case = DataLoader(dataset_train, batch_size = 64*batch_size,shuffle = True)
    val_loader_case = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    ################  define model, loss and optimizer ##########################
    # net = model.DPN92_3D()
    # r3d18_K_200ep.pth: --model resnet --model_depth 18 --n_pretrain_classes 700
    n_input_channels = 32
    net = model.generate_model(model_depth= 18,
                                  # n_classes= 2,
                                  n_classes=700,
                                  n_input_channels= 3,
                                  shortcut_type='B',
                                  conv1_t_size=7,
                                  conv1_t_stride=1,
                                  no_max_pool=True,
                                  widen_factor=1.0)

    # restore the pretrained model
    pretrain_path = "./r3d18_K_200ep.pth"
    n_finetune_classes = 2
    net = load_pretrained_model(net, pretrain_path, "resnet", n_finetune_classes)

    # revise the last fully connected layer
    output_class = 2
    net.fc = nn.Linear(net.feat_dim, output_class)

    # define loss
    entropy_loss = torch.nn.CrossEntropyLoss()
    # define optimizer
    #learnable_params = filter(lambda p: p.requires_grad, net.parameters())
    #optimizer = torch.optim.Adam(learnable_params, lr=1e-4)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-5)
    #print('learning rate', optimizer.state_dict()['param_groups'])


    ################  train  model ##########################
    num_epoch = 200
    best_loss = 9999.9
    for i in range(num_epoch):

        train_loss = train_one_epoch(net, train_loader_case, entropy_loss, optimizer)

        val_loss = validate_one_epoch(net, val_loader_case,entropy_loss)
        # save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), "best.pth")

        # save the last trained model
        torch.save(net.state_dict(), "checkpoint.pth")

    print("test complete")


if __name__ == '__main__':
    train()