import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from data import DataBowl3Classifier
import model
import argparse

import random

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic=True

def load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')

        # model.load_state_dict(pretrain['state_dict'])
        model.load_state_dict(pretrain)
        tmp_model = model
        if model_name == 'densenet':
            tmp_model.classifier = nn.Linear(tmp_model.classifier.in_features,
                                             n_finetune_classes)
        else:
            tmp_model.fc = nn.Linear(tmp_model.fc.in_features,
                                     n_finetune_classes)

    return model


def evaluate_one_epoch(net, test_loader, entropy_loss):
    # test the mode
    # one epoch
    net.eval()
    test_loss = 0
    count = 0
    correct = 0.0
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
    print('evaluate acc is {:.3f}'.format(correct / ((i + 1) * inputs.shape[0])))
    # get the average testing loss
    test_loss = test_loss / count
    print("evaluate loss= %f" % test_loss)
    return test_loss


def evaluate():
    batch_size = 1
    workers = 2
    test_path = "/data/yujwu/NSCLC/survival_estimate/survival_est_xh/data/test"

    dataset_evaluate = DataBowl3Classifier(test_path, phase='test')
    test_loader_case = DataLoader(dataset_evaluate, batch_size=batch_size, shuffle=True)

    output_class = 2
    net = torchvision.models.video.r3d_18(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False

    num_featdim = net.fc.in_features
    net.fc = nn.Linear(num_featdim, output_class)
    print(num_featdim)

    # restore the pretrained model
    pretrain_path = "./best.pth"
    n_finetune_classes = 2
    net = load_pretrained_model(net, pretrain_path, "resnet", n_finetune_classes)

    # define loss
    entropy_loss = torch.nn.CrossEntropyLoss()
    # define optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-5)

    evaluate_loss = evaluate_one_epoch(net, test_loader_case, entropy_loss)

    print("evaluate complete")

if __name__ == '__main__':
    evaluate()
