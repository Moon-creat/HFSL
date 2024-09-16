# ============================================================================
# HFSLV2 learning: ResNet18 on FMNIST
# ============================================================================
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import torchvision
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
from pandas import DataFrame

import random
import numpy as np
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))

# ===================================================================
program = "HSFLV1-"
print(f"---------{program}----------")  # this is to identify the program in the slurm outputs files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m".format(skk))


def prGreen(skk): print("\033[92m {}\033[00m".format(skk))


# ===================================================================
# No. of users
num_users = 5
epochs = 100
frac = 1  # participation of clients; if 1 then 100% clients participate in HFSLV2
lr = 0.0005

def branchBottleNeck(channel_in, channel_out, kernel_size):
    middle_channel = channel_out // 4
    return nn.Sequential(
        nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),

        nn.Conv2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=kernel_size),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),

        nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
    )


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv1x1(in_planes, planes, stride=1):
    return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)

        if self.downsample is not None:
            residual = self.downsample(x)

        output += residual
        output = self.relu(output)

        return output

# =====================================================================================================
#                           Client-side Model definition
# =====================================================================================================
# Model at client side
class ResNet18_client_side(nn.Module):
    def __init__(self):
        super(ResNet18_client_side, self).__init__()
        expansion = 1
        num_classes = 10
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        self.bottleneck1_1 = branchBottleNeck(64 * expansion, 512 * expansion, kernel_size=8)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc1 = nn.Linear(512 * expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        resudial1 = F.relu(self.layer1(x))
        out1 = self.layer2(resudial1)
        out1 = out1 + resudial1  # adding the resudial inputs -- downsampling not required in this layer
        resudial2 = F.relu(out1)
        middle_output1 = self.bottleneck1_1(resudial2)
        middle_output1 = self.avgpool1(middle_output1)
        middle1_fea = middle_output1
        middle_output1 = torch.flatten(middle_output1, 1)
        middle_output1 = self.middle_fc1(middle_output1)
        return resudial2, middle_output1, middle1_fea


class ResNet18_client1_side(nn.Module):
    def __init__(self):
        super(ResNet18_client1_side, self).__init__()
        expansion = 1
        num_classes = 10
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # self.layer2 = nn.Sequential  (
        #         nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
        #         nn.BatchNorm2d(64),
        #         nn.ReLU (inplace = True),
        #         nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
        #         nn.BatchNorm2d(64),
        #     )
        self.bottleneck1_1 = branchBottleNeck(64 * expansion, 512 * expansion, kernel_size=8)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc1 = nn.Linear(512 * expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        resudial1 = F.relu(self.layer1(x))
        resudial2 = resudial1
        middle_output1 = self.bottleneck1_1(resudial2)
        middle_output1 = self.avgpool1(middle_output1)
        middle1_fea = middle_output1
        middle_output1 = torch.flatten(middle_output1, 1)
        middle_output1 = self.middle_fc1(middle_output1)
        return resudial2, middle_output1, middle1_fea


class ResNet18_client2_side(nn.Module):
    def __init__(self):
        super(ResNet18_client2_side, self).__init__()
        expansion = 1
        num_classes = 10
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.bottleneck1_1 = branchBottleNeck(64 * expansion, 512 * expansion, kernel_size=8)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc1 = nn.Linear(512 * expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        resudial1 = F.relu(self.layer1(x))
        out1 = self.layer2(resudial1)
        out1 = out1 + resudial1  # adding the resudial inputs -- downsampling not required in this layer
        resudial2 = F.relu(out1)
        middle_output1 = self.bottleneck1_1(resudial2)
        middle_output1 = self.avgpool1(middle_output1)
        middle1_fea = middle_output1
        middle_output1 = torch.flatten(middle_output1, 1)
        middle_output1 = self.middle_fc1(middle_output1)
        return resudial2, middle_output1, middle1_fea


net_glob_client = ResNet18_client_side()
if torch.cuda.device_count() > 1:
    print("We use", torch.cuda.device_count(), "GPUs")
    net_glob_client = nn.DataParallel(net_glob_client)

net_glob_client.to(device)
print(net_glob_client)


# =====================================================================================================
#                           Server-side Model definition
# =====================================================================================================
# Model at server side
class Baseblock(nn.Module):
    expansion = 1

    def __init__(self, input_planes, planes, stride=1, dim_change=None):
        super(Baseblock, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, stride=stride, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride=1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dim_change = dim_change

    def forward(self, x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))

        if self.dim_change is not None:
            res = self.dim_change(res)

        output += res
        output = F.relu(output)

        return output


class ResNet18_server_side(nn.Module):
    def __init__(self, block, num_layers, classes):
        super(ResNet18_server_side, self).__init__()
        self.input_planes = 64
        expansion = 1
        num_classes = 10
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

        self.layer4 = self._layer(block, 128, num_layers[0], stride=2)
        self.layer5 = self._layer(block, 256, num_layers[1], stride=2)
        self.layer6 = self._layer(block, 512, num_layers[2], stride=2)
        self.averagePool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, classes)
        self.bottleneck1_1 = branchBottleNeck(64 * block.expansion, 512 * block.expansion, kernel_size=4)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc1 = nn.Linear(512 * block.expansion, num_classes)

        self.bottleneck2_1 = branchBottleNeck(128 * block.expansion, 512 * block.expansion, kernel_size=2)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc2 = nn.Linear(512 * block.expansion, num_classes)

        self.bottleneck3_1 = branchBottleNeck(256 * block.expansion, 512 * block.expansion, kernel_size=2)
        self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc3 = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _layer(self, block, planes, num_layers, stride=2):
        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = nn.Sequential(
                nn.Conv2d(self.input_planes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride=stride, dim_change=dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion

        return nn.Sequential(*netLayers)

    def forward(self, x):
        out2 = self.layer3(x)
        out2 = out2 + x  # adding the resudial inputs -- downsampling not required in this layer
        middle_output, middle_fea = [], []
        x3 = F.relu(out2)

        middle_output4 = x3
        middle_output4 = self.bottleneck1_1(x3)
        middle_output4 = self.avgpool1(middle_output4)
        middle_fea4 = middle_output4
        middle_output4 = torch.flatten(middle_output4, 1)
        middle_output4 = self.middle_fc1(middle_output4)

        x4 = self.layer4(x3)
        middle_output2 = x4
        middle_output2 = self.bottleneck2_1(x4)
        middle_output2 = self.avgpool2(middle_output2)
        middle_fea2 = middle_output2
        middle_output2 = torch.flatten(middle_output2, 1)
        middle_output2 = self.middle_fc2(middle_output2)

        x5 = self.layer5(x4)
        middle_output3 = x5
        middle_output3 = self.bottleneck3_1(x5)
        middle_output3 = self.avgpool3(middle_output3)
        middle_fea3 = middle_output3
        middle_output3 = torch.flatten(middle_output3, 1)
        middle_output3 = self.middle_fc3(middle_output3)
        x6 = self.layer6(x5)

        middle_output.append(middle_output4)
        middle_output.append(middle_output2)
        middle_output.append(middle_output3)
        middle_fea.append(middle_fea4)
        middle_fea.append(middle_fea2)
        middle_fea.append(middle_fea3)
        # print("x6_shape",x6.shape)
        x7 = F.avg_pool2d(x6, 2)
        final_fea = x7
        x8 = x7.view(x7.size(0), -1)
        y_hat = self.fc(x8)

        return y_hat, final_fea, middle_output, middle_fea


net_glob_server = ResNet18_server_side(Baseblock, [2, 2, 2], 10)  # 10 is my numbr of classes
if torch.cuda.device_count() > 1:
    print("We use", torch.cuda.device_count(), "GPUs")
    net_glob_server = nn.DataParallel(net_glob_server)  # to use the multiple GPUs

net_glob_server.to(device)
print(net_glob_server)

# ===================================================================================
# For Server Side Loss and Accuracy
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []

criterion = nn.CrossEntropyLoss()
count1 = 0
count2 = 0


# ====================================================================================================
#                                  Server Side Program
# ====================================================================================================
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc


# to print train - test together in each round-- these are made global
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

# client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False


# optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)

# Server-side function associated with Training
def kd_loss_function(output, target_output, temperature):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output = output / temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    return loss_kd


def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()


# Server-side function associated with Training
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch, middle_output1, middle1_fea, alpha, beta,
                 temperature):
    global net_glob_server, criterion, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect
    global loss_train_collect_user, acc_train_collect_user, lr

    net_glob_server.train()
    optimizer_server = torch.optim.Adam(net_glob_server.parameters(), lr=lr)

    # train and update
    optimizer_server.zero_grad()

    fx_client = fx_client.to(device)
    y = y.to(device)

    # ---------forward prop-------------
    fx_server, final_fea, middle_output, middle_fea = net_glob_server(fx_client)

    # calculate loss
    loss = criterion(fx_server, y)
    middlel_loss1 = criterion(middle_output1, y)
    middlel_loss2 = criterion(middle_output[0], y)
    middlel_loss3 = criterion(middle_output[1], y)
    middlel_loss4 = criterion(middle_output[2], y)

    temp4 = fx_server / temperature
    temp4 = torch.softmax(temp4, dim=1)

    loss1by1 = kd_loss_function(middle_output1, temp4.detach(), temperature) * (temperature ** 2)
    feature_loss_1 = feature_loss_function(middle1_fea, final_fea.detach())

    loss1by2 = kd_loss_function(middle_output[0], temp4.detach(), temperature) * (temperature ** 2)
    feature_loss_2 = feature_loss_function(middle_fea[0], final_fea.detach())

    loss1by3 = kd_loss_function(middle_output[1], temp4.detach(), temperature) * (temperature ** 2)
    feature_loss_3 = feature_loss_function(middle_fea[1], final_fea.detach())

    loss1by4 = kd_loss_function(middle_output[2], temp4.detach(), temperature) * (temperature ** 2)
    feature_loss_4 = feature_loss_function(middle_fea[2], final_fea.detach())

    total_loss = (1 - alpha) * (loss + middlel_loss1 + middlel_loss2 + middlel_loss3 + middlel_loss4) + \
                 alpha * (loss1by1 + loss1by2 + loss1by3 + loss1by4) + \
                 beta * (feature_loss_1 + feature_loss_2 + feature_loss_3 + feature_loss_4)

    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)

    # --------backward prop--------------

    total_loss.backward(retain_graph=True)
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()

    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())

    # count1: to track the completion of the local batch associated with one client
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)  # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)

        batch_acc_train = []
        batch_loss_train = []
        count1 = 0

        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train,
                                                                                      loss_avg_train))
        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch - 1:

            l_epoch_check = True  # to evaluate_server function - to check local epoch has completed or not
            
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train

            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)

            # collect the id of each new user
            if idx not in idx_collect:
                idx_collect.append(idx)

        # This is to check if all users are served for one round --------------------
        if len(idx_collect) == num_users:
            fed_check = True  # to evaluate_server function  - to check fed check has hitted
            
            idx_collect = []

            acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)

            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)

            acc_train_collect_user = []
            loss_train_collect_user = []

    # send gradients to the client
    return dfx_client


# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_glob_server, criterion, batch_acc_test, batch_loss_test
    global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train

    net_glob_server.eval()

    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device)
        # ---------forward prop-------------
        fx_server, final_fee, a, b = net_glob_server(fx_client)

        # calculate loss
        loss = criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)

        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())

        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test) / len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test) / len(batch_loss_test)

            batch_acc_test = []
            batch_loss_test = []
            count2 = 0

            prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test,
                                                                                             loss_avg_test))

            # if a local epoch is completed
            if l_epoch_check:
                l_epoch_check = False

                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test

                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)

            # if all users are served for one round ----------
            if fed_check:
                fed_check = False

                acc_avg_all_user = sum(acc_test_collect_user) / len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user) / len(loss_test_collect_user)

                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user = []

                print("====================== SERVER V2==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train,
                                                                                          loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user,
                                                                                         loss_avg_all_user))
                print("==========================================================")

    return


# ==============================================================================================================
#                                       Clients-side Program
# ==============================================================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, net, idx, lr, device, dataset_train=None, dataset_test=None, idxs=None, idxs_test=None):
        self.net = net
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        # self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=128, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=16, shuffle=True)
        self.net.to(self.device)

    def setparam(self, net_dict):
        dict_param = (self.net).state_dict()
        for key, val in net_dict.items():
            if key in dict_param:
                dict_param[key] = net_dict[key]
        self.net.load_state_dict(dict_param)

    def train(self):
        self.net.train()
        optimizer_client = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                # ---------forward prop-------------
                fx, middle_output1, middle1_fea = self.net(images)
                client_fx = fx.clone().detach().requires_grad_(True)

                # Sending activations to server and receiving gradients from server
                dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch, middle_output1,
                                   middle1_fea, 0.0001, 1e-6, 4)
                # --------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()

            # prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))

        return self.net.state_dict()

    def evaluate(self, ell):
        self.net.eval()

        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                # ---------forward prop-------------
                fx, a, b = self.net(images)

                # Sending activations to server
                evaluate_server(fx, labels, self.idx, len_batch, ell)

            # prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))

        return
    # =====================================================================================================


# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client

def dataset_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# =============================================================================
#                         Data loading
# =============================================================================
mean = [0.485]
std = [0.229]

transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                                      transforms.RandomRotation((-10, 10)),
transforms.CenterCrop(64),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std)
                                      ])

transform_test = transforms.Compose([
transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

dataset_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
dataset_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
dict_users = dataset_iid(dataset_train, num_users)
dict_users_test = dataset_iid(dataset_test, num_users)

# ------------ Training And Testing  -----------------
# Federation takes place after certain local epochs in train() client-side
# this epoch is global epoch, also known as rounds
for iter in range(epochs):
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)
    w_locals_client = []
    for idx in idxs_users:
        net_dict = net_glob_client.state_dict()
        local = None
        if (idx % 3 == 0):
            local = Client(ResNet18_client_side(), idx, lr, device, dataset_train=dataset_train,
                           dataset_test=dataset_test, idxs=dict_users[idx], idxs_test=dict_users_test[idx])
            local.setparam(net_glob_client.state_dict())
        elif (idx % 3 == 1):
            local = Client(ResNet18_client1_side(), idx, lr, device, dataset_train=dataset_train,
                           dataset_test=dataset_test, idxs=dict_users[idx], idxs_test=dict_users_test[idx])
            local.setparam(net_glob_client.state_dict())
        else:
            local = Client(ResNet18_client2_side(), idx, lr, device, dataset_train=dataset_train,
                           dataset_test=dataset_test, idxs=dict_users[idx], idxs_test=dict_users_test[idx])
            local.setparam(net_glob_client.state_dict())
        # Training ------------------
        w_client = local.train()
        # Testing -------------------
        local.evaluate(ell=iter)

        for key, val in w_client.items():
            net_dict[key] = val
        net_glob_client.load_state_dict(net_dict)
        w_locals_client.append(copy.deepcopy(net_glob_client.state_dict()))
    print("-----------------------------------------------------------")
    print("------ FedServer: Federation process at Client-Side ------- ")
    print("-----------------------------------------------------------")
    w_glob_client = FedAvg(w_locals_client)
    # Update client-side global model
    net_glob_client.load_state_dict(w_glob_client)

# ===================================================================================

print("Training and Evaluation completed!")

# ===============================================================================
# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, len(acc_train_collect) + 1)]
df = DataFrame({'round': round_process, 'acc_train': acc_train_collect, 'acc_test': acc_test_collect})
file_name = "HFSLV2-FMNIST" + ".xlsx"
df.to_excel(file_name, sheet_name="v1_test", index=False)

# =============================================================================
#                         Program Completed
# =============================================================================


