"""Train CIFAR10/100 with PyTorch."""

#------------------------------------------------------------------------------
# System module.
#------------------------------------------------------------------------------
import os
import random
import time
import copy
import argparse
import sys

#------------------------------------------------------------------------------
# Torch module.
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#------------------------------------------------------------------------------
# Numpy module.
#------------------------------------------------------------------------------
import numpy as np
import numpy.matlib

#------------------------------------------------------------------------------
# DNN module
#------------------------------------------------------------------------------
from models import *
from aegdm import *



def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dateset',
                        choices=['cifar10','cifar100'])
    parser.add_argument('--model', default='resnet32', type=str, help='model',
                        choices=['cifarnet', 'vgg16',
                        'resnet20','resnet32', 'resnet56','densenetbc12',
                        'squeezenet', 'googlenet'])
    parser.add_argument('--optim', default='SGDM', type=str, help='optimizer',
                        choices=['SGDM','Adam','AEGD', 'AEGDM'])
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGDM momentum term')
    parser.add_argument('--c', default=1, type=float, help='AEGD(M) constant')
    parser.add_argument('--milestones', type=int, default=[150],
                        help='list of epoch indices', nargs='+')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='multiplicative factor of learning rate decay')
    parser.add_argument('--bs', type=int, default=128, help='batch_size')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs to train')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay for optimizers')
    return parser


def build_dataset(args):
    print('==> Preparing dataset {}'.format(args.dataset))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=0)

    testset = dataloader(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=0)

    return train_loader, test_loader


def get_ckpt_name(dataset='cifar10', model='resnet56', optimizer='SGD', lr=0.01, ):
    name = {
        'SGDM': 'lr{}'.format(lr),
        'Adam': 'lr{}'.format(lr),
        'AEGD': 'lr{}'.format(lr),
        'AEGDM': 'lr{}'.format(lr)
        }[optimizer]
    return '{}-{}-{}-{}'.format(dataset, model, optimizer, name)


def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
    return torch.load(path)


def build_model(args, device, ckpt=None):
    print("==> Building model '{}'".format(args.model))
    net = {
        # cifar10 models
        'cifarnet': cifarnet,
        'vgg16': vgg16,
        'resnet20': resnet20,
        'resnet32': resnet32,
        'resnet56': resnet56,
        'densenetbc12': densenetbc12,
        # cifar100 models
        'squeezenet': squeezenet,
        'googlenet': googlenet
    }[args.model]()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt['net'])

    return net


def create_optimizer(args, model_params):
    if args.optim == 'SGDM':
        return optim.SGD([{'params': model_params, 'initial_lr': args.lr}],
                         args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        return optim.Adam([{'params': model_params, 'initial_lr': args.lr}],
                          args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'AEGD':
        return AEGDM([{'params': model_params, 'initial_lr': args.lr}],
                    args.lr, c=args.c, momentum=0, weight_decay=args.weight_decay)
    else:
        assert args.optim == 'AEGDM'
        return AEGDM([{'params': model_params, 'initial_lr': args.lr}],
                    args.lr, c=args.c, weight_decay=args.weight_decay)


def create_lr_scheduler(args, optimizer, start_epoch):
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                         gamma=args.gamma, last_epoch=start_epoch)




def train(args, net, epoch, device, data_loader, optimizer, criterion):
    print('\nEpoch: {}'.format(epoch))
    net.train()
    train_loss = []
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        if args.optim in {'AEGD', 'AEGDM'}:
            def closure():
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                return loss #, outputs
            loss = optimizer.step(closure)
            # use "loss, outputs = optimizer.step(closure)" instead
            # if you need "outputs" to evaluate training accuracy
            # also revise the output of the closure
            # and the output of AEGD.step in the algorithm correspondingly
        else:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if (batch_idx + 1) % 1 == 0:
            print('Epoch: [{}/{}], Step: [{}/{}], Loss: {}'.format(
                   epoch + 1, args.num_epochs, batch_idx + 1, len(data_loader),
                   loss.item()))

        train_loss.append(loss.item())
        #_, predicted = outputs.max(1)
        #total += targets.size(0)
        #correct += predicted.eq(targets).sum().item()

    epoch_loss = sum(train_loss) / len(train_loss)
    #accuracy = 100. * correct / total
    print('train loss: {:.4f}'.format(epoch_loss))

    return epoch_loss


def test(net, device, data_loader, criterion):
    net.eval()
    testloss = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            #testloss.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    #epochloss = sum(testloss)/len(testloss)
    accuracy = 100. * correct / total
    print('test acc: {:.4f}'.format(accuracy))

    return accuracy#, epochloss


def main():
    parser = get_parser()
    args = parser.parse_args()

    train_loader, test_loader = build_dataset(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_name = get_ckpt_name(dataset=args.dataset, model=args.model,
                              optimizer=args.optim, lr=args.lr)
    if args.resume:
        ckpt = load_checkpoint(ckpt_name)
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']
    else:
        ckpt = None
        best_acc = 0
        start_epoch = -1

    net = build_model(args, device, ckpt=ckpt)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, net.parameters())
    scheduler = create_lr_scheduler(args, optimizer, start_epoch)

    #train_accuracies = []
    train_losses = []
    test_accuracies = []
    #test_losses = []

    for epoch in range(start_epoch + 1, args.num_epochs):
        train_loss = train(args, net, epoch, device, train_loader, optimizer, criterion)
        test_acc = test(net, device, test_loader, criterion)
        scheduler.step()

        # Save checkpoint.
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, os.path.join('checkpoint', ckpt_name))
            best_acc = test_acc

        #train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        test_accuracies.append(test_acc)
        #test_losses.append(test_loss)

        if not os.path.isdir('curve'):
            os.mkdir('curve')
        torch.save({'train_loss': train_losses, 'test_acc': test_accuracies},
                    #'train_acc': train_accuracies 'test_loss': test_losses},
                   os.path.join('curve', ckpt_name))


if __name__ == '__main__':
    main()
