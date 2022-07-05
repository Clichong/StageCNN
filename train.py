import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os

from utils.dataset import get_dataset
from utils.seed import seed_everything
from utils.logger import get_logger, check_file
from utils.draw import list_to_draw
from model import *

# 定义全局变量参数
best_acc = 0    # best test accuracy
best_epoch = 0  # best test epoch
acc_lists = []
trainloss_lists = []
testloss_lists = []


# 功能: 传参
def get_args():

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--model', default='resnet50', type=str, help='choose the dataset to train')
    parser.add_argument('--dataset', default='pokemon', type=str, help='choose the model')
    parser.add_argument('--resize', default=224, type=int, help='train photo size')
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
    parser.add_argument('--D', default=0.2, type=float, help='Equal difference factor')
    parser.add_argument('--Li', default=3, type=int, help='fuse stage layer numbers')
    parser.add_argument('--device', default='0', type=str, metavar='PATH', help='path to dataset')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--reweight', default=False, type=bool, help='use the Weight redistribution')
    parser.add_argument('--lrweight', default=False, type=bool, help='use the Learnable redistribution')
    parser.add_argument('--mdl_path', default='./record/checkpoint', type=str, help='path to save weights file')
    parser.add_argument('--rec_path', default='./record/log', type=str, help='path to save logging file')
    parser.add_argument('--acc_path', default='./record/chart', type=str, help='path to save acc record photo')
    parser.add_argument('--loss_path', default='./record/loss', type=str, help='path to save train/test loss photo')
    parser.add_argument('--trainloss_path', type=str, help='path to save train/test loss photo')
    parser.add_argument('--textloss_path', type=str, help='path to save train/test loss photo')
    args = parser.parse_args()

    return args


# 功能: 选择模型
def get_model(model, dataset, device):

    # 根据数据集确定分类数目
    if dataset == 'cifar10':
        nclass = 10
    elif dataset == 'cifar100':
        nclass = 100
    elif dataset == 'caltech101':
        nclass = 101
    elif dataset == 'caltech256':
        nclass = 257
    elif dataset == 'pokemon':
        nclass = 5
    elif dataset == 'food101':
        nclass = 101
    else:
        raise ImportError("Setting num_class error. Have no such dataset setting.")

    # 挑选模型，动态设置分类数,
    # 1.最原始模型
    if model == 'resnet18':
        net = ResNet18(num_classes=nclass)
    elif model == 'resnet50':
        net = ResNet50(num_classes=nclass)
    elif model == 'resnext50':
        net = ResNeXt50_32x4d(num_classes=nclass)

    # 2.使用多尺度融合模型
    elif model == 'stage_resnet18':
        net = Stage_ResNet18(num_classes=nclass)
    elif model == 'stage_resnet50':
        net = Stage_ResNet50(num_classes=nclass)
    elif model == 'stage_resnext50':
        net = Stage_ResNeXt50_32x4d(num_classes=nclass)

    # 3. 权重重分配的等差系数D与层级L的消融实验
    elif model == 'stageRW_resnet18':
        net = StageRW_ResNet18(num_classes=nclass, rWeightFlag=args.reweight, D=args.D, use_layer=args.Li)
    elif model == 'stageRW_resnet50':
        net = StageRW_ResNet50(num_classes=nclass, rWeightFlag=args.reweight, D=args.D, use_layer=args.Li)
    elif model == 'stageRW_resnext50':
        net = StageRW_ResNeXt50_32x4d(num_classes=nclass, rWeightFlag=args.reweight, D=args.D, use_layer=args.Li)

    # 4. 可学习参数分配权重
    elif model == 'stageLW_resnet18':
        net = StageRW_ResNet18(num_classes=nclass, D=args.D, LearnableWeight=args.lrweight)
    elif model == 'stageLW_resnet50':
        net = StageRW_ResNet50(num_classes=nclass, D=args.D, LearnableWeight=args.lrweight)
    elif model == 'stageLW_resnext50':
        net = StageRW_ResNeXt50_32x4d(num_classes=nclass, D=args.D, LearnableWeight=args.lrweight)

    # 5. 对比实验，其他融合策略的结果
    elif model == 'DAG_CNN':
        net = DAGCNN_ResNet50(num_classes=nclass)
    elif model == 'CFN_NW':
        net = CFN_ResNet50(num_classes=nclass, mode='NW')
    elif model == 'CFN_SW':
        net = CFN_ResNet50(num_classes=nclass, mode='SW')
    elif model == 'CFN_NSW':
        net = CFN_ResNet50(num_classes=nclass, mode='NSW')

    else:
        raise ImportError("Have no such model, Check again the model name")

    net = net.to(device)

    if torch.cuda.device_count() > 1:
        print("Use device: ", torch.cuda.device_count(), "GPU \n")
        net = torch.nn.DataParallel(net)

    print(net)
    return net


def train_one_epoch(epoch, model, criterion, optimizer, train_loader, logger, device):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # get loss
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if batch_idx % (len(train_loader) // 2) == 0 or batch_idx == len(train_loader) - 1:
            train_info = "epoch:{}/{}, batch:{}/{}, loss:{}, acc:{} ({}/{})" \
                .format(epoch + 1, args.epochs, batch_idx, len(train_loader), train_loss / (batch_idx + 1),
                        correct / total, correct, total)
            print(train_info)
            logger.info(train_info)

    # add train loss
    trainloss_lists.append(train_loss / len(train_loader))


def eval_one_epoch(epoch, model, criterion, val_loader, logger, device, mdl_file):
    global best_acc
    global best_epoch
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    test_info = "Test ==> epoch:{}/{}, loss:{}, acc:{} ({}/{})" \
        .format(epoch + 1, args.epochs, test_loss / len(val_loader),
                correct / total, correct, total)
    print(test_info)
    logger.info(test_info)

    # Save checkpoint.
    testloss_lists.append(test_loss / len(val_loader))
    acc = 100. * correct / total
    acc_lists.append(acc)
    if acc >= best_acc:
        print('Saving..')
        checkpoint = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        torch.save(checkpoint, mdl_file)
        best_acc = acc
        best_epoch = epoch
        # print("best_acc: ", checkpoint['acc'], "best_epoch: ", checkpoint['epoch'])
    # print(infostat)


def train(agrs):

    # 获取记录文件
    logger = get_logger(record_file=args.rec_path)
    logger.info("args:\n{}".format(args))

    # 获取训练集与验证集
    train_data, val_data = get_dataset(dataset=agrs.dataset)
    print("train_data: {} val_data:{}".format(len(train_data), len(val_data)))
    train_loader = DataLoader(train_data, batch_size=agrs.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=agrs.batch_size, shuffle=True, pin_memory=True)
    print("train_loader: {} val_loader:{}".format(len(train_loader), len(val_loader)))

    # 获取模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model=agrs.model, dataset=agrs.dataset, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info(model)

    # 训练
    for epoch in range(args.epochs):
        train_one_epoch(epoch, model, criterion, optimizer, train_loader, logger, device)
        eval_one_epoch(epoch, model, criterion, val_loader, logger, device, args.mdl_path)
        logger.info("best_acc: {} best_epoch: {}".format(best_acc, best_epoch))

    # 可视化训练曲线
    list_to_draw(acc_lists, filename=args.acc_path, title='{}_acc'.format(args.model),
                 xlabel='Epochs', ylabel='Accuracy')
    list_to_draw(trainloss_lists, filename=args.trainloss_path, title='{}_trainloss'.format(args.model),
                 xlabel='Epochs', ylabel='Train Loss')
    list_to_draw(testloss_lists, filename=args.textloss_path, title='{}_testloss'.format(args.model),
                 xlabel='Epochs', ylabel='Test Loss')

    # 记录训练参数
    logger.info("acc_lists:\n{}".format(acc_lists))
    logger.info("trainloss_lists:\n{}".format(trainloss_lists))
    logger.info("trainloss_lists:\n{}".format(testloss_lists))


if __name__ == '__main__':
    seed_everything()
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    check_file(args)
    train(args)