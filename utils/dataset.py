from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

import numpy as np
import cv2
import os

from imutils import paths
from utils.seed import SEED


# 自定义数据集
class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None, mappings=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
        self.mappings = mappings

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X[i][:]
        label = self.mappings[self.y[i]]

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return data, label
        else:
            return data


# 功能: 根据数据集路径获取划分后的数据集与验证集
# 参数: 数据集的文件格式是目录下相同类别的图像存放在同一标好类别名称的文件夹下, root即为数据集目录的名称
def get_label_and_data(root):

    assert os.path.exists(root), "{} is not exists".format(root)

    image_paths = list(paths.list_images(root))
    data = []
    labels = []
    for image_path in image_paths:
        label = image_path.split(os.path.sep)[-2]
        if label == 'BACKGROUND_Google':
            continue
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data.append(image)
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    gle = LabelEncoder()
    genre_labels = gle.fit_transform(labels)
    genre_mappings = {label: index for index, label in enumerate(gle.classes_)}
    print(genre_mappings)

    # divide the data into train, validation, and test set
    (x_train, x_val, y_train, y_val) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=SEED)
    print(f"x_train examples: {x_train.shape} x_val examples: {x_val.shape}")

    return x_train, y_train, x_val, y_val, genre_mappings


# 功能: 获取特定的数据集，返回训练数据集与测试数据集
# 参数：dataset：选择指定数据集训练
#       transform：设置特定的转换格式
def get_dataset(dataset=None, transform=None, root=None):

    if transform is None:
        resize = 224
        transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    if dataset == "cifar10":
        if root is None:
            root = './dataset/cifar10'
        traindata = torchvision.datasets.CIFAR10(root=root,
                                                 train=True,
                                                 transform=transform,
                                                 download=False)
        testdata = torchvision.datasets.CIFAR10(root=root,
                                                train=False,
                                                transform=transform,
                                                download=False)

    elif dataset == "cifar100":
        if root is None:
            root = './dataset/cifar100'
        traindata = torchvision.datasets.CIFAR100(root=root,
                                                  train=True,
                                                  transform=transform,
                                                  download=False)
        testdata = torchvision.datasets.CIFAR100(root=root,
                                                 train=False,
                                                 transform=transform,
                                                 download=False)

    elif dataset == "food101":
        if root is None:
            root = './dataset/food-101/images/'
        resize = 224
        transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((resize, resize)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
        x_train, y_train, x_val, y_val, genre_mappings = \
            get_label_and_data(root=root)
        traindata = ImageDataset(x_train, y_train, transform, genre_mappings)
        testdata = ImageDataset(x_val, y_val, transform, genre_mappings)

    elif dataset == "caltech101":
        if root is None:
            root = './dataset/caltech101/101_ObjectCategories/'
        resize = 224
        transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((resize, resize)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
        x_train, y_train, x_val, y_val, genre_mappings = \
            get_label_and_data(root=root)
        traindata = ImageDataset(x_train, y_train, transform, genre_mappings)
        testdata = ImageDataset(x_val, y_val, transform, genre_mappings)

    elif dataset == "caltech256":
        if root is None:
            root = './dataset/caltech256/256_ObjectCategories/'
        resize = 224
        transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((resize, resize)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
        x_train, y_train, x_val, y_val, genre_mappings = \
            get_label_and_data(root=root)
        traindata = ImageDataset(x_train, y_train, transform, genre_mappings)
        testdata = ImageDataset(x_val, y_val, transform, genre_mappings)

    elif dataset == "pokemon":
        if root is None:
            root = './dataset/pokemon/'
        resize = 224
        transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((resize, resize)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
        x_train, y_train, x_val, y_val, genre_mappings = \
            get_label_and_data(root=root)
        traindata = ImageDataset(x_train, y_train, transform, genre_mappings)
        testdata = ImageDataset(x_val, y_val, transform, genre_mappings)

    else:
        raise ImportError("Error loading dataset. Have no such dataset.")

    return traindata, testdata


if __name__ == '__main__':

    traindata, testdata = get_dataset(dataset="cifar10", root='../dataset/cifar10')
    print(len(traindata), len(testdata))
    #
    # traindata, testdata = get_dataset(dataset="cifar100")
    # print(len(traindata), len(testdata))

    # traindata, testdata = get_dataset(dataset="food101")
    # print(len(traindata), len(testdata))