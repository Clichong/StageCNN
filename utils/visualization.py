from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import cv2
import torch
import io, os, copy
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from pathlib import Path

from utils.dataset import get_dataset
from utils.seed import SEED
from train import get_model
from model import *
from mlp_models import *

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


# 功能: 对数据集进行降维可视化
# 参数：dataset：选择可视化的数据集，这里会提取设定相关数据集的路径
#       mode：选择降维的方式，可以选择tSNE, PCA, LDA等常用降维方法(未补充完全)
#       showTrainFlag：控制对训练数据集展示或者是测试训练集展示，True表示使用训练集展示，False表示使用测试集展示
#       showVisMode：控制可视化展示的维度，可以使用2d或者3d来可视化
def dataset_to_visualization(dataset, mode='tSNE', showTrainFlag=True, showVisMode='2D'):

    root = None     # 数据集的路径，根据dataset自动设置
    # 数据集的路径设计(相对路径与绝对路径的相关问题)
    if dataset == 'caltach101':
        root = '../dataset/caltech101/101_ObjectCategories/'
    if dataset == 'caltach256':
        root = '../dataset/caltech256/256_ObjectCategories/'
    elif dataset == "food101":
        root = '../dataset/food-101/images/'
    elif dataset == 'cifar10' or dataset == 'cifar100' or dataset == 'pokemon':
        root = '../dataset/{}'.format(dataset)
    else:
        raise AssertionError("Error loading dataset. Have no such dataset.")
    print('Use Dataset: {}'.format(dataset), '|', "Root:{}".format(root))

    # 获取数据集, 不需要resice处理
    transform = transforms.Compose([
        # transforms.Resize((resize, resize)),
        transforms.ToTensor()
    ])
    train_data, val_data = get_dataset(dataset=dataset, root=root, transform=transform)
    print(len(train_data), len(val_data))

    # 这里对训练数据进行2D可视化展示
    X_train = train_data.data
    y_train = train_data.targets
    X_test = val_data.data
    y_test = val_data.targets

    # 选择训练数据或者是测试数据进行可视化展示
    if showTrainFlag == True:
        X = X_train
        y = y_train
    else:
        X = X_test
        y = y_test
    # 展平处理
    # X = X.reshape(len(X), -1)
    X = X[..., 0].reshape(len(X), -1)

    # 选择进行2D或者是3D的降维可视化展示, n_components是最后的维数
    showVisMode = showVisMode.lower()
    if showVisMode == '2d':
        n_components = 2
    elif showVisMode == '3d':
        n_components = 3
    else:
        raise ImportError('Error choose the visualiztion mode. Choose "2d" or "3d".')

    # t-SNE降维处理
    if mode == 'tSNE':
        tsne = TSNE(n_components=n_components, verbose=1, random_state=SEED)
        result = tsne.fit_transform(X)
    elif mode == 'Chi':
        result = VarianceThreshold(threshold=0.1).fit_transform(X)   # 对方差为0.01的特征进行过滤
        result = SelectKBest(chi2, k=300).fit_transform(result, y)    # 卡方过滤来进行第二次降维
        result = TSNE(n_components=n_components, verbose=1, random_state=SEED).fit_transform(result)
    else:
        raise AssertionError("Error choose visual mode. Have no such mode {}. Here suffer: tSNE, PCA...".format(mode))

    # 归一化处理
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    result = scaler.fit_transform(result)

    # 可视化结果
    fig = plt.figure(figsize=(14, 14))
    if showVisMode == '2d':
        ax = fig.add_subplot(111)
        ax.set_title('{} t-SNE process'.format(dataset))
        ax.scatter(result[:, 0], result[:, 1], c=y, s=10)
    elif showVisMode == '3d':
        ax = fig.add_subplot(projection=showVisMode)
        ax.set_title('{} t-SNE process'.format(dataset))
        ax.scatter(result[:, 0], result[:, 1], result[:, 2], c=y, s=10)
    fig.show()
    fig.savefig("../visual/dataset_visual/visual_{}_{}_OneChannel.jpg".format(dataset, showVisMode))


# 功能: 对图像继续注意力可视化，输出训练好的网络最在意的部分
# 参数：model： 训练好的模型
#       image_path：待注意力可视化的图像路径
def attention_to_visualization(model, image_path):

    use_cuda = False
    eigen_smooth = False
    aug_smooth = False
    target_layers = [model.layer4[-1]]

    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

    # targets = [e.g ClassifierOutputTarget(281)]
    targets = None

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=eigen_smooth,
                        aug_smooth=aug_smooth)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite('../visual/attention_visual/car.jpg',
                cam_image)
    print("process success.")


# 功能: 利用attention_to_visualization完成热力图的可视化
def attention_to_show():

    # 直接加载oom，显存不足，所以需要使用cpu来加载
    # model = get_model(model=model, dataset=dataset, device='0')
    # model.load_state_dict(
    #     torch.load('./record/checkpoint/{}/{}_{}.mdl'.format(dataset, dataset, model))
    # )
    # print(model)

    # 构建模型
    model = ResNet50(num_classes=101)
    # model = Stage_ResNet50(num_classes=101, rWeightFlag=True)

    # 将GPU上的参数转化为CPU再加载进模型
    # checkpoint = torch.load('../record/checkpoint/pokemon/pokemon_stage_resnet50.mdl',
    #                         map_location={'cuda:0':'cuda:1'})
    checkpoint = torch.load('../record/checkpoint/caltech101/caltech101_resnet50.mdl',
                            map_location=lambda storage, loc: storage)

    # 加载参数
    model.load_state_dict(checkpoint['model'])
    print("acc:", checkpoint['acc'], "epoch:", checkpoint['epoch'])
    # print(model)

    image_path = '../data/image/caltech101/car.jpg'
    attention_to_visualization(model, image_path=image_path)


# 功能: 对图像channels进行注意力可视化（构建4x4的图像，并保存）
def channels_to_visualization(feature_map, save_name):

    save_path = r"../visual/channel_visual"
    save_path = os.path.join(save_path, save_name)

    # 选择16张channel的特征图
    n = 16
    feature_map = feature_map.clone().detach()
    b, c, h, w = feature_map.shape
    index = np.random.randint(0, c, n)
    feature = feature_map.squeeze(dim=0)[index, :, :].permute(1,2,0)
    image = feature.detach().numpy()
    print("image shape:", image.shape)

    # 单张图像保存
    # plt.imshow(image)
    # plt.savefig(save_path)

    plt.figure(figsize=(8, 8))
    # plt.title(save_name.split('.')[0])
    # 多张图像保存
    for i in range(n):
        plt.subplot(4, 4, i + 1)
        # 将坐标轴隐藏
        fig = plt.gca()
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.imshow(image[:, :, i])
    plt.savefig(save_path)


# 功能: 输入一张图片, 对提取特征之后的channels进行可视化
def channels_to_show(mode=1):

    image_path = r"../data/image/pokemon/pikachu.png"
    assert os.path.exists(image_path), "image path:{} is not exists".format(image_path)

    model = SpinMLP(num_classes=5, num_blocks=1, spinflag=False)
    spin_model = SpinModule(hidden_dim=224)

    patch_embed = model.patch_embed
    stage_embed = model.stages


    # 图像处理，转换格式
    image = cv2.imread(image_path)
    orgin_size = image.shape
    image = cv2.resize(image, (224, 224))
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(dim=0)
    print('process before image shape: {}, process after image shape: {}'.format(orgin_size, image.shape))

    if mode == 1:
        patch_feature = patch_embed(image)
        print("feature map shape: ", patch_feature.shape)
        feature = patch_feature

    elif mode == 2:
        patch_feature = patch_embed(image)
        stage_feature = stage_embed(patch_feature.permute(0,2,3,1)).permute(0,3,1,2)
        print("feature map shape: ", stage_feature.shape)
        feature = stage_feature

    elif mode == 3:
        patch_feature = patch_embed(image)
        spin_feature = spin_model.spin(patch_feature.permute(0,2,3,1), group=2, direction=True).permute(0,3,1,2)
        print("feature map shape: ", spin_feature.shape)
        feature = spin_feature

    channels_to_visualization(feature, 'spin_feature_x.jpg')


if __name__ == '__main__':

    # 可视化数据集
    # dataset_to_visualization(dataset='cifar10', mode='tSNE', showTrainFlag=False, showVisMode='3D')

    # 可视化channel
    channels_to_show(mode=2)
