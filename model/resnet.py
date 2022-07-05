import torch
import torch.nn as nn
from thop import profile

# 分类数目
# num_class = 101

# 各层数目
resnet18_params = [2, 2, 2, 2]
resnet34_params = [3, 4, 6, 3]
resnet50_params = [3, 4, 6, 3]
resnet101_params = [3, 4, 23, 3]
resnet152_params = [3, 8, 36, 3]
resnext50_32x4d_params = [3, 4, 6, 3]
resnext101_32x8d_params = [3, 4, 23, 3]


# 定义Conv1层
def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


# 浅层的残差结构
class BasicBlock(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=1):
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        # torch.Size([1, 64, 56, 56]), stride = 1
        # torch.Size([1, 128, 28, 28]), stride = 2
        # torch.Size([1, 256, 14, 14]), stride = 2
        # torch.Size([1, 512, 7, 7]), stride = 2
        self.basicblock = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        # torch.Size([1, 64, 56, 56])
        # torch.Size([1, 128, 28, 28])
        # torch.Size([1, 256, 14, 14])
        # torch.Size([1, 512, 7, 7])
        # 每个大模块的第一个残差结构需要改变步长
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 实线分支
        residual = x
        out = self.basicblock(x)

        # 虚线分支
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# 深层的残差结构
class Bottleneck(nn.Module):

    # 注意:默认 downsampling=False
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            # torch.Size([1, 64, 56, 56])，stride=1
            # torch.Size([1, 128, 56, 56])，stride=1
            # torch.Size([1, 256, 28, 28]), stride=1
            # torch.Size([1, 512, 14, 14]), stride=1
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            # torch.Size([1, 64, 56, 56])，stride=1
            # torch.Size([1, 128, 28, 28]), stride=2
            # torch.Size([1, 256, 14, 14]), stride=2
            # torch.Size([1, 512, 7, 7]), stride=2
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            # torch.Size([1, 256, 56, 56])，stride=1
            # torch.Size([1, 512, 28, 28]), stride=1
            # torch.Size([1, 1024, 14, 14]), stride=1
            # torch.Size([1, 2048, 7, 7]), stride=1
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        # torch.Size([1, 256, 56, 56])
        # torch.Size([1, 512, 28, 28])
        # torch.Size([1, 1024, 14, 14])
        # torch.Size([1, 2048, 7, 7])
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 实线分支
        residual = x
        out = self.bottleneck(x)

        # 虚线分支
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 定义ResNeXt残差结构
class ResNeXtBlock(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=2, cardinality=32):
        super(ResNeXtBlock, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        # torch.Size([1, 256, 56, 56])
        # torch.Size([1, 512, 28, 28])
        # torch.Size([1, 1024, 14, 14])
        # torch.Size([1, 2048, 7, 7])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False,
                      groups=cardinality),  # 使用了组卷积
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 实现分支
        residual = x
        out = self.bottleneck(x)

        # 虚线分支
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, blocks, blockkinds, num_classes):
        super(ResNet, self).__init__()

        self.blockkinds = blockkinds
        self.conv1 = Conv1(in_planes=3, places=64)

        # 对应浅层网络结构
        if self.blockkinds == BasicBlock:
            self.expansion = 1
            # 64 -> 64
            self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
            # 64 -> 128
            self.layer2 = self.make_layer(in_places=64, places=128, block=blocks[1], stride=2)
            # 128 -> 256
            self.layer3 = self.make_layer(in_places=128, places=256, block=blocks[2], stride=2)
            # 256 -> 512
            self.layer4 = self.make_layer(in_places=256, places=512, block=blocks[3], stride=2)

            self.fc = nn.Linear(512, num_classes)
            print("blockkinds == BasicBlock")

        # 对应深层网络结构
        if self.blockkinds == Bottleneck:
            self.expansion = 4
            # 64 -> 64
            self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
            # 256 -> 128
            self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
            # 512 -> 256
            self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
            # 1024 -> 512
            self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

            self.fc = nn.Linear(2048, num_classes)
            print("blockkinds == Bottleneck")

        # 对应ResNeXt结构
        if self.blockkinds == ResNeXtBlock:
            self.expansion = 2
            # 64 -> 128
            self.layer1 = self.make_layer(in_places=64, places=128, block=blocks[0], stride=1)
            # 256 -> 256
            self.layer2 = self.make_layer(in_places=256, places=256, block=blocks[1], stride=2)
            # 512 -> 512
            self.layer3 = self.make_layer(in_places=512, places=512, block=blocks[2], stride=2)
            # 1024 -> 1024
            self.layer4 = self.make_layer(in_places=1024, places=1024, block=blocks[3], stride=2)

            self.fc = nn.Linear(2048, num_classes)
            print("blockkinds == ResNeXtBlock")

        self.avgpool = nn.AvgPool2d(7, stride=1)

        # 初始化网络结构
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 采用了何凯明的初始化方法
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):

        layers = []

        # torch.Size([1, 64, 56, 56])  -> torch.Size([1, 256, 56, 56])， stride=1 故w，h不变
        # torch.Size([1, 256, 56, 56]) -> torch.Size([1, 512, 28, 28])， stride=2 故w，h变
        # torch.Size([1, 512, 28, 28]) -> torch.Size([1, 1024, 14, 14])，stride=2 故w，h变
        # torch.Size([1, 1024, 14, 14]) -> torch.Size([1, 2048, 7, 7])， stride=2 故w，h变
        # 此步需要通过虚线分支，downsampling=True
        layers.append(self.blockkinds(in_places, places, stride, downsampling=True))

        # torch.Size([1, 256, 56, 56]) -> torch.Size([1, 256, 56, 56])
        # torch.Size([1, 512, 28, 28]) -> torch.Size([1, 512, 28, 28])
        # torch.Size([1, 1024, 14, 14]) -> torch.Size([1, 1024, 14, 14])
        # torch.Size([1, 2048, 7, 7]) -> torch.Size([1, 2048, 7, 7])
        # print("places*self.expansion:", places*self.expansion)
        # print("block:", block)
        # 此步需要通过实线分支，downsampling=False， 每个大模块的第一个残差结构需要改变步长
        for i in range(1, block):
            layers.append(self.blockkinds(places * self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):

        # conv1层
        x = self.conv1(x)  # torch.Size([1, 64, 56, 56])

        # conv2_x层
        x = self.layer1(x)  # torch.Size([1, 256, 56, 56])
        # conv3_x层
        x = self.layer2(x)  # torch.Size([1, 512, 28, 28])
        # conv4_x层
        x = self.layer3(x)  # torch.Size([1, 1024, 14, 14])
        # conv5_x层
        x = self.layer4(x)  # torch.Size([1, 2048, 7, 7])

        x = self.avgpool(x)  # torch.Size([1, 2048, 1, 1]) / torch.Size([1, 512])
        x = x.view(x.size(0), -1)  # torch.Size([1, 2048]) / torch.Size([1, 512])
        x = self.fc(x)  # torch.Size([1, 5])

        return x


def ResNet18(num_classes):
    return ResNet(resnet18_params, BasicBlock, num_classes)


def ResNet34(num_classes):
    return ResNet(resnet34_params, BasicBlock, num_classes)


def ResNet50(num_classes):
    return ResNet(resnet50_params, Bottleneck, num_classes)


def ResNet101(num_classes):
    return ResNet(resnet101_params, Bottleneck, num_classes)


def ResNet152(num_classes):
    return ResNet(resnet152_params, Bottleneck, num_classes)


def ResNeXt50_32x4d(num_classes):
    return ResNet(resnext50_32x4d_params, ResNeXtBlock, num_classes)


def ResNeXt101_32x8d(num_classes):
    return ResNet(resnext101_32x8d_params, ResNeXtBlock, num_classes)


if __name__ == '__main__':
    # model = torchvision.models.resnet50()

    # 模型测试
    # model = ResNet18(num_classes=1000)
    # model = ResNet34()
    model = ResNet50(num_classes=1000)
    # model = ResNet101()
    # model = ResNet152()
    # model = ResNeXt50_32x4d(num_classes=1000)
    # model = ResNeXt101_32x8d(num_classes=10)
    # print(model)

    input = torch.randn(128, 3, 224, 224)
    out = model(input)
    print(out.shape)
    print(model)

    flops, params = profile(model, inputs=(input,))
    print('flops:{} G'.format(flops / 1000000000))
    print('params:{} M'.format(params / 1000000))

