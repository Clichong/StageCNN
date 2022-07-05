import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import einops

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


class CFNEmbedding(nn.Module):
    def __init__(self, hidden_dim, mode):
        super(CFNEmbedding, self).__init__()

        self.mode = mode
        print('CFN Embedding choose mode: ', mode)

        # 128 -> 512 / 512 -> 2048
        self.conv_x0 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim // 4, out_channels=hidden_dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # 256 -> 512 / 1024 -> 2048
        self.conv_x1 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # 512 -> 512 / 2048 -> 2048
        self.conv_x2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )

        if self.mode == 'SW':
            self.fc = nn.Linear(3, 1)
        elif self.mode == 'NSW':
            self.fc = nn.Conv2d(1, 1, kernel_size=(3, 1))

    def forward(self, x):

        [x0, x1, x2] = x

        o0 = self.conv_x0(x0)
        o1 = self.conv_x1(x1)
        o2 = self.conv_x2(x2)

        if self.mode == "NW":       # no weight
            out = o0 + o1 + o2
            out = out.squeeze(-1).squeeze(-1)
        elif self.mode == 'SW':     # sharing weights
            'b c s 1 1 -> b c s'
            out = torch.stack([o0, o1, o2], dim=2).squeeze(-1).squeeze(-1)
            out = self.fc(out).squeeze(-1)
        elif self.mode == 'NSW':    # no sharing weights
            'b s c 1 1 ->  -> b 1 s c'
            out = torch.stack([o0, o1, o2], dim=1)
            out = einops.rearrange(out, 'b s c h w -> b h s (c w)')
            out = self.fc(out)
            out = einops.rearrange(out, 'b h w c -> b (h w c)')
        return out


class ResNet(nn.Module):
    def __init__(self, blocks, blockkinds, num_classes, mode):
        super(ResNet, self).__init__()

        self.blockkinds = blockkinds
        self.conv1 = Conv1(in_planes=3, places=64)

        # 对应浅层网络结构
        if self.blockkinds == BasicBlock:
            self.expansion = 1
            self.hidden_dim = 512
            # 64 -> 64
            self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
            # 64 -> 128
            self.layer2 = self.make_layer(in_places=64, places=128, block=blocks[1], stride=2)
            # 128 -> 256
            self.layer3 = self.make_layer(in_places=128, places=256, block=blocks[2], stride=2)
            # 256 -> 512
            self.layer4 = self.make_layer(in_places=256, places=512, block=blocks[3], stride=2)

            self.fc = nn.Linear(self.hidden_dim, num_classes)
            print("blockkinds == BasicBlock")

        # 对应深层网络结构
        if self.blockkinds == Bottleneck:
            self.expansion = 4
            self.hidden_dim = 2048
            # 64 -> 64
            self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
            # 256 -> 128
            self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
            # 512 -> 256
            self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
            # 1024 -> 512
            self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

            self.fc = nn.Linear(self.hidden_dim, num_classes)
            print("blockkinds == Bottleneck")

        # 对应ResNeXt结构
        if self.blockkinds == ResNeXtBlock:
            self.expansion = 2
            self.hidden_dim = 2048
            # 64 -> 128
            self.layer1 = self.make_layer(in_places=64, places=128, block=blocks[0], stride=1)
            # 256 -> 256
            self.layer2 = self.make_layer(in_places=256, places=256, block=blocks[1], stride=2)
            # 512 -> 512
            self.layer3 = self.make_layer(in_places=512, places=512, block=blocks[2], stride=2)
            # 1024 -> 1024
            self.layer4 = self.make_layer(in_places=1024, places=1024, block=blocks[3], stride=2)

            self.fc = nn.Linear(self.hidden_dim, num_classes)
            print("blockkinds == ResNeXtBlock")

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.cfn_embedding = CFNEmbedding(self.hidden_dim, mode)

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

        out = []

        # conv1层
        x = self.conv1(x)  # torch.Size([1, 64, 56, 56])

        # conv2_x层
        x = self.layer1(x)  # torch.Size([1, 256, 56, 56])
        # conv3_x层
        x = self.layer2(x)  # torch.Size([1, 512, 28, 28])
        out.append(x)
        # conv4_x层
        x = self.layer3(x)  # torch.Size([1, 1024, 14, 14])
        out.append(x)
        # conv5_x层
        x = self.layer4(x)  # torch.Size([1, 2048, 7, 7])
        out.append(x)

        x = self.cfn_embedding(out)
        # x = self.avgpool(x)  # torch.Size([1, 2048, 1, 1]) / torch.Size([1, 512])
        # x = x.view(x.size(0), -1)  # torch.Size([1, 2048]) / torch.Size([1, 512])
        x = self.fc(x)  # torch.Size([1, 5])

        return x


def CFN_ResNet18(num_classes):
    return ResNet(resnet18_params, BasicBlock, num_classes)


# def Stage_ResNet34(num_classes, rWeightFlag=False):
#     return ResNet(resnet34_params, BasicBlock, num_classes, rWeightFlag)


def CFN_ResNet50(num_classes, mode):
    return ResNet(resnet50_params, Bottleneck, num_classes, mode)


# def Stage_ResNet101(num_classes, rWeightFlag=False):
#     return ResNet(resnet101_params, Bottleneck, num_classes, rWeightFlag)
#
#
# def Stage_ResNet152(num_classes, rWeightFlag=False):
#     return ResNet(resnet152_params, Bottleneck, num_classes, rWeightFlag)


def CFN_ResNeXt50_32x4d(num_classes):
    return ResNet(resnext50_32x4d_params, ResNeXtBlock, num_classes)


# def Stage_ResNeXt101_32x8d(num_classes, rWeightFlag=False):
#     return ResNet(resnext101_32x8d_params, ResNeXtBlock, num_classes, rWeightFlag)


if __name__ == '__main__':
    # model = torchvision.models.resnet50()

    # 模型测试: MODE = 'NW' / 'SW' / 'NSW'
    model = CFN_ResNet50(num_classes=1000, mode='NSW')
    # print(model)

    input = torch.randn(128, 3, 224, 224)
    out = model(input)
    print(out.shape)

    flops, params = profile(model, inputs=(input,))
    print('flops:{} G'.format(flops / 1e9))
    print('params:{} M'.format(params / 1e6))

