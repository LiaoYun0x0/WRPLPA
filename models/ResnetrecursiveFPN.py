import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.swish = Swish()

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        # y = self.swish(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)
        # return self.swish(x+y)

class ResnetrecursiveFPN(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = 128
        block_dims = [128, 196, 256]

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(3, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        # self.relu = nn.ReLU(inplace=True)
        self.swish = Swish()

        self.layer1 = self._make_layer_withBlocksNum(block, block_dims[0], 3,stride=1)  # 1/2
        self.layer2 = self._make_layer_withBlocksNum(block, block_dims[1], 4,stride=2)  # 1/4
        self.layer3 = self._make_layer_withBlocksNum(block, block_dims[2], 8,stride=2)  # 1/8

        # self.layer1_2 = nn.Sequential(
        #         nn.Conv2d(initial_dim,block_dims[1],kernel_size=2,stride=2),
        #         nn.BatchNorm2d(block_dims[1], momentum=0.01, eps=1e-3),
        #     )
        # self.layer2_3 = nn.Sequential(
        #         nn.Conv2d(block_dims[0],block_dims[2],kernel_size=2,stride=4),
        #         nn.BatchNorm2d(block_dims[2], momentum=0.01, eps=1e-3),
        #     )
        # for i in range(1,2):
        #     self.layer1_2.append(block(block_dims[1],block_dims[1],stride=1))
        # for i in range(1,3):
        #     self.layer2_3.append(block(block_dims[2],block_dims[2],stride=1))

        # 3. FPN upsample
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Weight
        self.epsilon = 1e-4
        # self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # self.w1_relu = nn.ReLU()
        # self.w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # self.w2_relu = nn.ReLU()
        self.w3 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w3_relu = nn.ReLU()
        self.w4 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w4_relu = nn.ReLU()


    def _make_layer_withBlocksNum(self, block, dim, blocks,stride):
        layers = []
        layers.append(block(self.in_planes,dim,stride=stride))
        self.in_planes = dim
        for i in range(1,blocks):
            layers.append(block(self.in_planes,dim))
        return nn.Sequential(*layers)
    def forward(self, x):
        # x0 = self.relu(self.bn1(self.conv1(x)))
        x0 = self.swish(self.bn1(self.conv1(x)))
        # x0_copy=x0
        # x1_2 = self.layer1_2(x0_copy)
        x1 = self.layer1(x0)  # 1/2
        # x1_copy = x1
        # x2_3=self.layer2_3(x1_copy)
        x2 = self.layer2(x1)  # 1/4
        # w1 = self.w1_relu(self.w1)
        # weight = w1 / (torch.sum(w1, dim=0) + self.epsilon)
        # x2 = (weight[0]*x2+weight[1]*x1_2)
        x3 = self.layer3(x2)  # 1/8
        # w2 = self.w2_relu(self.w2)
        # weight = w2 / (torch.sum(w2, dim=0) + self.epsilon)
        # x3 = (weight[0]*x3+weight[1]*x2_3)

        # FPN
        x3_out = self.layer3_outconv(x3)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

        x1=x1_out
        x2=x2_out
        x3=x3_out
        # x1_copy = x1
        # x2_3=self.layer2_3(x1_copy)
        x2_fromLayer = self.layer2(x1)  # 1/4
        w3 = self.w3_relu(self.w3)
        weight = w3 / (torch.sum(w3, dim=0) + self.epsilon)
        # x2 = (weight[0]*x2+weight[1]*x1_2+weight[2]*x2_fromLayer)
        x2 = (weight[0]*x2+weight[1]*x2_fromLayer)
        x3_fromLayer = self.layer3(x2)  # 1/8
        w4 = self.w4_relu(self.w4)
        weight = w4 / (torch.sum(w4, dim=0) + self.epsilon)
        # x3 = (weight[0]*x3+weight[1]*x2_3+weight[2]*x3_fromLayer)
        x3 = (weight[0]*x3+weight[1]*x3_fromLayer)

        # FPN
        x3_out = self.layer3_outconv(x3)
        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

        return [x3_out, x1_out]