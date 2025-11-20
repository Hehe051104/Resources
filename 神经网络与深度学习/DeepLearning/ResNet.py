import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 残差块 ----------
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        # 主分支两层卷积
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)

        # shortcut：如果尺寸或通道不匹配，就加一个 1×1 卷积
        self.shortcut = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)

# ---------- 主体网络 ----------
class ResNet(nn.Module):
    def __init__(self, layers, num_classes=10):
        """
        layers: 每个 stage 的 block 数，例如 [2,2,2,2] 就是 ResNet-18
        """
        super().__init__()
        self.in_c = 64

        # stem：CIFAR 风格（小图用 3x3，不下采样）
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        # 四个 stage，每个 stage 第一个 block stride=2
        self.layer1 = self._make_layer(64,  layers[0], stride=1)  # 不降采样
        self.layer2 = self._make_layer(128, layers[1], stride=2)  # ↓ 空间减半
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_c, num_blocks, stride):
        """造一个 stage：首块 stride=2，其余 stride=1"""
        layers = []
        # 首块：可能下采样（stride=2）
        layers.append(BasicBlock(self.in_c, out_c, stride))
        self.in_c = out_c
        # 其余块：不下采样
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # stem
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ---------- 一个小构造函数 ----------
def resnet18(num_classes=10):
    return ResNet([2, 2, 2, 2], num_classes=num_classes)

# ---------- 快速测试 ----------
if __name__ == "__main__":
    model = resnet18()
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print("输出维度:", y.shape)  # [4, 10]
