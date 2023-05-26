import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        return self.conv_block(x)


class STEM(nn.Module):

    def __init__(self):
        super(STEM, self).__init__()
        self.conv1 = ConvolutionBlock(
            in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0)
        self.conv2 = ConvolutionBlock(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv3 = ConvolutionBlock(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv4 = ConvolutionBlock(
            in_channels=64, out_channels=80, kernel_size=1, stride=1, padding=0)
        self.conv5 = ConvolutionBlock(
            in_channels=80, out_channels=192, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # 149 x 149 x 64
        x = self.pool1(x)  # 147 x 147 x 64
        x = self.conv4(x)  # 147 x 147 x 80
        x = self.conv5(x)  # 147 x 147 x 192
        x = self.pool2(x)  # 73 x 73 x 192
        return x


class InceptionBlockA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlockA, self).__init__()

        self.branch1 = nn.Sequential(
            ConvolutionBlock(in_channels=in_channels, out_channels=64,
                             kernel_size=1, stride=1, padding=0),
            ConvolutionBlock(in_channels=64, out_channels=96,
                             kernel_size=3, stride=1, padding=1),
            ConvolutionBlock(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1))

        self.branch2 = nn.Sequential(
            ConvolutionBlock(in_channels=in_channels, out_channels=48,
                             kernel_size=1, stride=1, padding=0),
            ConvolutionBlock(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvolutionBlock(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0))

        self.branch4 = ConvolutionBlock(
            in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class ReductionBlockA(nn.Module):
    def __init__(self, in_channels):
        super(ReductionBlockA, self).__init__()

        self.branch1 = nn.Sequential(
            ConvolutionBlock(in_channels=in_channels, out_channels=64,
                             kernel_size=1, stride=1, padding=0),
            ConvolutionBlock(in_channels=64, out_channels=96,
                             kernel_size=3, stride=1, padding=1),
            ConvolutionBlock(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=0))

        self.branch2 = nn.Sequential(
            ConvolutionBlock(in_channels=in_channels, out_channels=384, kernel_size=2, stride=2, padding=0))

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)


class InceptionBlockB(nn.Module):
    def __init__(self, in_channels, nbr_kernel):
        super(InceptionBlockB, self).__init__()

        self.branch1 = nn.Sequential(
            ConvolutionBlock(in_channels=in_channels, out_channels=nbr_kernel,
                             kernel_size=1, stride=1, padding=0),
            ConvolutionBlock(in_channels=nbr_kernel, out_channels=nbr_kernel, kernel_size=(
                7, 1), stride=1, padding=(0, 3)),
            ConvolutionBlock(in_channels=nbr_kernel, out_channels=nbr_kernel, kernel_size=(
                1, 7), stride=1, padding=(3, 0)),
            ConvolutionBlock(in_channels=nbr_kernel, out_channels=nbr_kernel, kernel_size=(
                7, 1), stride=1, padding=(0, 3)),
            ConvolutionBlock(in_channels=nbr_kernel, out_channels=192, kernel_size=(1, 7), stride=1, padding=(3, 0)))

        self.branch2 = nn.Sequential(
            ConvolutionBlock(in_channels=in_channels, out_channels=nbr_kernel,
                             kernel_size=1, stride=1, padding=0),
            ConvolutionBlock(in_channels=nbr_kernel, out_channels=nbr_kernel, kernel_size=(
                1, 7), stride=1, padding=(0, 3)),
            ConvolutionBlock(in_channels=nbr_kernel, out_channels=192, kernel_size=(7, 1), stride=1, padding=(3, 0)))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvolutionBlock(in_channels=in_channels, out_channels=192, kernel_size=1, stride=1, padding=0))

        self.branch4 = ConvolutionBlock(
            in_channels=in_channels, out_channels=192, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class ReductionBlockB(nn.Module):
    def __init__(self, in_channels):
        super(ReductionBlockB, self).__init__()

        self.branch1 = nn.Sequential(
            ConvolutionBlock(in_channels=in_channels, out_channels=192,
                             kernel_size=1, stride=1, padding=0),
            ConvolutionBlock(in_channels=192, out_channels=192,
                             kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvolutionBlock(in_channels=192, out_channels=192,
                             kernel_size=(7, 1), stride=1, padding=(3, 0)),
            ConvolutionBlock(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=0))

        self.branch2 = nn.Sequential(
            ConvolutionBlock(in_channels=in_channels, out_channels=192,
                             kernel_size=1, stride=1, padding=0),
            ConvolutionBlock(in_channels=192, out_channels=320, kernel_size=3, stride=2, padding=0))

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)


class AuxiliaryClassifierBlock(nn.Module):
    def __init__(self, in_channels):
        super(AuxiliaryClassifierBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3, padding=0),
            ConvolutionBlock(in_channels=in_channels, out_channels=128,
                             kernel_size=1, stride=1, padding=0),
            ConvolutionBlock(in_channels=128, out_channels=768,
                             kernel_size=1, stride=1, padding=0),
        )

        self.fc1 = nn.LazyLinear(1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.branch1(x)
        x = torch.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return nn.Softmax()(x)


class InceptionBlockC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1 = nn.Sequential(
            ConvolutionBlock(in_channels=in_channels, out_channels=384, kernel_size=1, stride=1, padding=0))

        self.branch1_1 = nn.Sequential(
            ConvolutionBlock(in_channels=384, out_channels=384, kernel_size=(1, 3), stride=1, padding=(0, 1)))

        self.branch1_2 = nn.Sequential(
            ConvolutionBlock(in_channels=384, out_channels=384,
                             kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        self.branch2 = nn.Sequential(
            ConvolutionBlock(in_channels=in_channels, out_channels=448,
                             kernel_size=1, stride=1, padding=0),
            ConvolutionBlock(in_channels=448, out_channels=384,
                             kernel_size=3, stride=1, padding=1)
        )
        self.branch2_1 = ConvolutionBlock(
            in_channels=384, out_channels=384, kernel_size=(1, 3), stride=1, padding=(0, 1))

        self.branch2_2 = ConvolutionBlock(
            in_channels=384, out_channels=384, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvolutionBlock(in_channels=in_channels,
                             out_channels=192, kernel_size=1, stride=1, padding=0)
        )

        self.branch4 = ConvolutionBlock(
            in_channels=in_channels, out_channels=320, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return torch.cat([self.branch1_1(self.branch1(x)), self.branch1_2(self.branch1(x)), self.branch2_1(self.branch2(x)), self.branch2_2(self.branch2(x)), self.branch3(x), self.branch4(x)], dim=1)


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = STEM()

        self.InceptionA1 = InceptionBlockA(192)
        self.InceptionA2 = InceptionBlockA(288)
        self.InceptionA3 = InceptionBlockA(288)

        self.redA = ReductionBlockA(288)

        self.InceptionB1 = InceptionBlockB(768, 128)
        self.InceptionB2 = InceptionBlockB(768, 160)
        self.InceptionB3 = InceptionBlockB(768, 160)
        self.InceptionB4 = InceptionBlockB(768, 192)

        self.redB = ReductionBlockB(768)

        self.InceptionC1 = InceptionBlockC(1280)
        self.InceptionC2 = InceptionBlockC(2048)

        self.aux = AuxiliaryClassifierBlock(768)

        self.pool = nn.AvgPool2d(kernel_size=1)
        self.fc1 = nn.LazyLinear(1000)

        self.fc2 = nn.Linear(1000, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.stem(x)

        x = self.InceptionA1(x)
        x = self.InceptionA2(x)
        x = self.InceptionA2(x)

        x = self.redA(x)

        x = self.InceptionB1(x)
        x = self.InceptionB2(x)
        x = self.InceptionB3(x)
        x = self.InceptionB4(x)

        aux = self.aux(x)

        x = self.redB(x)

        x = self.InceptionC1(x)
        x = self.InceptionC2(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x
