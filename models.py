import torch
import torch.nn as nn


class ResidualUpBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int
                 ) -> None:
        super(ResidualUpBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.upconv1 = nn.ConvTranspose2d(in_features,
                                          out_features,
                                          kernel_size=(3, 3),
                                          stride=(2, 2),
                                          padding=(1, 1),
                                          output_padding=(1, 1),
                                          bias=False)
        self.bn1 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.upconv2 = nn.ConvTranspose2d(out_features,
                                          out_features,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          padding=(1, 1),
                                          bias=False)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.down_features = nn.ConvTranspose2d(in_features,
                                                out_features,
                                                kernel_size=(1, 1),
                                                stride=(1, 1),
                                                bias=False)
        self.upsample = nn.Upsample(scale_factor=2)

    @property
    def expand_features(self):
        return self.in_features != self.out_features

    def forward(self, x):
        identity = x
        out = self.upconv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.upconv2(out)
        out = self.bn2(out)
        if self.expand_features:
            identity = self.down_features(identity)
            identity = self.upsample(identity)
        return out + identity


class ResidualDownBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int) -> None:
        super(ResidualDownBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv2d(in_features,
                               out_features,
                               kernel_size=(3, 3),
                               stride=(2, 2),
                               padding=(1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_features,
                               out_features,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.downsample = nn.Conv2d(in_features,
                                    out_features,
                                    kernel_size=(1, 1),
                                    stride=(2, 2),
                                    bias=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(identity)
        return out + identity


class SeparableResidualDownBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int) -> None:
        super(SeparableResidualDownBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dconv1 = nn.Conv2d(in_features,
                                in_features,
                                groups=in_features,
                                kernel_size=(3, 3),
                                stride=(2, 2),
                                padding=(1, 1),
                                bias=False)
        self.pconv1 = nn.Conv2d(in_features,
                                out_features,
                                kernel_size=(1, 1),
                                bias=False)
        self.bn1 = nn.InstanceNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dconv2 = nn.Conv2d(out_features,
                                out_features,
                                groups=out_features,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=(1, 1),
                                bias=False)
        self.pconv2 = nn.Conv2d(out_features,
                                out_features,
                                kernel_size=(1, 1),
                                bias=False)
        self.bn2 = nn.InstanceNorm2d(out_features)
        self.downsample = nn.Conv2d(in_features,
                                    out_features,
                                    kernel_size=(1, 1),
                                    stride=(2, 2),
                                    bias=False)

    def forward(self, x):
        identity = x
        out = self.pconv1(self.dconv1(x))
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pconv2(self.dconv2(out))
        out = self.bn2(out)
        identity = self.downsample(identity)
        return out + identity


class ResidualDecoder(nn.Module):
    def __init__(self,
                 channels: int = 3,
                 features: int = 32) -> None:
        super(ResidualDecoder, self).__init__()
        self.liear_decoder = nn.Sequential(
            nn.Linear(1, 16 * 16 * 32 * features),
            nn.ReLU(inplace=True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32 * features, 16, 16))
        self.relu = nn.ReLU(inplace=True)
        self.block1 = ResidualUpBlock(32 * features, 16 * features)
        self.block2 = ResidualUpBlock(16 * features, 8 * features)
        self.block3 = ResidualUpBlock(8 * features, 4 * features)
        self.block4 = ResidualUpBlock(4 * features, 2 * features)
        self.block5 = ResidualUpBlock(2 * features, features)
        self.conv = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=(1, 1))

    def forward(self, x):
        # Image decoder
        out = self.liear_decoder(x)
        out = self.unflatten(out)
        out = self.relu(self.block1(out))
        out = self.relu(self.block2(out))
        out = self.relu(self.block3(out))
        out = self.relu(self.block4(out))
        out = self.relu(self.block5(out))
        out = torch.sigmoid(self.conv(out))
        return out


class ResidualEncoder(nn.Module):
    def __init__(self,
                 channels: int = 3,
                 features: int = 32,
                 n_frames: int = 6,
                 n_actions: int = 3) -> None:
        super(ResidualEncoder, self).__init__()
        self.n_frames = n_frames
        self.n_actions = n_actions
        self.relu = nn.ReLU(inplace=True)
        self.block1 = ResidualDownBlock(channels * self.n_frames, features)
        self.block2 = ResidualDownBlock(features, 2 * features)
        self.block3 = ResidualDownBlock(2 * features, 4 * features)
        self.block4 = ResidualDownBlock(4 * features, 8 * features)
        self.block5 = ResidualDownBlock(8 * features, 16 * features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(16 * features, self.n_actions)
        self.fc2 = nn.Linear(16 * features, self.n_actions)
        self.fc3 = nn.Linear(16 * features, self.n_actions)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        out = self.relu(self.block1(x))
        out = self.relu(self.block2(out))
        out = self.relu(self.block3(out))
        out = self.relu(self.block4(out))
        out = self.relu(self.block5(out))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out1 = self.fc1(out)
        out2 = self.fc2(out)
        out3 = self.fc3(out)
        return out1, out2, out3


# Deep Q-Network
class DeepQNet(nn.Module):
    def __init__(self,
                 channels: int = 3,
                 features: int = 32,
                 n_frames: int = 5,
                 n_actions: int = 3) -> None:
        super(DeepQNet, self).__init__()
        self.n_frames = n_frames
        self.n_actions = n_actions
        self.relu = nn.ReLU(inplace=True)
        self.block1 = SeparableResidualDownBlock(channels * self.n_frames, features)
        self.block2 = SeparableResidualDownBlock(features, 2 * features)
        self.block3 = SeparableResidualDownBlock(2 * features, 4 * features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_ss = nn.Linear(4 * features, 2)
        self.fc1 = nn.Linear(4 * features, self.n_actions)
        self.fc2 = nn.Linear(4 * features, self.n_actions)
        self.fc3 = nn.Linear(4 * features, self.n_actions)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        out = self.relu(self.block1(x))
        out = self.relu(self.block2(out))
        out = self.relu(self.block3(out))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out_ss = self.fc_ss(out)
        out1 = self.fc1(out)
        out2 = self.fc2(out)
        out3 = self.fc3(out)
        return out_ss, out1, out2, out3


# Deep Q-Network
class SoftSensor(nn.Module):
    def __init__(self,
                 channels: int = 3,
                 features: int = 32,
                 n_frames: int = 5,
                 n_actions: int = 27) -> None:
        super(SoftSensor, self).__init__()
        self.n_frames = n_frames
        self.n_actions = n_actions
        self.relu = nn.ReLU(inplace=True)
        self.block1 = SeparableResidualDownBlock(channels * self.n_frames, features)
        self.block2 = SeparableResidualDownBlock(features, 2 * features)
        self.block3 = SeparableResidualDownBlock(2 * features, 4 * features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc_ss = nn.Linear(4 * features, 2)
        # self.fc1 = nn.Linear(4 * features, self.n_actions)
        # self.fc2 = nn.Linear(4 * features, self.n_actions)
        # self.fc3 = nn.Linear(4 * features, self.n_actions)
        self.fc1 = nn.Linear(4 * features, n_actions)
        self.fc2 = nn.Linear(4 * features, 2)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        out = self.relu(self.block1(x))
        out = self.relu(self.block2(out))
        out = self.relu(self.block3(out))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        # out_ss = self.fc_ss(out)
        # out1 = self.fc1(out)
        # out2 = self.fc2(out)
        # out3 = self.fc3(out)
        out1 = self.fc1(out)
        out2 = self.fc2(out)
        return out1, out2