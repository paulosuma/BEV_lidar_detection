import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, first_stride=1):
        super(ResidualBlock, self).__init__()
        layers = []
        downsample = None

        if first_stride != 1 or in_channels!=out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=first_stride),
                nn.BatchNorm2d(out_channels)
            )
        layers.append(ResidualUnit(in_channels, out_channels, stride=first_stride, downsample=downsample))

        for i in range(1, num_layers):
            layers.append(ResidualUnit(out_channels, out_channels))

        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)
      

class Backbone(nn.Module):
    def __init__(self, in_channels):
        super(Backbone, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )                                               # Block 1 with 2 layers, stride 1, 32 channels
        
        self.block2 = ResidualBlock(32, 96, 3, first_stride=2)  # Block 2 with 3 layers, first stride 2, 96 channels
        self.block3 = ResidualBlock(96, 192, 6, first_stride=2)  # Block 3 with 6 layers, first stride 2, 192 channels
        self.block4 = ResidualBlock(192, 256, 6, first_stride=2)  # Block 4 with 6 layers, first stride 2, 256 channels
        self.block5 = ResidualBlock(256, 384, 3, first_stride=2)  # Block 5 with 3 layers, first stride 2, 384 channels

        #Upsampling 1
        self.up1_conv = nn.Conv2d(384, 196, kernel_size=1, stride=1, padding=0)
        self.up1_deconv = nn.ConvTranspose2d(196, 128, kernel_size=2, stride=2)
        self.up1_conv_corresponding = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)

        #upsampling 2
        self.up2_deconv = nn.ConvTranspose2d(128, 96, kernel_size=3, stride=2, padding=1, output_padding=(1, 0))
        self.up2_conv_corresponding = nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0)

    def upsample_1(self, x, correspoding_x):
        x = self.up1_conv(x)            # (196, 50, 44)
        x = self.up1_deconv(x)          # (128, 100, 88)
        correspoding_x = self.up1_conv_corresponding(correspoding_x) # (384, 50, 44)
        return x + correspoding_x
    
    def upsample_2(self, x, correspoding_x):
        x = self.up2_deconv(x)
        correspoding_x = self.up2_conv_corresponding(correspoding_x)
        return x + correspoding_x 

    
    def forward(self, x):
        x1 = self.block1(x)  # (32, 800, 700)
        x2 = self.block2(x1)  # (96, 400, 350)
        x3 = self.block3(x2)  # (192, 200, 175)
        x4 = self.block4(x3)  # (256, 100, 88)
        x5 = self.block5(x4)  # (384, 50, 44)
        
        x6 = self.upsample_1(x5, x4)
        x7 = self.upsample_2(x6, x3)
        return x7  # Final output of the network


class Header(nn.Module):
    def __init__(self, out_channels) :
        super(Header, self).__init__()
        basic_block = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True))
        self.conv1 = basic_block
        self.conv2 = copy.deepcopy(basic_block)
        self.conv3 = copy.deepcopy(basic_block)
        self.conv4 = copy.deepcopy(basic_block)
        self.sigmoid = nn.Sigmoid()
        self.classification = nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)
        self.regression = nn.Conv2d(out_channels, 6, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        cls = self.sigmoid((self.classification(x)))
        reg = self.regression(x)
        return cls, reg
    

class PixorNet(nn.Module):
    def __init__(self):
        super(PixorNet, self).__init__()
        self.backbone = Backbone(36)
        self.header = Header(96)

    def forward(self, lidar_input): 
        cls, reg = self.header(self.backbone(lidar_input))
        return torch.cat((cls, reg), dim=1)
    
    def print_no_of_parameters(self):
        total_backbone = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_header = sum(p.numel() for p in self.header.parameters() if p.requires_grad)
        total_params = total_backbone+total_header
        print(f"Total number of learnable parameters in the BackboneNet: {total_backbone}")
        print(f"Total number of learnable parameters in the HeaderNet: {total_header}")
        print(f"Total number of learnable parameters in entire PixorNet: {total_params}")


