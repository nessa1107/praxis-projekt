import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # Dropout
        self.dropout = nn.Dropout(p=0.5)
        # Contracting Path
        self.conv1 = self.conv_block(in_channels, 64)
        self.conv2 = self.conv_block(64, 128)
        self.conv3 = self.conv_block(128, 256)
        self.conv4 = self.conv_block(256, 512)
        self.conv5 = self.conv_block(512, 1024)
        # Expansive Path
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = self.conv_block(128, 64)
        # Output
        self.conv10 = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            self.dropout
        )
        return block

    def forward(self, x):
        # Contracting Path
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(nn.functional.max_pool2d(conv1_out, kernel_size=2, stride=2))
        conv3_out = self.conv3(nn.functional.max_pool2d(conv2_out, kernel_size=2, stride=2))
        conv4_out = self.conv4(nn.functional.max_pool2d(conv3_out, kernel_size=2, stride=2))
        conv5_out = self.conv5(nn.functional.max_pool2d(conv4_out, kernel_size=2, stride=2))
        # Expansive Path
        upconv4_out = self.upconv4(conv5_out)
        concat4_out = torch.cat((upconv4_out, conv4_out), dim=1)
        conv6_out = self.conv6(concat4_out)
        upconv3_out = self.upconv3(conv6_out)
        concat3_out = torch.cat((upconv3_out, conv3_out), dim=1)
        conv7_out = self.conv7(concat3_out)
        upconv2_out = self.upconv2(conv7_out)
        concat2_out = torch.cat((upconv2_out, conv2_out), dim=1)
        conv8_out = self.conv8(concat2_out)
        upconv1_out = self.upconv1(conv8_out)
        concat1_out = torch.cat((upconv1_out, conv1_out), dim=1)
        conv9_out = self.conv9(concat1_out)
        # Output
        output = self.conv10(conv9_out)
        return output


