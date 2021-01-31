import torch

__all__ = ['unet']

class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch,is_norm=True):
        super(DoubleConv, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.is_norm = is_norm
        if is_norm:
            self.bn1 = torch.nn.BatchNorm2d(out_ch)
            self.bn2 = torch.nn.BatchNorm2d(out_ch)

    def forward(self, input):
        if self.is_norm:
            out = self.relu(self.bn1(self.conv1(input)))
            out = self.relu(self.bn2(self.conv2(out)))
        else:
            out = self.relu(self.conv1(input))
            out = self.relu(self.conv2(out))
        return out

class unet(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128, is_norm=True)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256, is_norm=True)
        self.pool3 = torch.nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512, is_norm=True)
        self.pool4 = torch.nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024, is_norm=True)

        self.up6 = torch.nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512, is_norm=True)
        self.up7 = torch.nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256, is_norm=True)
        self.up8 = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128, is_norm=True)
        self.up9 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64, is_norm=False)
        self.conv10 = torch.nn.Conv2d(64, out_ch, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = self.relu(c10)
        return out
