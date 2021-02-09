import torch
from module.TorchLayer import SARTlayer

__all__ = ['attention_unet','aedsnn']

class attention(torch.nn.Module):
    def __init__(self,scale_size):
        super(attention, self).__init__()
        self.grad_x = torch.nn.Conv2d(1,1,kernel_size=(1,3),stride=1,padding=(0,1),bias=False)
        self.grad_y = torch.nn.Conv2d(1,1,kernel_size=(3,1),stride=1,padding=(1,0),bias=False)
        self.line1 = torch.nn.Conv2d(1,1,kernel_size=1,stride=1,padding=0)
        self.line2 = torch.nn.Conv2d(1,1,kernel_size=1,stride=1,padding=0)
        self.sigmoid = torch.nn.Sigmoid()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=scale_size,stride=scale_size)
        for step, p in enumerate(self.parameters()):
            if(step<2):
                p.requires_grad = False
        self.reset_parameters()

    def forward(self,input, attention_image):
        attention_image = self.maxpool(attention_image)
        d_x = 0.5*self.grad_x(attention_image)
        d_y = 0.5*self.grad_y(attention_image)
        d_x = self.sigmoid(self.line1(d_x))
        d_y = self.sigmoid(self.line2(d_y))
        out = torch.zeros_like(input,dtype=torch.float32)
        out[:,:input.shape[1]//2,:,:] = input[:,:input.shape[1]//2,:,:]*d_x
        out[:,input.shape[1]//2:,:,:] = input[:,input.shape[1]//2:,:,:]*d_y
        return out

    def reset_parameters(self):
        self.grad_x.weight.data = torch.Tensor([-1,0,1]).view(1,1,1,3)
        self.grad_y.weight.data = torch.Tensor([-1,0,1]).view(1,1,3,1)
        self.line1.weight.data = torch.Tensor([1]).view(1,1,1,1)
        self.line2.weight.data = torch.Tensor([1]).view(1,1,1,1)
        self.line1.bias.data = torch.Tensor([0.5])
        self.line2.bias.data = torch.Tensor([0.5])

class block(torch.nn.Module):
    def __init__(self,in_channel,out_channel, k,is_norm,is_x_dir):
        super(block, self).__init__()
        self.is_norm = is_norm
        if (is_x_dir):
            self.conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=(1, k), stride=1, padding=(0, k // 2))
            self.conv2 = torch.nn.Conv2d(out_channel, out_channel, kernel_size=(1, k), stride=1, padding=(0, k // 2))
        else:
            self.conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=(k, 1), stride=1, padding=(k // 2, 0))
            self.conv2 = torch.nn.Conv2d(out_channel, out_channel, kernel_size=(k, 1), stride=1, padding=(k // 2, 0))
        if (is_norm):
            self.bn1 = torch.nn.BatchNorm2d(out_channel)
            self.bn2 = torch.nn.BatchNorm2d(out_channel)
        self.relu = torch.nn.ReLU()

    def forward(self,input):
        if(self.is_norm):
            out = self.relu(self.bn1(self.conv1(input)))
            out = self.relu(self.bn2(self.conv2(out)))
        else:
            out = self.relu(self.conv1(input))
            out = self.relu(self.conv2(out))
        return out

class attention_unet(torch.nn.Module):
    def __init__(self,in_ch,out_ch,k1,k2):
        super(attention_unet, self).__init__()

        self.attention1 = attention(1)
        self.attention2 = attention(2)
        self.attention3 = attention(4)
        self.attention4 = attention(8)

        self.conv0 = torch.nn.Conv2d(in_ch, 64, kernel_size=(1, k1), stride=1, padding=(0, k1 // 2))
        self.conv1 = block(64, 64, k=k1, is_norm=True, is_x_dir=True)
        self.pool1 = torch.nn.AvgPool2d(2)
        self.conv2 = block(64, 128, k=k1, is_norm=True, is_x_dir=True)
        self.pool2 = torch.nn.AvgPool2d(2)
        self.conv3 = block(128, 256, k=k1, is_norm=True, is_x_dir=True)
        self.pool3 = torch.nn.AvgPool2d(2)
        self.conv4 = block(256, 512, k=k1, is_norm=True, is_x_dir=True)
        self.pool4 = torch.nn.AvgPool2d(2)
        self.conv5 = block(512, 1024, k=k1, is_norm=True, is_x_dir=True)

        self.up6 = torch.nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = block(1024, 512, k=k2, is_norm=False, is_x_dir=False)
        self.up7 = torch.nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = block(512, 256, k=k2, is_norm=False, is_x_dir=False)
        self.up8 = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = block(256, 128, k=k2, is_norm=False, is_x_dir=False)
        self.up9 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = block(128, 64, k=k2, is_norm=False, is_x_dir=False)
        self.conv10 = torch.nn.Conv2d(64, out_ch, 1)
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(2, 1, kernel_size=1, bias=False)

    def forward(self, x):
        c0 = self.relu(self.conv0(x))
        c1 = self.conv1(c0)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        c4 = self.attention4(c4,x)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        c3 = self.attention3(c3, x)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        c2 = self.attention2(c2, x)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        c1 = self.attention1(c1, x)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = self.relu(c10)
        # out = torch.cat((x, out), dim=1)
        # out = self.relu(self.conv(out))
        return out

class aedsnn(torch.nn.Module):
    def __init__(self,k1,k2,block=3):
        super(aedsnn, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.block = block
        self.sart = SARTlayer()
        self.layers = torch.nn.ModuleList([attention_unet(1,1,k1=k1,k2=k2).cuda() for i in range(block)])
        self.init_weight()
        
    def forward(self,input,proj):
        for step, layer in enumerate(self.layers):
            out = self.sart(input,proj)
            input = layer(out)
        return input

    def init_weight(self):
        for i in range(self.block):
            self.layer=torch.load(r'../checkpoints/attention_unet__{}_{}_{}.pt'.format(self.k1,self.k2,i))
