import torch
import torch.nn as nn
from network_module import *
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F

class Block_of_DMT1(nn.Module):
    def __init__(self):
        super(Block_of_DMT1,self).__init__()
 
        #DMT1
        self.conv1_1=nn.Conv2d(in_channels=160,out_channels=160,kernel_size=3,stride=1,padding=1)
        self.bn1_1=nn.BatchNorm2d(160, affine=True)
        self.relu1_1=nn.ReLU()
 
    def forward(self, x):
        output = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        return output 
 
class Block_of_DMT2(nn.Module):
    def __init__(self):
        super(Block_of_DMT2,self).__init__()
 
        #DMT1
        self.conv2_1=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn2_1=nn.BatchNorm2d(256, affine=True)
        self.relu2_1=nn.ReLU()
 
    def forward(self, x):
        output = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        return output 
 
class Block_of_DMT3(nn.Module):
    def __init__(self):
        super(Block_of_DMT3,self).__init__()
 
        #DMT1
        self.conv3_1=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn3_1=nn.BatchNorm2d(256, affine=True)
        self.relu3_1=nn.ReLU()
 
    def forward(self, x):
        output = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        return output 

class Block_of_DMT4(nn.Module):
    def __init__(self):
        super(Block_of_DMT4,self).__init__()
 
        #DMT1
        self.conv4_1=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn4_1=nn.BatchNorm2d(256, affine=True)
        self.relu4_1=nn.ReLU()
 
    def forward(self, x):
        output = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        return output 

class MWCNN(nn.Module):
    def __init__(self, opt):
        super(MWCNN, self).__init__()
        self.DWT = DWTForward(J=1, wave='haar').cuda() 
        self.IDWT = DWTInverse(wave='haar').cuda()
        # The generator is U shaped
        # Encoder
        self.E1 = Conv2dLayer(in_channels = 3*4,  out_channels = 160, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'relu', norm = opt.norm)
        self.E2 = Conv2dLayer(in_channels = 640, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'relu',norm = opt.norm)
        self.E3 = Conv2dLayer(in_channels = 1024, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'relu',norm = opt.norm)
        self.E4 = Conv2dLayer(in_channels = 1024, out_channels = 256, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'relu',norm = opt.norm)
        # Bottle neck
        self.BottleNeck = nn.Sequential(
            ResConv2dLayer(256, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(256, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(256, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(256, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        self.blockDMT1 = self.make_layer(Block_of_DMT1,3)
        self.blockDMT2 = self.make_layer(Block_of_DMT2,3)
        self.blockDMT3 = self.make_layer(Block_of_DMT3,3)
        self.blockDMT4 = self.make_layer(Block_of_DMT4,3)
        # Decoder
        self.D1 = Conv2dLayer(in_channels = 256, out_channels = 1024, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'relu', norm = opt.norm)
        self.D2 = Conv2dLayer(in_channels = 256, out_channels = 1024, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'relu', norm = opt.norm)
        self.D3 = Conv2dLayer(in_channels = 256, out_channels = 640,kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'relu', norm = opt.norm)
        self.D4 = Conv2dLayer(in_channels = 160, out_channels = 3*4, kernel_size=3, stride = 1, padding = 1, dilation = 1, pad_type = opt.pad, norm = 'none', activation = 'tanh')

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
 
    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:,:,i,:,:])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def _Itransformer(self,out):
        yh = []
        C=int(out.shape[1]/4)
        # print(out.shape[0])
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:,:,0].contiguous()
        yh.append(y[:,:,1:].contiguous())
 
        return yl, yh

    def forward(self, x):
        residual = x
        # print(x.shape)
        #DMT1
        DMT1_yl,DMT1_yh = self.DWT(x)
        DMT1 = self._transformer(DMT1_yl, DMT1_yh)
        # print(DMT1.shape)
        E1 = self.E1(DMT1)
        E1 = self.blockDMT1(E1)    # channel = 160        

        #DMT2
        DMT2_yl, DMT2_yh = self.DWT(E1)
        DMT2 = self._transformer(DMT2_yl, DMT2_yh)
        # print(DMT2.shape)
        E2 = self.E2(DMT2)
        E2 = self.blockDMT2(E2)    # channel = 256

        #DMT3
        DMT3_yl, DMT3_yh = self.DWT(E2)
        DMT3 = self._transformer(DMT3_yl, DMT3_yh)
        # print(DMT3.shape)
        E3 = self.E3(DMT3)
        E3 = self.blockDMT3(E3)     #channel = 256

        #DMT4
        DMT4_yl, DMT4_yh = self.DWT(E3)
        DMT4 = self._transformer(DMT4_yl, DMT4_yh)
        # print(DMT4.shape)
        E4 = self.E4(DMT4)
        E4 = self.blockDMT4(E4)     #channel = 256
        E4 = self.BottleNeck(E4)

        #IDMT4
        D1=self.blockDMT4(E4)
        D1=self.D1(D1)
        D1=self._Itransformer(D1)
        IDMT4=self.IDWT(D1)
        D1=IDMT4+E3
        #IDMT3
        D2=self.blockDMT3(D1)
        D2=self.D2(D2)
        D2=self._Itransformer(D2)
        IDMT3=self.IDWT(D2)
        D2=IDMT3+E2
        #IDMT2
        D3=self.blockDMT2(D2)
        D3=self.D3(D3)
        D3=self._Itransformer(D3)
        IDMT2=self.IDWT(D3)
        D3=IDMT2+E1
        #IDMT1
        D4=self.blockDMT1(D3)
        D4 = self.D4(D3)  
        D4 = self._Itransformer(D4)
        D4 = self.IDWT(D4)
        x = D4 + residual
        return x
