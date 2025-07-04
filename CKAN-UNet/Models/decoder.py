import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from Convkan.fastkanconv import FastKANConvLayer as ConvKAN
from utils.Attention import SELayer, SimAMLayer, ECALayer, DA_Block, GCSA

nonlinearity = partial(F.relu, inplace=True)
def BNReLU(num_features):
    
    return nn.Sequential(
                nn.BatchNorm2d(num_features),
                nn.ReLU()
            )

# ############################################## Drop block ###########################################

class Drop(nn.Module):
    # drop_rate : 1-keep_prob  (all droped feature points)
    # block_size : 
    def __init__(self, drop_rate=0.1, block_size=2):
        super(Drop, self).__init__()
 
        self.drop_rate = drop_rate
        self.block_size = block_size
 
    def forward(self, x):
    
        if not self.training:
            return x
        
        if self.drop_rate == 0:
            return x
            
        gamma = self.drop_rate / (self.block_size**2)
        # torch.rand(*sizes, out=None) 
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
 
        mask = mask.to(x.device)
 
        # compute block mask
        block_mask = self._compute_block_mask(mask)
        out = x * block_mask[:, None, :, :]
        out = out * block_mask.numel() / block_mask.sum()
        return out
 
    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size,
                                               self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask

# ############################################## MSIF_module ############################################################
class SPP_inception_block(nn.Module):
    def __init__(self, in_channels):
        super(SPP_inception_block, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[1, 1], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool4 = nn.MaxPool2d(kernel_size=[4, 4], stride=4)

        self.dilate1 = ConvKAN(in_channels, in_channels, kernel_size=1, dilation=1, padding=0)
        self.dilate2 = ConvKAN(in_channels, in_channels, kernel_size=3, dilation=1, padding=1)
        self.dilate3 = ConvKAN(in_channels, in_channels, kernel_size=3, dilation=2, padding=2)
        self.dilate4 = ConvKAN(in_channels, in_channels, kernel_size=3, dilation=3, padding=3)

    def forward(self, x):
        b, c, h, w = x.size()
        pool_1 = self.pool1(x).view(b, c, -1)
        pool_2 = self.pool2(x).view(b, c, -1)
        pool_3 = self.pool3(x).view(b, c, -1)
        pool_4 = self.pool4(x).view(b, c, -1)
        
        pool_cat = torch.cat([pool_1, pool_2, pool_3, pool_4], -1)
        
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(x))
        dilate3_out = nonlinearity(self.dilate3(x))
        dilate4_out = nonlinearity(self.dilate4(x))

        cnn_out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        cnn_out = cnn_out.view(b, c, -1)
        
        out = torch.cat([pool_cat, cnn_out], -1)
        out = out.permute(0, 2, 1)
       
        return out

class NonLocal_spp_inception_block(nn.Module):

    def __init__(self, in_channels=512, ratio= 2):
        super(NonLocal_spp_inception_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.value_channels = in_channels//ratio      # key == value
        self.query_channels = in_channels//ratio
            
        # self.f_value = nn.Sequential(
        #                            nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1, stride=1, padding=0),
        #                             BNReLU(self.value_channels),
        #                           )
        #
        # self.f_query = nn.Sequential(
        #                            nn.Conv2d(in_channels=self.in_channels, out_channels=self.query_channels, kernel_size=1, stride=1, padding=0),
        #                             BNReLU(self.query_channels),
        #                           )
        #
        # self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
        #                    kernel_size=1, stride=1, padding=0)

        self.f_value = nn.Sequential(
            ConvKAN(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1, stride=1,
                      padding=0),
            BNReLU(self.value_channels),
        )

        self.f_query = nn.Sequential(
            ConvKAN(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                      padding=0),
            BNReLU(self.in_channels),
        )

        self.W = ConvKAN(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
      
        self.spp_inception_v = SPP_inception_block(self.value_channels)  # key == value
        

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)     # [4, 272, 7, 7]

        x_v = self.f_value(x)                                  # [4, 136, 7, 7]
        value = self.spp_inception_v(x_v)                      # [4, 30+49, 136]
        
        # query = self.f_query(x).view(batch_size, self.value_channels, -1)       # [4, 136, 7, 7], [4, 136, 49]
        # query = query.permute(0, 2, 1)

        Q = self.f_query(x)
        Q1, Q2 = torch.split(Q, [self.query_channels, self.query_channels], dim=1)  #  [4, 136, 7, 7],
        query = (Q1-Q2).view(batch_size, self.value_channels, -1)               #  [4, 136, 49]
        query = query.permute(0, 2, 1)                                          #  [4, 49, 136]
        
        key_0 = value
        key = key_0.permute(0, 2, 1)                                            # [4, 136, 79]
              
        sim_map = torch.matmul(query, key)                                      # [4, 49, 79]
        sim_map = (self.value_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        
        context = torch.matmul(sim_map, value)                                  # [4, 49, 136]
        context = context.permute(0, 2, 1).contiguous()                   # [4, 136, 49]
        context = context.view(batch_size, self.value_channels, *x.size()[2:])  # [4, 136, 7, 7]
        context = self.W(context)                                               # [4, 272, 7, 7]

        return context

class MSIF(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels=512, ratio= 2, dropout=0.0):
        super(MSIF, self).__init__()

        self.NSIB = NonLocal_spp_inception_block(in_channels=in_channels, ratio= ratio)


    def forward(self, feats):
        att = self.NSIB(feats)
        output = att + feats
        
        return output

# ################################ DPIF at decoder stage ######################################################################
class DPIF_DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, rla_channel=32, Attention='Sim', size=None, reduction=16):
        super(DPIF_DecoderBlock, self).__init__()

        self.conv1 = ConvKAN(in_channels+rla_channel, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = ConvKAN(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity
        self.expansion = 1
        self.Attention = Attention
        
        self.deconv_h = nn.ConvTranspose2d(rla_channel, rla_channel, 3, stride=2, padding=1, output_padding=1)
        self.deconv_x = nn.ConvTranspose2d(in_channels, n_filters, 3, stride=2, padding=1, output_padding=1)

        if Attention == 'Sim':
            self.Sim = SimAMLayer(e_lambda=size)
        elif Attention == 'SE':
            self.se = SELayer(n_filters * self.expansion, reduction)
        # else:
        #     self.eca = GCSA(n_filters * self.expansion, 2, True)

        self.conv_out = nn.Conv2d(n_filters, rla_channel, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.norm4 = nn.BatchNorm2d(rla_channel)
        self.tanh = nn.Tanh()        

    def forward(self, x, h):
        identity = x
        x = torch.cat((x, h), dim=1)
    
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        
        out = self.deconv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.norm3(out)
        
        if self.Attention == 'Sim':
            out = self.Sim(out)
        elif self.Attention == 'SE':
            out = self.se(out)
        # else:
        #     out = self.eca(out)
        y = out
        
        identity = self.deconv_x(identity)
        out += identity
        out = self.relu3(out)

        y_out = self.conv_out(y)
        h = self.deconv_h(h)
        h = h + y_out
        h = self.norm4(h)
        h = self.tanh(h)

        return out, h
