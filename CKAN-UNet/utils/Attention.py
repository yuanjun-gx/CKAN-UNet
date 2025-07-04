import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange
import einops
from torch import einsum
from torch.nn import Module, Parameter, Softmax


# ################################ ECA ######################################################################
class ECALayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
        source: https://github.com/BangguWu/ECANet
    """
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        print(x.shape)

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


# ################################ SE ######################################################################
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SimAMLayer(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAMLayer, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        del x_minus_mu_square

        return x * self.activaton(y)


# ################################ CBAM ######################################################################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


# ################################ CPCA ######################################################################
class CPCA_ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(CPCA_ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)

        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)

        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class CPCA(nn.Module):
    def __init__(self, channels, channelAttention_reduce=4):
        super().__init__()

        self.ca = CPCA_ChannelAttention(input_channels=channels, internal_neurons=channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dconv1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3),
                                  groups=channels)
        self.dconv7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0),
                                  groups=channels)
        self.dconv1_11 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5),
                                   groups=channels)
        self.dconv11_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0),
                                   groups=channels)
        self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10),
                                   groups=channels)
        self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0),
                                   groups=channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        inputs = self.conv(inputs)
        inputs = self.act(inputs)
        inputs = self.ca(inputs)

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)

        x = x_1 + x_2 + x_3 + x_init

        spatial_att = self.conv(x)

        out = spatial_att * inputs

        out = self.conv(out)

        return out


# ################################ ECAv2 ######################################################################
class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)


# ################################ GCSA ######################################################################
class GCSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(GCSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


# ################################ DA ######################################################################
class DepthWiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(DepthWiseConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class PAM_Module(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = DepthWiseConv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv = DepthWiseConv2d(in_dim, in_dim, kernel_size=1)
        self.value_conv = DepthWiseConv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class DA_Block(nn.Module):
    def __init__(self, in_channels):
        super(DA_Block, self).__init__()
        inter_channels = in_channels // 16

        self.conv5a = nn.Sequential(DepthWiseConv2d(in_channels, inter_channels, 3, padding=1),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(DepthWiseConv2d(in_channels, inter_channels, 3, padding=1),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(DepthWiseConv2d(inter_channels, inter_channels, 3, padding=1),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(DepthWiseConv2d(inter_channels, inter_channels, 3, padding=1),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.05, False), DepthWiseConv2d(inter_channels, in_channels, 1),
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Dropout2d(0.05, False), DepthWiseConv2d(inter_channels, in_channels, 1),
                                   nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.05, False), DepthWiseConv2d(in_channels, in_channels, 1),
                                   nn.ReLU())

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output1 = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output2 = self.conv7(sc_conv)

        feat_sum = sa_output1 + sc_output2

        sasc_output = self.conv8(feat_sum)

        return sasc_output


# ################################ EAA ######################################################################
class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """
    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()
        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)
        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)
        query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
        key = torch.nn.functional.normalize(key, dim=-1) #BxNxD
        query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor # BxNx1
        A = torch.nn.functional.normalize(A, dim=1) # BxNx1
        G = torch.sum(A * query, dim=1) # BxD
        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        ) # BxNxD
        out = self.Proj(G * key) + query #BxNxD
        out = self.final(out) # BxNxD
        return out