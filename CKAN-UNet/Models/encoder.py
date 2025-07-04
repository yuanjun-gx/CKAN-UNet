import torch
import torch.nn as nn
from Convkan.fastkanconv import FastKANConvLayer as ConvKAN
from utils.Attention import SELayer, SimAMLayer, ECALayer, DA_Block, GCSA


def conv3x3(in_planes: int, out_planes: int, stride: int = 1,
            groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return ConvKAN(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                   groups=groups, dilation=dilation, use_base_update=False )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return ConvKAN(in_planes, out_planes, kernel_size=1, stride=stride, use_base_update=False)

# =================================== define bottleneck ====================================
class DPIF_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, Attention='Sim',
                 size=None, rla_channel=32, dilation=1, norm_layer=None, reduction=16):
        super(DPIF_BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes + rla_channel, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride
        self.Attention = Attention
        self.size = size

        if self.Attention == 'Sim':
            self.Sim = SimAMLayer(e_lambda=self.size)
        elif self.Attention == 'SE':
            self.se = SELayer(planes * self.expansion, reduction)
        # else:
        #     self.eca = GCSA(planes * self.expansion, 2, True)

        self.averagePooling = None
        if downsample is not None and stride != 1:
            self.averagePooling = nn.AvgPool2d((2, 2), stride=(2, 2))

    def forward(self, x, h):
        identity = x

        x = torch.cat((x, h), dim=1)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.Attention == 'Sim':
            out = self.Sim(out)
        elif self.Attention == 'SE':
            out = self.se(out)
        # else:
        #     out = self.eca(out)
        y = out

        if self.downsample is not None:
            identity = self.downsample(identity)
        if self.averagePooling is not None:
            h = self.averagePooling(h)

        out += identity
        out = self.relu(out)

        return out, y, h

# ================================ define encoder network ==================================
class EncoderNet(nn.Module):

    def __init__(self, block, layers, rla_channel=32, Attention='Sim',
                 size=None, channels=None, norm_layer=None):
        super(EncoderNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if channels is None:
            channels = [64, 64, 128, 256, 512]

        if size is None:
            size = [5e-3, 5e-3, 5e-3, 7e-3]

        self._norm_layer = norm_layer
        self.inplanes = channels[0]
        self.dilation = 1
        self.rla_channel = rla_channel

        self.conv1 = ConvKAN(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        conv_outs = [None] * 4
        stages = [None] * 4
        stage_bns = [None] * 4
        stages[0], stage_bns[0], conv_outs[0] = self.make_layer(block, channels[1], layers[0],
                                             rla_channel=rla_channel, Attention=Attention, size=size[0])

        stages[1], stage_bns[1], conv_outs[1] = self.make_layer(block, channels[2], layers[1],
                                             rla_channel=rla_channel, Attention=Attention, size=size[1], stride=2)

        stages[2], stage_bns[2], conv_outs[2] = self.make_layer(block, channels[3], layers[2],
                                            rla_channel=rla_channel, Attention=Attention, size=size[2], stride=2)

        stages[3], stage_bns[3], conv_outs[3] = self.make_layer(block, channels[4], layers[3],
                                            rla_channel=rla_channel, Attention=Attention, size=size[3], stride=2)

        self.conv_outs = nn.ModuleList(conv_outs)
        self.stages = nn.ModuleList(stages)
        self.stage_bns = nn.ModuleList(stage_bns)
        self.tanh = nn.Tanh()

    def make_layer(self, block, planes, blocks, rla_channel, Attention,
                   size, stride=1, dilate=False):

        conv_out = conv1x1(int(planes * block.expansion), rla_channel)

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != int(planes * block.expansion):
            downsample = nn.Sequential(
                conv1x1(self.inplanes, int(planes * block.expansion), stride),
                norm_layer(int(planes * block.expansion)),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, rla_channel=rla_channel,
                            Attention=Attention, size=size, dilation=previous_dilation, norm_layer=norm_layer))

        self.inplanes = int(planes * block.expansion)

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, rla_channel=rla_channel, Attention=Attention,
                                size=size, dilation=self.dilation, norm_layer=norm_layer))

        bns = [norm_layer(rla_channel) for _ in range(blocks)]

        return nn.ModuleList(layers), nn.ModuleList(bns), conv_out

    def _get_one_layer(self, layers, bns, conv_out, x, h):
        for layer, bn in zip(layers, bns):
            x, y, h = layer(x, h)
            y_out = conv_out(y)
            h = h + y_out
            h = bn(h)
            h = self.tanh(h)

        return x, h

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        batch, _, height, width = x.size()

        h = torch.zeros(batch, self.rla_channel, height, width,
                        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # layer_0
        layer_0 = self.stages[0]
        bns_0 = self.stage_bns[0]
        conv_out_0 = self.conv_outs[0]
        x_1, h_1 = self._get_one_layer(layer_0, bns_0, conv_out_0, x, h)
        # print(x_1.shape)

        # layer_1
        layer_1 = self.stages[1]
        bns_1 = self.stage_bns[1]
        conv_out_1 = self.conv_outs[1]
        x_2, h_2 = self._get_one_layer(layer_1, bns_1, conv_out_1, x_1, h_1)
        # print(x_2.shape)

        # layer_2
        layer_2 = self.stages[2]
        bns_2 = self.stage_bns[2]
        conv_out_2 = self.conv_outs[2]
        x_3, h_3 = self._get_one_layer(layer_2, bns_2, conv_out_2, x_2, h_2)
        # print(x_3.shape)

        # layer_3
        layer_3 = self.stages[3]
        bns_3 = self.stage_bns[3]
        conv_out_3 = self.conv_outs[3]
        x_4, h_4 = self._get_one_layer(layer_3, bns_3, conv_out_3, x_3, h_3)
        # print(x_4.shape)

        return x_1, x_2, x_3, x_4, h_1, h_2, h_3, h_4

    def forward(self, x):
        return self._forward_impl(x)

# =========================== available  encoder models ====================================
def Encoder(rla_channel, attention, layers,channel, k_size):

    model = EncoderNet(DPIF_BasicBlock, layers=layers, rla_channel=rla_channel,
                       Attention=attention, size=k_size, channels=channel)
    return model
