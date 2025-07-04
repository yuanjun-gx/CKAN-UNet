import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from Models.encoder import Encoder
from Convkan.fastkanconv import FastKANConvLayer as ConvKAN
from Models.decoder import MSIF, Drop, DPIF_DecoderBlock

nonlinearity = partial(F.relu, inplace=True)


class CKANUNet(nn.Module):
    def __init__(self, num_classes=2, channels=3):
        super(CKANUNet, self).__init__()

        self.rla_channel = 16
        self.attention = 'Sim'  # "SE", "ECA"
        # self.attention = 'ECA'  # "SE", "ECA"
        self.filters = [32, 32, 64, 128, 256]
        # self.size = [5e-3, 5e-3, 5e-3, 7e-3]
        self.size = [5e-2, 5e-3, 5e-3, 5e-2]
        self.layers = [2, 3, 3, 2]


        self.model = Encoder(rla_channel=self.rla_channel, layers=self.layers, attention=self.attention,
                             channel=self.filters, k_size=self.size)

        self.conv1 = ConvKAN(self.filters[4] + self.rla_channel, (self.filters[4] + self.rla_channel) // 2,
                             kernel_size=1, stride=1, use_base_update=False)

        self.conv2 = ConvKAN((self.filters[4] + self.rla_channel) // 2, self.filters[4] + self.rla_channel,
                             kernel_size=1, stride=1, use_base_update=False)

        self.flat_layer = MSIF(in_channels=(self.filters[4] + self.rla_channel) // 2, ratio=2)

        self.drop_block = Drop(drop_rate=0.2, block_size=2)

        self.decoder4 = DPIF_DecoderBlock(self.filters[4], self.filters[3], rla_channel=self.rla_channel,
                                          Attention=self.attention, size=self.size[3])
        self.decoder3 = DPIF_DecoderBlock(self.filters[3], self.filters[2], rla_channel=self.rla_channel,
                                          Attention=self.attention, size=self.size[2])
        self.decoder2 = DPIF_DecoderBlock(self.filters[2], self.filters[1], rla_channel=self.rla_channel,
                                          Attention=self.attention, size=self.size[1])
        self.decoder1 = DPIF_DecoderBlock(self.filters[1], self.filters[0], rla_channel=self.rla_channel,
                                          Attention=self.attention, size=self.size[0])

        self.finaldeconv1 = nn.ConvTranspose2d(self.filters[0] + self.rla_channel, 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv3 = ConvKAN(32, num_classes, 3, padding=1)

    def forward(self, x, training=True):
        # Encoder
        e1, e2, e3, e4, e_h1, e_h2, e_h3, e_h4 = self.model(x)

        # Center
        flat_feature = torch.cat((e4, e_h4), dim=1)
        flat_feature = self.conv1(flat_feature)
        flat_feature = self.flat_layer(flat_feature)
        flat_feature = self.drop_block(flat_feature)
        flat_feature = self.conv2(flat_feature)

        e4_flat, eh4_flat = torch.split(flat_feature, [self.filters[4], self.rla_channel], dim=1)

        # Decoder
        dh_4 = eh4_flat
        d3, dh_3 = self.decoder4(e4_flat, dh_4)

        # dh_4 = e_h4
        # d3, dh_3 = self.decoder4(e4, dh_4)

        d3 = d3 + e3
        dh_3 = dh_3 + e_h3

        d2, dh_2 = self.decoder3(d3, dh_3)
        d2 = d2 + e2
        dh_2 = dh_2 + e_h2

        d1, dh_1 = self.decoder2(d2, dh_2)
        d1 = d1 + e1
        dh_1 = dh_1 + e_h1

        d0, dh_0 = self.decoder1(d1, dh_1)
        d0_out = torch.cat((d0, dh_0), dim=1)

        out = self.finaldeconv1(d0_out)
        out = self.finalrelu1(out)
        out = self.finalconv3(out)

        if training:
            encoder = [e1, e2, e3, e4]
            decoder = [d0, d1, d2, d3]
            final = d0
            output = F.sigmoid(out)
            return output, encoder, decoder, final
        else:
            del e1, e2, e3, e4,d0, d1, d2, d3
            del e_h1, e_h2, e_h3, e_h4, dh_4, dh_3, dh_2, dh_1
            torch.cuda.empty_cache()
            return F.sigmoid(out)