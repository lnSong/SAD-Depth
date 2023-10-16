from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers import *
from timm.models.layers import trunc_normal_

device = "cuda"

class SAD_Decoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(3), num_output_channels=4, use_skips=True, uncer=False):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales
        self.uncert = uncer
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int')
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.dispconv_num_ch_in = []
        self.convs = OrderedDict()
        for i in range(2, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 2 else num_ch_in
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            if i <2 :
                self.dispconv_num_ch_in.append(num_ch_out)

            num_ch_in = (self.num_ch_dec[i]/4).astype('int')
            
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            if i ==0 :
                self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
                self.dispconv_num_ch_in.append(num_ch_out)
        self.dispconv_num_ch_in.reverse()
        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.dispconv_num_ch_in[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        self.outputs = {}
        x = input_features[-1]
        for i in range(2, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            
            if len(self.scales)>1 and i+1 in self.scales:
                f = self.pixel_shuffle(self.convs[("dispconv", i+1)](x))
                self.outputs[("disp", i+1)] = self.sigmoid(f)

            x_shuffle = self.pixel_shuffle(x)
            x = [x_shuffle]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            if i == 0:
                x = self.convs[("upconv", i, 1)](x)
                f = self.pixel_shuffle(self.convs[("dispconv", i)](x))
                if self.uncert:
                    self.outputs[("uncer", i)] = f
                else:
                    self.outputs[("disp", i)] = self.sigmoid(f)

        return self.outputs
