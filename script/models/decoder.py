import torch.nn as nn
import torch
from torchsummary import summary
from kornia.filters import filter2d

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)

class Decoder(nn.Module):
    ''' from Giraffe Nueral Rendering class

    Args:
        n_feat (int): number of features
        input_dim (int): input dimension; if not equal to n_feat,
            it is projected to n_feat with a 1x1 convolution
        out_dim (int): output dimension
        final_actvn (bool): whether to apply a final activation (sigmoid)
        min_feat (int): minimum features
        n_blocks (int): n blocks for upsampling, 15x27 -> 240x427 => 4 blocks
        use_rgb_skip (bool): whether to use RGB skip connections
        upsample_feat (str): upsampling type for feature upsampling
        upsample_rgb (str): upsampling type for rgb upsampling
        use_norm (bool): whether to use normalization
    '''

    def __init__(
            self, n_feat=128, input_dim=128, out_dim=3, final_actvn=True,
            min_feat=32, n_blocks=4, use_rgb_skip=True,
            upsample_feat="nn", upsample_rgb="bilinear", use_norm=False,
            **kwargs):
        super().__init__()
        self.final_actvn = final_actvn
        self.input_dim = input_dim
        self.use_rgb_skip = use_rgb_skip
        self.use_norm = use_norm
        self.h_dim = kwargs['h_dim'] # GT image size
        self.w_dim = kwargs['w_dim'] # GT image size
        self.n_blocks = n_blocks
        # n_blocks = int(log2(img_size) - 4) # 4 here represent 16x16 featuremap?

        assert(upsample_feat in ("nn"))
        self.upsample_2 = nn.Upsample(scale_factor=2.) # feature upsampling
        assert(upsample_rgb in ("bilinear"))
        self.upsample_rgb = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur()) # rgb upsampling 

        # for the last layer, we upsample to specified resolution, same with GT image
        self.upsample_feat_final = nn.Upsample(size=[self.h_dim, self.w_dim])
        self.upsample_rgb_final = nn.Sequential(nn.Upsample(
                size=[self.h_dim, self.w_dim], mode='bilinear', align_corners=False), Blur())

        if n_feat == input_dim:
            self.conv_in = lambda x: x
        else:
            self.conv_in = nn.Conv2d(input_dim, n_feat, 1, 1, 0)

        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(n_feat, max(n_feat // 2, min_feat), 3, 1, 1)] +
            [nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
                       max(n_feat // (2 ** (i + 2)), min_feat), 3, 1, 1)
                for i in range(0, n_blocks - 1)]
        )
        if use_rgb_skip:
            self.conv_rgb = nn.ModuleList(
                [nn.Conv2d(input_dim, out_dim, 3, 1, 1)] +
                [nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
                           out_dim, 3, 1, 1) for i in range(0, n_blocks)]
            )
        else:
            self.conv_rgb = nn.Conv2d(
                max(n_feat // (2 ** (n_blocks)), min_feat), 3, 1, 1)

        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, return_hier_rgbs=False):
        '''
        x: features
        return_hier_rgbs: return list of hidden levels of rgbs for photometric supervision if True
        '''

        net = self.conv_in(x)

        if self.use_rgb_skip: # 1st time upsample to rgb, should be bilinear
            if self.n_blocks>1:
                rgb = self.upsample_rgb(self.conv_rgb[0](x))
            else:
                rgb = self.upsample_rgb_final(self.conv_rgb[0](x))

        if return_hier_rgbs==True:
            rgbs = []

        for idx, layer in enumerate(self.conv_layers):
            # print("idx", idx)

            if idx < len(self.conv_layers) - 1:
                hid = layer(self.upsample_2(net))
            else:
                hid = layer(self.upsample_feat_final(net))

            net = self.actvn(hid)

            if self.use_rgb_skip:
                rgb = rgb + self.conv_rgb[idx + 1](net)

                if return_hier_rgbs==True:
                    rgbs.append(rgb)

                if idx < len(self.conv_layers) - 2:
                    rgb = self.upsample_rgb(rgb)
                elif idx == len(self.conv_layers) - 2:
                    rgb = self.upsample_rgb_final(rgb)

        if not self.use_rgb_skip:
            rgb = self.conv_rgb(net)

        if return_hier_rgbs==True:
            # do not apply final activation
            torch.clamp(rgbs[-1], 0., 1.)
            return rgbs

        if self.final_actvn:
            rgb = torch.sigmoid(rgb)

        return rgb

def main():
    """
    test decoders
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_feat=128
    n_blocks=3
    kwargs = dict(h_dim=240, w_dim=427)

    decoder = Decoder(n_feat=128, input_dim=128, out_dim=3, final_actvn=True,
            min_feat=32, n_blocks=n_blocks, use_rgb_skip=True,
            upsample_feat="nn", upsample_rgb="bilinear", use_norm=False, **kwargs)
    decoder = decoder.to(device)
    # summary(decoder, (128, 15, 27))
    summary(decoder, (128, 30, 54))

    # f_in = torch.rand(1, 128, 15, 27).to(device) # B,C,H,W
    # pdb.set_trace()
    # out = decoder(f_in)


if __name__ == '__main__':
  main()
