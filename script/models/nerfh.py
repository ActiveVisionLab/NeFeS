import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import repeat
from models.nerf import run_network, NeRF
from models.decoder import Decoder
from einops import rearrange, reduce, repeat

import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.tensor([10.], device=device))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# moved from old rendering_helper.py
# transfer model output to color --- my version
def raw2outputs_NeRFW(raw, z_vals, raw_noise_std=0, output_transient=False, beta_min=0.1, white_bkgd=False, test_time=False, static_only=True, typ="coarse", loss_mode=0, store_rgb=False): # softplus_on_beta=False
    ''' Convert NeRFW fine network output to rendered colors
    This version is implemented in nerf_pl https://github.com/kwea123/nerf_pl/tree/nerfw
    Inputs:
        raw: torch.Tensor() [N_rays, N_samples, 9]

    '''

    if typ=="coarse" and test_time and (store_rgb == False):
        static_sigmas = raw[..., 0]
        transient_sigmas = None
    else:
        if output_transient==False:
            N_rays, N_samples, ch = raw.size()
            ch_rgbs = ch - 1 # rgb sigma - sigma channel. for rgb: ch_rgbs = 3, for feature: ch_rgbs = args.out_channel_size
        else:
            N_rays, N_samples, ch = raw.size()
            ch_rgbs = (ch - 3) // 2 # rgb sigma - static_sigma - transient_sigmas - transient_betas channels

        static_rgbs = raw[..., :ch_rgbs] # (N_rays, N_samples, 3), [..., 0:3]
        static_sigmas = raw[..., ch_rgbs] # (N_rays, N_samples) [..., 3]
        if output_transient:
            transient_rgbs = raw[..., ch_rgbs + 1: 2 * ch_rgbs + 1] # [..., 4:7]
            transient_sigmas = raw[..., 2 * ch_rgbs + 1] # [..., 7]
            transient_betas = raw[..., 2 * ch_rgbs + 2] # [..., 8]
        else:
            transient_sigmas = None

    # Convert these values using volume rendering
    deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
    delta_inf = 1e2 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity, nerf used 1e10

    # In original NeRF, Multiply each distance by the norm of its corresponding direction ray
    # but not in this implementation
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_) [32768, 128], 0.3175

    if output_transient:
        static_alphas = 1-torch.exp(-deltas*static_sigmas) # this is [0,1]
        transient_alphas = 1-torch.exp(-deltas*transient_sigmas)
        alphas = 1-torch.exp(-deltas*(static_sigmas+transient_sigmas))
    else:
        noise = torch.randn_like(static_sigmas) * raw_noise_std
        alphas = 1-torch.exp(-deltas*(static_sigmas+noise)) # opacity, as a CDF
        # alphas = 1-torch.exp(-deltas*torch.relu(static_sigmas+noise)) # opacity, as a CDF. Obsolete, activation should be before static_sigmas

    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas], -1) # [1, 1-a1, 1-a2, ...]
    transmittance = torch.cumprod(alphas_shifted[:, :-1], -1) # [1, 1-a1, (1-a1)(1-a2), ...] transmittance Tau_n

    if output_transient:
        static_weights = static_alphas * transmittance
        transient_weights = transient_alphas * transmittance

    weights = alphas * transmittance # This is like a PDF

    weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum')
    acc_map = weights_sum

    if typ=="coarse" and test_time and (store_rgb == False):
        rgb_map=None
        disp_map=None
        depth_map=None
        beta=None
        return rgb_map, disp_map, acc_map, weights, depth_map, transient_sigmas, beta

    if output_transient:
        static_rgb_map = reduce(rearrange(static_weights, 'n1 n2 -> n1 n2 1')*static_rgbs,
                                'n1 n2 c -> n1 c', 'sum') # _rgb_fine_static
        if white_bkgd:
            static_rgb_map += 1-rearrange(weights_sum, 'n -> n 1')
        
        transient_rgb_map = \
            reduce(rearrange(transient_weights, 'n1 n2 -> n1 n2 1')*transient_rgbs,
                   'n1 n2 c -> n1 c', 'sum') # _rgb_fine_transient
        # breakpoint()
        beta = reduce(transient_weights*transient_betas, 'n1 n2 -> n1', 'sum')

        # Add beta_min AFTER the beta composition. Different from eq 10~12 in the paper.
        # See "Notes on differences with the paper" in README.
        # beta += beta_min

        if loss_mode==1:
            # nerfw paper, log(1+exp(beta)) is nn.softplus(). However, we add this to further restrict fine color loss's maximum weights
            beta = beta_min + torch.log(1+torch.exp(beta)) 
        elif loss_mode==0:
            # Add beta_min AFTER the beta composition. Different from eq 10~12 in the paper.
            # See "Notes on differences with the paper" in README.
            beta += beta_min

        try:
            # for debugging
            assert not torch.isnan(beta).any()
            assert not torch.isinf(beta).any()
        except:
            print("transient_betas.max()", transient_betas.max(), "transient_betas.min()", transient_betas.min())
            print("beta.max()", beta.max(), "beta.min()", beta.min())
            breakpoint()

        # the rgb maps here are when both fields exist
        rgb_fine = static_rgb_map + transient_rgb_map
        rgb_map = rgb_fine

        if test_time and static_only: # output static rgbs only, need testing
            # Compute static rgbs when only one field exists.
            # The result is different from when both fields exist, since the transimttance
            # will change.
            static_alphas_shifted = \
                torch.cat([torch.ones_like(static_alphas[:, :1]), 1-static_alphas], -1)
            static_transmittance = torch.cumprod(static_alphas_shifted[:, :-1], -1)
            static_weights_ = static_alphas * static_transmittance
            static_rgb_map_ = \
                reduce(rearrange(static_weights_, 'n1 n2 -> n1 n2 1')*static_rgbs,
                       'n1 n2 c -> n1 c', 'sum')
            if white_bkgd:
                static_rgb_map_ += 1-rearrange(weights_sum, 'n -> n 1')
            rgb_fine = static_rgb_map_
            # mod0
            # depth_map = reduce(static_weights_*z_vals, 'n1 n2 -> n1', 'sum')
            # disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)) 
    
            # disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(static_weights_, -1)) # 
            depth_map = reduce(weights*z_vals, 'n1 n2 -> n1', 'sum')
            disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)) # 
            return rgb_map, disp_map, acc_map, weights, depth_map, transient_sigmas, beta

    else:
        rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')*static_rgbs,
                             'n1 n2 c -> n1 c', 'sum')
        if white_bkgd:
            rgb_map += 1-rearrange(weights_sum, 'n -> n 1')

        beta=torch.Tensor([0]).repeat(weights_sum.shape[0]) # this is useless, just to fake a return

    # compute depth_map and disp_map
    depth_map = torch.sum(weights * z_vals, -1) # this is like the weighted average of depth
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    return rgb_map, disp_map, acc_map, weights, depth_map, transient_sigmas, beta

def run_network_NeRFH(inputs, viewdirs, ts, fn, embed_fn, embeddirs_fn, 
                    typ, embedding_a, embedding_t, output_transient, 
                    netchunk=1024*64, test_time=False):
    ''' We need a new query function, Coarse = NeRF, Fine = NeRF-W 
    Inputs:
        inputs: torch.Tensor() [N_rays,N_samples,3]
        viewdirs: torch.Tensor() [N_rays, 3]
        ts: latent code from img_idxs [N_rays]
        fn: NeRFH object
        embed_fn: embedder for position
        embeddirs_fn: embedder for view directions
        typ: 'coarse' or 'fine'
        embedding_a: NeRFH appearance embedding layer
        embedding_t: NeRFH transient embedding layer
        output_transient: True/False
        netchunk: chunk size to inference
        test_time: True/False
    '''
    # breakpoint()
    out_chunks = []
    N_rays, N_samples = inputs.shape[0], inputs.shape[1]
    # print("typ: {}, test_time: {}".format(typ, test_time))
    # embed inputs like NeRF
    if typ == 'coarse' and test_time: # mod1
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

        # Feed NeRF-W coarse train
        for i in range(0, inputs_flat.shape[0], netchunk):
            embedded_inputs = [embed_fn(inputs_flat[i: i+netchunk])]
            out_chunks += [fn(torch.cat(embedded_inputs, 1), sigma_only=True)]
            
        out = torch.cat(out_chunks, 0) # [N_rays*N_samples, 4]
        out = torch.reshape(out, list(inputs.shape[:-1]) + [out.shape[-1]]) # [N_rays, N_samples, 4]
        return out
    if typ == 'coarse': # case: coarse + train
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

        # Feed NeRF-W coarse train
        for i in range(0, inputs_flat.shape[0], netchunk):
            embedded_inputs = [embed_fn(inputs_flat[i: i+netchunk]), embeddirs_fn(input_dirs_flat[i:i+netchunk])]
            out_chunks += [fn(torch.cat(embedded_inputs, 1), output_transient=output_transient)]
        out = torch.cat(out_chunks, 0) # [N_rays*N_samples, 4]
        out = torch.reshape(out, list(inputs.shape[:-1]) + [out.shape[-1]]) # [N_rays, N_samples, 4]
        return out

    elif typ == 'fine':
        # print("typ: {}, test_time: {}".format(typ, test_time)) ### TODO: maybe more efficient implementation when test_time == True, since we don't need transient network
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

        # embed encode_a and encode_t
        a_embedded = embedding_a(ts.long()) # [N_rays, 48]

        # if output_transient:
        t_embedded = embedding_t(ts.long()) # [N_rays, 16]

        # for embedding histogram experiement
        if len(a_embedded.size()) == 3:
            a_embedded = a_embedded.reshape(N_rays, -1) # [N_rays, hist_bin*N_a]
        if len(t_embedded.size()) == 3:
            t_embedded = t_embedded.reshape(N_rays, -1) # [N_rays, hist_bin*N_tau]

        a_embedded_ = repeat(a_embedded, 'n1 c -> (n1 n2) c', n2=N_samples)
        t_embedded_ = repeat(t_embedded, 'n1 c -> (n1 n2) c', n2=N_samples)

        # Feed NeRF-W fine train
        for i in range(0, inputs_flat.shape[0], netchunk):
            # inputs for original NeRF
            embedded_inputs = [embed_fn(inputs_flat[i: i+netchunk]), embeddirs_fn(input_dirs_flat[i:i+netchunk])]

            # additional inputs for NeRF-W
            embedded_inputs += [a_embedded_[i:i+netchunk]]
            if output_transient:
                embedded_inputs += [t_embedded_[i:i+netchunk]]

            out_chunks += [fn(torch.cat(embedded_inputs, 1), output_transient=output_transient)]
        out = torch.cat(out_chunks, 0) # [N_rays*N_samples, 9]
        out = torch.reshape(out, list(inputs.shape[:-1]) + [out.shape[-1]]) # [N_rays, N_samples, 9]
        return out

# Positional encoding (section 5.1 of NERF)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.N_freqs = 0
        self.N = -1 # epoch to max frequency, for Nerfie embedding only
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        self.N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=self.N_freqs) # tensor([  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256., 512.])
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=self.N_freqs) 

        for freq in freq_bands: # 10 iters for 3D location, 4 iters for 2D direction
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        # inputs [65536, 3]
        if self.kwargs['max_freq_log2'] != 0:
            ret = torch.cat([fn(inputs) for fn in self.embed_fns], -1) # cos, sin embedding # ret.shape [65536, 63]
        else:
            ret = inputs
        return ret

    def get_embed_weight(self, epoch, num_freqs, N):
        ''' Nerfie Paper Eq.(8) '''
        alpha = num_freqs * epoch / N
        W_j = []
        for i in range(num_freqs):
            tmp = torch.clamp(torch.Tensor([alpha - i]), 0, 1)
            tmp2 = (1 - torch.cos(torch.Tensor([np.pi]) * tmp)) / 2
            W_j.append(tmp2)
        return W_j

    def embed_DNeRF(self, inputs, epoch):
        ''' Nerfie paper section 3.5 Coarse-to-Fine Deformation Regularization '''
        # get weight for each frequency band j
        W_j = self.get_embed_weight(epoch, self.N_freqs, self.N) # W_j: [W_0, W_1, W_2, ..., W_{m-1}]
        
        # Fourier embedding
        out = []
        for fn in self.embed_fns: # 17, embed_fns:[input, cos, sin, cos, sin, ..., cos, sin]
            out.append(fn(inputs))

        # apply weighted positional encoding, only to cos&sins
        for i in range(len(W_j)):
            out[2*i+1] = W_j[i] * out[2*i+1]
            out[2*i+2] = W_j[i] * out[2*i+2]
        ret = torch.cat(out, -1)
        return ret

    def update_N(self, N):
        self.N=N


def get_embedder(multires, i=0, reduce_mode=-1, epochToMaxFreq=-1):
    if i == -1:
        return nn.Identity(), 3
    
    if reduce_mode == 0:
        # reduce embedding
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : (multires-1)//2,
                    'num_freqs' : multires//2,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }
    elif reduce_mode == 1:
        # remove embedding
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : 0,
                    'num_freqs' : 0,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }
    elif reduce_mode == 2:
        # DNeRF embedding
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }
    else:
        # paper default
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }

    embedder_obj = Embedder(**embed_kwargs)
    if reduce_mode == 2:
        embedder_obj.update_N(epochToMaxFreq)
        embed = lambda x, epoch, eo=embedder_obj: eo.embed_DNeRF(x, epoch)
    else: 
        embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim, embedder_obj# 63 for pos, 27 for view dir

# Model
class NeRFH(nn.Module):
    def __init__(self, typ,
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27,
                 encode_appearance=False, in_channels_a=48,
                 encode_transient=False, in_channels_t=16,
                 beta_min=0.1, out_ch_size=3):
        """
        ---Parameters for the original NeRF---
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        in_channels_t: number of input channels for t

        ---Parameters for NeRF-W (used in fine model only as per section 4.3)---
        ---cf. Figure 3 of the paper---
        encode_appearance: whether to add appearance encoding as input (NeRF-A)
        in_channels_a: appearance embedding dimension. n^(a) in the paper
        encode_transient: whether to add transient encoding as input (NeRF-U)
        in_channels_t: transient embedding dimension. n^(tau) in the paper
        beta_min: minimum pixel color variance
        """
        super().__init__()
        torch.manual_seed(0)
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_transient = False if typ=='coarse' else encode_transient
        self.in_channels_t = in_channels_t
        self.beta_min = beta_min

        # original NeRF-Hist
        # xyz encoding layers
        for i in range(D):
            if i == 0:
                # tmp = torch.ones(63)
                layer = nn.Linear(in_channels_xyz, W)
                # print("layer(tmp): ", layer(tmp))
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                        nn.Linear(W+in_channels_dir+self.in_channels_a, W//2), nn.ReLU(True))

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        
        if out_ch_size == 3: #NeRF, NeRFH output rgb
            self.static_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
        else: # output features like giraffe
            self.static_rgb = nn.Sequential(nn.Linear(W//2, out_ch_size))

        if self.encode_transient:
            # transient encoding layers
            self.transient_encoding = nn.Sequential(
                                        nn.Linear(W+in_channels_t, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True))
            # transient output layers
            self.transient_sigma = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())
            if out_ch_size == 3: #NeRF, NeRFH output rgb
                self.transient_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
            else: # output features like giraffe
                self.transient_rgb = nn.Sequential(nn.Linear(W//2, out_ch_size))
            self.transient_beta = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())
        
        # pdb.set_trace()

    def forward(self, x, sigma_only=False, output_transient=True):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: the embedded vector of position (+ direction + appearance + transient)
            sigma_only: whether to infer sigma only.
            has_transient: whether to infer the transient component.

        Outputs (concatenated):
            if sigma_ony:
                static_sigma
            elif output_transient:
                static_rgb, static_sigma, transient_rgb, transient_sigma, transient_beta
            else:
                static_rgb, static_sigma
        """
        if sigma_only:
            input_xyz = x
        elif output_transient:
            input_xyz, input_dir_a, input_t = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir+self.in_channels_a,
                                self.in_channels_t], dim=-1)
        else:
            input_xyz, input_dir_a = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir+self.in_channels_a], dim=-1)
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        static_sigma = self.static_sigma(xyz_) # (B, 1)

        if sigma_only:
            return static_sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        static_rgb = self.static_rgb(dir_encoding) # (B, 3)
        static = torch.cat([static_rgb, static_sigma], 1) # (B, 4)

        if not output_transient:
            return static

        transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
        transient_encoding = self.transient_encoding(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding) # (B, 1)
        transient_rgb = self.transient_rgb(transient_encoding) # (B, 3)
        transient_beta = self.transient_beta(transient_encoding) # (B, 1)

        transient = torch.cat([transient_rgb, transient_sigma,
                               transient_beta], 1) # (B, 5)

        return torch.cat([static, transient], 1) # (B, 9)

def create_decoder(args,h_dim,w_dim):
    """ 
    create decoder inspired from GIRAFFE style training
    """
    kwargs = dict(h_dim=h_dim, w_dim=w_dim, n_blocks=args.n_blocks)
    # decoder_coarse = Decoder(n_feat=args.n_feat, input_dim=args.input_dim, min_feat=args.min_feat, **kwargs)
    # decoder_fine = Decoder(n_feat=args.n_feat, input_dim=args.input_dim, min_feat=args.min_feat, **kwargs)
    decoder_coarse = Decoder(n_feat=args.out_channel_size, input_dim=args.out_channel_size, min_feat=args.min_feat, **kwargs)
    decoder_fine = Decoder(n_feat=args.out_channel_size, input_dim=args.out_channel_size, min_feat=args.min_feat, **kwargs)
    return decoder_coarse, decoder_fine

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """

    # initialize embedding functions
    embed_fn, input_ch, _ = get_embedder(args.multires, args.i_embed, args.reduce_embedding) # input_ch.shape=63

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views, _ = get_embedder(args.multires_views, args.i_embed, args.reduce_embedding) # input_ch_views.shape=27
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    device = torch.device("cuda")

    # initialize NeRF model
    if args.NeRFW:
        if args.encode_hist: # embedding histogram
            embedding_a = torch.nn.Embedding(args.N_vocab, 5) # N_vocab > len(dataset), N_a = 5
            embedding_a = embedding_a.to(device)  # static appearance embedder
            embedding_t = torch.nn.Embedding(args.N_vocab, 2) # N_vocab > len(dataset), N_tau = 2
            embedding_t = embedding_t.to(device) # transient appearance embedder

        if args.ffmlp:
            from models.nerfh_ffmlp import NeRFH_FFMLP
            model = NeRFH_FFMLP('coarse', D=args.netdepth, W=args.netwidth, skips=skips, in_channels_xyz=input_ch, in_channels_dir=input_ch_views)
        else:
            model = NeRFH('coarse', D=args.netdepth, W=args.netwidth, skips=skips, in_channels_xyz=input_ch, in_channels_dir=input_ch_views)
    else:
        model = NeRF(D=args.netdepth, W=args.netwidth, input_ch=input_ch, output_ch=output_ch, skips=skips, input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)

    if args.multi_gpu:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    grad_vars = list(model.parameters())

    model_fine = None

    if args.N_importance > 0:
        if args.NeRFW:
            if args.ffmlp:
                model_fine = NeRFH_FFMLP('fine', D=args.netdepth, W=args.netwidth, skips=skips, 
                in_channels_xyz=input_ch, in_channels_dir=input_ch_views,
                encode_appearance=True, encode_transient=True,
                in_channels_a=args.in_channels_a, in_channels_t=args.in_channels_t)
            else:
                model_fine = NeRFH('fine', D=args.netdepth, W=args.netwidth, skips=skips, 
                    in_channels_xyz=input_ch, in_channels_dir=input_ch_views,
                    encode_appearance=True, encode_transient=True,
                    in_channels_a=args.in_channels_a, in_channels_t=args.in_channels_t, out_ch_size=args.out_channel_size)
        else:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine, input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        if args.multi_gpu:
            model_fine = torch.nn.DataParallel(model_fine).to(device)
        else:
            model_fine = model_fine.to(device)
        grad_vars += list(model_fine.parameters())
        if args.encode_hist:
            grad_vars += list(embedding_a.parameters())
            grad_vars += list(embedding_t.parameters())

    if args.NeRFW:
        network_query_fn = lambda inputs, viewdirs, ts, network_fn, \
                typ, embedding_a, embedding_t, output_transient, test_time : run_network_NeRFH(inputs, viewdirs, ts, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                typ=typ, 
                                                                embedding_a=embedding_a, 
                                                                embedding_t=embedding_t, 
                                                                output_transient=output_transient,
                                                                netchunk=args.netchunk,
                                                                test_time=test_time)
    else:
        network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer if NeRF is need to be trained, otherwise returns None
    if args.no_grad_update:
        grad_vars = None
        optimizer = None
    else:
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])

        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
            if args.NeRFW:
                embedding_a.load_state_dict(ckpt['embedding_a_state_dict'])
                embedding_t.load_state_dict(ckpt['embedding_t_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'test_time' : False,
        'args' : args,
        }

    if model_fine is not None:
        render_kwargs_train['network_fine'] = model_fine
        if args.NeRFW:
            render_kwargs_train['embedding_a'] = embedding_a
            render_kwargs_train['embedding_t'] = embedding_t

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['test_time'] = True

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer