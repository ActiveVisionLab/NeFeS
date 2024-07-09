import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat
import tinycudann as tcnn # for NeRF acceleration
from models.activation import trunc_exp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from einops import rearrange, reduce, repeat

# NeRF-hist with hash encoding and MLP acceleration
class NeRFH_TCNN(nn.Module):
    def __init__(self, typ,
                 W=64, N_vocab=1000, hash_level=16, 
                 encode_appearance=False, in_channels_a=48,
                 encode_transient=False, in_channels_t=16,
                 beta_min=0.1, bound=25): # beta_min=0.03
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
        self.W = W # hidden_dim
        self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_transient = False if typ=='coarse' else encode_transient
        self.in_channels_t = in_channels_t
        self.beta_min = beta_min
        # bound=25 # what's this?
        # print("Warning: bound is set to 25, add an argument here!")
        self.bound = bound
        print("Warning: bound is set to ", self.bound)
        self.hash_level=hash_level
        # 2+3 layers configuration
        self.num_layers = 2
        self.num_layers_color = 3
        ### 8 layers configuration, from instant-ngp paper, we should use fewer layers
        # self.num_layers = 4
        # self.num_layers_color = 4

        min_resolution = 16
        max_resolution = 2048
        # according to instance-ngp paper, eq. 3: b = exp([ln(N_{max}) - ln(N_{min})]/(L - 1))
        per_level_scale = np.exp(np.log(max_resolution / min_resolution) / (self.hash_level - 1))

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.hash_level,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )

        self.in_channels_xyz = 2*self.hash_level

        self.sigma_net = tcnn.Network(
            n_input_dims=self.in_channels_xyz,
            n_output_dims=W+1, # # W: feature width, 1: sigma,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64, # should be W
                "n_hidden_layers": self.num_layers - 1, # num_layers-1
            },
        )

        # static output layers
        self.static_sigma = nn.Sequential(nn.Softplus()) # TODO: not used
        ### color network ###
        self.num_layers_color = 3 # 3 num_layers_color
        self.hidden_dim_color = 64 # 64 hidden_dim_color

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.in_channels_dir = self.encoder_dir.n_output_dims

        self.embedding_a = torch.nn.Embedding(N_vocab, 5) # args.N_vocab

        self.in_dim_color = W+self.in_channels_dir+self.in_channels_a

        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.hidden_dim_color,
                "n_hidden_layers": self.num_layers_color - 1,
            },
        )

        ### transient color network ###
        if self.encode_transient:
            self.embedding_t = torch.nn.Embedding(N_vocab, 2)

            self.in_dim_transient_color = W+self.in_channels_dir+self.in_channels_t

            self.transient_color_net = tcnn.Network(
                n_input_dims=self.in_dim_transient_color,
                n_output_dims=5,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim_color,
                    "n_hidden_layers": self.num_layers_color,
                },
            )

            # transient output layers, mode0
            # self.transient_sigma = nn.Sequential(nn.Softplus())
            # self.transient_rgb = nn.Sequential(nn.Sigmoid())
            # self.transient_beta = nn.Sequential(nn.Softplus())

            # mod1
            self.transient_sigma = nn.Sequential(nn.ReLU())
            self.transient_rgb = nn.Sequential(nn.Sigmoid())
            self.transient_beta = nn.Sequential(nn.ReLU())

    def input_norm(self, x):
        '''
        normalize input sampled position to [0, 1]
        '''

        return (x + self.bound) / (2 * self.bound) # to [0, 1]

    def density(self, x, norm_input=True):
        '''
        x: xyz sample point coordinate
        norm_input: whether to normalize input x to [0, 1]. Set to false if the input x is already normalized
        '''

        # x: [N, 3], in [-bound, bound]
        if norm_input:
            x = self.input_norm(x)

        if x.max() > 1. or x.min() < 0.: # temporary sanity check
            print("Error: bound is not well set, please reset the self.bound. x.max(): {}, x.min(): {}", x.max(), x.min())
            breakpoint()

        x = self.encoder(x) # [N_samples, 3]
        h = self.sigma_net(x)

        sigma = F.relu(h[..., 0]) # defualt without error
        # sigma = trunc_exp(h[..., 0]) # trunc_exp may produce error
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def color(self, x, d, ts=None, mask=None, geo_feat=None, transient=False, norm_input=True):
        '''
        # x: [N, 3] in [-bound, bound]
        # ts: histogram []
        # mask: [N,], bool, indicates where we actually needs to compute rgb.
        norm_input: whether to normalize input x to [0, 1]. Set to false if the input x is already normalized
        '''

        if norm_input:
            x = self.input_norm(x)

        if x.max() > 1. or x.min() < 0.: # temporary sanity check
            print("bound is not well set, please reset the self.bound. x.max(): {}, x.min(): {}", x.max(), x.min())
            breakpoint()
        # x = torch.clamp(x, min=0., max=1.)
        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        # color
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        if self.encode_appearance:
            a_embedded = self.embedding_a(ts.long()).reshape(ts.shape[0],-1)
            h = torch.cat([d, geo_feat, a_embedded], dim=-1)
        else:
            h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)

        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        # transient logic
        if transient:
            t_embedded = self.embedding_t(ts.long()).reshape(ts.shape[0],-1)
            t = torch.cat([d, geo_feat, t_embedded], dim=-1)
            t = self.transient_color_net(t)

            # sigmoid activation for rgb
            t_sigma = self.transient_sigma(t[...,0:1]) # (B, 1)
            t_rgb = self.transient_rgb(t[...,1:4]) # (B, 3)
            t_beta = self.transient_beta(t[...,4:5]) # (B, 1)
            # t_beta = F.relu(t[...,4:5]) # instead of predict beta in nerfw, we directly predict log(sigma^2) like in kendall17 geo.posenet eq.4
            # t_beta = t[...,4:5]

            transient = torch.cat([t_rgb, t_sigma,
                               t_beta], 1) # (B, 5)

            return torch.cat([rgbs, transient], 1) # (B, 8)
        return rgbs

    def forward(self, x, d, ts=None, sigma_only=False, output_transient=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py
        Benchmark Log: 
            -origianl implementation: 36.3s/it
            -Mixed Precision (pytorch float32 + tcnn float16): 21.30s/it
            Ref: More Mixed Precision for pytorch https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
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

        density_outputs = self.density(x) # [65536, 3]
        sigma = density_outputs['sigma']
        if sigma_only:
            return sigma[...,None]

        geo_feat = density_outputs['geo_feat'] # [65536, 64]

        rgbs = self.color(x, d, ts=ts, geo_feat=geo_feat, transient=output_transient)

        if output_transient==False:
            rgbs = rgbs.float()
            static = torch.cat([rgbs, sigma[...,None]], 1)
            return static
        else:
            static = torch.cat([rgbs[...,:3], sigma[...,None]], 1)
            transient = rgbs[...,3:]
            return torch.cat([static, transient], 1) # (B, 9)

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """

    device = torch.device("cuda")

    # initialize NeRF model
    model = NeRFH_TCNN('coarse', hash_level=args.hash_level, bound=args.bound)
    model = model.to(device)
    grad_vars = list(model.parameters())

    model_fine = None

    if args.N_importance > 0:
        model_fine = NeRFH_TCNN('fine', hash_level=args.hash_level, encode_appearance=True, encode_transient=True,
        in_channels_a=args.in_channels_a, in_channels_t=args.in_channels_t, bound=args.bound)

        model_fine = model_fine.to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, ts, network_fn, typ, output_transient, test_time, store_rgb : \
                        run_NeRFH_TCNN(inputs, viewdirs, ts, network_fn, typ=typ,
                                                                output_transient=output_transient,
                                                                netchunk=args.netchunk,
                                                                test_time=test_time, store_rgb=store_rgb)

    # Create optimizer if NeRF is need to be trained, otherwise returns None
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999), eps=1e-06) # try solution: https://github.com/ashawkey/torch-ngp/issues/76

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
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])

        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'test_time' : False,
        'args' : args,
    }

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

def run_NeRFH_TCNN(inputs, viewdirs, ts, fn, 
                    typ, output_transient, 
                    netchunk=1024*64, test_time=False, store_rgb=False):
    ''' We need a new query function, Coarse = NeRF, Fine = NeRF-W 
    Inputs:
        inputs: torch.Tensor() [N_rays,N_samples,3]
        viewdirs: torch.Tensor() [N_rays, 3]
        ts: latent code from img_idxs [N_rays]
        fn: NeRFW object
        embed_fn: embedder for position
        embeddirs_fn: embedder for view directions
        typ: 'coarse' or 'fine'
        embedding_a: NeRFW appearance embedding layer
        embedding_t: NeRFW transient embedding layer
        output_transient: True/False
        netchunk: chunk size to inference
        test_time: True/False
        store_rgb: True/False. Indicate whether we have to return rgb output. Used only when N_importance==0
    '''
    out_chunks = []
    N_rays, N_samples = inputs.shape[0], inputs.shape[1]
    # print("typ: {}, test_time: {}".format(typ, test_time))

    # embed inputs like NeRF
    if typ == 'coarse' and test_time and (store_rgb==False):

        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        # Feed NeRF-W coarse train
        for i in range(0, inputs_flat.shape[0], netchunk):
            out_chunks += [fn(inputs_flat[i: i+netchunk], None, sigma_only=True)]
        out = torch.cat(out_chunks, 0) # [N_rays*N_samples, 1]
        out = torch.reshape(out, list(inputs.shape[:-1]) + [out.shape[-1]]) # [N_rays, N_samples, 1]
        return out
    if typ == 'coarse': # case: coarse + train
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

        # Feed NeRF-W coarse train
        for i in range(0, inputs_flat.shape[0], netchunk):
            out_chunks += [fn(inputs_flat[i: i+netchunk], input_dirs_flat[i:i+netchunk], output_transient=output_transient)]

        out = torch.cat(out_chunks, 0) # [N_rays*N_samples, 4]
        out = torch.reshape(out, list(inputs.shape[:-1]) + [out.shape[-1]]) # [N_rays, N_samples, 4]
        return out
    
    elif typ == 'fine':
        # if test_time:
        #     breakpoint()
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

        if N_samples==1 and test_time: # probably using sparse volume
            ts = repeat(ts[0:1], 'n1 c -> (n2 n1) c', n2=N_rays) # this is ugly, assuming ts[:] are the same
        else:
            ts = repeat(ts, 'n1 c -> (n1 n2) c', n2=N_samples)

        # Feed NeRF-W fine train
        for i in range(0, inputs_flat.shape[0], netchunk):
            out_chunks += [fn(inputs_flat[i: i+netchunk], input_dirs_flat[i:i+netchunk], ts=ts[i:i+netchunk], output_transient=output_transient)]
        out = torch.cat(out_chunks, 0) # [N_rays*N_samples, 9]
        out = torch.reshape(out, list(inputs.shape[:-1]) + [out.shape[-1]]) # [N_rays, N_samples, 9]
        return out
