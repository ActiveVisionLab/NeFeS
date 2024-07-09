"""
The Multi-Scene TransPoseNet model
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch3d.transforms as transforms3d
from torch import nn
from mstransformer.transformer import Transformer
from mstransformer.pencoder import nested_tensor_from_tensor_list
from mstransformer.backbone import build_backbone
import os.path as osp
import numpy as np
from dataset_loaders.load_Cambridge import rot_phi

class PoseRegressor(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, decoder_dim, output_dim, use_prior=False):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the outpur dimension
        use_prior: (bool) whether to use prior information
        """
        super().__init__()
        ch = 1024
        self.fc_h = nn.Linear(decoder_dim, ch)
        self.use_prior = use_prior
        if self.use_prior:
            self.fc_h_prior = nn.Linear(decoder_dim * 2, ch)
        self.fc_o = nn.Linear(ch, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        """
        if self.use_prior:
            x = F.gelu(self.fc_h_prior(x))
        else:
            x = F.gelu(self.fc_h(x))

        return self.fc_o(x)


class MSTransPoseNet(nn.Module):

    def __init__(self, config, pretrained_path):
        """ Initializes the model.
        default config values from the public repo:
        {'batch_size': 8, 'equalize_scenes': False, 'num_scenes': 7, 'learnable': True, 's_x': 0.0, 's_q': -3.0, 'lr': 0.0001, 'norm': 2, 
        'weight_decay': 0.0001, 'eps': 1e-10, 'lr_scheduler_gamma': 0.1, 'lr_scheduler_step_size': 10, 'n_epochs': 30, 
        'num_t_encoder_layers': 6, 'num_t_decoder_layers': 6, 'num_rot_encoder_layers': 6, 'num_rot_decoder_layers': 6, 
        'dim_feedforward': 256, 'hidden_dim': 256, 'dropout': 0.1, 'nheads': 4, 'reduction': ['reduction_4', 'reduction_3'], 
        'freeze': False, 'freeze_exclude_phrase': 'regressor_head_rot', 'refine_orientation': 'affine', 'single_scene': False, 
        'shallow_mlp': False, 'apply_positional_encoding': True, 'n_freq_print': 5, 'n_freq_checkpoint': 10, 'n_workers': 4, 'device_id': 'cuda:0'}
        """
        super().__init__()

        config["backbone"] = pretrained_path
        config["learn_embedding_with_pose_token"] = False
        num_scenes = config.get("num_scenes")
        self.backbone = build_backbone(config)

        config_t = {**config}
        config_t["num_encoder_layers"] = config["num_t_encoder_layers"] # 6
        config_t["num_decoder_layers"] = config["num_t_decoder_layers"] # 6
        config_rot = {**config}
        config_rot["num_encoder_layers"] = config["num_rot_encoder_layers"] # 6
        config_rot["num_decoder_layers"] = config["num_rot_decoder_layers"] # 6
        self.transformer_t = Transformer(config_t)
        self.transformer_rot = Transformer(config_rot)

        decoder_dim = self.transformer_t.d_model

        self.input_proj_t = nn.Conv2d(self.backbone.num_channels[0], decoder_dim, kernel_size=1)
        self.input_proj_rot = nn.Conv2d(self.backbone.num_channels[1], decoder_dim, kernel_size=1)

        self.query_embed_t = nn.Embedding(num_scenes, decoder_dim)
        self.query_embed_rot = nn.Embedding(num_scenes, decoder_dim)

        self.log_softmax = nn.LogSoftmax(dim=1)

        self.scene_embed = nn.Linear(decoder_dim*2, 1)
        self.regressor_head_t = nn.Sequential(*[PoseRegressor(decoder_dim, 3) for _ in range(num_scenes)])
        self.regressor_head_rot = nn.Sequential(*[PoseRegressor(decoder_dim, 4) for _ in range(num_scenes)])
        
        self.img_transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
        ])

    def forward_transformers(self, data):
        """
        Forward of the Transformers
        The forward pass expects a dictionary with key-value 'img' -- NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED
        return a dictionary with the following keys--values:
            global_desc_t: latent representation from the position encoder
            global_dec_rot: latent representation from the orientation encoder
            scene_log_distr: the log softmax over the scenes
            max_indices: the index of the max value in the scene distribution
        """

        samples = data.clone() # data.get('img')
        scene_indices = None
        batch_size = samples.shape[0]

        # check dimension of the input image
        if len(samples.shape) == 3:
            samples = samples.unsqueeze(0)
        elif len(samples.shape) == 4:
            pass

        imgs = []
        # normalize the images
        for img in samples:
            imgs.append(self.img_transform(img))
        samples = torch.stack(imgs, dim=0).to(device=data.device)

        # Handle data structures
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # Extract the features and the position embedding from the visual backbone
        features, pos = self.backbone(samples)

        src_t, mask_t = features[0].decompose() # [1, 112, 14, 14], mask_t [1,14,14] of False...? What is mask for?
        src_rot, mask_rot = features[1].decompose() # [1, 40, 28, 28], mask_rot [1,28,28] of False...? What is mask for?

        # Run through the transformer to translate to "camera-pose" language
        assert mask_t is not None
        assert mask_rot is not None
        local_descs_t = self.transformer_t(self.input_proj_t(src_t), mask_t, self.query_embed_t.weight, pos[0])[0][0]
        local_descs_rot = self.transformer_rot(self.input_proj_rot(src_rot), mask_rot, self.query_embed_rot.weight, pos[1])[0][0]

        # Get the scene index with FC + log-softmax
        scene_log_distr = self.log_softmax(self.scene_embed(torch.cat((local_descs_t, local_descs_rot), dim=2))).squeeze(2)
        _, max_indices = scene_log_distr.max(dim=1)
        if scene_indices is not None:
            max_indices = scene_indices
        # Take the global latents by zeroing other scene's predictions and summing up
        w = local_descs_t*0
        w[range(batch_size),max_indices, :] = 1
        global_desc_t = torch.sum(w * local_descs_t, dim=1)
        global_desc_rot = torch.sum(w * local_descs_rot, dim=1)

        return {'global_desc_t':global_desc_t,
                'global_desc_rot':global_desc_rot,
                'scene_log_distr':scene_log_distr,
                'max_indices':max_indices}


    def forward_heads(self, transformers_res):
        """
        Forward pass of the MLP heads
        The forward pass execpts a dictionary with two keys-values:
        global_desc_t: latent representation from the position encoder
        global_dec_rot: latent representation from the orientation encoder
        scene_log_distr: the log softmax over the scenes
        max_indices: the index of the max value in the scene distribution
        returns: dictionary with key-value 'pose'--expected pose (NX7) and scene_log_distr
        """
        global_desc_t = transformers_res.get('global_desc_t')
        global_desc_rot = transformers_res.get('global_desc_rot')
        max_indices = transformers_res.get('max_indices') # We can only use the max index for weights selection
        batch_size = global_desc_t.shape[0]
        expected_pose = torch.zeros((batch_size,7)).to(global_desc_t.device).to(global_desc_t.dtype)
        for i in range(batch_size):
            x_t = self.regressor_head_t[max_indices[i]](global_desc_t[i].unsqueeze(0))
            x_rot = self.regressor_head_rot[max_indices[i]](global_desc_rot[i].unsqueeze(0))
            expected_pose[i, :] = torch.cat((x_t, x_rot), dim=1)
        return {'pose':expected_pose, 'scene_log_distr':transformers_res.get('scene_log_distr')}

    def forward(self, data):
        """ The forward pass expects a dictionary with the following keys-values
         'img' -- NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED
         'scene_indices': ground truth scene indices for each image (can be None)

        returns a dictionary with the following keys-values;
        'pose': expected pose (NX7)
        'log_scene_distr': (log) probability distribution over scenes
        """
        transformers_res = self.forward_transformers(data)
        # Regress the pose from the image descriptors

        heads_res = self.forward_heads(transformers_res)

        return heads_res


class EMSTransPoseNet(MSTransPoseNet):

    def __init__(self, config, pretrained_path, args):
        """ Initializes the model.
        """
        super().__init__(config, pretrained_path)

        # scene coord. transformation for cambridge, we transform the network output coordinates to OpenGL coordinates
        self.scene = args.dataset_type

        pose_avg_stats_file = osp.join(args.datadir, 'pose_avg_stats.txt')
        self.pose_avg_stats = np.loadtxt(pose_avg_stats_file)

        # Ms-Transformer Initialization
        decoder_dim = self.transformer_t.d_model # 256
        self.regressor_head_t = PoseRegressor(decoder_dim, 3)
        self.regressor_head_rot = PoseRegressor(decoder_dim, 4)
    
    def fix_coord_Cambridge2OpenGL(self, pose, pose_avg_stats, device=None):
        '''
        # this is to transform the network output coordinates to OpenGL coordinates. 
        # Since Ms-Transformer official model is trained in Cambrdige Coordinates
        Inputs:
            pose: pose [N, 3, 4]
            pose_avg_stats: pre-computed pose average stats
            device: cpu or gpu
        Outputs:
            out: converted Pose in shape [N, 3, 4]
        '''

        # assert(args.PoseEstimatorType == 'MsTransformer' and args.dataset_type == 'Cambridge')
        pose_avg_homo = torch.eye(4)
        pose_avg_homo[:3] = torch.Tensor(pose_avg_stats)
        last_row = torch.tile(torch.Tensor([0, 0, 0, 1]), (len(pose), 1, 1)).to(device)
        poses_homo = torch.cat([pose, last_row], 1)
        poses_centered = torch.linalg.inv(pose_avg_homo.to(device)) @ poses_homo

        # rotate tpose 90 degrees at x axis # only corrected translation position
        poses = torch.Tensor(rot_phi(180/180.*np.pi)).to(device) @ poses_centered

        # correct view direction except mirror with gt view
        poses[:,:3,:3] = -poses[:,:3,:3]

        # camera direction mirror at x axis mod1 R' = R @ mirror matrix 
        # ref: https://gamedev.stackexchange.com/questions/149062/how-to-mirror-reflect-flip-a-4d-transformation-matrix
        poses[:,:3,:3] = poses[:,:3,:3] @ torch.Tensor([[-1,0,0],[0,1,0],[0,0,1]]).to(device)
        out = poses[:, :3]
        
        return out

    def forward_heads(self, transformers_res):
        """
        Forward pass of the MLP heads
        The forward pass execpts a dictionary with two keys-values:
        global_desc_t: latent representation from the position encoder
        global_dec_rot: latent representation from the orientation encoder
        scene_log_distr: the log softmax over the scenes
        max_indices: the index of the max value in the scene distribution
        returns: dictionary with key-value 'pose'--expected pose (NX7) and scene_log_distr
        """
        global_desc_t = transformers_res.get('global_desc_t')
        global_desc_rot = transformers_res.get('global_desc_rot')

        x_t = self.regressor_head_t(global_desc_t)
        x_rot = self.regressor_head_rot(global_desc_rot)
        expected_pose = torch.cat((x_t, x_rot), dim=1)

        R = transforms3d.quaternion_to_matrix(expected_pose[:, 3:])
        output = torch.zeros((expected_pose.shape[0], 3, 4), device=expected_pose.device)
        output[:, :3, :3] = R
        output[:, :, 3] = expected_pose[:, :3]

        if self.scene == 'Cambridge':
            # for some reason, R needs to take a transpose. 
            # This might be due to our dataset processing method from DSAC or MapNet
            output[:, :3, :3] = output[:, :3, :3].transpose(1, 2)
            output = self.fix_coord_Cambridge2OpenGL(output, self.pose_avg_stats, device=output.device)

        return output