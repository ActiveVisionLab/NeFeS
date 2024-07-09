import cv2
import sys

import torch
import torch.nn as nn
import torch.nn.init
import numpy as np

from models.nerfh import img2mse
from models.ray_utils import get_rays
''' FeatureNet models '''
from feature.dfnet import DFNet as FeatureNet

''' APR Models '''
from feature.dfnet import DFNet
from utils.utils import freeze_bn_layer

def disable_model_grad(model):
    ''' set whole model to requires_grad=False, this is for nerf model '''
    print("disable_model_grad...")
    for module in model.modules():
        # print("this is a layer:", module)
        if hasattr(module, 'weight'):
            module.weight.requires_grad_(False)
        if hasattr(module, 'bias'):
            module.bias.requires_grad_(False)
    return model

def svd_reg(pose):
    '''
    Refer to Direct-PN supp. Orthogonalize the Rotation Matrix
    :param: pose [B, 3, 4]
    return pose [B, 3, 4]
    '''
    R_torch = pose[:,:3,:3].clone() # debug
    u,s,v=torch.svd(R_torch)
    Rs = torch.matmul(u, v.transpose(-2,-1))
    pose[:,:3,:3] = Rs
    return pose

def inference_pose_regression(args, data, device, model):
    """
    Inference the Pose Regression Network
    Inputs:
        args: parsed argument
        data: Input image in shape (batchsize, channels, H, W)
        device: gpu device
        model: PoseNet model
    Outputs:
        pose: Predicted Pose in shape (batchsize, 3, 4)
    """
    inputs = data.to(device)
    predict_pose = model(inputs)
    pose = predict_pose.reshape(args.batch_size, 3, 4)

    pose = svd_reg(pose) if args.svd_reg else pose # only needed for models that predict SE(3)
    return pose

def rgb_loss(rgb, target, extras):
    ''' Compute RGB MSE Loss, original from NeRF Paper '''
    # Compute MSE loss between predicted and true RGB.
    img_loss = img2mse(rgb, target)
    loss = img_loss

    # Add MSE loss for coarse-grained model
    if 'rgb0' in extras:
        img_loss0 = img2mse(extras['rgb0'], target)
        loss += img_loss0
    return loss

def PoseLoss(args, pose_, pose, device):
    loss_func = nn.MSELoss()
    predict_pose = pose_.reshape(args.batch_size, 12).to(device) # maynot need reshape
    pose_loss = loss_func(predict_pose, pose)
    return pose_loss

def load_PoseNet(args, device):
    from dm.pose_model import PoseNet_res34, EfficientNetB3, PoseNetV2
    # create model
    if args.resnet34: # for paper experiment table1
        model = PoseNet_res34(droprate=args.dropout)
    elif args.efficientnet:
        model = EfficientNetB3() # vanilla posenet
    else: # default mobilenetv2 backbone
        model = PoseNetV2()

    if args.pretrain_model_path:
        print("load pretrained model from: ", args.pretrain_model_path)
        model.load_state_dict(torch.load(args.pretrain_model_path))

    # Freeze BN to not updating gamma and beta
    if args.freezeBN:
        model = freeze_bn_layer(model)
    model.to(device)
    return model

def load_exisiting_model(args, isFeatureNet=False):
    ''' Load a pretrained PoseNet model '''

    if isFeatureNet==False: # load apr
        model = DFNet()
        model.load_state_dict(torch.load(args.pretrain_model_path))
        return model
    else: # load dfnet for feature extraction
        model=FeatureNet()
        model.load_state_dict(torch.load(args.pretrain_featurenet_path))
        return model

def load_FeatureNet(args, device):
    # load pretrained FeatureNet model
    if args.pretrain_featurenet_path == '':
        print('check FeatureNet model path')
        sys.exit()
    else:
        feat_model = load_exisiting_model(args, isFeatureNet=True)
    feat_model.eval()
    feat_model.to(device)
    return feat_model

def load_MsTransformer(args, device):
    ''' load pretrained MsTransformer Models using the offical code '''
    from mstransformer.transposenet import EMSTransPoseNet
    torch.nn.Module.dump_patches = True
    import json
    # load MSTransformer
    if '7Scenes' in args.dataset_type:
        config_path = '../logs/mstransformer/7scenes_config.json'
    elif 'Cambridge' in args.dataset_type:
        config_path = '../logs/mstransformer/CambridgeLandmarks_config.json'

    with open(config_path, "r") as read_file:
        config = json.load(read_file)
    
    model_params = config['ems-transposenet']
    general_params = config['general']
    config = {**model_params, **general_params}

    backbone_path = '../logs/mstransformer/efficient-net-b0.pth'
    ckpt_path = args.pretrain_model_path
    model = EMSTransPoseNet(config, backbone_path, args=args)
    model.load_state_dict(torch.load(ckpt_path))
    
    model.to(device)

    return model


def load_APR_and_FeatureNet(args, device):
    ''' Load both APR and FeatureNet models '''

    ### pose regression module
    if args.pretrain_model_path == '':
        print('check pretrained model path')
        sys.exit()
    elif args.PoseEstimatorType == 'PoseNet': 
        model = load_PoseNet(args, device)
    elif args.PoseEstimatorType == 'MsTransformer':
        model = load_MsTransformer(args, device)
    elif args.PoseEstimatorType == 'DFNet':
        model = load_exisiting_model(args) # load pretrained DFNet model
        if args.freezeBN:
            model = freeze_bn_layer(model)
        model.to(device)
    elif args.PoseEstimatorType == 'NetVlad':
        print('loading NetVlad')
        model = None
    else:
        NotImplementedError

    ### feature extraction module
    feat_model = load_FeatureNet(args, device)

    return model, feat_model

def prepare_batch_render(args, pose, batch_size, target_, H, W, focal, half_res=True, rand=True):
    ''' Break batch of images into rays '''
    target_ = target_.permute(0, 2, 3, 1).numpy()#.squeeze(0) # convert to numpy image
    if half_res:
        N_rand = batch_size * (H//2) * (W//2)
        target_half = np.stack([cv2.resize(target_[i], (H//2, W//2), interpolation=cv2.INTER_AREA) for i in range(batch_size)], 0)
        target_half = torch.Tensor(target_half)
        
        rays = torch.stack([torch.stack(get_rays(H//2, W//2, focal/2, pose[i]), 0) for i in range(batch_size)], 0) # [N, ro+rd, H, W, 3] (130, 2, 100, 100, 3)
        rays_rgb = torch.cat((rays, target_half[:, None, ...]), 1)

    else:
        # N_rand = batch_size * H * W
        N_rand = args.N_rand
        target_ = torch.Tensor(target_)
        rays = torch.stack([torch.stack(get_rays(H, W, focal, pose[i]), 0) for i in range(batch_size)], 0) # [N, ro+rd, H, W, 3] (130, 2, 200, 200, 3)
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = torch.cat([rays, target_[:, None, ...]], 1)

    # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.permute(0, 2, 3, 1, 4)
    
    # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = torch.reshape(rays_rgb, (-1, 3, 3))

    if 1:
        #print('shuffle rays')
        rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]

    # Random over all images
    batch = rays_rgb[:N_rand].permute(1, 0 , 2) # [B, 2+1, 3*?] # (4096, 3, 3)
    batch_rays, target_s = batch[:2], batch[2] # [2, 4096, 3], [4096, 3]

    return batch_rays, target_s

def fix_coord_supp(args, pose, world_setup_dict, device=None):
    # this function needs to be fixed because it is taking args.pose_scale
    '''supplementary fix_coord() for direct matching
    Inputs:
        args: parsed argument
        pose: pose [N, 3, 4]
        device: cpu or gpu
    Outputs:
        pose: converted Pose in shape [N, 3, 4]
    '''
    if not torch.is_tensor(pose):
        pose = torch.Tensor(pose).to(device)
    sc=world_setup_dict['pose_scale'] # manual tuned factor, align with colmap scale
    if device is None:
        move_all_cam_vec = torch.Tensor(world_setup_dict['move_all_cam_vec'])
    else:
        move_all_cam_vec = torch.Tensor(world_setup_dict['move_all_cam_vec']).to(device)
    sc2 = world_setup_dict['pose_scale2']
    pose[:,:3,3] *= sc
    # move center of camera pose
    pose[:, :3, 3] += move_all_cam_vec
    pose[:,:3,3] *= sc2
    return pose

