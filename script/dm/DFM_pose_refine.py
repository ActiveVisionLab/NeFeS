import time
from copy import deepcopy
import os
import os.path as osp
import torch
import torch.nn.init
import numpy as np

from dm.direct_pose_model import fix_coord_supp
from models.nerfh import img2mse
from models.rendering import render
from models.poses import LearnPose
from utils.utils import tensor_hwc2nchw, set_default_to_cuda
from torchvision.utils import save_image
from utils.utils import save_image_saliancy
from dm.pose_model import compute_pose_error
import torch.optim as optim

def tmp_plot2(target_in, rgb_in, features_target, features_rgb, i=0):
    '''
    print 1 pair of batch of salient feature map
    :param: target_in [B, 3, H, W]
    :param: rgb_in [B, 3, H, W]
    :param: features_target [B, C, H, W]
    :param: features_rgb [B, C, H, W]
    :param: frame index i of batch
    '''
    print("for debug only...")

    if target_in != None:
        save_image(target_in[i], './tmp/target_in.png')
    if rgb_in != None:
        save_image(rgb_in[i], './tmp/rgb_in.png')
    features_t = features_target[i].clone()[:, None, :, :]
    features_r = features_rgb[i].clone()[:, None, :, :]
    save_image_saliancy(features_t, './tmp/target', True)
    save_image_saliancy(features_r, './tmp/rgb', True)

def plot_features_batch(features_target, features_rgb, fn1, fn2, i=0):
    '''
    print 1 pair of batch of salient feature map
    :param: features_target [B, C, H, W]
    :param: features_rgb [B, C, H, W]
    :param: fn1 filename of target feature map
    :param: fn2 filename of rgb feature map
    :param: frame index i of batch
    '''
    # print("for debug only...")
    features_t = features_target[i].clone()[:, None, :, :]
    features_r = features_rgb[i].clone()[:, None, :, :]
    save_image_saliancy(features_t, fn1, True)
    save_image_saliancy(features_r, fn2, True)

def plot_features(features_target, features_rgb, fn1, fn2, i=0):
    '''
    print 1 pair of 1 sample of salient feature map
    :param: features_target [B, C, H, W]
    :param: features_rgb [B, C, H, W]
    :param: fn1 filename of target feature map
    :param: fn2 filename of rgb feature map
    :param: frame index i of batch
    '''
    # print("for debug only...")
    features_t = features_target[i].clone()[:, None, :, :]
    features_r = features_rgb[i].clone()[:, None, :, :]
    save_image_saliancy(features_t[0], fn1, True)
    save_image_saliancy(features_r[0], fn2, True)

def load_NeRF_model(args, near, far):
    '''
    load pretrained NeRF-Hist model.
    :param: args, near, far
    return render_kwargs_test
    '''
    # load NeRF model
    from models.nerfh_nff import create_nerf

    _, render_kwargs_test, start, _, _ = create_nerf(args)

    global_step = start
    if args.reduce_embedding==2:
        render_kwargs_test['i_epoch'] = global_step
    
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_test.update(bds_dict)

    if (args.tcnn == False or args.NeRFW == False) and args.nerfh_nff == False and args.nerfh_nff2 == False:
        render_kwargs_test['embedding_a'] = disable_model_grad(render_kwargs_test['embedding_a'])
        render_kwargs_test['embedding_t'] = disable_model_grad(render_kwargs_test['embedding_t'])
    render_kwargs_test['network_fn'] = disable_model_grad(render_kwargs_test['network_fn'])
    render_kwargs_test['network_fine'] = disable_model_grad(render_kwargs_test['network_fine'])
    return render_kwargs_test

def preprocess_features_for_loss(feature):
    '''
    transform output features from the network to required shape for computing loss
    :param: feature [L, B, C, H, W] # L stands for level of features (we currently use 3)
    return feature' [B,L*C,H,W]
    '''
    feature = feature.permute(1,0,2,3,4)
    B, L, C, H, W = feature.size()
    feature = feature.reshape((B,L*C,H,W))
    return feature

def disable_model_grad(model):
    ''' set whole model to requires_grad=False, this is for nerf model '''
    # print("disable_model_grad...")
    for module in model.modules():
        # print("this is a layer:", module)
        if hasattr(module, 'weight'):
            module.weight.requires_grad_(False)
        if hasattr(module, 'bias'):
            module.bias.requires_grad_(False)
    return model

def svd_reg(pose):
    '''
    Refer to Direct-PN supplementary. Section: Orthogonalize the Rotation Matrix
    :param: pose [B, 3, 4]
    return pose [B, 3, 4]
    '''
    R_torch = pose[:,:3,:3].clone() # debug
    u,s,v=torch.svd(R_torch)
    Rs = torch.matmul(u, v.transpose(-2,-1))
    pose[:,:3,:3] = Rs
    return pose

@torch.cuda.amp.autocast(dtype=torch.float32)
def inference_pose_regression(args, data, device, model, retFeature=False, isSingleStream=True, return_pose=True, H=None, W=None):
    """
    Inference the Pose Regression Network only.
    Inputs:
        args: parsed argument
        data: Input image in shape (batchsize, channels, H, W)
        device: gpu device
        model: PoseNet model
    Outputs:
        pose: Predicted Pose in shape (batchsize, 3, 4)
    """
    inputs = data.to(device)
    if args.PoseEstimatorType=='DFNet':
        features, predict_pose = model(inputs, return_feature=retFeature, isSingleStream=isSingleStream, return_pose=return_pose, upsampleH=H, upsampleW=W)
        pose = predict_pose.reshape(inputs.shape[0], 3, 4)
        pose = svd_reg(pose) if args.svd_reg else pose # only needed for models that predict SE(3)
    elif args.PoseEstimatorType=='PoseNet':
        predict_pose = model(inputs) # predict_pose: [1, 12]
        pose = predict_pose.reshape(inputs.shape[0], 3, 4)
        pose = svd_reg(pose) if args.svd_reg else pose # only needed for models that predict SE(3)
    elif args.PoseEstimatorType=='MapNet':
        predict_pose = model(inputs) # predict_pose: [1, 12]
        pose = predict_pose.reshape(inputs.shape[0], 3, 4)
    elif args.PoseEstimatorType=='MsTransformer':
        pose = model(inputs) # predict_pose: [1, 3, 4]
    else:
        features, predict_pose = model(inputs, isTrain=retFeature, isSingleStream=isSingleStream) # features: (1, [1, 1, 320, 8, 14]), predict_pose: [1, 12]
        pose = predict_pose.reshape(inputs.shape[0], 3, 4)
        pose = svd_reg(pose) if args.svd_reg else pose # only needed for models that predict SE(3)
    return pose

def inference_pose_feature_extraction(args, data, device, model, retFeature=False, isSingleStream=True, return_pose=True, H=None, W=None):
    """
    Inference the Pose Regression Network and Feature Extraction Network.
    Inputs:
        args: parsed argument
        data: Input image in shape (batchsize, channels, H, W)
        device: gpu device
        model: PoseNet model
    Outputs:
        pose: Predicted Pose in shape (batchsize, 3, 4)
    """
    inputs = data.to(device)
    if H ==None and W ==None:
        _,_,H,W = data.size()
    assert (H!=None)
    assert (W!=None)

    if args.dfnet:
        features, predict_pose = model(inputs, return_feature=retFeature, isSingleStream=isSingleStream, return_pose=return_pose, upsampleH=H, upsampleW=W) # features: , predict_pose: [1, 12]
    else:
        features, predict_pose = model(inputs, isTrain=retFeature, isSingleStream=isSingleStream) # features: (1, [1, 1, 320, 8, 14]), predict_pose: [1, 12]
    
    if return_pose==False:
        return features, None

    pose = predict_pose.reshape(inputs.shape[0], 3, 4)
    pose = svd_reg(pose) if args.svd_reg else pose # only needed for models that predict SE(3)
    return features, pose

def rgb_loss(rgb, target, extras):
    ''' Compute RGB MSE Loss'''
    # Compute MSE loss between predicted and true RGB.
    img_loss = img2mse(rgb, target)
    loss = img_loss
    return loss

def normalize_features(tensor, value_range=None, scale_each: bool = False):
    ''' Find unit norm of channel wise feature 
        :param: tensor, img tensor (C,H,W)
    '''
    tensor = tensor.clone()  # avoid modifying tensor in-place
    C,H,W = tensor.size()

    # normlaize the features with l2 norm
    tensor = tensor.reshape(C, H*W)
    tensor = torch.nn.functional.normalize(tensor)
    return tensor

def feature_loss(feature_rgb, feature_target, img_in=True, per_pixel=False):
    ''' Compute Feature MSE Loss 
    :param: feature_rgb, [C,H,W] or [C, N_rand]
    :param: feature_target, [C,H,W] or [C, N_rand]
    :param: img_in, True: input is feature maps, False: input is rays
    :param: random, True: randomly using per pixel or per channel cossimilarity loss
    '''
    if img_in:
        C,H,W = feature_rgb.size()
        fr = feature_rgb.reshape(C, H*W)
        ft = feature_target.reshape(C, H*W)
    else:
        fr = feature_rgb
        ft = feature_target

    # cosine loss
    if per_pixel:
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6) # H*W normlized pixels
    else:
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    # minimize average cosine similarity
    loss = 1 - cos(fr, ft).mean()
    return loss

class FeatureLoss(torch.nn.Module):
    def __init__(self, img_in=True, per_pixel=False):
        super().__init__()
        self.img_in = img_in
        self.per_pixel = per_pixel
        if self.per_pixel:
            self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6) # H*W normlized pixels
        else:
            self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, feature_rgb, feature_target):
        if self.img_in:
            C,H,W = feature_rgb.size()
            fr = feature_rgb.reshape(C, H*W)
            ft = feature_target.reshape(C, H*W)
        else:
            fr = feature_rgb
            ft = feature_target
        loss = 1 - self.cos(fr, ft).mean()
        return loss

def masked_feature_loss(feature_rgb, feature_target, mask, img_in=True, per_pixel=False):
    ''' Compute Feature MSE Loss with semantic mask
    :param: feature_rgb, [C,H,W] or [C, N_rand]
    :param: feature_target, [C,H,W] or [C, N_rand]
    :param: mask, [1,H,W]
    :param: img_in, True: input is feature maps, False: input is rays
    :param: random, True: randomly using per pixel or per channel cossimilarity loss
    '''

    _mask = mask.reshape(-1)
    valid_inds = torch.nonzero(_mask>0, as_tuple=True)[0]

    if img_in:
        C,H,W = feature_rgb.size()
        fr = feature_rgb.reshape(C, H*W)
        ft = feature_target.reshape(C, H*W)
    else:
        fr = feature_rgb
        ft = feature_target

    # apply mask
    fr = fr[:, valid_inds]
    ft = ft[:, valid_inds]

    # cosine loss
    if per_pixel:
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6) # H*W normlized pixels
    else:
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = 1 - cos(fr, ft).mean()

    return loss

def DFM_optimization_NFF(args, batch_idx, data, pose_param_net, hist, hwf, optimizer, device, world_setup_dict, ft_pre, render_kwargs_test, mask):
    '''
    core optimization for DFM post-processing 
    :param: args,
    :param: batch_idx, query image idx
    :param: data, torch.Tensor, [N,C,H,W]
    :param: feat_model, feature extractor
    :param: pose_param_net,
    :param: hist, torch.Tensor, [1, num_bins]
    :param: hwf
    :param: optimizer
    :param: device
    :param: world_setup_dict
    :param: ft_pre, previously computed feature_target
    :param: **render_kwargs_test
    return iter_loss, iter_psnr, pose_param_net
    '''

    batch_size = data.shape[0]
    H, W, focal = hwf
    pose_param_net.train()

    pose = pose_param_net(cam_id=batch_idx)[None, :3,:4]
    pose_nerf = fix_coord_supp(args, pose, world_setup_dict, device=device)

    hist = hist.to(device)

    # every new tensor from onward is in GPU, here memory cost is a bottleneck
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float32)

    rgb, _, _, extras = render(int(H//args.tinyscale), int(W//args.tinyscale), focal/args.tinyscale, chunk=args.chunk, c2w=pose_nerf[0,:3,:4], img_idx=hist, **render_kwargs_test)

    if args.nerfh_nff: # fusion part
        if args.encode_hist:
            affine_color_transform = render_kwargs_test['network_fn'].affine_color_transform
            rgb = affine_color_transform(args, rgb, hist, batch_size)
        # NeRF feature + RGB -> CNN Fusion -> Feature
        Fusion_Net = render_kwargs_test['network_fn'].run_fusion_net
        render_rgb, render_feature, feature_rgb = Fusion_Net(rgb, extras['feat_map'], int(H//args.tinyscale), int(W//args.tinyscale), batch_size) # (1,3,120,213), (1,16,120,213), (1,16,120,213)
    else:
        # convert feature to B,C,H,W
        feat_map = extras['feat_map'].reshape((int(H//args.tinyscale), int(W//args.tinyscale), -1))
        feature_rgb = tensor_hwc2nchw(feat_map)

    feature_target = deepcopy(ft_pre) # [1, 16, 120, 213]

    ### Loss Design Here ###
    loss = feature_loss(feature_rgb[0], feature_target[0], per_pixel=False) # False for DFNet paper model

    ### Loss Design End
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # end of every new tensor from onward is in GPU
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float32)
    return pose_param_net

def DFM_post_processing2(args, predict_poses, feat_model, render_kwargs_test, hwf, device, test_dl=None, world_setup_dict=None):
    '''
    Use Direct Feature Matching as a Post-processing. We optimize poses from unknown pose estimator in this loop.
    Here I didn't use the fail safe verification step, but it is possible to do it.

    predict_poses: [N, 3, 4]
    feat_model: feature extractor
    render_kwargs_test: kwargs for the nefes model
    hwf: [H, W, focal]
    device: gpu device
    test_dl: test dataloader
    world_setup_dict: world setup dict
    '''

    # set nerf model requires_grad to False to accelerate the refinement?
    render_kwargs_test['network_fn'].requires_grad_(False)
    render_kwargs_test['network_fine'].requires_grad_(False)
    feat_model.eval().requires_grad_(False)

    # initialize pose in 4x4 using predicted pose like nerf--
    predict_poses = torch.Tensor(predict_poses)
    # convert predicted pose to 4x4
    last_row = torch.tile(torch.Tensor([0, 0, 0, 1]), (len(test_dl.dataset), 1, 1))  # (N_images, 1, 4)
    predict_poses = torch.cat([predict_poses, last_row], 1)
    pose_param_net = LearnPose(len(test_dl.dataset), True, True, predict_poses, lietorch=True).to(device)
    
    init_iter = args.opt_iter
    print("perform pose refinement for {} iters...".format(init_iter))

    # profiling
    for batch_idx, batch_data in enumerate(test_dl):
        data = batch_data['img']
        hist = batch_data['hist']
        if args.semantic:
            mask = batch_data['mask']
        else:
            mask = None

        if batch_idx % 10 == 0:
            print("renders {}/total {}".format(batch_idx, len(test_dl.dataset)))

        # # set optimizer for post processing model (BAD Implementation)
        list_params = []
        for name,param in pose_param_net.named_parameters():
            if name == 'r':
                list_params.append({'params': param, 'lr': args.lr_r}) # shop
            elif name == 't':
                list_params.append({'params': param, 'lr': args.lr_t}) # shop
        optimizer = optim.Adam(list_params)

        # extract feature for the query image
        with torch.no_grad():
            # DFM + NeFeS Features
            feature_target, _ = inference_pose_feature_extraction(args, data, device, feat_model, retFeature=True, isSingleStream=True, return_pose=False, H=int(hwf[0]//args.tinyscale), W=int(hwf[1]//args.tinyscale)) # (120, 213)
            feature_target = feature_target[0] # [3, 1, 128, 240, 427]

        # DFM + NeFeS Features
        # preprocessing target feature here
        indices = torch.tensor(args.feature_matching_lvl).to(device)
        feature_target = torch.index_select(feature_target, 0, indices) # torch.Size([1, 1, 16, 240, 417])
        feature_target = preprocess_features_for_loss(feature_target) # torch.Size([1, 16, 240, 417])

        for i in range(init_iter):
            pose_param_net = DFM_optimization_NFF(args, batch_idx, data.to(device), pose_param_net, hist.to(device), hwf, optimizer, device, world_setup_dict, feature_target, render_kwargs_test, mask)

    SAVE_DFM_RESULT = True
    if SAVE_DFM_RESULT:
        pose_results = []

    ## evaluate the post-processing results
    results_pp = np.zeros((len(test_dl.dataset), 2))
    for i in range(len(test_dl.dataset)):
        ### inference estimated pose_param_net
        pose = test_dl.dataset.poses[i:i+1].reshape(1,3,4) # label

        # evaluation
        pose_param_net.eval()
        with torch.no_grad():
            predict_pose_DFM = pose_param_net(cam_id=i)[None,:3,:4].cpu().detach().numpy() # predicted pose after DFM post-processing

        # compute pose error between the ground truth and the network predicted pose
        error_x, theta = compute_pose_error(torch.Tensor(pose), torch.Tensor(predict_pose_DFM))
        results_pp[i,:] = [error_x, theta]
        if SAVE_DFM_RESULT:
            pose_results.append(predict_pose_DFM)

    median_result = np.median(results_pp,axis=0)
    mean_result = np.mean(results_pp,axis=0)

    # standard log
    print ('Median error {}m and {} degrees.'.format(median_result[0], median_result[1]))
    print ('Mean error {}m and {} degrees.'.format(mean_result[0], mean_result[1]))

    if SAVE_DFM_RESULT:
        print('saving pose results...')
        pose_results = np.concatenate(pose_results).reshape(-1,12)
        scene = osp.split(args.datadir)[-1]
        save_folder = f'tmp/{args.PoseEstimatorType}_NeFeS{init_iter:01d}_{args.dataset_type}/{scene}/'

        if osp.exists(save_folder) is False:
            os.makedirs(save_folder)
        np.savetxt(os.path.join(f'{save_folder}/{args.PoseEstimatorType}_{scene}_NeFeS{init_iter:01d}_pose_pose_results.txt'), pose_results)

    return pose_param_net
