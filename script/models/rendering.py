import os
import torch
import torch.nn.functional as F
import numpy as np
import imageio
import time
from tqdm import tqdm
from models.ray_utils import get_rays, ndc_rays
from models.nerfh_nff import to8b, raw2outputs_NeRFH_NFF

from utils.utils import tensor_hwc2nchw,tensor_nchw2hwc
from utils.utils import save_image_saliancy, set_default_to_cuda
''' 
Render Procedure:
render()->batchify_rays()->render_rays()->raw2outputs()->sample_pdf()
'''

PROFILE_TIME=False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):

    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF (ex. y-axis cdf, x-axis pdf. Finding pdf indexes from CDF sampling)
    u = u.contiguous()
    inds = torch.searchsorted(cdf.detach(), u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom

    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                i_epoch=-1,
                embedding_a=None,
                embedding_t=None,
                test_time=False,
                args=None,
                volume=None):

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,8:11] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    img_idxs = ray_batch[...,11:] # same as ts from nerf_pl-nerfw code [N_rays]
    
    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals) # sample in depth
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals)) # sample in diparity disparity = 1/depth

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1]) # find mid points of each interval
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape) # randomly choose in intervals

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3] # Sample 3D point print it out, [batchsize, 64 samples, 3 axis]

    # If we only use coarse network, we have to return rgb output here. This is ugly, need to optimize code here
    store_rgb=False
    if N_importance == 0: # only coarse network is used
        store_rgb=True  # store rgb output

    ### inference coarse MLPs ###
    raw = network_query_fn(pts, viewdirs, None, network_fn, 'coarse', False, test_time=test_time, store_rgb=store_rgb)

    ### produce volume rendering outputs ###
    rgb_map, feat_map, disp_map, acc_map, weights, depth_map, _, _ = raw2outputs_NeRFH_NFF(raw, z_vals, raw_noise_std=raw_noise_std, white_bkgd=white_bkgd, test_time=test_time, typ="coarse", store_rgb=store_rgb)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        if args.nerfh_nff:
            feat_map_0 = feat_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)

        z_samples = z_samples.detach()

        if args.use_fine_only: # mod 0, use only fine sample
            z_vals = z_samples
        else:
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
        ### Inference by neural network
        output_transient=False
        if args.NeRFW:
            output_transient=True

        ### inference fine NeFeS MLPs ###
        raw = network_query_fn(pts, viewdirs, img_idxs, network_fine, 'fine', output_transient, test_time=test_time, store_rgb=store_rgb)
        ### End of inference by neural network

        ### produce volume rendering outputs ###
        rgb_map, feat_map, disp_map, acc_map, weights, depth_map, transient_sigmas, beta = raw2outputs_NeRFH_NFF(raw, z_vals, raw_noise_std=raw_noise_std, output_transient=output_transient, \
                                                                                    beta_min=network_fine.beta_min, white_bkgd=white_bkgd, test_time=test_time, typ="fine", transient_at_test=args.transient_at_test)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if args.nerfh_nff:
        ret['feat_map'] = feat_map

    if (N_importance > 0 and test_time) or (N_importance == 0):
        pass
    elif N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        if args.NeRFW: # only when using nerfh # and args.nerfh_nff2==False
            ret['transient_sigmas'] = transient_sigmas
            ret['beta'] = beta
        if args.nerfh_nff:
            if feat_map_0 != None:
                ret['feat0'] = feat_map_0

    ### need to uncomment later ###
    # for k in ret:
    #     if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
    #         print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)

        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None, img_idx=torch.Tensor(0),
                  **kwargs):

    # profiling point 0
    time0 = time.time()
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1) # [1024, 11]

    # for NeRFH, we need to add frame index as input
    if img_idx.shape[0] != rays.shape[0]:
        img_idx = img_idx.repeat(rays.shape[0],1) # [1024, 1]
    rays = torch.cat([rays, img_idx], 1) # [1024, 12]


    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

### This is for validation ###
def render_path(args, render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, single_gt_img=False, img_ids=torch.Tensor(0)):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = int(H//render_factor)
        W = int(W//render_factor)
        focal = focal/render_factor

    rgbs = []
    disps = []
    psnr = []

    if PROFILE_TIME:
        t = time.time()
        for i, c2w in enumerate(render_poses):
            breakpoint()
            # TODO: double check rendering size later
            rgb, disp, acc, extras = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], img_idx=img_ids[i], **render_kwargs)

        print("time: ", (time.time()-t))
        return None, None

    for i, c2w in enumerate(tqdm(render_poses)):

        with torch.cuda.amp.autocast(enabled=(args.ffmlp or args.tcnn)):
            rgb, disp, acc, extras = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], img_idx=img_ids[i], **render_kwargs)

        if args.encode_hist and (args.sh_nff or args.nerfh_nff or args.nerfh_nff2):
            affine_color_transform = render_kwargs['network_fn'].affine_color_transform
            rgb = affine_color_transform(args, rgb, img_ids[i:i+1], 1)

        rgbs.append(rgb.reshape((H,W,3)).cpu().numpy())
        disps.append(disp.reshape((H,W)).cpu().numpy())

        if i==0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None:

            if single_gt_img:
                if torch.is_tensor(gt_imgs): # convert to numpy if gt_imgs is a torch tensor
                    gt = gt_imgs.cpu().numpy()
                else:
                    gt = gt_imgs
            else:
                if torch.is_tensor(gt_imgs):
                    gt = gt_imgs[i].cpu().numpy()
                else:
                    gt = gt_imgs[i]

            p = -10. * np.log10(np.mean(np.square(rgbs[i] - gt)))
            psnr.append(p)#print(p)

        if savedir is not None:

            rgb8_f = to8b(rgbs[-1]) # save coarse+fine img
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8_f)

            ### Need validate code
            rgb_gt = to8b(gt_imgs[i]) # save GT img here
            filename = os.path.join(savedir, '{:03d}_GT.png'.format(i))
            imageio.imwrite(filename, rgb_gt)

            rgb_disp = to8b(disps[-1] / np.max(disps[-1])) # save GT img here
            filename = os.path.join(savedir, '{:03d}_disp.png'.format(i))
            imageio.imwrite(filename, rgb_disp)

    psnr = np.mean(psnr,0)
    print("Mean PSNR of this run is:", psnr)
    return rgbs, disps

def render_test(args, train_dl, val_dl, hwf, start, render_kwargs_test, feat_model=None, pose_param_net=None):

    ### Eval Training set result
    trainsavedir = os.path.join(args.basedir, args.expname, 'evaluate_train_{}_{:06d}'.format('test' if args.render_test else 'path', start))
    os.makedirs(trainsavedir, exist_ok=True)
    images_train = []
    poses_train = []
    index_train = []

    # views from validation set
    for batch_data in train_dl:
        img = batch_data['img']
        pose = batch_data['pose']
        img_idx = batch_data['hist']
        batch_size = img.shape[0]

        for i in range(batch_size):
            img_val = img[i:i+1].permute(0,2,3,1) # (1,240,360,3)
            pose_val = torch.zeros(1,4,4)
            pose_val[0,:3,:4] = pose[i:i+1].reshape(3,4)[:3,:4] # (1,3,4))
            pose_val[0,3,3] = 1.
            images_train.append(img_val)
            poses_train.append(pose_val)
            index_train.append(img_idx[i:i+1])

    images_train = torch.cat(images_train, dim=0).numpy()
    poses_train = torch.cat(poses_train, dim=0).to(device)
    index_train = torch.cat(index_train, dim=0).to(device)
    print('train poses shape', poses_train.shape)

    with torch.no_grad():
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)
        if args.color_feat_loss or args.color_feat_fusion_loss or args.color_feat_fusion_nerfw_loss:
            rgbs, disps = render_path_with_feature(args, poses_train.to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images_train, savedir=trainsavedir, img_ids=index_train, feat_model=feat_model, global_step=start)
        else:
            rgbs, disps = render_path(args, poses_train.to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images_train, savedir=trainsavedir, img_ids=index_train)
        torch.set_default_device('cpu')
        torch.set_default_dtype(torch.float32)
    print('Saved train set')
    if args.render_video_train:
        print('Saving trainset as video', rgbs.shape, disps.shape)
        moviebase = os.path.join(args.basedir, args.expname, '{}_trainset_{:06d}_'.format(args.expname, start))
        imageio.mimwrite(moviebase + 'train_rgb.mp4', to8b(rgbs), fps=15, quality=8)
        imageio.mimwrite(moviebase + 'train_disp.mp4', to8b(disps / np.max(disps)), fps=15, quality=8)
    del images_train
    del poses_train
    # clean GPU memory after testing
    torch.cuda.empty_cache()

    ### Eval Validation set result
    testsavedir = os.path.join(args.basedir, args.expname, 'evaluate_val_{}_{:06d}'.format('test' if args.render_test else 'path', start))
    os.makedirs(testsavedir, exist_ok=True)
    images_val = []
    poses_val = []
    index_val = []
    # views from validation set
    for batch_idx, batch_data in enumerate(val_dl):

        img = batch_data['img']
        pose = batch_data['pose']
        img_idx = batch_data['hist']
        batch_size = img.shape[0]

        for i in range(batch_size):
            img_val = img[i:i+1].permute(0,2,3,1) # (1,240,360,3)
            pose_val = torch.zeros(1,4,4)
            pose_val[0,:3,:4] = pose[i:i+1].reshape(3,4)[:3,:4] # (1,3,4))
            pose_val[0,3,3] = 1.
            images_val.append(img_val)
            poses_val.append(pose_val)
            index_val.append(img_idx[i:i+1])

    images_val = torch.cat(images_val, dim=0).numpy()
    poses_val = torch.cat(poses_val, dim=0).to(device)
    index_val = torch.cat(index_val, dim=0).to(device)
    print('test poses shape', poses_val.shape)
    with torch.no_grad():
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)
        if args.color_feat_loss or args.color_feat_fusion_loss or args.color_feat_fusion_nerfw_loss:
            rgbs, disps = render_path_with_feature(args, poses_val.to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images_val, savedir=testsavedir, img_ids=index_val, feat_model=feat_model, global_step=start)
        else:
            rgbs, disps = render_path(args, poses_val.to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images_val, savedir=testsavedir, img_ids=index_val)
        torch.set_default_device('cpu')
        torch.set_default_dtype(torch.float32)
    print('Saved test set')
    if args.render_video_test:
        print('Saving testset as video', rgbs.shape, disps.shape)
        moviebase = os.path.join(args.basedir, args.expname, '{}_test_{:06d}_'.format(args.expname, start))
        imageio.mimwrite(moviebase + 'test_rgb.mp4', to8b(rgbs), fps=15, quality=8)
        imageio.mimwrite(moviebase + 'test_disp.mp4', to8b(disps / np.max(disps)), fps=15, quality=8)
    del images_val
    del poses_val
    return

def render_path_upsample(args, render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, single_gt_img=False, img_ids=torch.Tensor(0), target_size=[-1,-1]):

    start_frame = 0 # start to save from this frame, default should be 0

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = int(H//render_factor)
        W = int(W//render_factor)
        focal = focal/render_factor

    rgbs = []
    disps = []

    for i, c2w in enumerate(tqdm(render_poses)):

        if i < start_frame:
            continue

        with torch.cuda.amp.autocast(enabled=(args.ffmlp or args.tcnn)):
            rgb, disp, _, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], img_idx=img_ids[i], **render_kwargs)

        if (target_size[0] != W) or (target_size[1] != H):
            rgb = tensor_hwc2nchw(rgb)
            rgb = F.interpolate(rgb, size=(target_size[1],target_size[0]), mode='bicubic')
            rgb = tensor_nchw2hwc(rgb)
        rgbs.append(rgb.cpu().numpy()) # [H,W,3]
        # disps.append(disp.cpu().numpy()) # [H,W]

        if i==0:
            print(rgb.shape, disp.shape)

        if savedir is not None:

            rgb8_f = to8b(rgbs[-1]) # save coarse+fine img
            filename = os.path.join(savedir, 'frame{:05d}.png'.format(i+1-start_frame))
            imageio.imwrite(filename, rgb8_f)

    rgbs = np.stack(rgbs, 0)
    # disps = np.stack(disps, 0)
    return rgbs, disps

def render_test_upsample(args, val_dl, hwf, render_kwargs_test, target_size=[-1,-1]):
    ''' render the val poses with size of hwf and upsample to the target_size'''

    ### Eval Validation set result
    testsavedir = os.path.join(args.basedir, args.expname, 'testset_renders')
    os.makedirs(testsavedir, exist_ok=True)
    images_val = []
    poses_val = []
    index_val = []
    # views from validation set
    for img, pose, img_idx in val_dl: 
        img_val = img.permute(0,2,3,1) # (1,240,360,3)
        pose_val = torch.zeros(1,4,4)
        pose_val[0,:3,:4] = pose.reshape(3,4)[:3,:4] # (1,3,4))
        pose_val[0,3,3] = 1.
        images_val.append(img_val)
        poses_val.append(pose_val)
        index_val.append(img_idx)

    images_val = torch.cat(images_val, dim=0).numpy()
    poses_val = torch.cat(poses_val, dim=0).to(device)
    index_val = torch.cat(index_val, dim=0).to(device)
    print('test poses shape', poses_val.shape)
    with torch.no_grad():
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)
        rgbs, disps = render_path_upsample(args, poses_val.to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images_val, savedir=testsavedir, img_ids=index_val, target_size=target_size)
    print('Saved test set')
    del images_val
    del poses_val
    return

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

def render_path_with_feature(args, render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, single_gt_img=False, img_ids=torch.Tensor(0), feat_model=None, global_step=None):
    ''' Render a path of images with feature map for neural radiance field'''
    from dm.DFM_pose_refine import inference_pose_feature_extraction, feature_loss
    LARGE_FEATURE_SIZE = True # whether to evaluate the upsampled feature map

    H, W, focal = hwf

    assert (feat_model!=None)

    if render_factor!=0:
        # Render downsampled for speed
        H = int(H//render_factor)
        W = int(W//render_factor)
        focal = focal/render_factor

    rgbs = []
    disps = []
    psnr = []
    feats_psnr = []

    if PROFILE_TIME:
        t = time.time()
        for i, c2w in enumerate(render_poses):
            with torch.cuda.amp.autocast(enabled=(args.ffmlp or args.tcnn)):
                rgb, disp, acc, extras = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], img_idx=img_ids[i], **render_kwargs)
            if args.encode_hist:
                affine_color_transform = render_kwargs['network_fn'].affine_color_transform
                rgb = affine_color_transform(args, rgb, img_ids[i], 1)
        print("time: ", (time.time()-t))
        return None, None

    for i, c2w in enumerate(tqdm(render_poses)):

        with torch.cuda.amp.autocast(enabled=(args.ffmlp or args.tcnn)):
            rgb, disp, acc, extras = render(int(H//args.tinyscale), int(W//args.tinyscale), focal/args.tinyscale, chunk=chunk, c2w=c2w[:3,:4], img_idx=img_ids[i], **render_kwargs)
        if args.encode_hist:
            affine_color_transform = render_kwargs['network_fn'].affine_color_transform
            rgb = affine_color_transform(args, rgb, img_ids[i:i+1], 1)

        target = tensor_hwc2nchw(torch.Tensor(gt_imgs[i]))

        if global_step>=200: # fusion part. Chould change threshold to 400
            # NeRF feature + RGB -> CNN Fusion -> Feature
            Fusion_Net = render_kwargs['network_fn'].run_fusion_net

            render_rgb, render_feature, fusion_output = Fusion_Net(rgb, extras['feat_map'],int(H//args.tinyscale), int(W//args.tinyscale), B=1) # (1,3,60,106), (1,16,60,106), (1,16,60,106)
            fusion_output = tensor_nchw2hwc(fusion_output) # (1,16,120,213) # very high activation on stairs 
            feats = fusion_output
        else:
            render_rgb = tensor_hwc2nchw(rgb.reshape((int(H//args.tinyscale), int(W//args.tinyscale),3)))
            feats = extras['feat_map'].reshape((int(H//args.tinyscale), int(W//args.tinyscale),-1))

        if LARGE_FEATURE_SIZE:
            feats = tensor_hwc2nchw(feats)
            feat_map = torch.nn.Upsample(size=(H, W), mode='bicubic')(feats)
        else:
            feats = tensor_hwc2nchw(feats)
            feat_map = feats
        

        if i==0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None:

            data = tensor_hwc2nchw(torch.Tensor(gt_imgs[i]))
            # extract features using DFNet
            with torch.no_grad():
                if LARGE_FEATURE_SIZE:
                    hf = H
                    wf = W
                else:
                    hf = int(H//args.tinyscale)

                gt_feat, _ = inference_pose_feature_extraction(args, data, device, feat_model, retFeature=True, isSingleStream=True, return_pose=False, H=hf, W=wf)
                gt_feat = gt_feat[0][0].detach() # [1,16,120,213]

                # upsample rgb to hwf size
                render_rgb_ = torch.nn.Upsample(size=(H, W), mode='bicubic')(render_rgb) # we used bicubic for DFM RGB+CNN feature. This is mimicking that

        # remove the invalid contour by cropping edge of feautre map
        if 1:
            gt_feat = gt_feat[:, :, 10:-10, 10:-10]
            feat_map = feat_map[:, :, 10:-10, 10:-10]

        p = -10. * np.log10(np.mean(np.square((render_rgb_ - target).cpu().numpy())))
        psnr.append(p)

        if savedir is not None:

            rgb8_f = to8b(tensor_nchw2hwc(render_rgb_).cpu().numpy())
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8_f)

            rgb_gt = to8b(gt_imgs[i]) # save GT img here
            filename = os.path.join(savedir, '{:03d}_GT.png'.format(i))
            imageio.imwrite(filename, rgb_gt)

            disp = disp.reshape((int(H//args.tinyscale), int(W//args.tinyscale))).cpu().numpy()
            rgb_disp = to8b(disp / np.max(disp)) # save GT img here
            filename = os.path.join(savedir, '{:03d}_disp.png'.format(i))
            imageio.imwrite(filename, rgb_disp)

            # save feature map
            fn1 = os.path.join(savedir, '{:03d}_feature_gt.png'.format(i))
            fn2 = os.path.join(savedir, '{:03d}_feature.png'.format(i))
            plot_features(gt_feat, feat_map, fn1, fn2, i=0)

            fp = feature_loss(feat_map[0], gt_feat[0], img_in=True, per_pixel=True).cpu().numpy()
            feats_psnr.append(fp)

    rgbs=None
    disps=None
    psnr = np.mean(psnr,0)
    print("Mean PSNR of this run is:", psnr)
    feats_psnr = np.mean(feats_psnr,0)
    # print("Mean Feature PSNR of this run is:", feats_psnr)
    print("Feature cosine similarity loss:", feats_psnr)

    return rgbs, disps

