# neural feature field training
import utils.set_sys_path
import os
import numpy as np
import random
import torch

from tqdm import tqdm, trange

from models.ray_utils import get_rays_batch
from models.options import config_parser
from models.rendering import render, render_test, render_path, render_test_upsample, render_path_with_feature
from models.nerfh_nff import img2mse, mse2psnr
from dataset_loaders.load_7Scenes import load_7Scenes_dataloader_NeRF
from dataset_loaders.load_Cambridge import load_Cambridge_dataloader_NeRF
from dataset_loaders.load_7Scenes_colmap import load_7Scenes_dataloader_NeRF_colmap

# losses
from models.losses import loss_dict
from dm.DFM_pose_refine import inference_pose_feature_extraction
from utils.utils import tensor_nchw2nhwc, set_default_to_cuda, set_default_to_cpu

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(0)
import random
random.seed(0)

parser = config_parser()
args = parser.parse_args()

### Section 3.2 Progressive Training ###
if args.new_schedule == 2: # for step2 of training color+feature+fusion
    PRE_TRAIN_COLOR_ONLY_EPOCH=0
    EPOCH_TO_FUSION=200+PRE_TRAIN_COLOR_ONLY_EPOCH
else: # step1 of training color only
    PRE_TRAIN_COLOR_ONLY_EPOCH=args.epochs
    EPOCH_TO_FUSION=200+PRE_TRAIN_COLOR_ONLY_EPOCH

def render_nerf_random_ray(args, H, W, focal, pose, valid_inds, N_rand, target, feature_target, hist, render_kwargs_train):
    ''' Render a random ray batch given a NeRF model. '''

    rays_o, rays_d = get_rays_batch(H, W, focal, torch.Tensor(pose)) # (B, H, W, 3), (B, H, W, 3))

    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='ij'), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
    # filter out dynamic coords and extract valid GT rays
    select_coords=[]
    for i in range(target.shape[0]):
        if args.semantic:
            try:
                coords_ = coords[valid_inds[i]] # filter out temporal coordinates of image
                select_inds = np.random.choice(coords_.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords.append(coords_[select_inds].long())  # (N_rand, 2)
            except:
                breakpoint()
        else:
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords.append(coords[select_inds].long())  # (N_rand, 2)
    select_coords = torch.stack(select_coords, dim=0)  # (B, N_rand, 2)
    # extract valid GT rays
    rays_o = torch.cat([rays_o[k,select_coords[k,:,0], select_coords[k,:,1]] for k in range(target.shape[0])])  # (B*N_rand, 3)
    rays_d = torch.cat([rays_d[k,select_coords[k,:,0], select_coords[k,:,1]] for k in range(target.shape[0])])  # (B*N_rand, 3)
    batch_rays = torch.stack([rays_o, rays_d], dim=0) # (2,B*N_rand,3)

    target_s = torch.cat([target[k,select_coords[k,:,0], select_coords[k,:,1]] for k in range(target.shape[0])])  # (B*N_rand, 3)
    target_f = torch.cat([feature_target[k,select_coords[k,:,0], select_coords[k,:,1]] for k in range(target.shape[0])])  # (B*N_rand, 3)

    hist = torch.cat([hist[k].expand(N_rand,10) for k in range(target.shape[0])])  # (B*N_rand, 10)

    # #####  Core optimization loop  #####
    with torch.cuda.amp.autocast(enabled=(args.ffmlp or args.tcnn)):
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, retraw=True, img_idx=hist, **render_kwargs_train)
    return rgb, disp, acc, extras, target_s, target_f, select_coords

def render_nerf_random_patch(args, H, W, focal, pose, target, feature_target, hist, render_kwargs_train):
    ''' Render random patches given a NeRF model. '''
    batch_size = target.shape[0]

    rays_o, rays_d = get_rays_batch(H, W, focal, torch.Tensor(pose)) # (B, H, W, 3), (B, H, W, 3))
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='ij'), -1)  # (H, W, 2)

    # randomly select coordinates, lazy impelmentation, select same pixels coordinates for all images in batch
    num_crops = 7 # 7 patches per image in batch
    crop_size = 16 # 16x16 pixels
    H_coord = torch.randint(0, H-crop_size, (num_crops,))
    W_coord = torch.randint(0, W-crop_size, (num_crops,))

    select_coords = []
    for i in range(num_crops):
        select_coords.append(coords[H_coord[i]:H_coord[i]+crop_size, W_coord[i]:W_coord[i]+crop_size].long())
    select_coords = torch.stack(select_coords, dim=0).reshape([-1,2])  # (num_crops*crop_size*crop_size, 2)

    # create ray batches
    rays_o = torch.cat([rays_o[k,select_coords[:,0], select_coords[:,1]] for k in range(batch_size)])  # (B*N_rand, 3)
    rays_d = torch.cat([rays_d[k,select_coords[:,0], select_coords[:,1]] for k in range(batch_size)])  # (B*N_rand, 3)
    batch_rays = torch.stack([rays_o, rays_d], dim=0) # (2,batch_size*num_crops*crop_size*crop_size,3)

    target_s = torch.cat([target[k,select_coords[:,0], select_coords[:,1]] for k in range(batch_size)])  # (B*N_rand, 3)
    target_f = torch.cat([feature_target[k,select_coords[:,0], select_coords[:,1]] for k in range(batch_size)])  # (B*N_rand, 3)

    hist = torch.cat([hist[k].expand(num_crops*crop_size*crop_size,10) for k in range(batch_size)])  # (B*N_rand, 10)
    with torch.cuda.amp.autocast(enabled=(args.ffmlp or args.tcnn)):
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, retraw=True, img_idx=hist, **render_kwargs_train)

    return rgb, disp, acc, extras, target_s, target_f, select_coords, num_crops, crop_size

def render_nerf_full_img(H, W, focal, pose, hist, render_kwargs_train):
    ''' Render a full image given a NeRF model. '''
    with torch.cuda.amp.autocast(enabled=(args.ffmlp or args.tcnn)):
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose, img_idx=hist, **render_kwargs_train) # render (120x213,3)
    return rgb, disp, acc, extras

def train_nerf_on_epoch(args, train_dl, H, W, focal, N_rand, optimizer, loss_func, global_step, render_kwargs_train, feat_model, pose_param_net=None, scaler=None):
    # Random from one image for 7 Scenes
    for batch_idx, batch_data in enumerate(train_dl):
        img, pose, hist = batch_data['img'], batch_data['pose'], batch_data['hist']
        batch_size = img.shape[0] # current true batch size

        optimizer.zero_grad()
        pose = pose.reshape(-1,3,4).to(device) # [B,3,4]
    
        data = img.to(device) # (1,3,240,427)
        hist = hist.to(device)

        if args.semantic:
            # find pixels which are static
            mask = batch_data['mask']
            mask = mask.reshape(-1,int(H//args.tinyscale) * int(W//args.tinyscale)).to(device) # [B,H*W]
            valid_inds = [torch.nonzero(mask[i]>0, as_tuple=True)[0] for i in range(batch_size)] # list of valid static indices (B,[Ni_valid]), notice each Ni_valid could have different amount of values
        else:
            valid_inds = None

        # extract features using DFNet
        with torch.no_grad():
            # CNN must input at least 224x224 image
            feature_target, _ = inference_pose_feature_extraction(args, data, device, feat_model, retFeature=True, isSingleStream=True, return_pose=False, H=int(H//args.tinyscale), W=int(W//args.tinyscale))
            feature_target = feature_target[0][0].permute(0,2,3,1).detach() # [B,120,213,16]

        # render features with neural feature field
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)

        downsample_factor = int(args.tinyscale) # we use this factor to downsample the target image
        target = F.interpolate(data, size=(H//downsample_factor, W//downsample_factor), mode='bilinear', align_corners=False)
        target = tensor_nchw2nhwc(target)

        if global_step>=EPOCH_TO_FUSION: # patch-wise training custom random crop

            rgb, disp, acc, extras, target_s, target_f, select_coords, num_crops, crop_size = render_nerf_random_patch(args, int(H//args.tinyscale), int(W//args.tinyscale), focal/args.tinyscale, pose, target, feature_target, hist, render_kwargs_train)
            if args.encode_hist:
                affine_color_transform = render_kwargs_train['network_fn'].affine_color_transform
                rgb = affine_color_transform(args, rgb, hist, batch_size)

            # NeRF feature + RGB -> CNN Fusion -> Feature
            Fusion_Net = render_kwargs_train['network_fn'].run_fusion_net

            render_rgb, render_feature, fusion_output = Fusion_Net(rgb, extras['feat_map'], crop_size, crop_size, batch_size*num_crops) # (B,3,120,213), (B,16,120,213), (B,16,120,213)
            fusion_output = tensor_nchw2nhwc(fusion_output).reshape([-1,fusion_output.shape[1]])  # (B,120,213,16) #.reshape((-1,16))->(B*120*213,16)

            # filter out invalid pixels
            if args.semantic:
                mask = batch_data['mask']
                select_mask = torch.cat([mask[k,select_coords[:,0], select_coords[:,1]] for k in range(batch_size)])
                valid_inds = torch.nonzero(select_mask>0, as_tuple=True)[0]

                rgbs = rgb[valid_inds]
                rgb0s = extras['rgb0'][valid_inds]
                feats = extras['feat_map'][valid_inds]
                fusion_output = fusion_output[valid_inds]
                target_s = target_s[valid_inds]
                target_f = target_f[valid_inds]
                if 'static_sigmas' in extras:
                    static_sigmas = extras['static_sigmas'][valid_inds]
                if 'beta' in extras:
                    beta = extras['beta'][valid_inds]
                if 'transient_sigmas' in extras:
                    transient_sigmas = extras['transient_sigmas'][valid_inds]
            else:
                rgbs = rgb
                rgb0s = extras['rgb0']
                feats = extras['feat_map']
                target_s = target_s
                target_f = target_f

                if 'static_sigmas' in extras:
                    static_sigmas = extras['static_sigmas']
                if 'beta' in extras:
                    beta = extras['beta']
                if 'transient_sigmas' in extras:
                    transient_sigmas = extras['transient_sigmas']

            ### prepare for loss
            assert(rgbs.shape[0]==target_s.shape[0])
            assert(feats.shape[0]==target_f.shape[0])
            results = {}
            results['rgb_fine'] = rgbs
            if 'rgb0' in extras:
                results['rgb_coarse'] = rgb0s
            if 'feat_map' in extras:
                results['feat_fine'] = feats
            results['feat_fusion'] = fusion_output

            if 'beta' in extras:
                results['beta'] = beta
            if 'transient_sigmas' in extras:
                results['transient_sigmas'] = transient_sigmas
        else:
            # random sampling
            rgb, disp, acc, extras, target_s, target_f, select_coords = render_nerf_random_ray(args, int(H//args.tinyscale), int(W//args.tinyscale), focal/args.tinyscale, pose, valid_inds, N_rand, target, feature_target, hist, render_kwargs_train)

            # perform histogram matching if needed
            if args.encode_hist:
                affine_color_transform = render_kwargs_train['network_fn'].affine_color_transform
                rgb = affine_color_transform(args, rgb, hist, batch_size)

            ### prepare for loss
            results = {}
            results['rgb_fine'] = rgb
            rgbs = rgb

            if 'rgb0' in extras:
                results['rgb_coarse'] = extras['rgb0']
            if 'feat_map' in extras:
                results['feat_fine'] = extras['feat_map']

            if 'static_sigmas' in extras:
                static_sigmas = extras['static_sigmas']

            if 'beta' in extras:
                results['beta'] = extras['beta']
            if 'transient_sigmas' in extras:
                results['transient_sigmas'] = extras['transient_sigmas']

        # compute loss
        if global_step >= EPOCH_TO_FUSION: # train fusion network
            target_ = {'rgb': target_s, 'feat': target_f}
            loss_rgb, loss_f, loss_fusion = loss_func(results, target_, switch_on=True, color_only_switch=False)
            loss_f = 0.02 * loss_f
            loss_fusion = 0.02 * loss_fusion
            loss = loss_rgb + loss_f + loss_fusion
        elif global_step >=PRE_TRAIN_COLOR_ONLY_EPOCH: # train like color_feat_loss
            target_ = {'rgb': target_s, 'feat': target_f}
            loss_rgb, loss_f = loss_func(results, target_, switch_on=False, color_only_switch=False)
            loss_f = 0.04 * loss_f
            loss = loss_rgb + loss_f
        else: # train color only
            target_ = {'rgb': target_s}
            loss = loss_func(results, target_, switch_on=False, color_only_switch=True)

        try:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        except:
            print("loss.backward() error")
            breakpoint()

        with torch.no_grad():
            img_loss = img2mse(rgbs, target_s)
            psnr = mse2psnr(img_loss)

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000 # this can be manually simplified TODO
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))  # max epochs=600, decay_steps=754
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        torch.set_default_device('cpu')
        torch.set_default_dtype(torch.float32)
    return loss, psnr

def train_nerf(args, train_dl, val_dl, hwf, i_split, near, far, render_poses=None, render_img=None):
    if args.nerfh_nff:
        from models.nerfh_nff import create_nerf
    else:
        print("ERROR: check your model settings")
        breakpoint()

    i_train, i_val, i_test = i_split
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    if args.reduce_embedding==2:
        render_kwargs_train['i_epoch'] = global_step
        render_kwargs_test['i_epoch'] = global_step

    # # load pretrained DFNet model
    from dm.direct_pose_model import load_FeatureNet
    feat_model = load_FeatureNet(args, device)
    pose_param_net = None

    if args.new_schedule == 2: # for step2 of training color+feature+fusion
        if args.render_test!=True:
            start = 0 # for finetuning
            global_step=0 # for finetuning

    if args.render_test:
        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)
        render_test(args, train_dl, val_dl, hwf, start, render_kwargs_test, feat_model=feat_model, pose_param_net=pose_param_net)
        return

    if args.render_pose_only:
        # exp.1, render nerf-Hist result and upsample to Wt x Ht
        # pixloc: 1920x1080, dsac*: 854x480
        Wt = 1920 # 854
        Ht = 1080 # 480
        render_test_upsample(args, val_dl, hwf, render_kwargs_test, target_size=[Wt,Ht])
        return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand

    N_epoch = args.epochs + 1 # epoch
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # loss function
    if args.color_loss_only==True:
        loss_func = loss_dict['color'](coef=1)
    elif args.color_feat_loss:
        loss_func = loss_dict['color_feat'](coef=1, L1_loss=True)
    elif args.color_feat_fusion_loss:
        loss_func = loss_dict['color_feat_fusion'](coef=1, L1_loss=True, cos_loss=False)
    elif args.color_feat_fusion_nerfw_loss:
        loss_func = loss_dict['color_feat_fusion_nerfw'](coef=1, L1_loss=True)
    else:
        print("Check loss setting")
        NotImplementedError

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler(enabled=(args.ffmlp or args.tcnn))

    for i in trange(start, N_epoch):

        if args.reduce_embedding==2:
            render_kwargs_train['i_epoch'] = i
        loss, psnr = train_nerf_on_epoch(args, train_dl, H, W, focal, N_rand, optimizer, loss_func, global_step, render_kwargs_train, feat_model, pose_param_net=pose_param_net, scaler=scaler)

        # Rest is logging
        if i%args.i_weights==0 and i!=0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))

            model_save_dict={'global_step': global_step,
                            'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),}
            if args.N_importance > 0: # have fine sample network
                model_save_dict['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()

            torch.save(model_save_dict, path)
            print('Saved checkpoints at', path)

        if i%args.i_testset==0 and i > 0: # run thru all validation set

            # clean GPU memory before testing, try to avoid OOM
            torch.cuda.empty_cache()

            if args.reduce_embedding==2:
                render_kwargs_test['i_epoch'] = i
            trainsavedir = os.path.join(basedir, expname, 'trainset_{:06d}'.format(i))
            os.makedirs(trainsavedir, exist_ok=True)
            images_train = []
            poses_train = []
            index_train = []
            j_cnt = 0
            j_skip = 10 # save holdout view render result Trainset/j_skip
            # randomly choose some holdout views from training set
            for batch_idx, batch_data in enumerate(train_dl):

                img = batch_data['img']
                pose = batch_data['pose']
                img_idx = batch_data['hist']
                batch_size = img.shape[0]

                for j in range(batch_size):
                    if j_cnt % j_skip != 0:
                        j_cnt += 1
                        continue
                    img_val = img[j:j+1].permute(0,2,3,1) # (1,240,360,3)
                    pose_val = torch.zeros(1,4,4)
                    pose_val[0,:3,:4] = pose[j:j+1].reshape(3,4)[:3,:4] # (1,3,4))
                    pose_val[0,3,3] = 1.
                    images_train.append(img_val)
                    poses_train.append(pose_val)
                    index_train.append(img_idx[j:j+1])
                    j_cnt += 1

            images_train = torch.cat(images_train, dim=0).numpy()
            poses_train = torch.cat(poses_train, dim=0).to(device)
            index_train = torch.cat(index_train, dim=0).to(device)
            print('train poses shape', poses_train.shape)

            with torch.no_grad():
                torch.set_default_device('cuda')
                torch.set_default_dtype(torch.float32)
                if args.color_feat_loss or args.color_feat_fusion_loss or args.color_feat_fusion_nerfw_loss:
                    render_path_with_feature(args, poses_train, hwf, args.chunk, render_kwargs_test, gt_imgs=images_train, savedir=trainsavedir, img_ids=index_train, feat_model=feat_model, global_step=global_step)
                else:
                    render_path(args, poses_train, hwf, args.chunk, render_kwargs_test, gt_imgs=images_train, savedir=trainsavedir, img_ids=index_train)
                torch.set_default_device('cpu')
                torch.set_default_dtype(torch.float32)
            print('Saved train set')
            del images_train
            del poses_train

            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            images_val = []
            poses_val = []
            index_val = []
            # views from validation set
            for batch_data in val_dl:
                img = batch_data['img']
                pose = batch_data['pose']
                img_idx = batch_data['hist']
                batch_size = img.shape[0]

                for j in range(batch_size):
                    img_val = img[j:j+1].permute(0,2,3,1) # (1,240,360,3)
                    pose_val = torch.zeros(1,4,4)
                    pose_val[0,:3,:4] = pose[j:j+1].reshape(3,4)[:3,:4] # (1,3,4))
                    pose_val[0,3,3] = 1.
                    images_val.append(img_val)
                    poses_val.append(pose_val)
                    index_val.append(img_idx[j:j+1])
                    j_cnt += 1

            images_val = torch.cat(images_val, dim=0).numpy()
            poses_val = torch.cat(poses_val, dim=0).to(device)
            index_val = torch.cat(index_val, dim=0).to(device)
            print('test poses shape', poses_val.shape)

            with torch.no_grad():
                torch.set_default_device('cuda')
                torch.set_default_dtype(torch.float32)
                if args.color_feat_loss or args.color_feat_fusion_loss or args.color_feat_fusion_nerfw_loss:
                    render_path_with_feature(args, poses_val, hwf, args.chunk, render_kwargs_test, gt_imgs=images_val, savedir=testsavedir, img_ids=index_val, feat_model=feat_model, global_step=global_step)
                else:
                    render_path(args, poses_val, hwf, args.chunk, render_kwargs_test, gt_imgs=images_val, savedir=testsavedir, img_ids=index_val)
                torch.set_default_device('cpu')
                torch.set_default_dtype(torch.float32)
            print('Saved test set')

            # clean GPU memory after testing
            torch.cuda.empty_cache()
            del images_val
            del poses_val

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1

def train():

    print(parser.format_values())

    # Load data
    if args.dataset_type == '7Scenes':
        train_dl, val_dl, hwf, i_split, bds, render_poses, render_img = load_7Scenes_dataloader_NeRF(args)
        near = bds[0]
        far = bds[1]
        if args.set_near_far:
            print('use customized near_far')
            near = args.near_far[0]
            far = args.near_far[1]

        print('NEAR FAR Bound', near, far, args.bound)
        train_nerf(args, train_dl, val_dl, hwf, i_split, near, far, render_poses, render_img)
        return

    elif args.dataset_type == 'Cambridge':

        train_dl, val_dl, hwf, i_split, bds, render_poses, render_img = load_Cambridge_dataloader_NeRF(args)
        near = bds[0]
        far = bds[1]
        if args.set_near_far:
            print('use customized near_far')
            near = args.near_far[0]
            far = args.near_far[1]

        print('NEAR FAR Bound', near, far, args.bound)
        train_nerf(args, train_dl, val_dl, hwf, i_split, near, far, render_poses, render_img)
        return

    elif args.dataset_type == '7Scenes_colmap':

        train_dl, val_dl, hwf, i_split, bds, render_poses, render_img = load_7Scenes_dataloader_NeRF_colmap(args) # dataset with GT poses from COLMAP
        near = bds[0]
        far = bds[1]
        if args.set_near_far:
            print('use customized near_far')
            near = args.near_far[0]
            far = args.near_far[1]

        print('NEAR FAR Bound', near, far, args.bound)
        train_nerf(args, train_dl, val_dl, hwf, i_split, near, far, render_poses, render_img)
        return
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

if __name__=='__main__':

    train()