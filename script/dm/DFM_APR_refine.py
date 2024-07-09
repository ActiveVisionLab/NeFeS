import time, gc
from copy import deepcopy
import os, sys
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.init
import numpy as np

from dm.pose_model import get_error_in_q, compute_pose_error_SE3
from dm.DFM_pose_refine import inference_pose_regression, inference_pose_feature_extraction, load_NeRF_model, FeatureLoss
from dm.direct_pose_model import fix_coord_supp
from models.nerfh import img2mse, mse2psnr
from models.rendering import render

from torchvision.utils import save_image
from utils.utils import save_image_saliancy, SSIM, set_default_to_cuda, set_default_to_cpu

# # try to be deterministic
# np.random.seed(0)
# torch.manual_seed(0)
# import random
# random.seed(0)

PROFILING_DFM = False
def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))

def plot_rgb_n_batch_salient_feature(target_in, rgb_in, features_target, features_rgb, i=0):
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

def PoseLoss(args, pose_, pose, device):
    loss_func = nn.MSELoss()
    predict_pose = pose_.reshape(args.batch_size, 12).to(device) # maynot need reshape
    pose_loss = loss_func(predict_pose, pose)
    return pose_loss

# scaler = torch.cuda.amp.GradScaler(enabled=True)

def train_on_batch(args, data, model, feat_model, feature_target, pose, img_idx, hwf, optimizer, device, world_setup_dict, render_kwargs_test, feature_loss, iter_i=None):
    ''' Perform 1 step of training '''

    H, W, focal = hwf
    data = data.to(device) # [1, 3, 240, 427] non_blocking=Truen

    # pose regression module
    pose_ = inference_pose_regression(args, data, device, model) # here returns predicted pose [1, 3, 4] # real img features and predicted pose # features: (1, [3, 1, 128, 240, 427]), predict_pose: [1, 3, 4]
    
    pose_nerf = pose_.clone()

    # direct matching module
    # rescale the predicted pose to nerf scales
    pose_nerf = fix_coord_supp(args, pose_nerf, world_setup_dict, device=device)

    pose = pose.to(device)
    img_idx = img_idx.to(device)

    # every new tensor from onward is in GPU, here memory cost is a bottleneck
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float32)
    assert(args.nerfh_nff)

    rgb, _, _, extras = render(int(H//args.tinyscale), int(W//args.tinyscale), focal/args.tinyscale, chunk=args.chunk, c2w=pose_nerf[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
    if args.encode_hist:
        affine_color_transform = render_kwargs_test['network_fn'].affine_color_transform
        rgb = affine_color_transform(args, rgb, img_idx, 1)
    # NeRF feature + RGB -> CNN Fusion -> Feature
    Fusion_Net = render_kwargs_test['network_fn'].run_fusion_net
    render_rgb, render_feature, feature_rgb = Fusion_Net(rgb, extras['feat_map'], int(H//args.tinyscale), int(W//args.tinyscale), 1) # (1,3,120,213), (1,16,120,213), (1,16,120,213)
    feature_rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(feature_rgb)

    # VERIFICATION_STEP exp. 
    rgb = render_rgb.reshape(1, 3, int(H//args.tinyscale), int(W//args.tinyscale))
    rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
    # end of VERIFICATION_STEP exp.


    # crop potential invalid region on the edge of feautre map, to compensate zero padding in CNN
    feature_target = feature_target[:, :, 10:-10, 10:-10]
    feature_rgb = feature_rgb[:, :, 10:-10, 10:-10]
    gt_img = data[:, :, 10:-10, 10:-10]
    rgb = rgb[:, :, 10:-10, 10:-10]

    # only use feature loss
    loss = feature_loss(feature_rgb[0], feature_target[0])

    ### Loss Design End
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # end of every new tensor from onward is in GPU
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float32)

    device_cpu = torch.device('cpu')
    iter_loss = loss.to(device_cpu).detach().numpy()
    iter_loss = np.array([iter_loss])

    if args.nerfh_nff:
        try:
            psnr = mse2psnr(img2mse(rgb, gt_img))
            iter_psnr = psnr.to(device_cpu).detach().numpy()
            compute_ssim = SSIM().to(device)
            ssim = compute_ssim(rgb, gt_img).mean()
        except:
            print('check if 1 paddings are removed')
    else:
        psnr = mse2psnr(img2mse(rgb.cpu(), data.cpu()))
        iter_psnr = psnr.to(device_cpu).detach().numpy()
        ssim = 0
    return iter_loss, iter_psnr, ssim

def DFM_post_processing(args, model, feat_model, hwf, near, far, device, test_dl=None):
    ''' Use Direct Feature Matching as a Post-processing. We optimize APR model in this loop

    predict_poses: [N, 3, 4]
    feat_model: feature extractor
    render_kwargs_test: kwargs for the nefes model
    hwf: [H, W, focal]
    device: gpu device
    test_dl: test dataloader
    world_setup_dict: world setup dict
    '''
    SAVE_DFM_RESULT = True
    VERIFICATION_STEP = not args.no_verification_step

    if SAVE_DFM_RESULT:
        pose_results = []

    # load NeFeS model
    render_kwargs_test = load_NeRF_model(args, near, far)

    world_setup_dict = {
        'pose_scale' : test_dl.dataset.pose_scale,
        'pose_scale2' : test_dl.dataset.pose_scale2,
        'move_all_cam_vec' : test_dl.dataset.move_all_cam_vec,
    }

    model.eval()

    # Benchmark inital pose precision
    print("Initial Precision:")
    get_error_in_q(args, test_dl, model, len(test_dl.dataset), device, batch_size=1)
    model.train() # # TODO: resume gradiant update

    # set nerf model requires_grad to False to accelerate the refinement?
    render_kwargs_test['network_fn'].requires_grad_(False)
    render_kwargs_test['network_fine'].requires_grad_(False)
    feat_model.eval().requires_grad_(False)

    import torch.optim as optim

    results = np.zeros((len(test_dl.dataset), 2))

    ### Core optimization loop exp. per image per model ###
    init_iter = args.opt_iter
    # profiling
    start_timer()
    for batch_idx, batch_data in enumerate(test_dl):
        data, pose, img_idx = batch_data['img'], batch_data['pose'], batch_data['hist']

        if batch_idx % 10 == 0:
            print("renders {}/total {}".format(batch_idx, len(test_dl.dataset)), flush=True)
        pp_model = deepcopy(model)

        # set optimizer for post processing model (BAD Implementation)
        optimizer_pp = optim.Adam(pp_model.parameters(), lr=args.learning_rate) #weight_decay=weight_decay, **kwargs
        feature_loss = FeatureLoss(per_pixel=args.per_pixel).to(device)

        # We move the query image feature extraction to here for acceleration. This only need once per image.
        LARGE_FEATURE_SIZE = True # use 240x427 feature size, or 60x106
        if LARGE_FEATURE_SIZE:
            feature_list, _ = inference_pose_feature_extraction(args, data, device, feat_model, retFeature=True, isSingleStream=True, return_pose=False, H=int(hwf[0]), W=int(hwf[1]))
        else:
            feature_list, _ = inference_pose_feature_extraction(args, data, device, feat_model, retFeature=True, isSingleStream=True, return_pose=False, H=int(hwf[0]//args.tinyscale), W=int(hwf[1]//args.tinyscale)) # here returns GT img and nerf img features (2, [3, 1, 128, 240, 427])
        feature_target = feature_list[0][0]

        for i in range(init_iter):
            loss, psnr, ssim = train_on_batch(args, data, pp_model, feat_model, feature_target, pose, img_idx, hwf, optimizer_pp, device, world_setup_dict, render_kwargs_test, feature_loss, i)

            if VERIFICATION_STEP:
                if i==0:
                    init_psnr = psnr
                    init_ssim = ssim
                
                elif i % (init_iter-1) == 0:
                    end_psnr = psnr
                    end_ssim = ssim

        ### inference pp_model
        data = data.to(device) # input
        pose = pose.reshape(1,3,4) # label

        predict_pose = inference_pose_regression(args, data, device, pp_model)
        predict_pose = predict_pose.reshape(1,3,4).cpu().detach()

        if VERIFICATION_STEP: # this is a fail safe mechanism to prevent degradation after refinement
            retreat=False
            if end_psnr < init_psnr:
                retreat=True
            if end_ssim < init_ssim:
                retreat=True
            if retreat:
                predict_pose = inference_pose_regression(args, data, device, model)
                predict_pose = predict_pose.reshape(1,3,4).cpu().detach()

        # compute pose error between the ground truth and the network predicted pose
        error_x, theta = compute_pose_error_SE3(pose, predict_pose) # we recently update this for better error precision
        results[batch_idx,:] = [error_x, theta]

        if SAVE_DFM_RESULT:
            pose_results.append(predict_pose.numpy())

    end_timer_and_print("Mixed precision:")

    median_result = np.median(results,axis=0)
    mean_result = np.mean(results,axis=0)

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
        np.savetxt(os.path.join(f'{save_folder}/{args.PoseEstimatorType}_{scene}_NeFeS{init_iter:01d}_APR_pose_results.txt'), pose_results)
    sys.exit()
# 