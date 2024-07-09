import torch
from torch import nn
import pytorch3d.transforms as transforms
import numpy as np
import math
# from utils.utils import plot_features, save_image_saliancy
from dm.direct_pose_model import fix_coord_supp
# from dm.pose_model import vis_pose
from models.rendering import render
import time
import os
import os.path as osp
from copy import deepcopy
# from torchvision.utils import save_image
from utils.utils import tensor_nhwc2nchw, set_default_to_cuda, set_default_to_cpu

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# translation z axis
trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]], dtype=np.float)

# x rotation
rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]], dtype=np.float)

# y rotation
rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]], dtype=np.float)

# z rotation
rot_psi = lambda psi : np.array([
    [np.cos(psi),-np.sin(psi),0,0],
    [np.sin(psi),np.cos(psi),0,0],
    [0,0,1,0],
    [0,0,0,1]], dtype=np.float)

def compute_error_in_q(args, dl, model, device, results, batch_size=1):
    use_SVD=True # Turn on for Direct-PN and Direct-PN+U reported result, despite it makes minuscule differences
    time_spent = []
    predict_pose_list = []
    gt_pose_list = []
    ang_error_list = []
    i = 0
    for batch in dl:
        if (type(batch) is dict):
            data = batch['img']
            pose = batch['pose']
        elif args.NeRFW:
            data, pose, img_idx = batch
        else:
            data, pose = batch
        data = data.to(device) # input
        pose = pose.reshape((batch_size,3,4)).numpy() # label

        if use_SVD:
            # using SVD to make sure predict rotation is normalized rotation matrix
            with torch.no_grad():
                _, predict_pose = model(data)
                R_torch = predict_pose.reshape((batch_size, 3, 4))[:,:3,:3] # debug
                predict_pose = predict_pose.reshape((batch_size, 3, 4)).cpu().numpy()

                R = predict_pose[:,:3,:3]
                res = R@np.linalg.inv(R)
                # print('R@np.linalg.inv(R):', res)

                u,s,v=torch.svd(R_torch)
                Rs = torch.matmul(u, v.transpose(-2,-1))
            predict_pose[:,:3,:3] = Rs[:,:3,:3].cpu().numpy()
        else:
            start_time = time.time()
            # inference NN
            with torch.no_grad():
                predict_pose = model(data)
                predict_pose = predict_pose.reshape((batch_size, 3, 4)).cpu().numpy()
            time_spent.append(time.time() - start_time)

        pose_q = transforms.matrix_to_quaternion(torch.Tensor(pose[:,:3,:3]))#.cpu().numpy() # gnd truth in quaternion
        pose_x = pose[:, :3, 3] # gnd truth position
        predicted_q = transforms.matrix_to_quaternion(torch.Tensor(predict_pose[:,:3,:3]))#.cpu().numpy() # predict in quaternion
        predicted_x = predict_pose[:, :3, 3] # predict position
        pose_q = pose_q.squeeze() 
        pose_x = pose_x.squeeze() 
        predicted_q = predicted_q.squeeze() 
        predicted_x = predicted_x.squeeze()
        
        #Compute Individual Sample Error 
        q1 = pose_q / torch.linalg.norm(pose_q)
        q2 = predicted_q / torch.linalg.norm(predicted_q)
        d = torch.abs(torch.sum(torch.matmul(q1,q2))) 
        d = torch.clamp(d, -1., 1.) # acos can only input [-1~1]
        theta = (2 * torch.acos(d) * 180/math.pi).numpy()
        error_x = torch.linalg.norm(torch.Tensor(pose_x-predicted_x)).numpy()
        results[i,:] = [error_x, theta]
        #print ('Iteration: {} Error XYZ (m): {} Error Q (degrees): {}'.format(i, error_x, theta)) 

        # save results for visualization
        predict_pose_list.append(predicted_x)
        gt_pose_list.append(pose_x)
        ang_error_list.append(theta)
        i += 1
    predict_pose_list = np.array(predict_pose_list)
    gt_pose_list = np.array(gt_pose_list)
    ang_error_list = np.array(ang_error_list)
    vis_info_ret = {"pose": predict_pose_list, "pose_gt": gt_pose_list, "theta": ang_error_list}
    return results, vis_info_ret

# # pytorch
def get_error_in_q(args, dl, model, sample_size, device, batch_size=1):
    ''' Convert Rotation matrix to quaternion, then calculate the location errors. original from PoseNet Paper '''
    model.eval()
    
    results = np.zeros((sample_size, 2))
    results, vis_info = compute_error_in_q(args, dl, model, device, results, batch_size)
    median_result = np.median(results,axis=0)
    mean_result = np.mean(results,axis=0)

    # standard log
    print ('Median error {}m and {} degrees.'.format(median_result[0], median_result[1]))
    print ('Mean error {}m and {} degrees.'.format(mean_result[0], mean_result[1]))

    # visualize results
    # vis_pose(vis_info)

def get_render_error_in_q(args, model, sample_size, device, targets, rgbs, poses, batch_size=1):
    ''' use nerf render imgs instead of use real imgs '''
    model.eval()
    
    results = np.zeros((sample_size, 2))
    print("to be implement...")

    predict_pose_list = []
    gt_pose_list = []
    ang_error_list = []

    for i in range(sample_size):
        data = rgbs[i:i+1].permute(0,3,1,2)
        pose = poses[i:i+1].reshape(batch_size, 12)

        data = data.to(device) # input
        pose = pose.reshape((batch_size,3,4)).numpy() # label

        # using SVD to make sure predict rotation is normalized rotation matrix
        with torch.no_grad():
            _, predict_pose = model(data)
            R_torch = predict_pose.reshape((batch_size, 3, 4))[:,:3,:3] # debug
            predict_pose = predict_pose.reshape((batch_size, 3, 4)).cpu().numpy()

            R = predict_pose[:,:3,:3]
            res = R@np.linalg.inv(R)
            # print('R@np.linalg.inv(R):', res)

            u,s,v=torch.svd(R_torch)
            Rs = torch.matmul(u, v.transpose(-2,-1))
        predict_pose[:,:3,:3] = Rs[:,:3,:3].cpu().numpy()

        pose_q = transforms.matrix_to_quaternion(torch.Tensor(pose[:,:3,:3]))#.cpu().numpy() # gnd truth in quaternion
        pose_x = pose[:, :3, 3] # gnd truth position
        predicted_q = transforms.matrix_to_quaternion(torch.Tensor(predict_pose[:,:3,:3]))#.cpu().numpy() # predict in quaternion
        predicted_x = predict_pose[:, :3, 3] # predict position
        pose_q = pose_q.squeeze() 
        pose_x = pose_x.squeeze() 
        predicted_q = predicted_q.squeeze() 
        predicted_x = predicted_x.squeeze()
        
        #Compute Individual Sample Error 
        q1 = pose_q / torch.linalg.norm(pose_q)
        q2 = predicted_q / torch.linalg.norm(predicted_q)
        d = torch.abs(torch.sum(torch.matmul(q1,q2))) 
        d = torch.clamp(d, -1., 1.) # acos can only input [-1~1]
        theta = (2 * torch.acos(d) * 180/math.pi).numpy()
        error_x = torch.linalg.norm(torch.Tensor(pose_x-predicted_x)).numpy()
        results[i,:] = [error_x, theta]
        #print ('Iteration: {} Error XYZ (m): {} Error Q (degrees): {}'.format(i, error_x, theta)) 

        # save results for visualization
        predict_pose_list.append(predicted_x)
        gt_pose_list.append(pose_x)
        ang_error_list.append(theta)

    predict_pose_list = np.array(predict_pose_list)
    gt_pose_list = np.array(gt_pose_list)
    ang_error_list = np.array(ang_error_list)
    vis_info_ret = {"pose": predict_pose_list, "pose_gt": gt_pose_list, "theta": ang_error_list}

    median_result = np.median(results,axis=0)
    mean_result = np.mean(results,axis=0)

    # standard log
    print ('Median error {}m and {} degrees.'.format(median_result[0], median_result[1]))
    print ('Mean error {}m and {} degrees.'.format(mean_result[0], mean_result[1]))

    return

def render_nerfw_imgs(args, dl, hwf, device, render_kwargs_test, world_setup_dict, NFF=False, verbose=True):
    ''' render nerfw imgs, save unscaled pose and results'''
    H, W, focal = hwf
    target_list = []
    rgb_list = []
    pose_list = []
    img_idx_list = []
    # profiling code
    # time0 = time.time()
    
    # inference nerfw and save rgb, target, pose
    for batch_idx, batch_data in enumerate(dl):
        target = batch_data['img']
        pose = batch_data['pose']
        img_idx = batch_data['hist']

        if batch_idx % 100 == 0 and verbose:
            print("renders {}/total {}".format(batch_idx, len(dl.dataset)))

        target = target[0].permute(1,2,0).to(device) # (240,360,3)
        pose = pose.reshape(3,4) # reshape to 3x4 rot matrix

        img_idx = img_idx.to(device)
        pose_nerf = pose.clone()


        # rescale the predicted pose to nerf scales
        pose_nerf = fix_coord_supp(args, pose_nerf[None,...], world_setup_dict)

        # generate nerf image
        with torch.no_grad():
            torch.set_default_device('cuda')
            torch.set_default_dtype(torch.float32)
            if args.tinyimg:
                rgb, _, _, _ = render(int(H//args.tinyscale), int(W//args.tinyscale), focal/args.tinyscale, chunk=args.chunk, c2w=pose_nerf[0,:3,:4].to(device), retraw=True, img_idx=img_idx, **render_kwargs_test)
                if args.encode_hist and NFF:
                    affine_color_transform = render_kwargs_test['network_fn'].affine_color_transform
                    rgb = affine_color_transform(args, rgb, img_idx, 1)
                rgb = rgb.reshape((int(H//args.tinyscale), int(W//args.tinyscale), 3))

                # convert rgb to B,C,H,W
                rgb = rgb[None,...].permute(0,3,1,2)
                # breakpoint()
                # save_image(rgb, './tmp/rgb1.png')
                # upsample rgb to hwf size
                rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
                # save_image(rgb, './tmp/rgb2.png')
                # breakpoint()
                # convert rgb back to H,W,C format
                rgb = rgb[0].permute(1,2,0)

            else:
                rgb, _, _, _ = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf[0,:3,:4].to(device), retraw=True, img_idx=img_idx, **render_kwargs_test)
                rgb = rgb.reshape((H, W, 3))
            torch.set_default_device('cpu')
            torch.set_default_dtype(torch.float32)

        target_list.append(target.cpu())
        rgb_list.append(rgb.cpu())
        pose_list.append(pose.cpu())
        img_idx_list.append(img_idx.cpu())
    # print("time spending: ", time.time()-time0)

    targets = torch.stack(target_list).detach() # [46, 240, 427, 3]
    rgbs = torch.stack(rgb_list).detach() # [46, 240, 427, 3]
    poses = torch.stack(pose_list).detach()
    img_idxs = torch.stack(img_idx_list).detach()
    return targets, rgbs, poses, img_idxs

def render_virtual_imgs(args, pose_perturb, img_idxs, hwf, device, render_kwargs_test, world_setup_dict, NFF=False):
    ''' render nerfw imgs, save unscaled pose and results'''
    H, W, focal = hwf
    rgb_list = []

    # inference nerfw and save rgb, target, pose
    for batch_idx in range(pose_perturb.shape[0]):
        # if batch_idx % 10 == 0:
            # print("renders RVS {}/total {}".format(batch_idx, pose_perturb.shape[0]))
        # breakpoint()
        pose = pose_perturb[batch_idx]
        img_idx = img_idxs[batch_idx].to(device)
        pose_nerf = pose.clone()
        # if args.NeRFW:
        # rescale the predicted pose to nerf scales
        pose_nerf = fix_coord_supp(args, pose_nerf[None,...].cpu(), world_setup_dict)

        # generate nerf image
        with torch.no_grad():
            torch.set_default_device('cuda')
            torch.set_default_dtype(torch.float32)
            if args.tinyimg:
                rgb, _, _, _ = render(int(H//args.tinyscale), int(W//args.tinyscale), focal/args.tinyscale, chunk=args.chunk, c2w=pose_nerf[0,:3,:4].to(device), retraw=False, img_idx=img_idx, **render_kwargs_test)
                if args.encode_hist and NFF:
                    affine_color_transform = render_kwargs_test['network_fn'].affine_color_transform
                    rgb = affine_color_transform(args, rgb, img_idx, 1)
                rgb = rgb.reshape((int(H//args.tinyscale), int(W//args.tinyscale), 3))
                # convert rgb to B,C,H,W
                rgb = rgb[None,...].permute(0,3,1,2)
                # upsample rgb to hwf size
                rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
                # convert rgb back to H,W,C format
                rgb = rgb[0].permute(1,2,0)

            else:
                rgb, _, _, _ = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf[0,:3,:4].to(device), retraw=False, img_idx=img_idx, **render_kwargs_test)
            torch.set_default_device('cpu')
            torch.set_default_dtype(torch.float32)
        rgb_list.append(rgb.cpu())

    rgbs = torch.stack(rgb_list).detach()
    return rgbs


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        batch_size = tensor.size(0)
        for i in range(batch_size):
            for t, m, s in zip(tensor[i], self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
        return tensor

class L2Loss(nn.Module):
    """
    simple L_2-Loss
    """
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, input, target):
        return torch.sqrt(torch.mean(torch.abs(input - target).pow(2)))

def PoseLoss(args, pose_, pose, device):
    loss_func = nn.MSELoss()
    predict_pose = pose_.to(device) # maynot need reshape
    pose_loss = loss_func(predict_pose, pose)
    return pose_loss

def to_img(x):
    x = (x + 1.) * 0.5
    x = x.clamp(0, 1)
    return x

def masked_mse_loss(f1, f2, valid_mask):
    ''' 
    compute loss only in masked region, with MSE loss
    :param criterion: loss function
    :param f1: [N, C, H, W]
    :param f2: [N, C, H, W]
    :param valid_mask: [N, H, W]
    :return:
        loss
    '''
    # filter out masked area
    valid_mask = valid_mask[0].reshape(-1) # [H*W]
    valid_mask_ind = torch.nonzero(valid_mask>0, as_tuple=True)[0] # [N_valid]

    # apply mask on f1
    # f1 = f1 * (valid_mask[:,None,...])
    # f2 = f2
    N, C, H, W = f1.shape
    f1 = f1.reshape(N,C,H*W)[...,valid_mask_ind] # [N, C, N_valid]
    f2 = f2.reshape(N,C,H*W)[...,valid_mask_ind]

    # correct implementation
    # mse_loss = nn.MSELoss(reduction='none')
    # loss = mse_loss(f1, f2)
    # loss = (loss * valid_mask[:,None,...]).sum()
    # loss = loss / torch.sum(valid_mask)

    mse_loss = nn.MSELoss(reduction='mean')
    loss = mse_loss(f1, f2)
    return loss

def masked_triplet_loss(f1, f2, f3, valid_mask):
    ''' 
    compute loss only in masked region. 
    For simplicity, here we assume each warped image in the batch is warped from the same homo matrix
    :param f1: GT image features, full resolution. [N, C, H, W]
    :param f2: Inverse Warped Image Features, with blank area caused by inverse warp [N, C, H, W]
    :param f2: Warped Image Features, used for negative samples [N, C, H, W]
    :param valid_mask: [N, H, W] # we assume valid mask is the same for all images in the batch
    :return:
        loss
    '''

    # filter out masked area
    valid_mask = valid_mask[0].reshape(-1) # [H*W]
    valid_mask_ind = torch.nonzero(valid_mask>0, as_tuple=True)[0] # [N_valid]

    N, C, H, W = f1.shape
    # # visualize masked features
    # plot_features(f1, 'f1.png', False)
    # plot_features(f2, 'f2.png', False)

    # apply mask on f1
    # f1 = f1.reshape(N,C,H*W)[...,valid_mask_ind] # [N, C, N_valid]
    # f2 = f2.reshape(N,C,H*W)[...,valid_mask_ind]
    # f3 = f3.reshape(N,C,H*W)[...,valid_mask_ind]

    anchor = f1
    positive = f2
    negative = f3

    # #my implementation for debug
    # tmp1 = torch.sqrt(((anchor-positive)**2.0).sum(dim=-1))
    # tmp2 = torch.sqrt(((anchor-negative)**2.0).sum(dim=-1))
    # tmp_triplet = torch.mean(torch.max(tmp1-tmp2+1.0, torch.zeros_like(tmp1))) # 0.9979
    # breakpoint()

    # triplet loss
    criterion = nn.TripletMarginLoss(margin=1., reduction='mean')
    loss = criterion(anchor, positive, negative)
    return loss

def triplet_loss(f1, f2, margin=1.):
    ''' 
    naive implementation of triplet loss
    :param criterion: loss function
    :param f1: [lvl, B, C, H, W]
    :param f2: [lvl, B, C, H, W]
    :return:
        loss
    '''
    criterion = nn.TripletMarginLoss(margin=margin, reduction='mean')
    anchor = f1
    positive = f2
    negative = torch.roll(f2, shifts=1, dims=1)
    loss = criterion(anchor, positive, negative)
    return loss

def triplet_loss_hard_negative_mining(f1, f2, margin=1.):
    ''' 
    triplet loss with hard negative mining, inspired by http://www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf section3.3 
    :param criterion: loss function
    :param f1: [lvl, B, C, H, W]
    :param f2: [lvl, B, C, H, W]
    :return:
        loss
    '''
    criterion = nn.TripletMarginLoss(margin=margin, reduction='mean')
    anchor = f1
    anchor_negative = torch.roll(f1, shifts=1, dims=1)
    positive = f2
    negative = torch.roll(f2, shifts=1, dims=1)

    # select in-triplet hard negative, reference: section3.3 
    mse = nn.MSELoss(reduction='mean')
    with torch.no_grad():
        case1 = mse(anchor, negative)
        case2 = mse(positive, anchor_negative)
    
    # perform anchor swap if necessary
    if case1 < case2:
        loss = criterion(anchor, positive, negative)
    else:
        loss = criterion(positive, anchor, anchor_negative)
    return loss

def triplet_loss_hard_negative_mining_plus(f1, f2, margin=1.):
    ''' 
    triplet loss with hard negative mining, four cases. inspired by http://www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf section3.3 
    :param criterion: loss function
    :param f1: [lvl, B, C, H, W] or [B, C, H, W]
    :param f2: [lvl, B, C, H, W] or [B, C, H, W]
    :return:
        loss
    '''

    criterion = nn.TripletMarginLoss(margin=margin, reduction='mean')

    assert(len(f1.shape)==4 or len(f1.shape)==5)
    if len(f1.shape) == 5:
        anchor = f1
        anchor_negative = torch.roll(f1, shifts=1, dims=1)
        positive = f2
        negative = torch.roll(f2, shifts=1, dims=1)
    elif len(f1.shape) == 4:
        anchor = f1
        anchor_negative = torch.roll(f1, shifts=1, dims=0)
        positive = f2
        negative = torch.roll(f2, shifts=1, dims=0)

    # breakpoint()
    # #my implementation for debugging
    # tmp1 = torch.sqrt(((anchor-positive)**2.0).sum(dim=-1))
    # tmp2 = torch.sqrt(((anchor-negative)**2.0).sum(dim=-1))
    # tmp_triplet = torch.mean(torch.max(tmp1-tmp2+1.0, torch.zeros_like(tmp1))) # 0.9979
    # breakpoint()

    # select in-triplet hard negative, reference: section3.3 
    mse = nn.MSELoss(reduction='mean')
    with torch.no_grad():
        case1 = mse(anchor, negative)
        case2 = mse(positive, anchor_negative)
        case3 = mse(anchor, anchor_negative)
        case4 = mse(positive, negative)
        distance_list = torch.stack([case1,case2,case3,case4])
        loss_case = torch.argmin(distance_list)
    
    # perform anchor swap if necessary
    # print("loss_case:", loss_case)
    if loss_case == 0:
        loss = criterion(anchor, positive, negative)
    elif loss_case == 1:
        loss = criterion(positive, anchor, anchor_negative)
    elif loss_case == 2:
        loss = criterion(anchor, positive, anchor_negative)
    elif loss_case == 3:
        loss = criterion(positive, anchor, negative)
    else:
        raise NotImplementedError
    return loss

def normalize_features(f, value_range=None, scale_each: bool = False):
    ''' Find unit norm of channel wise feature 
        :param: f, feature tensor (C,H,W)
    '''
    tensor = f.clone()  # avoid modifying tensor in-place
    C,H,W = tensor.size()

    # normlaize the features with l2 norm
    tensor = tensor.reshape(C, H*W)
    tensor = torch.nn.functional.normalize(tensor)
    tensor = tensor.reshape(C,H,W)
    return tensor

def normalize_features2(f, value_range=None, scale_each: bool = False):
    ''' Find unit norm of channel wise feature 
        :param: f, feature tensor (lvl,B,C,H,W)
        return f, normalized feature tensor (lvl,B,C,H,W)
    '''
    f = f/(torch.norm(f, dim=2, keepdim=True) + 1e-8)
    return f

def triplet_loss_norm_along_channel(f1, f2, margin=1.):
    ''' 
    triplet loss with hard negative mining, four cases. inspired by http://www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf section3.3 
    :param criterion: loss function
    :param f1: [lvl, B, C, H, W]
    :param f2: [lvl, B, C, H, W]
    :return:
        loss
    '''
    # normalize features on per-pixel basis (along channel axis)
    f1 = normalize_features2(f1)
    f2 = normalize_features2(f2)

    criterion = nn.TripletMarginLoss(margin=margin, reduction='mean')
    anchor = f1
    anchor_negative = torch.roll(f1, shifts=1, dims=1)
    positive = f2
    negative = torch.roll(f2, shifts=1, dims=1)

    # select in-triplet hard negative, reference: section3.3 
    mse = nn.MSELoss(reduction='mean')
    with torch.no_grad():
        case1 = mse(anchor, negative)
        case2 = mse(positive, anchor_negative)
        case3 = mse(anchor, anchor_negative)
        case4 = mse(positive, negative)
        distance_list = torch.stack([case1,case2,case3,case4])
        loss_case = torch.argmin(distance_list)
    
    # perform anchor swap if necessary
    # print("loss_case:", loss_case)
    if loss_case == 0:
        loss = criterion(anchor, positive, negative)
    elif loss_case == 1:
        loss = criterion(positive, anchor, anchor_negative)
    elif loss_case == 2:
        loss = criterion(anchor, positive, anchor_negative)
    elif loss_case == 3:
        loss = criterion(positive, anchor, negative)
    else:
        raise NotImplementedError
    return loss

def perturb_rotation(c2w, theta, phi, psi=0):
    last_row = np.array([[0, 0, 0, 1]])# np.tile(np.array([0, 0, 0, 1]), (1, 1))  # (N_images, 1, 4)
    c2w = np.concatenate([c2w, last_row], 0)  # (N_images, 4, 4) homogeneous coordinate
    # print("c2w", c2w)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = rot_psi(psi/180.*np.pi) @ c2w
    c2w = c2w[:3,:4]
    # print("updated c2w", c2w)
    return c2w

def perturb_single_render_pose(poses, x, angle):
    """
    Inputs:
        poses: (3, 4)
        x: translational perturb range
        angle: rotation angle perturb range in degrees
    Outputs:
        new_c2w: (N_views, 3, 4) new poses
    """
    c2w=poses
    
    N_views = 1 # number of views 
    new_c2w = np.zeros((N_views, 3, 4))

    for i in range(N_views):
        new_c2w[i] = c2w
        loc = deepcopy(new_c2w[i,:,3]) # this is a must
        # print("new_c2w[i,:,3]", new_c2w[i,:,3])

        # perturb rotation pose
        rot_rand=np.random.uniform(-angle,angle,3) # in degrees
        theta, phi, psi = rot_rand

        new_c2w[i] = perturb_rotation(new_c2w[i], theta, phi, psi)
        # print("rot new_c2w[i,:,3]", new_c2w[i,:,3])

        # perturb translational pose
        trans_rand = np.random.uniform(-x,x,3) # random number of 3 axis pose perturbation
        # print("trans_rand", trans_rand)

        # # normalize 3 axis perturbation to x, this will make sure sum of 3-axis perturbation to be x constant
        # trans_rand = trans_rand / abs(trans_rand).sum() * x
        new_c2w[i,:,3] = loc + trans_rand # perturb pos between -1 to 1
        # print("new new_c2w[i,:,3]", new_c2w[i,:,3])

    return new_c2w

def perturb_single_render_pose_norm(poses, x, angle):
    """
    Inputs:
        poses: (3, 4)
        x: translational perturb range
        angle: rotation angle perturb range in degrees
    Outputs:
        new_c2w: (N_views, 3, 4) new poses
    """
    c2w=poses
    
    N_views = 1 # number of views 
    new_c2w = np.zeros((N_views, 3, 4))

    # perturb translational pose
    for i in range(N_views):
        new_c2w[i] = c2w
        trans_rand = np.random.uniform(-x,x,3) # random number of 3 axis pose perturbation

        # normalize 3 axis perturbation to x, this will make sure sum of 3-axis perturbation to be x constant
        trans_rand = trans_rand / abs(trans_rand).sum() * x
        new_c2w[i,:,3] = new_c2w[i,:,3] + trans_rand # perturb pos between -1 to 1

        # perturb rotation pose
        theta=np.random.uniform(-angle,angle,1) # in degrees
        phi=np.random.uniform(-angle,angle,1) # in degrees
        psi=np.random.uniform(-angle,angle,1) # in degrees
        
        rot_rand = np.array([theta, phi, psi])
        rot_rand = rot_rand / abs(rot_rand).sum() * angle
        theta, phi, psi = rot_rand

        # normalize 3 axis perturbation to x, this will make sure sum of 3-axis perturbation to be x constant
        new_c2w[i] = perturb_rotation(new_c2w[i], theta, phi, psi)
    return new_c2w

def get_validation_loss(args, feat_model, val_dl, hwf, render_kwargs_test, world_setup_dict, device, NFF=False, verbose=False, option_list=[]):
    """
    option_list: ['pose', 'feature_mse', 'feature_triplet', 'feature_cosine_similarity']
    """
    # validation
    assert(len(option_list)!=0)
    feat_model.eval()
    loss_dict = {}
    mse = nn.MSELoss(reduction='mean')

    # compute validation loss
    if 'pose' in option_list: # Pose Loss as val
        val_loss_epoch = []
        for batch_data in val_dl:
            data = batch_data['img']
            pose = batch_data['pose']

            inputs = data.to(device)
            labels = pose.to(device)
            _, predict = feat_model(inputs)
            loss = mse(predict, labels)
            val_loss_epoch.append(loss.item())
        val_loss = np.mean(val_loss_epoch)
        loss_dict['poss'] = val_loss

    if 'feature_mse' in option_list or 'feature_triplet' in option_list or 'feature_cosine_similarity' in option_list:

        if 'feature_mse' in option_list: # mse Feature Loss as val
            targets_val, rgbs_val, _, _ = render_nerfw_imgs(args, val_dl, hwf, device, render_kwargs_test, world_setup_dict, NFF=NFF, verbose=False)
            targets_val = tensor_nhwc2nchw(targets_val)
            rgbs_val = tensor_nhwc2nchw(rgbs_val)
            val_loss_epoch = []
            for i in range(targets_val.shape[0]):
                val_in = torch.stack([targets_val[i], rgbs_val[i]], dim=0).to(device) # [2, 3, H, W]
                features, _ = feat_model(val_in, return_feature=True, isSingleStream=False, return_pose=False, upsampleH=hwf[0], upsampleW=hwf[1]) # {N, lvl, B, C, H, W}
                f_label = features[0][0] # [1,C,H,W]
                f_predict = features[1][0] # [1,C,H,W]
                loss = mse(f_predict, f_label)
                val_loss_epoch.append(loss.item())
            val_loss = np.mean(val_loss_epoch)
            loss_dict['feature_mse'] = val_loss
            # clean GPU memory after testing
            torch.cuda.empty_cache()
        
        if 'feature_triplet' in option_list:
            targets_val, rgbs_val, _, _ = render_nerfw_imgs(args, val_dl, hwf, device, render_kwargs_test, world_setup_dict, NFF=NFF, verbose=False)
            targets_val = tensor_nhwc2nchw(targets_val)
            rgbs_val = tensor_nhwc2nchw(rgbs_val)
            val_loss_epoch = []
            B_size = args.featurenet_batch_size
            for i in range(0, targets_val.shape[0], B_size):
                features_labels = []
                features_predict = []
                for j in range(B_size):
                    val_in = torch.stack([targets_val[j], rgbs_val[j]], dim=0).to(device) # [2, 3, H, W]
                    features, _ = feat_model(val_in, return_feature=True, isSingleStream=False, return_pose=False, upsampleH=hwf[0], upsampleW=hwf[1]) # {N, lvl, B, C, H, W}
                    features_labels.append(features[0]) # [lvl,1,C,H,W]
                    features_predict.append(features[1]) # [lvl,1,C,H,W]
                features_labels = torch.stack(features_labels)
                features_predict = torch.stack(features_predict)
                f_label = features_labels[:,0,0] # [B,C,H,W]
                f_predict = features_predict[:,0,0] # [B,C,H,W]
                loss = triplet_loss_hard_negative_mining_plus(f_predict, f_label, margin=args.triplet_margin)
                val_loss_epoch.append(loss.item())
            val_loss = np.mean(val_loss_epoch)
            loss_dict['feature_triplet'] = val_loss
            # clean GPU memory after testing
            torch.cuda.empty_cache()

        if 'feature_cosine_similarity' in option_list:
            from dm.DFM_pose_refine import feature_loss
            targets_val, rgbs_val, _, _ = render_nerfw_imgs(args, val_dl, hwf, device, render_kwargs_test, world_setup_dict, NFF=NFF, verbose=False)
            targets_val = tensor_nhwc2nchw(targets_val)
            rgbs_val = tensor_nhwc2nchw(rgbs_val)
            val_loss_epoch = []
            for i in range(targets_val.shape[0]):
                val_in = torch.stack([targets_val[i], rgbs_val[i]], dim=0).to(device) # [2, 3, H, W]
                features, _ = feat_model(val_in, return_feature=True, isSingleStream=False, return_pose=False, upsampleH=hwf[0], upsampleW=hwf[1]) # {N, lvl, B, C, H, W}
                f_label = features[0][0,0]
                f_predict = features[1][0,0]
                loss = feature_loss(f_predict, f_label, img_in=True, per_pixel=True)
                val_loss_epoch.append(loss.item())
            val_loss = np.mean(val_loss_epoch)
            loss_dict['feature_cosine_similarity'] = val_loss
            # clean GPU memory after testing
            torch.cuda.empty_cache()

    return loss_dict

def save_dfnet_results_to_txt(args, model, test_dl, device):
    # save dfnet results to txt
    predict_pose_list = []
    for batch in test_dl:
        if (type(batch) is dict):
            data = batch['img']
            pose = batch['pose']
        data = data.to(device) # input
        pose = pose.reshape((1,3,4)).numpy() # label

        with torch.no_grad():
            if args.PoseEstimatorType=='DFNet':
                _, predict_pose = model(data)
            else:
                predict_pose = model(data)
            R_torch = predict_pose.reshape((1, 3, 4))[:,:3,:3] # debug
            predict_pose = predict_pose.reshape((1, 3, 4)).cpu().numpy()

            R = predict_pose[:,:3,:3]
            u,s,v=torch.svd(R_torch)
            Rs = torch.matmul(u, v.transpose(-2,-1))
        predict_pose[:,:3,:3] = Rs[:,:3,:3].cpu().numpy()
        predict_pose_list.append(predict_pose)
    pose_results = np.concatenate(predict_pose_list).reshape(-1,12)

    test_gt_pose = test_dl.dataset.poses
    test_gt_c_imgs = test_dl.dataset.c_imgs

    scene = osp.split(args.datadir)[-1]

    if osp.exists(f'tmp/{args.PoseEstimatorType}_{args.dataset_type}') is False:
        os.makedirs(f'tmp/{args.PoseEstimatorType}_{args.dataset_type}')
    # save APR predicted results
    np.savetxt(os.path.join(f'tmp/{args.PoseEstimatorType}_{args.dataset_type}/{args.PoseEstimatorType}_{scene}_results.txt'), pose_results)
    # save ground truth
    np.savetxt(os.path.join(f'tmp/{args.PoseEstimatorType}_{args.dataset_type}/{scene}_test_gt.txt'), test_gt_pose)
    # save ground truth image names
    np.savetxt(os.path.join(f'tmp/{args.PoseEstimatorType}_{args.dataset_type}/{scene}_test_gt_filename.txt'), test_gt_c_imgs, fmt='%s')

def save_train_gt_to_txt(args, train_dl):
    train_gt_pose = train_dl.dataset.poses
    train_gt_c_imgs = train_dl.dataset.c_imgs
    scene = osp.split(args.datadir)[-1]
    if osp.exists(f'tmp/{args.PoseEstimatorType}_{args.dataset_type}') is False:
        os.makedirs(f'tmp/{args.PoseEstimatorType}_{args.dataset_type}')
    # save ground truth
    np.savetxt(os.path.join(f'tmp/{args.PoseEstimatorType}_{args.dataset_type}/{scene}_train_gt.txt'), train_gt_pose)
    np.array(train_gt_c_imgs)
    # save ground truth image names
    np.savetxt(os.path.join(f'tmp/{args.PoseEstimatorType}_{args.dataset_type}/{scene}_train_gt_filename.txt'), train_gt_c_imgs, fmt='%s')