'''
helper functions to train robust feature extractors
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from math import pi
import cv2

class SSIM(nn.Module):
    """Layer to compute the SSIM between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        k = 7
        self.mu_x_pool = nn.AvgPool2d(k, 1)
        self.mu_y_pool = nn.AvgPool2d(k, 1)
        self.sig_x_pool = nn.AvgPool2d(k, 1)
        self.sig_y_pool = nn.AvgPool2d(k, 1)
        self.sig_xy_pool = nn.AvgPool2d(k, 1)

        self.refl = nn.ReflectionPad2d(k//2)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
            (sigma_x + sigma_y + self.C2)

        # return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1) # this is DSSIM, disimilarity of SSIM. Or SSIM loss
        return torch.clamp((SSIM_n/SSIM_d), 0, 1)

def freeze_bn_layer(model):
    ''' freeze bn layer by not require grad but still behave differently when model.train() vs. model.eval() '''
    print("Freezing BatchNorm Layers...")
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            # print("this is a BN layer:", module)
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
    return model

def freeze_bn_layer_train(model):
    ''' set batchnorm to eval() 
        it is useful to align train and testing result 
    '''
    # model.train()
    # print("Freezing BatchNorm Layers...")
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    return model

def save_image_saliancy(tensor, path, normalize: bool = False, scale_each: bool = False,):
    """
    Modification based on TORCHVISION.UTILS
    ::param: tensor (batch, channel, H, W)
    """
    # grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=32)
    grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=6)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    fig = plt.figure()
    plt.imshow(ndarr[:,:,0], cmap='jet') # viridis, plasma
    plt.axis('off')
    fig.savefig(path, bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    plt.close()

def save_image_saliancy_single(tensor, path, normalize: bool = False, scale_each: bool = False,):
    """
    Modification based on TORCHVISION.UTILS, save single feature map
    ::param: tensor (batch, channel, H, W)
    """
    # grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=32)
    grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=1)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    fig = plt.figure()
    # plt.imshow(ndarr[:,:,0], cmap='plasma') # viridis, jet
    plt.imshow(ndarr[:,:,0], cmap='jet') # viridis, jet
    plt.axis('off')
    fig.savefig(path, bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    plt.close()

def print_feature_examples(features, path):
    """
    print feature maps
    ::param: features
    """
    kwargs = {'normalize' : True, } # 'scale_each' : True
    
    for i in range(len(features)):
        fn = path + '{}.png'.format(i)
        # save_image(features[i].permute(1,0,2,3), fn, **kwargs)
        save_image_saliancy(features[i].permute(1,0,2,3), fn, normalize=True)
    # pdb.set_trace()
    ###

def plot_features(features, path='f', isList=True):
    """
    print feature maps
    :param features: (3, [batch, H, W]) or [3, batch, H, W]
    :param path: save image path
    :param isList: wether the features is an list
    :return:
    """
    kwargs = {'normalize' : True, } # 'scale_each' : True
    
    if isList:
        dim = features[0].dim()
    else:
        dim = features.dim()
    assert(dim==3 or dim==4)

    if dim==4 and isList:
        print_feature_examples(features, path)
    elif dim==4 and (isList==False):
        fn = path
        lvl, b, H, W = features.shape
        for i in range(features.shape[0]):
            fn = path + '{}.png'.format(i)
            save_image_saliancy(features[i][None,...].permute(1,0,2,3).cpu(), fn, normalize=True)

        # # concat everything
        # features = features.reshape([-1, H, W])
        # # save_image(features[None,...].permute(1,0,2,3).cpu(), fn, **kwargs)
        # save_image_saliancy(features[None,...].permute(1,0,2,3).cpu(), fn, normalize=True) 

    elif dim==3 and isList: # print all images in the list
        for i in range(len(features)):
            fn = path + '{}.png'.format(i)
            # save_image(features[i][None,...].permute(1,0,2,3).cpu(), fn, **kwargs)
            save_image_saliancy(features[i][None,...].permute(1,0,2,3).cpu(), fn, normalize=True)
    elif dim==3 and (isList==False):
            fn = path
            save_image_saliancy(features[None,...].permute(1,0,2,3).cpu(), fn, normalize=True)

def warp_points(points, homographies, device='cpu'):
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2(x, y))).
        homography: batched or not (shapes (B, 3, 3) and (...) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2(x, y)) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.

    """
    # expand points len to (x, y, 1)
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies

    batch_size = homographies.shape[0]
    points = torch.cat((points.float(), torch.ones((points.shape[0], 1)).to(device)), dim=1)
    points = points.to(device)
    homographies = homographies.view(batch_size*3,3)

    warped_points = homographies@points.transpose(0,1)

    # normalize the points
    warped_points = warped_points.view([batch_size, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points[0,:,:] if no_batches else warped_points

def inv_warp_image_batch(img, mat_homo_inv, device='cuda', mode='bilinear'):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [batch_size, 1, H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [batch_size, 3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [batch_size, 1, H, W]
    '''
    # compute inverse warped points
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1,1,img.shape[0], img.shape[1])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1,3,3)

    Batch, channel, H, W = img.shape
    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H), indexing='ij'), dim=2)
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous()

    src_pixel_coords = warp_points(coor_cells.view([-1, 2]), mat_homo_inv, device)
    src_pixel_coords = src_pixel_coords.view([Batch, H, W, 2])
    src_pixel_coords = src_pixel_coords.float()

    warped_img = F.grid_sample(img, src_pixel_coords, mode=mode, align_corners=True)
    return warped_img

def compute_valid_mask(image_shape, inv_homography, device='cpu', erosion_radius=0):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.

    Arguments:
        input_shape: Tensor of rank 2 representing the image shape, i.e. `[H, W]`.
        homography: Tensor of shape (B, 8) or (8,), where B is the batch size.
        `erosion_radius: radius of the margin to be discarded.

    Returns: a Tensor of type `tf.int32` and shape (H, W).
    """

    if inv_homography.dim() == 2:
        inv_homography = inv_homography.view(-1, 3, 3)
    batch_size = inv_homography.shape[0]
    mask = torch.ones(batch_size, 1, image_shape[0], image_shape[1]).to(device)
    mask = inv_warp_image_batch(mask, inv_homography, device=device, mode='nearest')
    mask = mask.view(batch_size, image_shape[0], image_shape[1])
    mask = mask.cpu().numpy()
    if erosion_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius*2,)*2)
        for i in range(batch_size):
            mask[i, :, :] = cv2.erode(mask[i, :, :], kernel, iterations=1)

    return torch.tensor(mask).to(device)

def Kalman1D(observations,damping=1):
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1
    initial_value_guess
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state

def Kalman3D(observations,damping=1):
    '''
    In:
    observation: Nx3
    Out:
    pred_state: Nx3
    '''
    # To return the smoothed time series data
    observation_covariance = damping
    transition_matrix = 1
    transition_covariance = 0.1
    initial_value_guess_x = observations[0,0]
    initial_value_guess_y = observations[0,1] # ?
    initial_value_guess_z = observations[0,2] # ?
    
    # perform 1D smooth for each axis
    kfx = KalmanFilter(
            initial_state_mean=initial_value_guess_x,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state_x, state_cov_x = kfx.smooth(observations[:, 0])
    
    kfy = KalmanFilter(
            initial_state_mean=initial_value_guess_y,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state_y, state_cov_y = kfy.smooth(observations[:, 1])
    
    kfz = KalmanFilter(
            initial_state_mean=initial_value_guess_z,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state_z, state_cov_z = kfy.smooth(observations[:, 2])
    
    pred_state = np.concatenate((pred_state_x, pred_state_y, pred_state_z), axis=1)
    return pred_state

def tensor_hwc2nchw(t):
    """
    permute a torch tensor from H x W x C to N x C x H x W, where N=1

    Arguments:
        t: Tensor of [H, W, C].

    Returns: Tensor of [1, C, H, W].
    """
    assert(len(t.size()) == 3)
    return t[None,...].permute(0,3,1,2)

def tensor_nhwc2nchw(t):
    """
    permute a torch tensor from N x H x W x C to N x C x H x W, where N>=1

    Arguments:
        t: Tensor of [N, H, W, C].

    Returns: Tensor of [N, C, H, W].
    """
    assert(len(t.size()) == 4)
    return t.permute(0,3,1,2)

def tensor_nchw2hwc(t):
    """
    permute a torch tensor from N x C x H x W to H x W x C, where N must be 1

    Arguments:
        t: Tensor of [1, C, H, W]

    Returns: Tensor of [H, W, C]
    """
    assert(len(t.size()) == 4)
    return t[0].permute(1,2,0)
    

def tensor_nchw2nhwc(t):
    """
    permute a torch tensor from N x C x H x W to N x H x W x C, where N >=1

    Arguments:
        t: Tensor of [N,C,H,W].

    Returns: Tensor of [N,H,W,C]
    """
    assert(len(t.size()) == 4)

    return t.permute(0,2,3,1)

def save_nchwimg(t, path):
    """
    save a pytorch nchw image
    Arguments:
        t: Tensor of [N,C,H,W].
        path: path to save the image.
    """
    assert(len(t.size()) == 4)
    from models.nerfh import to8b
    t = tensor_nchw2hwc(t)
    t = to8b(t.detach().cpu().numpy())
    import imageio
    imageio.imwrite(path, t)

def inv_warp_imgs(imgs, homographies, batch_size, device):
    ''' 
    inverse the homography matrices and warped the features
    :param imgs: batch of imgs in tensor [batch_size, C, H, W]
    :param homographies: [batch_size, 3, 3]
    :param batch_size: batch_size
    :return:
        inv_warped_features [batch_size, C, H, W]
        inv_homographies [batch_size, 3, 3]
    '''
    inv_homographies = torch.stack([torch.inverse(homographies[i, :, :]) for i in range(batch_size)]).to(device)
    inv_warped_imgs = inv_warp_image_batch(imgs, inv_homographies, device=device, mode='bilinear').squeeze() 

    return inv_warped_imgs, inv_homographies

def inv_warp_features(features, homographies, batch_size, device):
    ''' 
    inverse the homography matrices and warped the features
    :param features: batch of features in list (3, [batch_size, H, W])
    :param homographies: [batch_size, 3, 3]
    :param batch_size: batch_size
    :return:
        inv_warped_features (3, [batch_size, H, W])
        inv_homographies [batch_size, 3, 3]
    '''

    inv_homographies = torch.stack([torch.inverse(homographies[i, :, :]) for i in range(batch_size)]).to(device)
    inv_warped_features = inv_warp_image_batch(features, inv_homographies, device=device, mode='bilinear').squeeze()

    return inv_warped_features, inv_homographies

def sample_homography_np(
        shape, shift=0, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
        perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=pi/2,
        allow_artifacts=False, translation_overflow=0.):
    """Sample a random valid homography.

    Computes the homography transformation between a random patch in the original image
    and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
    transformed input point (original patch).
    The original patch, which is initialized with a simple half-size centered crop, is
    iteratively projected, scaled, rotated and translated.

    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography. (like crop size)
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.

    Returns:
        A `Tensor` of shape `[1, 8]` corresponding to the flattened homography transform.
    """

    # Corners of the output image
    pts1 = np.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + np.array([[0, 0], [0, patch_ratio],
                                 [patch_ratio, patch_ratio], [patch_ratio, 0]])

    from numpy.random import normal
    from numpy.random import uniform
    from scipy.stats import truncnorm

    # Random perspective and affine perturbations
    # lower, upper = 0, 2
    std_trunc = 2

    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        perspective_displacement = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_y/2).rvs(1)
        h_displacement_left = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        h_displacement_right = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = truncnorm(-1*std_trunc, std_trunc, loc=1, scale=scaling_amplitude/2).rvs(n_scales)
        scales = np.concatenate((np.array([1]), scales), axis=0)

        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if allow_artifacts:
            valid = np.arange(n_scales)  # all scales are valid except scale=1
        else:
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx,:,:]

    # Random translation
    if translation:
        # pdb.set_trace()
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += np.array([uniform(-t_min[0], t_max[0],1), uniform(-t_min[1], t_max[1], 1)]).T

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, num=n_angles)
        angles = np.concatenate((angles, np.array([0.])), axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul( (pts2 - center)[np.newaxis,:,:], rot_mat) + center
        if allow_artifacts:
            valid = np.arange(n_angles)  # all scales are valid except scale=1
        else: # find multiple valid option and choose the valid one
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx,:,:]

    # Rescale to actual size
    shape = shape[::-1]  # different convention [y, x]
    pts1 *= shape[np.newaxis,:]
    pts2 *= shape[np.newaxis,:]

    homography = cv2.getPerspectiveTransform(np.float32(pts1+shift), np.float32(pts2+shift))
    return homography

def create_warped_img(img, batch_size, device):
    ''' 
    randomly generate homographies and warp the img 
    :param img: batch of images [batch_size, 3, H, W]
    :param batch_size: batch_size
    :return:
        batch of warped images
        tensor [batch_size, 3, 3]
    '''
    # create warped img
    # different homographies for each image in the batch
    # homographies = np.stack([sample_homography_np(np.array([2, 2]), shift=-1, 
    #     perspective=True, scaling=True, rotation=True, translation=True,
    #     n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.2,
    #     perspective_amplitude_y=0.2, patch_ratio=0.8, max_angle=pi/6,
    #     allow_artifacts=False, translation_overflow=0.
    #     ) for i in range(batch_size)])

    # same homographies for all images in the batch
    homographies = sample_homography_np(np.array([2, 2]), shift=-1, 
        perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.2,
        perspective_amplitude_y=0.2, patch_ratio=0.8, max_angle=pi/6,
        allow_artifacts=False, translation_overflow=0.
        )
    homographies = np.stack([homographies for i in range(batch_size)])

    homographies = torch.tensor(homographies, dtype=torch.float32, device=device)
    warped_img = inv_warp_image_batch(img.squeeze(), homographies, mode='bilinear', device=device).unsqueeze(0)
    warped_img = warped_img.squeeze()

    # save_image(img, './tmp.png')
    # save_image(warped_img, './tmp_warped.png')
    # inv_warped_img, inv_homographies = inv_warp_imgs(warped_img.to(device), homographies, batch_size, device)
    # save_image(inv_warped_img, './tmp_inv_warped_img.png')
    # breakpoint()
    return warped_img, homographies

def compute_none_ATE_error(pose1, pose2, dl):
    '''
    plot and compute pose error from two trajectories, without ATE alignment
    :param pose1/refined_pose:  (N0, 3/4, 4) torch tensor
    :param pose2/gt_pose:  (N0, 3/4, 4) torch tensor
    :param dl: dataloader
    '''

    from dm.pose_model import compute_pose_error, vis_pose
    assert(pose1.shape == pose2.shape)
    results2 = np.zeros((pose2.shape[0], 2))
    ind2 = 0

    pose1_list = []
    pose2_list = []
    ang_error_list = []

    for i in range(pose2.shape[0]):

        poses_gt = pose2[i:i+1]
        poses_pred = pose1[i:i+1]

        pose1_list.append(poses_pred[:,:3,3].squeeze())
        pose2_list.append(poses_gt[:,:3,3].squeeze())

        error_x, theta = compute_pose_error(poses_gt, poses_pred)
        results2[ind2,:] = [error_x, theta]

        ang_error_list.append(theta)
        ind2 += 1

    median_result = np.median(results2,axis=0)
    mean_result = np.mean(results2,axis=0)
    # standard log
    print ('pose Median error {}m and {} degrees.'.format(median_result[0], median_result[1]))
    print ('pose Mean error {}m and {} degrees.'.format(mean_result[0], mean_result[1]))

    pose1_list = np.array(torch.stack(pose1_list))
    pose2_list = np.array(torch.stack(pose2_list))
    ang_error_list = np.array(ang_error_list)
    vis_info_ret = {"pose": pose1_list, "pose_gt": pose2_list, "theta": ang_error_list}
    vis_pose(vis_info_ret)

def compute_ATE_error(refined_pose, gt_pose, dl):
    '''
    plot and compute ATE pose error from two trajectories
    :param pose1/refined_pose:  (N0, 3/4, 4) torch tensor
    :param pose2/gt_pose:  (N0, 3/4, 4) torch tensor
    :param dl: dataloader
    '''
    ### Apply ATE alignment
    from utils.align_traj import align_ate_c2b_use_a2b # compute_ate
    from dm.pose_model import compute_pose_error, vis_pose

    # ATE Align poses
    refined_pose_aligned = align_ate_c2b_use_a2b(refined_pose, gt_pose)  # (N, 4, 4)
    
    results2 = np.zeros((dl.dataset.poses.shape[0], 2))
    ind2 = 0

    predict_pose_list2 = []
    gt_pose_list2 = []
    ang_error_list2 = []

    for i in range(dl.dataset.poses.shape[0]):

        poses_gt = gt_pose[i:i+1]
        poses_pred = refined_pose_aligned[i:i+1]

        predict_pose_list2.append(poses_pred[:,:3,3].squeeze())
        gt_pose_list2.append(poses_gt[:,:3,3].squeeze())

        error_x, theta = compute_pose_error(poses_gt, poses_pred)
        results2[ind2,:] = [error_x, theta]

        ang_error_list2.append(theta)
        ind2 += 1

    median_result = np.median(results2,axis=0)
    mean_result = np.mean(results2,axis=0)
    # standard log
    print ('nerfmm pose ATE Median error {}m and {} degrees.'.format(median_result[0], median_result[1]))
    print ('nerfmm pose ATE Mean error {}m and {} degrees.'.format(mean_result[0], mean_result[1]))

    predict_pose_list2 = np.array(torch.stack(predict_pose_list2))
    gt_pose_list2 = np.array(torch.stack(gt_pose_list2))
    ang_error_list2 = np.array(ang_error_list2)
    vis_info_ret = {"pose": predict_pose_list2, "pose_gt": gt_pose_list2, "theta": ang_error_list2}
    vis_pose(vis_info_ret)

# x rotation
rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).astype(float)

# y rotation
rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).astype(float)

# z rotation
rot_psi = lambda psi : np.array([
    [np.cos(psi),-np.sin(psi),0,0],
    [np.sin(psi),np.cos(psi),0,0],
    [0,0,1,0],
    [0,0,0,1]]).astype(float)

# def perturb_rotation(c2w, theta, phi, psi=0):
#     last_row = np.tile(np.array([0, 0, 0, 1]), (1, 1))  # (N_images, 1, 4)
#     trans = c2w[:3, 3]
#     c2w = np.concatenate([c2w, last_row], 0)  # (N_images, 4, 4) homogeneous coordinate
#     c2w = rot_phi(phi/180.*np.pi) @ c2w
#     c2w = rot_theta(theta/180.*np.pi) @ c2w
#     c2w = rot_psi(psi/180.*np.pi) @ c2w
#     c2w = c2w[:3,:4]
#     c2w[:3, 3] = trans
#     return c2w

def perturb_rotation(c2w, delta_rot):
    c2w[:3,:3] = delta_rot @ c2w[:3,:3]
    return c2w

def perturb_render_pose(poses, x, angle):
    """
    Inputs:
        poses: (3, 4)
        x: translational perturb range
        angle: rotation angle perturb range in degrees
    Outputs:
        new_c2w: (N_views, 3, 4) new poses
    """
    idx = np.random.choice(poses.shape[0])
    c2w=poses[idx]
    # c2w=poses
    
    N_views = 5 # number of views in video    
    new_c2w = np.zeros((N_views, 3, 4))

    for i in range(N_views):
        new_c2w[i] = c2w

         # perturb translational pose
        if x < 1e-9:
            trans_rand=np.zeros(3)
        else:
            trans_rand = np.random.uniform(-x,x,3) # random number of 3 axis pose perturbation
            trans_rand = x * trans_rand/np.linalg.norm(trans_rand)
            # trans_rand = x,y,z,/ np.linalg.norm(np.array([x, y, z]))

            # normalize 3 axis perturbation to x, this will make sure sum of 3-axis perturbation to be x constant
            # trans_rand = trans_rand / abs(trans_rand).sum() * x
            new_c2w[i,:,3] = new_c2w[i,:,3] + trans_rand # perturb pos between -1 to 1

        # breakpoint()
        # perturb rotation pose
        if angle < 1e-9:
            # theta, phi, psi = 0, 0, 0
            continue
        else:
            from scipy.spatial.transform import Rotation as R
            # random generate x y z
            xx = np.random.uniform(-1,1,1)
            yy = np.random.uniform(-1,1,1)
            zz = np.random.uniform(-1,1,1)
            vector = np.array([xx, yy, zz])/np.linalg.norm(np.array([xx, yy, zz]))
            vector = vector[...,0]
            r = R.from_rotvec(angle * vector,degrees=True) # np.array([x, y, z]) np.linalg.norm(np.array([x, y, z]))
            delta_rot = r.as_matrix()
            new_c2w[i] = perturb_rotation(new_c2w[i], delta_rot)

    return new_c2w, idx

def set_default_to_cuda():
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float32)

def set_default_to_cpu():
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float32)