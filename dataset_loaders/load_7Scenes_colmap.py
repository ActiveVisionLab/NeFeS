import os
import os.path as osp
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch.cuda
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_loaders.seven_scenes_colmap import SevenScenes_colmap
# from seven_scenes_colmap import SevenScenes_colmap # for local testing

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# translation z axis
trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).astype(float)

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

def initK(f, cx, cy):
    K = np.eye(3, 3)
    K[0, 0] = K[1, 1] = f
    K[0, 2] = cx
    K[1, 2] = cy
    return K

def perturb_rotation(c2w, theta, phi, psi=0):
    last_row = np.tile(np.array([0, 0, 0, 1]), (1, 1))  # (N_images, 1, 4)
    c2w = np.concatenate([c2w, last_row], 0)  # (N_images, 4, 4) homogeneous coordinate
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = rot_psi(psi/180.*np.pi) @ c2w
    c2w = c2w[:3,:4]
    return c2w

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)
    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)
    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)
    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)
    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    return pose_avg

def center_poses(poses, pose_avg_from_file=None):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)
        pose_avg_from_file: if not None, pose_avg is loaded from pose_avg_stats.txt

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """


    if pose_avg_from_file is None:
        pose_avg = average_poses(poses)  # (3, 4) # this need to be fixed throughout dataset
    else:
        pose_avg = pose_avg_from_file

    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation (4,4)
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    # print('poses_homo:', poses_homo)
    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    # # debug
    # inv_poses_centered = pose_avg_homo @ poses_centered
    # print('inv_poses_centered:', inv_poses_centered)
    # breakpoint()

    return poses_centered, pose_avg #np.linalg.inv(pose_avg_homo)

def fix_coord(args, train_set, val_set, pose_avg_stats_file='', rescale_coord=True):
    ''' fix coord for 7 Scenes to align with llff style dataset '''

    # get all poses (train+val)
    train_poses = train_set.poses
    val_poses = val_set.poses
    all_poses = np.concatenate([train_poses, val_poses])

    all_poses = all_poses.reshape(all_poses.shape[0], 3, 4)

    # Correct axis from COLMAP (OPENCV) coordinates -> LLFF coordinates -> NeRF coordinates
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(all_poses), 1, 1))  # (N_images, 1, 4)
    all_poses = np.concatenate([all_poses, last_row], 1)

    # COLMAP -> NeRF in 1 step: switch from [r, -u, t] to [r, u, -t]
    all_poses = np.concatenate([all_poses[:, 0:1, :], -all_poses[:, 1:2, :], -all_poses[:, 2:3, :], all_poses[:, 3:4, :]], 1)

    # w2c -> c2w opengl
    all_poses = np.linalg.inv(all_poses) # [4,4]
    all_poses = all_poses[:,:3,:4]

    # This is only to store a pre-calculated pose average stats of the dataset
    if args.save_pose_avg_stats:
        breakpoint()
        if pose_avg_stats_file == '':
            print('pose_avg_stats_file location unspecified, please double check...')
            sys.exit()

        all_poses = train_set.poses
        all_poses = all_poses.reshape(all_poses.shape[0], 3, 4)
        all_poses, pose_avg = center_poses(all_poses)

        # save pose_avg to pose_avg_stats.txt
        np.savetxt(pose_avg_stats_file, pose_avg)
        print('pose_avg_stats_colmap.txt successfully saved')
        sys.exit()

    # Here we use either pre-stored pose average stats or calculate pose average stats on the flight to center the poses
    if args.load_pose_avg_stats:
        pose_avg_from_file = np.loadtxt(pose_avg_stats_file)
        all_poses, pose_avg = center_poses(all_poses, pose_avg_from_file)
    else:
        print("compute center_poses online, check fix_coords() settings...")
        all_poses, pose_avg = center_poses(all_poses)

    # # debug reverse center_poses
    # all_poses, pose_avg = reverse_poses_transformation(all_poses, pose_avg_from_file)
    # breakpoint()
    bounds = np.array([train_set.near, train_set.far]) # manual tuned

    # print("remove rescale coord for debugging")
    # if 0:
    if rescale_coord:

        sc=train_set.pose_scale # manual tuned factor, align with colmap scale
        all_poses[:,:3,3] *= sc

        ### quite ugly ### 
        # move center of camera pose
        if train_set.move_all_cam_vec != [0.,0.,0.]:
            all_poses[:, :3, 3] += train_set.move_all_cam_vec

        if train_set.pose_scale2 != 1.0:
            all_poses[:,:3,3] *= train_set.pose_scale2
        # end of new mod1

    # Return all poses to dataset loaders
    all_poses = all_poses.reshape(all_poses.shape[0], 12)
    train_set.poses = all_poses[:train_poses.shape[0]]
    val_set.poses = all_poses[train_poses.shape[0]:]
    return train_set, val_set, bounds

def load_7Scenes_dataloader_colmap(args):
    ''' Data loader for Pose Regression Network '''
    if args.pose_only: # if train posenet is true
        pass
    else:
        raise Exception('load_7Scenes_dataloader() currently only support PoseNet Training, not NeRF training')
    data_dir, scene = osp.split(args.datadir) # ../data/7Scenes, chess
    dataset_folder, dataset = osp.split(data_dir) # ../data, 7Scenes

    # transformer
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    target_transform = transforms.Lambda(lambda x: torch.Tensor(x))

    data_dir = osp.join(dataset_folder, 'deepslam_data', dataset) # ../data/deepslam_data/7Scenes

    ret_idx = False # return frame index
    fix_idx = False # return frame index=0 in training
    ret_hist = False

    if 'NeRFW' in args:
        if args.NeRFW == True:
            ret_idx = True
            if args.fix_index:
                fix_idx = True

    # encode hist experiment
    if args.encode_hist:
        ret_idx = False
        fix_idx = False
        ret_hist = True

    kwargs = dict(args=args, scene=scene, data_path=data_dir, 
        transform=data_transform, target_transform=target_transform, 
        df=args.df, ret_idx=ret_idx, fix_idx=fix_idx,
        ret_hist=ret_hist, hist_bin=args.hist_bin)

    train_set = SevenScenes_colmap(train=True, trainskip=args.trainskip, **kwargs)
    val_set = SevenScenes_colmap(train=False, testskip=args.testskip, **kwargs)
    L = len(train_set)

    i_train = train_set.gt_idx
    i_val = val_set.gt_idx
    i_test = val_set.gt_idx

    train_shuffle=True
    if args.eval:
        train_shuffle=False

    # use a pose average stats computed earlier to unify posenet and nerf training
    if args.save_pose_avg_stats or args.load_pose_avg_stats:
        pose_avg_stats_file = osp.join(args.datadir, 'pose_avg_stats_colmap.txt')
        train_set, val_set, bounds = fix_coord(args, train_set, val_set, pose_avg_stats_file, rescale_coord=False)
    else:
        train_set, val_set, bounds = fix_coord(args, train_set, val_set, rescale_coord=False)

    train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=train_shuffle, num_workers=4) # debug
    # train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=5, pin_memory=False)
    val_dl = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=2)
    test_dl = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    hwf = [train_set.H, train_set.W, train_set.focal]
    i_split = [i_train, i_val, i_test]
    return train_dl, val_dl, test_dl, hwf, i_split, bounds.min(), bounds.max()

def load_7Scenes_dataloader_NeRF_colmap(args):
    ''' Data loader for NeRF '''
    assert(args.semantic==False) # do not support yet

    data_dir, scene = osp.split(args.datadir) # ../data/7Scenes, chess
    dataset_folder, dataset = osp.split(data_dir) # ../data, 7Scenes

    data_transform = transforms.Compose([
        transforms.ToTensor()])
    target_transform = transforms.Lambda(lambda x: torch.Tensor(x))

    data_dir = osp.join('..', 'data', 'deepslam_data', dataset) # ../data/deepslam_data/7Scenes

    ret_idx = False # return frame index
    fix_idx = False # return frame index=0 in training
    ret_hist = False

    if 'NeRFW' in args:
        ret_idx = True
        if args.fix_index:
            fix_idx = True

    # encode hist experiment
    if args.encode_hist:
        ret_hist = True

    kwargs = dict(args=args, scene=scene, data_path=data_dir,
        transform=data_transform, target_transform=target_transform, 
        df=args.df, ret_idx=ret_idx, fix_idx=fix_idx, ret_hist=ret_hist, hist_bin=args.hist_bin)

    train_set = SevenScenes_colmap(train=True, trainskip=args.trainskip, **kwargs)
    val_set = SevenScenes_colmap(train=False, testskip=args.testskip, **kwargs)

    i_train = train_set.gt_idx
    i_val = val_set.gt_idx
    i_test = val_set.gt_idx

    # use a pose average stats computed earlier to unify posenet and nerf training
    if args.save_pose_avg_stats or args.load_pose_avg_stats:
        pose_avg_stats_file = osp.join(args.datadir, 'pose_avg_stats_colmap.txt')
        train_set, val_set, bounds = fix_coord(args, train_set, val_set, pose_avg_stats_file)
    else:
        train_set, val_set, bounds = fix_coord(args, train_set, val_set)

    render_poses = None
    render_img = None 

    train_shuffle=True
    if args.render_video_train or args.render_test:
        train_shuffle=False
    train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=train_shuffle, num_workers=4)
    # print("debug: train shuffle = False")
    # train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4) # for debug
    val_dl = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    hwf = [train_set.H, train_set.W, train_set.focal]

    i_split = [i_train, i_val, i_test]

    return train_dl, val_dl, hwf, i_split, bounds, render_poses, render_img

def main():
    """
    visualizes the dataset
    """
    import torchvision.transforms as transforms
    import sys
    sys.path.append('../script/')

    # from script.models.options import config_parser
    from models.options import config_parser

    # use this to run the script
    # python load_7Scenes_colmap.py --config ../script/config/7Scenes/config_chess.txt
    # python load_7Scenes_colmap.py --config ../script/config/7Scenes/config_heads.txt

    parser = config_parser()
    args = parser.parse_args()
    print(parser.format_values())

    # train_dl, val_dl, hwf, i_split, bounds, render_poses, render_img = load_7Scenes_dataloader_NeRF_colmap(args)

    # pose_gt = []
    # for batch_data in train_dl:
    #     img, pose, hist = batch_data['img'], batch_data['pose'], batch_data['hist']
    #     pose_gt.append(pose.reshape(3,4))
    # pose_gt = torch.stack(pose_gt, dim=0)

    # # visualize the GT poses
    # # # create figure object
    # # plot translation traj.
    # fig = plt.figure(figsize = (8,6))
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    # ax1 = fig.add_axes([0, 0.2, 0.9, 0.85], projection='3d')

    # ax1.scatter(pose_gt[:,0,3], pose_gt[:,1,3], zs=pose_gt[:,2,3], c='g', s=3**2,depthshade=0) # GT
    # ax1.view_init(30, 120)
    # ax1.set_xlabel('x (m)')
    # ax1.set_ylabel('y (m)')
    # ax1.set_zlabel('z (m)')

    # ax1.set_xlim(-0.75, 0.75)
    # ax1.set_ylim(-0.75, 0.75)
    # ax1.set_zlim(-0.25, 1.75)

    # fname = '../script/vis_pose_colmap.png'
    # plt.savefig(fname, dpi=100)

    #### estimate trajectroy error between GT and Colmap poses ####
    assert(args.load_pose_avg_stats==False)
    from load_7Scenes import load_7Scenes_dataloader_NeRF

    train_dl1, _, _, _, _, _, _ = load_7Scenes_dataloader_NeRF_colmap(args)
    train_dl2, _, _, _, _, _, _ = load_7Scenes_dataloader_NeRF(args)

    pose_colmap = []
    pose_gt = []
    for batch_data in train_dl1:
        pose = batch_data['pose']
        pose_colmap.append(pose.reshape(3,4))
    pose_colmap = torch.stack(pose_colmap, dim=0)

    for batch_data in train_dl2:
        pose = batch_data['pose']
        pose_gt.append(pose.reshape(3,4))
    pose_gt = torch.stack(pose_gt, dim=0)

    from utils.utils import compute_ATE_error, compute_none_ATE_error
    # compute_none_ATE_error(pose_colmap, pose_gt, train_dl1)
    compute_ATE_error(pose_colmap, pose_gt, train_dl1)

if __name__ == '__main__':
  main()