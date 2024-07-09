import utils.set_sys_path
import os.path as osp
import numpy as np
import torch
import math


from dm.prepare_data import load_dataset
from dm.options import config_parser
from dataset_loaders.load_7Scenes import load_7Scenes_dataloader
from dataset_loaders.load_7Scenes_colmap import load_7Scenes_dataloader_colmap
from dataset_loaders.load_Cambridge import load_Cambridge_dataloader
import cv2

parser = config_parser()
args = parser.parse_args()
device = torch.device('cuda:0') # this is really controlled in train.sh

scene = osp.split(args.datadir)[-1]

# DFNet vs. DFNet+NeFeS
if args.dataset_type == '7Scenes':
    APR_folder = '../paper_result/DFNet_NeFeS50_7scenes/'
    APR_filename=APR_folder+scene+f'/DFNet_{scene}_NeFeS50_APR_pose_results.txt'
elif args.dataset_type == '7Scenes_colmap':
    APR_folder = '../paper_result/DFNet_NeFeS50_7Scenes_colmap/'
    APR_filename=APR_folder+scene+f'/DFNet_{scene}_NeFeS50_APR_pose_results.txt'
elif args.dataset_type == 'Cambridge':
    APR_folder = '../paper_result/DFNet_NeFeS50_Cambridge/'
    APR_filename=APR_folder+scene+f'/DFNet_{scene}_NeFeS50_APR_pose_results.txt'
else:
    NotImplementedError

def compute_pose_error_SE3(pose, predict_pose):
    '''
    compute pose error between two SE(3) pose
    pose: (4,4) or (3,4)
    predict_pose: (4,4) or (3,4)
    return: t_err, R_err
    '''
    predict_pose = predict_pose.squeeze()
    pose = pose.squeeze()
    # torch.set_printoptions(precision=32)
    t_error = float(torch.norm(pose[0:3,3] - predict_pose[0:3,3]))

    pose_R = pose[0:3,0:3].numpy()
    predict_pose_R = predict_pose[0:3,0:3].numpy()

    r_error = np.matmul(predict_pose_R, np.transpose(pose_R))
    r_error = np.linalg.norm(cv2.Rodrigues(r_error)[0])*180/math.pi
    return t_error, r_error

def compute_accuracy_stats_on_errors(t_R_errors):
    '''
    compute stats on errors
    t_R_errors: (N, 2) numpy array
    '''
    pct500_10 = 0 # 500cm/10deg
    pct50_5 = 0 # 50cm/5deg
    pct25_2 = 0 # 25cm/2deg
    pct10_5 = 0 # 10cm/5deg
    pct5 = 0    # 5cm/5deg
    pct2 = 0    # 2cm/2deg
    pct1 = 0    # 1cm/1deg

    total_frames = t_R_errors.shape[0]
    for i in range(total_frames):
        if t_R_errors[i,0] < 5 and t_R_errors[i,1] < 10:
            pct500_10 += 1
        if t_R_errors[i,0] < 0.5 and t_R_errors[i,1] < 5:
            pct50_5 += 1
        if t_R_errors[i,0] < 0.25 and t_R_errors[i,1] < 2:
            pct25_2 += 1
        if t_R_errors[i,0] < 0.1 and t_R_errors[i,1] < 5:
            pct10_5 += 1
        if t_R_errors[i,0] < 0.05 and t_R_errors[i,1] < 5:
            pct5 += 1
        if t_R_errors[i,0] < 0.02 and t_R_errors[i,1] < 2:
            pct2 += 1
        if t_R_errors[i,0] < 0.01 and t_R_errors[i,1] < 1:
            pct1 += 1
    print("=============================================")
    print("Accuracy:")
    print(f"500cm/10deg: {pct500_10/total_frames*100:.1f}%", )
    print(f"50cm/5deg: {pct50_5/total_frames*100:.1f}%", )
    print(f"25cm/2deg: {pct25_2/total_frames*100:.1f}%", )
    print(f"10cm/5deg: {pct10_5/total_frames*100:.1f}%", )
    print(f"5cm/5deg: {pct5/total_frames*100:.1f}%", )
    print(f"2cm/2deg: {pct2/total_frames*100:.1f}%", )
    print(f"1cm/1deg: {pct1/total_frames*100:.1f}%", )

def compute_none_ATE_error(pose1, pose2):
    '''
    plot and compute pose error from two trajectories, without ATE alignment
    :param pose1/refined_pose:  (N0, 3/4, 4) torch tensor
    :param pose2/gt_pose:  (N0, 3/4, 4) torch tensor
    '''

    from dm.pose_model import vis_pose
    assert(pose1.shape == pose2.shape)
    t_R_errors = np.zeros((pose2.shape[0], 2))
    ind2 = 0

    pose1_list = []
    pose2_list = []
    ang_error_list = []

    for i in range(pose2.shape[0]):

        poses_gt = pose2[i:i+1]
        poses_pred = pose1[i:i+1]

        pose1_list.append(poses_pred[:,:3,3].squeeze())
        pose2_list.append(poses_gt[:,:3,3].squeeze())

        error_x, theta = compute_pose_error_SE3(torch.Tensor(poses_gt), torch.Tensor(poses_pred))
        t_R_errors[ind2,:] = [error_x, theta]

        ang_error_list.append(theta)
        ind2 += 1
    median_result = np.median(t_R_errors,axis=0)
    mean_result = np.mean(t_R_errors,axis=0)
    # standard log
    print ('pose Median error {}m and {} degrees.'.format(median_result[0], median_result[1]))
    print ('pose Mean error {}m and {} degrees.'.format(mean_result[0], mean_result[1]))

    pose1_list = np.array(pose1_list)
    pose2_list = np.array(pose2_list)
    ang_error_list = np.array(ang_error_list)
    vis_info_ret = {"pose": pose1_list, "pose_gt": pose2_list, "theta": ang_error_list}
    # vis_pose(vis_info_ret)
    compute_accuracy_stats_on_errors(t_R_errors)
    return vis_info_ret


print(parser.format_values())
MODE = args.pose_only
# Load data
if args.dataset_type == '7Scenes':
    train_dl, val_dl, test_dl, hwf, i_split, near, far = load_7Scenes_dataloader(args)
    if args.set_near_far:
        print('use customized near_far')
        near = args.near_far[0]
        far = args.near_far[1]
elif args.dataset_type == '7Scenes_colmap':
    train_dl, val_dl, test_dl, hwf, i_split, near, far = load_7Scenes_dataloader_colmap(args)
    if args.set_near_far:
        print('use customized near_far')
        near = args.near_far[0]
        far = args.near_far[1]
elif args.dataset_type == 'Cambridge':
    train_dl, val_dl, test_dl, hwf, i_split, near, far = load_Cambridge_dataloader(args)
    if args.set_near_far:
        print('use customized near_far')
        near = args.near_far[0]
        far = args.near_far[1]
else:
    images, poses_train, render_poses, hwf, i_split, near, far = load_dataset(args)
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    if args.set_near_far:
        print('use customized near_far')
        near = args.near_far[0]
        far = args.near_far[1]

i_train, i_val, i_test = i_split
print('TRAIN views are', i_train)
print('TEST views are', i_test)
print('VAL views are', i_val)

# load GT pose results
gt_pose = test_dl.dataset.poses[i_test]
if torch.is_tensor(gt_pose):
    gt_pose = gt_pose.numpy().reshape(gt_pose.shape[0], 3, 4).astype(np.float32)
else:
    gt_pose = gt_pose.reshape(gt_pose.shape[0], 3, 4).astype(np.float32)

# load APR pose results
apr_pose = np.loadtxt(APR_filename)
apr_pose = apr_pose.reshape(apr_pose.shape[0], 3, 4).astype(np.float32)

# # apply KS Filtering
# from lck.ks_filter import ks_filter
# ks_filter(apr_pose, gt_pose, KS_filename, th=0.95)

vis_info_ret = compute_none_ATE_error(apr_pose, gt_pose)
