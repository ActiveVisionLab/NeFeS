from ast import Not
import utils.set_sys_path
import numpy as np
import random
import torch
import os
from dm.pose_model import get_error_in_q
from dm.direct_pose_model import load_APR_and_FeatureNet
from dm.prepare_data import load_dataset
from dm.options import config_parser
from dm.DFM_APR_refine import DFM_post_processing
from dm.DFM_pose_refine import DFM_post_processing2, load_NeRF_model
from dataset_loaders.load_7Scenes import load_7Scenes_dataloader
from dataset_loaders.load_7Scenes_colmap import load_7Scenes_dataloader_colmap
from dataset_loaders.load_Cambridge import load_Cambridge_dataloader

parser = config_parser()
args = parser.parse_args()
device = torch.device('cuda:0') # this is really controlled in train.sh

# # try to be deterministic
# np.random.seed(0)
# torch.manual_seed(0)
# import random
# random.seed(0)

# os.system("ulimit -n 8192")
torch.multiprocessing.set_sharing_strategy('file_system')

def train():
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

    if MODE==2: # APR Refinement with NeFeS
        model, feat_model = load_APR_and_FeatureNet(args, device)

        # start training
        DFM_post_processing(args, model, feat_model, hwf, near, far, device, test_dl=test_dl)

    elif MODE==3: # Pose Refinement with NeFeS

        print ('Inital Pose Error...')
        ### load or inference the predicted camera poses and Feature Extraction model
        model, feat_model = load_APR_and_FeatureNet(args, device)

        # compute initial pose error
        vis_info = get_error_in_q(args, test_dl, model, len(i_test), device, batch_size=1, ret_vis_info=True)
        predict_poses = vis_info["pose_result_raw"]
        poses_gt = vis_info["pose_GT"]


        ### load NeRF
        world_setup_dict = {
            'pose_scale' : test_dl.dataset.pose_scale,
            'pose_scale2' : test_dl.dataset.pose_scale2,
            'move_all_cam_vec' : test_dl.dataset.move_all_cam_vec,
        }

        render_kwargs_test = load_NeRF_model(args, near, far)

        ### Perform DFM post-processing
        pose_param_net = DFM_post_processing2(args, predict_poses, feat_model, render_kwargs_test, hwf, device, test_dl=test_dl, world_setup_dict=world_setup_dict)

if __name__ == '__main__':
    if args.eval:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        # eval()
    else:
        train()
