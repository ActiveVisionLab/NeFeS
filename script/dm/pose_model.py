import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import numpy as np
from torchvision import models
from efficientnet_pytorch import EfficientNet
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

import math
import time
import os.path as osp

import pytorch3d.transforms as transforms
import cv2

def preprocess_data(inputs, device):
    # normalize inputs according to https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(device) # per channel subtraction
    std = torch.Tensor([0.229, 0.224, 0.225]).to(device) # per channel division
    inputs = (inputs - mean[None,:,None,None])/std[None,:,None,None]
    return inputs

def filter_hook(m, g_in, g_out):
    g_filtered = []
    for g in g_in:
        g = g.clone()
        g[g != g] = 0
        g_filtered.append(g)
    return tuple(g_filtered)

def vis_pose(vis_info, scene=None):
    '''
    visualize predicted pose result vs. gt pose
    '''

    pose = vis_info['pose']
    pose_gt = vis_info['pose_gt']
    theta = vis_info['theta']
    ang_threshold=10
    seq_num = theta.shape[0]
    # # create figure object
    # plot translation traj.
    fig = plt.figure(figsize = (8,6))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax1 = fig.add_axes([0, 0.2, 0.9, 0.85], projection='3d')
    ax1.scatter(pose[10:,0],pose[10:,1],zs=pose[10:,2], c='r', s=3**2,depthshade=0) # predict
    ax1.scatter(pose_gt[:,0], pose_gt[:,1], zs=pose_gt[:,2], c='g', s=3**2,depthshade=0) # GT
    ax1.scatter(pose[0:10,0],pose[0:10,1],zs=pose[0:10,2], c='k', s=3**2,depthshade=0) # predict
    ax1.view_init(30, 120)
    # ax1.view_init(0, -30)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('z (m)')

    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)

    # plot angular error
    ax2 = fig.add_axes([0.1, 0.05, 0.75, 0.2])
    err = theta.reshape(1, seq_num)
    err = np.tile(err, (20, 1))
    ax2.imshow(err, vmin=0,vmax=ang_threshold, aspect=3)
    ax2.set_yticks([])
    ax2.set_xticks([0, seq_num*1/5, seq_num*2/5, seq_num*3/5, seq_num*4/5, seq_num])
    if scene==None:
        fname = 'tmp/vis_pose.png'
    else:
        fname=f'tmp/vis_pose_{scene}.png'
    plt.savefig(fname, dpi=100)

def compute_pose_error_SE3(pose, predict_pose):
    '''
    compute pose error between two SE(3) pose
    pose: (4,4) or (3,4)
    predict_pose: (4,4) or (3,4)
    return: t_err, R_err
    '''
    predict_pose = predict_pose.squeeze()
    pose = pose.squeeze()
    torch.set_printoptions(precision=32)
    t_error = float(torch.norm(pose[0:3,3] - predict_pose[0:3,3]))

    pose_R = pose[0:3,0:3].numpy()
    predict_pose_R = predict_pose[0:3,0:3].numpy()

    r_error = np.matmul(predict_pose_R, np.transpose(pose_R))
    r_error = np.linalg.norm(cv2.Rodrigues(r_error)[0])*180/math.pi
    return t_error, r_error

def compute_pose_error(pose, predict_pose):
    ''' calculate rotation and translation error between 2 SE(3) camera poses
    :param: pose: GT pose, torch.Tensor [B, 3, 4]
    :param: predict_pose: predicted pose, torch.Tensor [B, 3, 4]
    return: t_error, r_error: translation error in m, rotation error in degree
    '''

    pose_q = transforms.matrix_to_quaternion(pose[:,:3,:3])#.cpu().numpy() # gnd truth in quaternion
    pose_x = pose[:, :3, 3] # gnd truth position
    predicted_q = transforms.matrix_to_quaternion(predict_pose[:,:3,:3])#.cpu().numpy() # predict in quaternion
    predicted_x = predict_pose[:, :3, 3] # predict position
    pose_q = pose_q.squeeze() 
    pose_x = pose_x.squeeze() 
    predicted_q = predicted_q.squeeze() 
    predicted_x = predicted_x.squeeze()

    #Compute Individual Sample Error 
    q1 = pose_q / torch.linalg.norm(pose_q)
    q2 = predicted_q / torch.linalg.norm(predicted_q)
    d = torch.abs(torch.sum(torch.matmul(q1,q2)))

    d = torch.clamp(d, -1.0, 1.0)
    r_error = (2 * torch.acos(d) * 180/math.pi).numpy() # somehow there exists a precision problem when d is really close to 1.0
    t_error = torch.linalg.norm(torch.Tensor(pose_x-predicted_x)).numpy()

    return t_error, r_error

def print_compute_pose_error(poses_gt, predict_poses):
    '''
    Calculate rotation and translation error between 2 SE(3) camera poses, given blackbox pose prediction.
    Then print the result.
    poses_gt: GT pose, numpy array [B, 3, 4]
    predict_poses: predicted pose, numpy array [B, 3, 4]

    '''
    assert(poses_gt.shape[0]==predict_poses.shape[0])

    num = poses_gt.shape[0]
    
    # compute initial pose error
    results = np.zeros((num, 2))
    for i in range(num):
        # error_x, theta = compute_pose_error(torch.Tensor(poses_gt[i:i+1]), torch.Tensor(predict_poses[i:i+1]))
        error_x, theta = compute_pose_error_SE3(torch.Tensor(poses_gt[i:i+1]), torch.Tensor(predict_poses[i:i+1]))
        results[i,:] = [error_x, theta]
    median_result = np.median(results,axis=0)
    mean_result = np.mean(results,axis=0)

    # standard log
    print ('Median error {}m and {} degrees.'.format(median_result[0], median_result[1]))
    print ('Mean error {}m and {} degrees.'.format(mean_result[0], mean_result[1]))
    ### end of load initial pose from pixloc

def compute_error_in_q(args, dl, model, device, results, batch_size=1):
    predict_pose_list = []
    gt_pose_list = []
    ang_error_list = []
    pose_result_raw = []
    pose_GT = []
    i = 0

    for batch in dl:
        if args.NeRFW:
            data, pose, img_idx = batch['img'], batch['pose'], batch['hist']
        else:
            data, pose = batch['img'], batch['pose']
        data = data.to(device) # input
        pose = pose.reshape((batch_size,3,4)).numpy() # label

        with torch.no_grad():
            if args.PoseEstimatorType=='DFNet':
                _, predict_pose = model(data)
            else:
                predict_pose = model(data)
            # using SVD to make sure predict rotation is normalized rotation matrix (TODO: only necessary for models predicts SE(3))
            R_torch = predict_pose.reshape((batch_size, 3, 4))[:,:3,:3]
            predict_pose = predict_pose.reshape((batch_size, 3, 4)).cpu().numpy()

            u,s,v=torch.svd(R_torch)
            Rs = torch.matmul(u, v.transpose(-2,-1))
        predict_pose[:,:3,:3] = Rs[:,:3,:3].cpu().numpy()

        # compute pose error between the ground truth and the network predicted pose
        # error_x, theta = compute_pose_error(torch.Tensor(pose), torch.Tensor(predict_pose))
        error_x, theta = compute_pose_error_SE3(torch.Tensor(pose), torch.Tensor(predict_pose))
        results[i,:] = [error_x, theta]

        # save results for visualization
        predict_pose_list.append(predict_pose[:,:3,3].squeeze())
        gt_pose_list.append(pose[:,:3,3].squeeze())
        ang_error_list.append(theta)
        pose_result_raw.append(predict_pose)
        pose_GT.append(pose)
        i += 1

    predict_pose_list = np.array(predict_pose_list)
    gt_pose_list = np.array(gt_pose_list)
    ang_error_list = np.array(ang_error_list)
    pose_result_raw = np.asarray(pose_result_raw)[:,0,:,:]
    pose_GT = np.asarray(pose_GT)[:,0,:,:]
    vis_info_ret = {"pose": predict_pose_list, "pose_gt": gt_pose_list, "theta": ang_error_list, "pose_result_raw": pose_result_raw, "pose_GT": pose_GT}
    return results, vis_info_ret

# # pytorch
def get_error_in_q(args, dl, model, sample_size, device, batch_size=1, ret_vis_info=False):
    ''' Convert Rotation matrix to quaternion, then calculate the location errors. original from PoseNet Paper '''
    model.eval()
    
    results = np.zeros((sample_size, 2))
    results, vis_info = compute_error_in_q(args, dl, model, device, results, batch_size)
    median_result = np.median(results,axis=0)
    mean_result = np.mean(results,axis=0)

    # standard log
    print ('Median error {}m and {} degrees.'.format(median_result[0], median_result[1]))
    print ('Mean error {}m and {} degrees.'.format(mean_result[0], mean_result[1]))

    # standard log2
    # num_translation_less_5cm = np.asarray(np.where(results[:,0]<0.05))[0]
    # num_rotation_less_5 = np.asarray(np.where(results[:,1]<5))[0]
    # print ('translation error less than 5cm {}/{}.'.format(num_translation_less_5cm.shape[0], results.shape[0]))
    # print ('rotation error less than 5 degree {}/{}.'.format(num_rotation_less_5.shape[0], results.shape[0]))
    # print ('results:', results)

    # visualize results
    _,scene=osp.split(args.datadir)
    vis_pose(vis_info, scene)

    if ret_vis_info:
        return vis_info
    return

class EfficientNetB3(nn.Module):
    ''' EfficientNet-B3 backbone,
    model ref: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py 
    '''
    def __init__(self, feat_dim=12):
        super(EfficientNetB3, self).__init__()
        self.backbone_net = EfficientNet.from_pretrained('efficientnet-b3')
        self.feature_extractor = self.backbone_net.extract_features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_pose = nn.Linear(1536, feat_dim) # 1280 for efficientnet-b0, 1536 for efficientnet-b3

    def forward(self, Input):

        # vanilla posenet
        x = self.feature_extractor(Input)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        predict = self.fc_pose(x)
        return predict

# PoseNet (SE(3)) w/ mobilev2 backbone
class PoseNetV2(nn.Module):
    def __init__(self, feat_dim=12):
        super(PoseNetV2, self).__init__()
        self.backbone_net = models.mobilenet_v2(pretrained=True)
        self.feature_extractor = self.backbone_net.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_pose = nn.Linear(1280, feat_dim)

    def forward(self, Input):
        x = self.feature_extractor(Input)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        predict = self.fc_pose(x)
        return predict

# PoseNet (SE(3)) w/ resnet34 backnone. We found dropout layer is unnecessary, so we set droprate as 0 in reported results.
class PoseNet_res34(nn.Module):
    def __init__(self, droprate=0.5, pretrained=True,
        feat_dim=2048):
        super(PoseNet_res34, self).__init__()
        self.droprate = droprate

        # replace the last FC layer in feature extractor
        self.feature_extractor = models.resnet34(pretrained=True)
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
        self.fc_pose = nn.Linear(feat_dim, 12)

        # initialize
        if pretrained:
          init_modules = [self.feature_extractor.fc]
        else:
          init_modules = self.modules()

        for m in init_modules:
          if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
              nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(x)
        if self.droprate > 0:
          x = F.dropout(x, p=self.droprate)
        predict = self.fc_pose(x)
        return predict

# from MapNet paper CVPR 2018
class PoseNet(nn.Module):
    def __init__(self, droprate=0.5, pretrained=True,
        feat_dim=2048, filter_nans=False):
        super(PoseNet, self).__init__()
        self.droprate = droprate

        # replace the last FC layer in feature extractor
        self.feature_extractor = models.resnet34(pretrained=True)
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        self.fc_xyz  = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)
        if filter_nans:
          self.fc_wpqr.register_backward_hook(hook=filter_hook)
        # initialize
        if pretrained:
          init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
        else:
          init_modules = self.modules()

        for m in init_modules:
          if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
              nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(x)
        if self.droprate > 0:
          x = F.dropout(x, p=self.droprate)

        xyz  = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), 1)

class MapNet(nn.Module):
    """
    Implements the MapNet model (green block in Fig. 2 of paper)
    """
    def __init__(self, mapnet):
        """
        :param mapnet: the MapNet (two CNN blocks inside the green block in Fig. 2
        of paper). Not to be confused with MapNet, the model!
        """
        super(MapNet, self).__init__()
        self.mapnet = mapnet
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def forward(self, x):
        """
        Notice that the offical code was tested with s=3 frames. Which was our reported results from MapNet Colmap (see section 3.5 of Brahmbhatt et al.)
        For simplicity, we only inference the pose of the middle frame for DFM.
        :param x: image blob (N x C x H x W)
        :return: pose outputs
         (N x 6)->(N x 12)
        """
        mean, std = x.new_tensor(self.mean), x.new_tensor(self.std)
        x = (x - mean[:, None, None]) / std[:, None, None]

        poses = self.mapnet(x) # [N, 6] in [x,y,z,log(qx),log(qy),log(qz)]

        # convert log quaternion to SE(3) matrix. The official code was trained in log quaternion space.
        q = self.qexp(poses[:,3:]) # here output (1, 4)
        poses = torch.hstack((poses[:, :3], q)) # here output (1, 7)
        R = transforms.quaternion_to_matrix(q)
        output = torch.zeros((x.shape[0], 3, 4), device=x.device)
        output[:, :3, :3] = R
        output[:, :, 3] = poses[:, :3]
        return output.reshape(-1, 12)

    def qexp(self, q):
        """
        Applies the exponential map to q
        :param q: (N, 3)
        :return: (N, 4)
        """

        n = torch.norm(q, keepdim=True)
        q = torch.hstack((torch.cos(n), torch.sinc(n/np.pi)*q))
        return q

def eval_on_epoch(args, dl, model, optimizer, loss_func, device):
    model.eval()
    val_loss_epoch = []
    for batch_data in dl:
        data, pose = batch_data['img'], batch_data['pose']
        inputs = data.to(device)
        labels = pose.to(device)
        if args.preprocess_ImgNet:
            inputs = preprocess_data(inputs, device)
        predict = model(inputs)
        loss = loss_func(predict, labels)
        val_loss_epoch.append(loss.item())
    val_loss_epoch_mean = np.mean(val_loss_epoch)
    return val_loss_epoch_mean


def train_on_epoch(args, dl, model, optimizer, loss_func, device):
    model.train()
    train_loss_epoch = []
    for batch_data in dl:
        data, pose = batch_data['img'], batch_data['pose']
        inputs = data.to(device) # (N, Ch, H, W) ~ (4,3,200,200), 7scenes [4, 3, 256, 341] wierd shape...
        labels = pose.to(device)
        if args.preprocess_ImgNet:
            inputs = preprocess_data(inputs, device)

        predict = model(inputs)
        loss = loss_func(predict, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_epoch.append(loss.item())
    train_loss_epoch_mean = np.mean(train_loss_epoch)
    return train_loss_epoch_mean

def train_posenet(args, train_dl, val_dl, model, epochs, optimizer, loss_func, scheduler, device, early_stopping):
    # writer = SummaryWriter()
    model_log = tqdm(total=0, position=1, bar_format='{desc}')
    for epoch in tqdm(range(epochs), desc='epochs'):
        
        # train 1 epoch
        train_loss = train_on_epoch(args, train_dl, model, optimizer, loss_func, device)
        # writer.add_scalar("Loss/train", train_loss, epoch)
        
        # validate every epoch
        val_loss = eval_on_epoch(args, val_dl, model, optimizer, loss_func, device)
        # writer.add_scalar("Loss/val", val_loss, epoch)
        
        # reduce LR on plateau
        scheduler.step(val_loss)
        # writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        # logging
        tqdm.write('At epoch {0:6d} : train loss: {1:.4f}, val loss: {2:.4f}'.format(epoch, train_loss, val_loss))
                
        # check wether to early stop
        early_stopping(val_loss, model, epoch=epoch, save_multiple=(not args.no_save_multiple), save_all=args.save_all_ckpt)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        model_log.set_description_str(f'Best val loss: {early_stopping.val_loss_min:.4f}')

        if epoch % args.i_eval == 0:
            get_error_in_q(args, val_dl, model, len(val_dl.dataset), device, batch_size=1)


    # writer.flush()
