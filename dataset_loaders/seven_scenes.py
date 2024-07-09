"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
pytorch data loader for the 7-scenes dataset
"""
import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import sys
import pickle
import pdb,copy
import cv2

sys.path.insert(0, '../')
import transforms3d.quaternions as txq
# see for formulas:
# https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-801-machine-vision-fall-2004/readings/quaternions.pdf
# and "Quaternion and Rotation" - Yan-Bin Jia, September 18, 2016
from dataset_loaders.utils.color import rgb_to_yuv
import json

def RT2QT(poses_in, mean_t, std_t):
  """
  processes the 1x12 raw pose from dataset by aligning and then normalizing
  :param poses_in: N x 12
  :param mean_t: 3
  :param std_t: 3
  :return: processed poses (translation + quaternion) N x 7
  """
  poses_out = np.zeros((len(poses_in), 7))
  poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

  # align
  for i in range(len(poses_out)):
    R = poses_in[i].reshape((3, 4))[:3, :3]
    q = txq.mat2quat(R)
    q = q/(np.linalg.norm(q) + 1e-12) # normalize
    q *= np.sign(q[0])  # constrain to hemisphere
    poses_out[i, 3:] = q

  # normalize translation
  poses_out[:, :3] -= mean_t
  poses_out[:, :3] /= std_t
  return poses_out

def qlog(q):
  """
  Applies logarithm map to q
  :param q: (4,)
  :return: (3,)
  """
  if all(q[1:] == 0):
    q = np.zeros(3)
  else:
    q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
  return q

import transforms3d.quaternions as txq # Warning: outdated package

def process_poses_rotmat(poses_in, mean_t, std_t, align_R, align_t, align_s):
  """
  processes the 1x12 raw pose from dataset by aligning and then normalizing
  produce logq
  :param poses_in: N x 12
  :return: processed poses N x 12
  """
  return poses_in

def process_poses_q(poses_in, mean_t, std_t, align_R, align_t, align_s):
  """
  processes the 1x12 raw pose from dataset by aligning and then normalizing
  produce logq
  :param poses_in: N x 12
  :param mean_t: 3
  :param std_t: 3
  :param align_R: 3 x 3
  :param align_t: 3
  :param align_s: 1
  :return: processed poses (translation + log quaternion) N x 6
  """
  poses_out = np.zeros((len(poses_in), 6)) # (1000,6)
  poses_out[:, 0:3] = poses_in[:, [3, 7, 11]] # x,y,z position
  # align
  for i in range(len(poses_out)):
    R = poses_in[i].reshape((3, 4))[:3, :3] # rotation
    q = txq.mat2quat(np.dot(align_R, R))
    q *= np.sign(q[0])  # constrain to hemisphere, first number, +1/-1, q.shape (1,4)
    poses_out[i, 3:] = q # logq rotation
    t = poses_out[i, :3] - align_t
    poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

  # normalize translation
  poses_out[:, :3] -= mean_t #(1000, 6)
  poses_out[:, :3] /= std_t
  return poses_out

def process_poses_logq(poses_in, mean_t, std_t, align_R, align_t, align_s):
  """
  processes the 1x12 raw pose from dataset by aligning and then normalizing
  produce logq
  :param poses_in: N x 12
  :param mean_t: 3
  :param std_t: 3
  :param align_R: 3 x 3
  :param align_t: 3
  :param align_s: 1
  :return: processed poses (translation + log quaternion) N x 6
  """
  poses_out = np.zeros((len(poses_in), 6)) # (1000,6)
  poses_out[:, 0:3] = poses_in[:, [3, 7, 11]] # x,y,z position
  # align
  for i in range(len(poses_out)):
    R = poses_in[i].reshape((3, 4))[:3, :3] # rotation
    q = txq.mat2quat(np.dot(align_R, R))
    q *= np.sign(q[0])  # constrain to hemisphere, first number, +1/-1, q.shape (1,4)
    q = qlog(q) # (1,3)
    poses_out[i, 3:] = q # logq rotation
    t = poses_out[i, :3] - align_t
    poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

  # normalize translation
  poses_out[:, :3] -= mean_t #(1000, 6)
  poses_out[:, :3] /= std_t
  return poses_out

from torchvision.datasets.folder import default_loader
def load_image(filename, loader=default_loader):
  try:
    img = loader(filename)
  except IOError as e:
    print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
    return None
  except:
    print('Could not load image {:s}, unexpected error'.format(filename))
    return None
  return img

def load_depth_image(filename):
  try:
    img_depth = Image.fromarray(np.array(Image.open(filename)).astype("uint16"))
  except IOError as e:
    print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
    return None
  return img_depth

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def normalize_recenter_pose(poses, sc, hwf):
  ''' normalize xyz into [-1, 1], and recenter pose ''' # BUG train and val should use same normalization!!!
  target_pose = poses.reshape(poses.shape[0],3,4)
  target_pose[:,:3,3] = target_pose[:,:3,3] * sc

  x_norm = target_pose[:,0,3]
  y_norm = target_pose[:,1,3]
  z_norm = target_pose[:,2,3]

  tpose_ = target_pose+0

  # find the center of pose
  center = np.array([x_norm.mean(), y_norm.mean(), z_norm.mean()])
  bottom = np.reshape([0,0,0,1.], [1,4])

  # pose avg
  vec2 = normalize(tpose_[:, :3, 2].sum(0)) 
  up = tpose_[:, :3, 1].sum(0)
  hwf=np.array(hwf).transpose()
  c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
  c2w = np.concatenate([c2w[:3,:4], bottom], -2)

  bottom = np.tile(np.reshape(bottom, [1,1,4]), [tpose_.shape[0],1,1])
  poses = np.concatenate([tpose_[:,:3,:4], bottom], -2)
  poses = np.linalg.inv(c2w) @ poses
  return poses[:,:3,:].reshape(poses.shape[0],12)

class SevenScenes(data.Dataset):
    def __init__(self, args, scene, data_path, train, transform=None,
                 target_transform=None, mode=0, seed=7,
                 df=1., trainskip=1, testskip=1, hwf=[480,640,585.], 
                 ret_idx=False, fix_idx=False, ret_hist=False, hist_bin=10):
      """
      :param scene: scene name ['chess', 'pumpkin', ...]
      :param data_path: root 7scenes data directory.
      Usually '../data/deepslam_data/7Scenes'
      :param train: if True, return the training images. If False, returns the
      testing images
      :param transform: transform to apply to the images
      :param target_transform: transform to apply to the poses
      :param mode: (Obsolete) 0: just color image, 1: color image in NeRF 0-1 and resized.
      :param df: downscale factor
      :param trainskip: due to 7scenes are so big, now can use less training sets # of trainset = 1/trainskip
      :param testskip: skip part of testset, # of testset = 1/testskip
      :param hwf: H,W,Focal from COLMAP
      :param ret_idx: bool, currently only used by NeRF-W
      """

      self.transform = transform
      self.target_transform = target_transform
      self.df = df

      self.H, self.W, self.focal = hwf
      self.H = int(self.H)
      self.W = int(self.W)
      np.random.seed(seed)

      self.train = train
      self.ret_idx = ret_idx
      self.fix_idx = fix_idx
      self.ret_hist = ret_hist
      self.hist_bin = hist_bin # histogram bin size

      # directories
      base_dir = osp.join(osp.expanduser(data_path), scene) # '../data/deepslam_data/7Scenes'
      deepslam_data_dir, _ = osp.split(data_path)
      data_dir, _ = osp.split(deepslam_data_dir)

      data_dir = osp.join(data_dir, '7Scenes', scene) # '../data/7Scenes/chess'
      world_setup_fn = data_dir + '/world_setup.json'

      # read json file
      with open(world_setup_fn, 'r') as myfile:
        data=myfile.read()

      # parse json file
      obj = json.loads(data)
      self.near = obj['near']
      self.far = obj['far']
      self.pose_scale = obj['pose_scale']
      self.pose_scale2 = obj['pose_scale2']
      self.move_all_cam_vec = obj['move_all_cam_vec']

      # decide which sequences to use
      if train:
        split_file = osp.join(base_dir, 'TrainSplit.txt')
      else:
        split_file = osp.join(base_dir, 'TestSplit.txt')
      with open(split_file, 'r') as f:
        seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')] # parsing

      # read poses and collect image names
      self.c_imgs = []
      self.d_imgs = []
      self.gt_idx = np.empty((0,), dtype=np.int)
      ps = {}
      vo_stats = {}
      gt_offset = int(0)
      for seq in seqs:
        seq_dir = osp.join(base_dir, 'seq-{:02d}'.format(seq))
        seq_data_dir = osp.join(data_dir, 'seq-{:02d}'.format(seq))
        
        p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if n.find('pose') >= 0]
        idxes = [int(n[6:12]) for n in p_filenames]

        frame_idx = np.array(sorted(idxes))


        # trainskip and testskip
        if train and trainskip > 1:
          frame_idx_tmp = frame_idx[::trainskip]
          frame_idx = frame_idx_tmp
        elif not train and testskip > 1:
          frame_idx_tmp = frame_idx[::testskip]
          frame_idx = frame_idx_tmp

        pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.
          format(i))).flatten()[:12] for i in frame_idx] # all the 3x4 pose matrices
        ps[seq] = np.asarray(pss) # list of all poses in file No. seq
        vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

        self.gt_idx = np.hstack((self.gt_idx, gt_offset+frame_idx))
        gt_offset += len(p_filenames)
        c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i)) for i in frame_idx]
        d_imgs = [osp.join(seq_dir, 'frame-{:06d}.depth.png'.format(i)) for i in frame_idx]
        self.c_imgs.extend(c_imgs)
        self.d_imgs.extend(d_imgs)

      pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
      if train:
        mean_t = np.zeros(3)  # optionally, use the ps dictionary to calc stats
        std_t = np.ones(3)
        np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
      else:
        mean_t, std_t = np.loadtxt(pose_stats_filename)

      # convert pose to translation + log quaternion
      logq = False
      quat = False
      if logq: # (batch_num, 6)
        self.poses = np.empty((0, 6))
      elif quat: # (batch_num, 7) 
        self.poses = np.empty((0, 7))
      else: # (batch_num, 12)
        self.poses = np.empty((0, 12))

      for seq in seqs:
        if logq:
          pss = process_poses_logq(poses_in=ps[seq], mean_t=mean_t, std_t=std_t, align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'], align_s=vo_stats[seq]['s']) # here returns t + logQed R
          self.poses = np.vstack((self.poses, pss))
        elif quat:
          pss = RT2QT(poses_in=ps[seq], mean_t=mean_t, std_t=std_t) # here returns t + quaternion R
          self.poses = np.vstack((self.poses, pss))
        else:
          pss = process_poses_rotmat(poses_in=ps[seq], mean_t=mean_t, std_t=std_t, align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'], align_s=vo_stats[seq]['s'])
          self.poses = np.vstack((self.poses, pss))

      # debug read one img and get the shape of the img
      img = load_image(self.c_imgs[0])
      img_np = (np.array(img) / 255.).astype(np.float32) # (480,640,3)
      self.H, self.W = img_np.shape[:2]
      if self.df != 1.:
        self.H = int(self.H//self.df)
        self.W = int(self.W//self.df)
        self.focal = self.focal/self.df

    def __len__(self):
      return self.poses.shape[0]

    # TODO: modify return to same as Cambridge dataset
    def __getitem__(self, index):
      # print("index:", index)
      img = load_image(self.c_imgs[index]) # chess img.size = (640,480)
      if self.df != 1.:
        img_np = (np.array(img) / 255.).astype(np.float32)
        dims = (self.W, self.H)
        img_half_res = cv2.resize(img_np, dims, interpolation=cv2.INTER_AREA) # (H, W, 3)
        img = img_half_res
      if self.transform is not None:
          img = self.transform(img)

      pose = self.poses[index]
      if self.target_transform is not None:
        pose = self.target_transform(pose)

      out = {}
      out['img'] = img
      out['pose'] = pose

      if self.ret_hist:
        yuv = rgb_to_yuv(img)
        y_img = yuv[0] # extract y channel only
        hist = torch.histc(y_img, bins=self.hist_bin, min=0., max=1.) # compute intensity histogram
        hist = hist/(hist.sum())*100 # convert to histogram density, in terms of percentage per bin
        hist = torch.round(hist)
        out['hist'] = hist

      if self.ret_idx:
        out['idx'] = index
        if self.fix_idx:
          out['idx'] = 0

      return out

class SevenScenesMF(SevenScenes):
    """
    Multi-frame consecutive frame dataloader for VLocNet
    Use SevenScenesMF as SevenScenes' child class
    """
    def __init__(self, train, half_res, trainskip=1, testskip=1, steps=2, **kwargs):
        super().__init__(train=train, half_res=half_res, trainskip=trainskip, testskip=testskip, **kwargs)
        self.steps = steps

    def processMF(self):
      ''' regrouping the dataset '''
      img = self.c_imgs
      pose = self.poses
      img_list = []
      pose_list = []
      img_list.append(copy.deepcopy(img))
      pose_list.append(copy.deepcopy(pose))

      for i in range(1, self.steps):
        img_list.append(copy.deepcopy(img))

        for j in range(i):
          img_list[j].pop(len(self.c_imgs)-1) # pop the last img
          img_list[j].insert(0, img_list[j][0]) # insert the first img in first spot


      for i in range(1, self.steps+1):
        pose_list.append(copy.deepcopy(pose))

        for j in range(i):
          pose_list[j] = np.delete(pose_list[j], len(self.c_imgs)-1, axis=0)
          pose_list[j] = np.insert(pose_list[j], 0, pose_list[j][0], axis=0)

      # the image queue now should be [[0,0] [0,1] [1,2] [2,3], ... [prev, cur]]
      self.c_imgs = img_list
      # the pose queue now should be [[0,0,0] [0,0,1] [0,1,2] [1,2,3], ... [prev-1, prev, cur]]
      self.poses = pose_list
      return

    def __getitem__(self, index):
      # grep 2 images and 3 poses at a time, convert pose to quaternion

      img_p = None # img previous
      img_c = None # img current
      while (img_p is None) or (img_c is None):
        img_p = load_image(self.c_imgs[0][index]) # chess img.size = (640,480)
        img_c = load_image(self.c_imgs[1][index])
        pose_p_p = self.poses[0][index] # prev-1
        pose_p = self.poses[1][index] # prev
        pose_c = self.poses[2][index] # cur
        index += 1
      img = [img_p, img_c]
      pose = [pose_p_p, pose_p, pose_c]
      index -= 1

      if self.transform is not None:
        img = [self.transform(i) for i in img]

      if self.target_transform is not None:
        pose = [self.target_transform(i) for i in pose]
      return img, pose

    def __len__(self):
      return self.poses[0].shape[0]

class SevenScenesMFDGR(SevenScenes):
    """
    Multi-frame consecutive frame dataloader for DGRNet
    Use SevenScenesMF as SevenScenes' child class
    """
    def __init__(self, train, half_res, trainskip=1, testskip=1, steps=2, **kwargs):
        super().__init__(train=train, half_res=half_res, trainskip=trainskip, testskip=testskip, **kwargs)
        self.steps = steps

    def shiftRight(self, img, pose, step_size):
      ''' right shift img and pose list, copy and insert 0th element to the left end '''
      for i in range(step_size):
        img.pop(len(self.c_imgs)-1) # pop the last img
        img.insert(0, img[0]) # insert the first img in first spot

        pose = np.delete(pose, len(self.c_imgs)-1, axis=0)
        pose = np.insert(pose, 0, pose[0], axis=0)
      return img, pose

    def shiftLeft(self, img, pose, step_size):
      ''' left shift img and pose list, copy and insert last element to the right end '''
      for i in range(step_size):
        img.insert(len(self.c_imgs), img[len(self.c_imgs)-1]) # insert the last img to the end of the list
        img.pop(0) # pop the first img

        pose = np.insert(pose, len(self.c_imgs), pose[len(self.c_imgs)-1], axis=0)
        pose = np.delete(pose, 0, axis=0)
      return img, pose

    def processMF(self):
      ''' regrouping the dataset '''
      img = self.c_imgs
      pose = self.poses
      img_list = []
      pose_list = []

      # ex: steps=5, [[prev-2], [prev-1], [cur], [cur+1], [cur+2]]
      # this implementation assume an odd number of steps
      assert(self.steps % 2 == 1 and self.steps > 1)

      cur_list_index = self.steps // 2 # ex: 0,1,2(current frame index),3,4
      for i in range(self.steps):
        # copy self.steps members to the list
        img_list.append(copy.deepcopy(img))
        pose_list.append(copy.deepcopy(pose))

        # start to shift img and pose list according to 
        if i < cur_list_index:
          img_list[i], pose_list[i] = self.shiftRight(img_list[i], pose_list[i], step_size=cur_list_index - i)
        elif i > cur_list_index:
          img_list[i], pose_list[i] = self.shiftLeft(img_list[i], pose_list[i], step_size=i - cur_list_index)

      self.c_imgs = img_list
      self.poses = pose_list
      return

    def __getitem__(self, index):
      # grep self.steps images and poses at a time, convert pose to quaternion
      # TODO!!!
      img = []
      pose = []
      for i in range(self.steps):
        img.append(load_image(self.c_imgs[i][index]))
        pose.append(self.poses[i][index])

      #pdb.set_trace()
      if self.transform is not None:
        img = [self.transform(i) for i in img]

      if self.target_transform is not None:
        pose = [self.target_transform(i) for i in pose]
      return img, pose

    def __len__(self):
      return self.poses[0].shape[0]


def main():
  """
  visualizes the dataset
  """
  # from common.vis_utils import show_batch, show_stereo_batch
  from torchvision.utils import make_grid
  import torchvision.transforms as transforms
  seq = 'heads'
  mode = 1
  num_workers = 6
  transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  target_transform = transforms.Lambda(lambda x: torch.Tensor(x))
  dset = SevenScenes(seq, '../data/deepslam_data/7Scenes', True, transform, target_transform=target_transform, mode=mode)
  print('Loaded 7Scenes sequence {:s}, length = {:d}'.format(seq, len(dset)))
  pdb.set_trace()

  data_loader = data.DataLoader(dset, batch_size=4, shuffle=True, num_workers=num_workers)

  batch_count = 0
  N = 2
  for batch in data_loader:
    print('Minibatch {:d}'.format(batch_count))
    pdb.set_trace()
    # if mode < 2:
    #   show_batch(make_grid(batch[0], nrow=1, padding=25, normalize=True))
    # elif mode == 2:
    #   lb = make_grid(batch[0][0], nrow=1, padding=25, normalize=True)
    #   rb = make_grid(batch[0][1], nrow=1, padding=25, normalize=True)
    #   show_stereo_batch(lb, rb)

    batch_count += 1
    if batch_count >= N:
      break

if __name__ == '__main__':
  main()
