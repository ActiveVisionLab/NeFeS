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
# import pickle
# import pdb,copy
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

import transforms3d.quaternions as txq # Warning: outdated package

def process_poses_quat2mat(poses_in):
  """
  processes the raw pose from dataset [Qw, Qx, Qy, Qz, Tx, Ty, Tz] to [R, T]
  produce logq
  :param poses_in: N x 7
  :return: processed poses N x 12
  """
  poses_out = np.zeros((len(poses_in), 3, 4)) # (1000,12)
  for i in range(len(poses_out)):
    q = poses_in[i,:4]

    # use transforms3d ([Qw, Qx, Qy, Qz])
    R = txq.quat2mat(q)

    # # use scipy, same result as transform3d
    # from scipy.spatial.transform import Rotation as R
    # # convert [Qw, Qx, Qy, Qz] -> [Qx, Qy, Qz, Qw]
    # q = np.array([q[1], q[2], q[3], q[0]])
    # R2 = R.from_quat(q) # scipy takes (x, y, z, w) -> R
    # R2 = R2.as_matrix()

    poses_out[i,:3,:3] = R
    poses_out[i,:3,3] = poses_in[i,4:]
  poses_out = poses_out.reshape(poses_out.shape[0],12)
  return poses_out.reshape(poses_out.shape[0],12)

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

class SevenScenes_colmap(data.Dataset):
    def __init__(self, args, scene, data_path, train, transform=None,
                 target_transform=None, mode=0, seed=7,
                 df=1., trainskip=1, testskip=1, hwf=[480,640,585.], 
                 ret_idx=False, fix_idx=False, ret_hist=False, hist_bin=10):
      """
      load 7scenes data with COLMAP poses from Brachmann21 "On the Limits of Pseudo Ground Truth in Visual Camera Re-localisation"
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
      self.gt_idx = np.empty((0,), dtype=int)
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
          frame_idx = frame_idx[::trainskip]
        elif not train and testskip > 1:
          frame_idx = frame_idx[::testskip]


        self.gt_idx = np.hstack((self.gt_idx, gt_offset+frame_idx))
        gt_offset += len(p_filenames)
        c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i)) for i in frame_idx]
        d_imgs = [osp.join(seq_dir, 'frame-{:06d}.depth.png'.format(i)) for i in frame_idx]
        self.c_imgs.extend(c_imgs)
        self.d_imgs.extend(d_imgs)

      ## parsing GT poses from colmap (for img size of 480x640x3) ###
      print("using colmap psudo poses")
      if train:
        colmap_pose_file = osp.join(base_dir, scene+'_train.txt')
      else:
        colmap_pose_file = osp.join(base_dir, scene+'_test.txt')

      colmap_poses = np.loadtxt(colmap_pose_file, dtype=str)

      # making a dictionary based on colmap GT files, i.e., {'seq-01/frame-000685.color.png': array(['0.941876', '0.087411', '-0.323503', '0.0239663', '0.525913', '1.011429', '-0.233122', '526.22']}
      colmap_pose_dict = {}
      for i in range(len(colmap_poses)):
        colmap_pose_dict[colmap_poses[i, 0]] = colmap_poses[i, 1:]

      # we fetch the quaternion rot, translation t, and focal length f from the dictionary based on c_imgs
      self.poses_qtf = [] # qw qx qy qz tx ty tz f
      for names in self.c_imgs:
        dir1, f_name = osp.split(names)
        dir2, seq_name = osp.split(dir1)
        c_imgs_tmp = osp.join(seq_name,f_name)
        try:
          qtf_data = colmap_pose_dict[c_imgs_tmp].astype(np.float32)
        except:
          print("Error: ", c_imgs_tmp, "check if the colmap file exists")
          breakpoint()
        self.poses_qtf.append(qtf_data)
      self.poses_qtf = np.asarray(self.poses_qtf)
      assert(self.poses_qtf.shape[0] == len(self.c_imgs))

      self.focal = self.poses_qtf[0, -1]
      # convert quaternion to rotation matrix
      self.poses = process_poses_quat2mat(self.poses_qtf[:,:7])

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

def main():
  """
  visualizes the dataset
  """
  # from common.vis_utils import show_batch, show_stereo_batch
  from torchvision.utils import make_grid
  import torchvision.transforms as transforms
  import sys
  sys.path.append('../script/')

  # from script.models.options import config_parser
  from models.options import config_parser

  # use this to run the script
  # python seven_scenes_colmap.py --config ../script/config/7Scenes/config_chess.txt

  parser = config_parser()
  args = parser.parse_args()
  print(parser.format_values())

  data_dir, scene = osp.split(args.datadir) # ../data/7Scenes, chess
  dataset_folder, dataset = osp.split(data_dir) # ../data, 7Scenes
  data_dir = osp.join(dataset_folder, 'deepslam_data', dataset) # ../data/deepslam_data/7Scenes

  num_workers = 6
  transform = transforms.Compose([
        transforms.ToTensor()])
  target_transform = transforms.Lambda(lambda x: torch.Tensor(x))

  kwargs = dict(args=args, scene=scene, data_path=data_dir,
        transform=transform, target_transform=target_transform, 
        df=2, ret_idx=True, fix_idx=False, ret_hist=True, hist_bin=10)
  dset = SevenScenes_colmap(train=True, trainskip=args.trainskip, **kwargs)
  print('Loaded 7Scenes sequence {:s}, length = {:d}'.format(scene, len(dset)))
  breakpoint()

  data_loader = data.DataLoader(dset, batch_size=4, shuffle=False, num_workers=num_workers)

  batch_count = 0
  N = 2
  for batch in data_loader:
    print('Minibatch {:d}'.format(batch_count))

    batch_count += 1
    if batch_count >= N:
      break

if __name__ == '__main__':
  main()
