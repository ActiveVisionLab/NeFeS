import utils.set_sys_path
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from dataset_loaders.utils.color import rgb_to_yuv
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def prepare_data(args, images, poses_train, i_split, hist):
    ''' prepare data for ready to train posenet, return dataloaders '''

    i_train, i_val, i_test = i_split

    img_train = torch.Tensor(images[i_train]).permute(0, 3, 1, 2) # now shape is [N, CH, H, W]
    pose_train = torch.Tensor(poses_train[i_train])
    hist_train = torch.Tensor(hist[i_train])

    trainset = TensorDataset(img_train, pose_train, hist_train)
    if args.render_test == True:
        train_dl = DataLoader(trainset, batch_size=1, shuffle=False)
    else:
        train_dl = DataLoader(trainset, batch_size=1, shuffle=True)
    
    img_val = torch.Tensor(images[i_val]).permute(0, 3, 1, 2) # now shape is [N, CH, H, W]
    pose_val = torch.Tensor(poses_train[i_val])
    hist_val = torch.Tensor(hist[i_val])
    
    valset = TensorDataset(img_val, pose_val, hist_val)
    val_dl = DataLoader(valset, shuffle=False)

    img_test = torch.Tensor(images[i_test]).permute(0, 3, 1, 2) # now shape is [N, CH, H, W]
    pose_test = torch.Tensor(poses_train[i_test])
    hist_test = torch.Tensor(hist[i_test])

    testset = TensorDataset(img_test, pose_test, hist_test)
    test_dl = DataLoader(testset, shuffle=False)

    return train_dl, val_dl, test_dl

def load_Colmap_dataset(args):
    ''' load training data in llff style, currently only support exp. for heads '''
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, factor=args.df,
                                                                  recenter=True, bd_factor=None,
                                                                  spherify=args.spherify, path_zflat=False)
        breakpoint()
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        # if not isinstance(i_test, list):
        #     i_test = [i_test]

        # if args.llffhold > 0:
        #     print('Auto LLFF holdout,', args.llffhold)
        #     i_test = np.arange(images.shape[0])[::args.llffhold]

        i_test = np.arange(231, 334, 1, dtype=int)
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        i_train = i_train[::args.trainskip]
        i_val = i_val[::args.testskip]
        i_test = i_test[::args.testskip]

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.

        i_split = [i_train, i_val, i_test]
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    poses_train = poses[:,:3,:].reshape((poses.shape[0],12)) # get rid of last row [0,0,0,1]
    print("images.shape {}, poses_train.shape {}".format(images.shape, poses_train.shape))

    INPUT_SHAPE = images[0].shape
    H = images[0].shape[0]
    W = images[0].shape[1]
    print("=====================================================================")
    print("INPUT_SHAPE:", INPUT_SHAPE)

    hist=None
    if args.encode_hist:
        imgs = torch.Tensor(images).permute(0,3,1,2)
        yuv = rgb_to_yuv(imgs)
        y_img = yuv[:,0] # extract y channel only
        hist = [torch.histc(y_img[i], bins=args.hist_bin, min=0., max=1.) for i in np.arange(imgs.shape[0])] # basically same as other dataloaders but in batch
        hist = torch.stack(hist)
        hist = torch.round(hist/(H*W)*100) # convert to histogram density, in terms of percentage per bin
        hist = np.asarray(hist)
    return images, poses_train, render_poses, hwf, i_split, near, far, hist

def load_dataset(args):
    ''' load posenet training data '''
    if args.dataset_type == 'llff':
        if args.no_bd_factor:
            bd_factor = None
        else:
            bd_factor = 0.75
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=bd_factor,
                                                                  spherify=args.spherify)

        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.

        i_split = [i_train, i_val, i_test]
    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, True, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        # breakpoint()
        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:]) # [400, 400, 400, 3]
        else:
            images = images[...,:3] # 
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    poses_train = poses[:,:3,:].reshape((poses.shape[0],12)) # get rid of last row [0,0,0,1]
    print("images.shape {}, poses_train.shape {}".format(images.shape, poses_train.shape))

    INPUT_SHAPE = images[0].shape
    print("=====================================================================")
    print("INPUT_SHAPE:", INPUT_SHAPE)

    hist=None
    # if args.encode_hist:
    hist = np.zeros((images.shape[0],10))
    return images, poses_train, render_poses, hwf, i_split, near, far, hist