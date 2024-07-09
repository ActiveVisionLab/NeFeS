import configargparse
def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
    parser.add_argument("--device", type=int, default=-1, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument("--multi_gpu", action='store_true', help='use multiple gpu on the server')
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='../logs', help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', help='input data directory')

    # data loader
    parser.add_argument("--trainskip", type=int, default=1, help='will load 1/N images from train sets, useful for large datasets like 7 Scenes')
    parser.add_argument("--df", type=float, default=1., help='image downscale factor')
    parser.add_argument("--reduce_embedding", type=int, default=-1, help='fourier embedding mode: -1: paper default, \
                                                                        0: reduce by half, 1: remove embedding, 2: DNeRF embedding')
    parser.add_argument("--epochToMaxFreq", type=int, default=-1, help='DNeRF embedding mode: (based on Nerfie paper): \
                                                                        hyper-parameter for when α should reach the maximum number of frequencies m')
    parser.add_argument("--render_pose_only", action='store_true', help='render a spiral video for 7 Scene')
    parser.add_argument("--save_pose_avg_stats", action='store_true', help='save a pose avg stats to unify NeRF, posenet, direct-pn training')
    parser.add_argument("--load_pose_avg_stats", action='store_true', help='load precomputed pose avg stats to unify NeRF, posenet, nerf tracking training')
    parser.add_argument("--train_local_nerf", type=int, default=-1, help='train local NeRF with ith training sequence only, ie. Stairs can pick 0~3')
    parser.add_argument("--render_video_train", action='store_true', help='render train set NeRF and save as video, make sure render_test is True')
    parser.add_argument("--render_video_test", action='store_true', help='render val set NeRF and save as video,  make sure render_test is True')
    parser.add_argument("--frustum_overlap_th", type=float, help='frustsum overlap threshold')
    parser.add_argument("--no_DNeRF_viewdir", action='store_true', default=False, help='will not use DNeRF in viewdir encoding')
    parser.add_argument("--load_unique_view_stats", action='store_true', help='load unique views frame index')
    
    # NeRF training options
    parser.add_argument("--netdepth", type=int, default=8, help='layers in network')
    parser.add_argument("--netwidth", type=int, default=128, help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=128, help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=1536, help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=float, default=0.754, help='Exponential learning rate decay. This can be flexible. In my experience both 0.754 or 5 are fine.')
    parser.add_argument("--chunk", type=int, default=1024*32, help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=2097152, help='number of pts sent through network in parallel, decrease if running out of memory, default: 2^21')
    # parser.add_argument("--no_batching", action='store_true', help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, help='specific weights npy file to reload for coarse network')
    parser.add_argument("--no_grad_update", action='store_true', default=False, help='do not update nerf in training')
    parser.add_argument("--nerfh_nff", action='store_true', default=True, help='Neural Feature Field (nff) with pure pytorch NeRF-H')
    parser.add_argument("--nerfh_nff2", action='store_true', default=False, help='Neural Feature Field (nff) with pure pytorch NeRF-H, without temporal network')
    parser.add_argument("--ffmlp", action='store_true', default=False, help='using NeRF with tiny-cuda-nn acceleration on mlp')
    parser.add_argument("--tcnn", action='store_true', default=False, help='using NeRF with tiny-cuda-nn acceleration, with hash encoding')
    parser.add_argument("--sh_tcnn", action='store_true', default=False, help='using NeRF with tiny-cuda-nn acceleration, with hash encoding')
    parser.add_argument("--sh_nff", action='store_true', default=False, help='Neural Feature Field (nff) with sh+tcnn+feature')
    parser.add_argument("--hash_level", type=int, default=16, help='hash encoding Number of levels')
    parser.add_argument("--transient_at_test", action='store_true', default=False, help='determine whether to predict transient appearance at the test time')
    parser.add_argument("--loss_mode", type=int, default=0, help='experiment on NeRF loss configuration, try to solve the bad reload issue. 0 is defualt, 1 is my mod')
    parser.add_argument("--tvloss", action='store_true', default=False, help='total variation loss for better geometry')
    parser.add_argument("--tvloss_w", type=float, default=1e-6, help='weight of tv loss, defautl 1e-6')
    parser.add_argument("--l1normloss", action='store_true', default=False, help='L1 Norm Loss for better geometry')
    parser.add_argument("--l1normloss_w", type=float, default=1e-6, help='weight of L1 Norm loss, defautl 1e-6')
    parser.add_argument("--bound", type=float, default=25, help='set the bound value of the scene before hash encoding. Should be able to shrink sampling points value to 0-1')
    parser.add_argument("--set_near_far", action='store_true', default=False, help='use customized near and far for training, must use with --near_far')
    parser.add_argument("--near_far", nargs='+', type=float, default=[0.5, 2.5], help='setting near, far params, NDC [0., 1.], no NDC [0.5, 2.5]')
    parser.add_argument("--semantic", action='store_true', default=False, help='using semantic segmentation mask to remove temporal noise at training time')
    parser.add_argument("--feature_dim", type=int, default=128, help='dimension of nff output feature, ex. 16 or 128')
    parser.add_argument("--depth", action='store_true', default=False, help='using monocular depth to improve geometry quality')
    parser.add_argument("--new_schedule", type=int, default=1, help='1: train color only, 2: train color+feature/color+feauture+fusion')

    # NeRF-w training options
    parser.add_argument("--NeRFW", action='store_true', default=True, help='my implementation for NeRFW, to enable NeRF-Hist training, please add --encode_hist. TODO: obsolete this arg')
    parser.add_argument("--N_vocab", type=int, default=1000,
                        help='''number of vocabulary (number of images) 
                                in the dataset for nn.Embedding''')
    parser.add_argument("--fix_index", action='store_true', help='fix training frame index as 0')
    parser.add_argument("--encode_hist", default=True, action='store_true', help='encode histogram instead of frame index. TODO: obsolete this arg')
    parser.add_argument("--hist_bin", type=int, default=10, help='image histogram bin size')
    parser.add_argument("--in_channels_a", type=int, default=50, help='appearance embedding dimension, hist_bin*N_a when embedding histogram')
    parser.add_argument("--in_channels_t", type=int, default=20, help='transient embedding dimension, hist_bin*N_tau when embedding histogram')
    parser.add_argument("--color_loss_only", action='store_true', default=False, help='specify to use photometric loss only, for experimental purpose.')
    parser.add_argument("--color_feat_loss", action='store_true', default=False, help='specify to use photometric loss and feature loss')
    parser.add_argument("--color_feat_fusion_loss", action='store_true', default=False, help='specify to use photometric loss and feature loss, with additional CNN feature fusion loss')
    parser.add_argument("--color_feat_fusion_nerfw_loss", action='store_true', default=False, help='specify to use nerfw photometric loss and feature loss, with additional CNN feature fusion loss')
    parser.add_argument("--sigma_sparsity_loss", action='store_true', default=False, help='cauchy sigma sparsity loss')
    parser.add_argument("--sigma_sparsity_loss2", action='store_true', default=False, help='cauchy sigma sparsity loss')
    parser.add_argument("--batch_size", type=int, default=4, help='# of train images to fetch from the dataloader every iteration.')
    parser.add_argument("--val_batch_size", type=int, default=1, help='batch size for validating NeRF')

    # Neural Feature Field training options
    parser.add_argument("--out_channel_size", type=int, default=3, help='channels size of output of nerf, either 3 for rgb, or 128 for features')
    parser.add_argument("--NFF", default=False, action='store_true', help='neural feature field like GIRAFFE (scheme2)')
    parser.add_argument("--n_blocks", type=int, default=4, help='decoder blocks for upsampling, default 4 (i.e. 15x27 -> 240x427)')
    parser.add_argument("--min_feat", type=int, default=32, help='minimum features of decoder')
    parser.add_argument("--tinyimg", action='store_true', default=False, help='render nerf img in a tiny scale image, this is a temporal compromise for direct feature matching, must FIX later')
    parser.add_argument("--tinyscale", type=float, default=4., help='determine the scale of downsizing nerf rendring, must FIX later')
    parser.add_argument("--use_fusion_res", default=False, action='store_true', help='add residual connection to the fusion block')
    parser.add_argument("--no_fusion_BN", default=False, action='store_true', help='no batchnorm for fusion block')

    # NeRF rendering options
    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=64,help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", default=True, action='store_true', help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_only", action='store_true', help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--use_sv", action='store_true', default=False, help='Use sparse voxel volume')
    parser.add_argument("--sv_file", type=str, default='', help='path to SV file')
    parser.add_argument("--retreat", action='store_true', default=False, help='Whether to retreat to original NeRF sampling strategy when no valid voxel along the ray')
    parser.add_argument("--use_fine_only", action='store_true', default=False, help='only use inverse CDF fine sampling for fine MLP')

    ##################### DFNet Settings ########################
    parser.add_argument("--dfnet", action='store_true', default=False, help='use dfnet')
    parser.add_argument("--pretrain_featurenet_path", type=str, default='', help='model path of pretrained featurenet model')
    parser.add_argument("--freezeBN", action='store_true', help='Freeze the Batch Norm layer at training PoseNet')
    parser.add_argument("--svd_reg", default=False, action='store_true', help='use svd regularize output at training')

    # legacy mesh options
    parser.add_argument("--mesh_only", action='store_true', help='do not optimize, reload weights and save mesh to a file')
    parser.add_argument("--mesh_grid_size", type=int, default=80,help='number of grid points to sample in each dimension for marching cubes')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,default=.5, help='fraction of img taken for central crops')
    parser.add_argument("--epochs", type=int, default=600,help='number of epochs to train')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', help='options: llff / 7Scenes')
    parser.add_argument("--testskip", type=int, default=1, help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## legacy blender flags
    parser.add_argument("--white_bkgd", action='store_true', help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,  help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument("--no_bd_factor", action='store_true', default=False, help='do not use bd factor')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=10, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=200, help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=200, help='frequency of evaluating on testset and save testset renderings')
    parser.add_argument("--i_video",   type=int, default=50000, help='frequency of render_poses video saving')

    return parser