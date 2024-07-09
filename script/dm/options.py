import configargparse
def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
    parser.add_argument("--multi_gpu", action='store_true', help='use multiple gpu on the server')

    # data loader
    parser.add_argument("--trainskip", type=int, default=1, help='will load 1/N images from train sets, useful for large datasets like 7 Scenes')
    parser.add_argument("--df", type=float, default=1., help='image downscale factor')
    parser.add_argument("--reduce_embedding", type=int, default=-1, help='fourier embedding mode: -1: paper default, \
                                                                        0: reduce by half, 1: remove embedding, 2: DNeRF embedding')
    parser.add_argument("--epochToMaxFreq", type=int, default=-1, help='DNeRF embedding mode: (based on DNeRF paper): \
                                                                        hyper-parameter for when α should reach the maximum number of frequencies m')
    parser.add_argument("--render_pose_only", action='store_true', help='render a spiral video for 7 Scene')
    parser.add_argument("--save_pose_avg_stats", action='store_true', help='save a pose avg stats to unify NeRF, posenet, nerf tracking training')
    parser.add_argument("--load_pose_avg_stats", action='store_true', help='load precomputed pose avg stats to unify NeRF, posenet, nerf tracking training')
    parser.add_argument("--i_eval",   type=int, default=50, help='frequency of eval posenet result')
    parser.add_argument("--save_all_ckpt", action='store_true', help='save all ckpts for each epoch')
    parser.add_argument("--train_local_nerf", type=int, default=-1, help='train local NeRF with ith training sequence only, ie. Stairs can pick 0~3')
    parser.add_argument("--render_video_train", action='store_true', help='render train set NeRF and save as video, make sure i_eval is True')
    parser.add_argument("--render_video_test", action='store_true', help='render val set NeRF and save as video,  make sure i_eval is True')
    parser.add_argument("--no_DNeRF_viewdir", action='store_true', default=False, help='will not use DNeRF in viewdir encoding')
    parser.add_argument("--val_on_psnr", action='store_true', default=False, help='EarlyStopping with max validation psnr')
    parser.add_argument("--feature_matching_lvl", nargs='+', type=int, default=[0], 
                        help='lvl of features used for feature matching, default use lvl [0,1,2]')

    ##################### APR Settings ########################
    parser.add_argument("--pose_only", type=int, default=0, help='APR refinement, \
                        2: DFM as post-processing to APR \
                        3: DFM post-processing to predicted poses')
    parser.add_argument("--learning_rate", type=float, default=0.00001, help='learning rate')
    parser.add_argument("--batch_size", type=int, default=1, help='train posenet only')
    parser.add_argument("--pretrain_model_path", type=str, default='', help='model path of pretrained pose regrssion model')
    parser.add_argument("--pretrain_featurenet_path", type=str, default='', help='model path of pretrained featurenet model')
    parser.add_argument("--model_name", type=str, help='pose model output folder name')
    parser.add_argument("--patience", nargs='+', type=int, default=[200, 50], help='set training schedule for patience [EarlyStopping, reduceLR]')
    parser.add_argument("--resize_factor", type=int, default=2, help='image resize downsample ratio')
    parser.add_argument("--freezeBN", action='store_true', help='Freeze the Batch Norm layers they exist in the APR model')
    parser.add_argument("--preprocess_ImgNet", action='store_true', help='Normalize input data for PoseNet')
    parser.add_argument("--eval", action='store_true', help='eval model')
    parser.add_argument("--no_save_multiple", action='store_true', help='default, save multiple posenet model, if true, save only one posenet model')
    parser.add_argument("--resnet34", action='store_true', default=False, help='use resnet34 backbone instead of mobilenetV2')
    parser.add_argument("--efficientnet", action='store_true', default=False, help='use efficientnet-b3 backbone instead of mobilenetV2')
    parser.add_argument("--efficientnet_block", type=int, default=6, help='choose which features from feature block (0-6) of efficientnet to use')
    parser.add_argument("--dropout", type=float, default=0.5, help='dropout rate for resnet34 backbone')
    parser.add_argument("--dfnet", action='store_true', default=False, help='use dfnet')
    # parser.add_argument("--DFNetEB0", action='store_true', default=False, help='use DFNet_EB0')
    parser.add_argument("--val_batch_size", type=int, default=1, help='batch_size for validation, higher number leads to faster speed')

    ##################### Forward Network Method settings ########################
    parser.add_argument("--fw_net_method", type=str, default='APR', choices=('APR', 'NetVlad', 'pixloc', 'DSAC') ,help='method of forward networks before post-processing, options: APR, pixloc, dsacstar. TODO: TO BE Obsolete')

    ##################### NeRF Settings ########################
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='../logs/', help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, help='layers in network')
    parser.add_argument("--netwidth", type=int, default=128, help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=128, help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=1536, help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=2097152, help='number of pts sent through network in parallel, defualt is 2^21, consider reduce this if GPU OOM')
    parser.add_argument("--no_batching", action='store_true', help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, help='specific weights npy file to reload for coarse network')
    parser.add_argument("--no_grad_update", default=True, help='do not update nerf in training')
    parser.add_argument("--per_pixel", default=False, action='store_true', help='using per pixel cosine similarity loss instead of per channel, default False')

    # NeRF-w training options
    parser.add_argument("--NeRFW", action='store_true', default=True, help='new implementation for NeRFW')
    parser.add_argument("--N_vocab", type=int, default=1000,
                        help='''number of vocabulary (number of images) 
                                in the dataset for nn.Embedding''')
    parser.add_argument("--fix_index", action='store_true', help='fix training frame index as 0')
    parser.add_argument("--encode_hist", default=True, action='store_true', help='encode histogram instead of frame index')
    parser.add_argument("--hist_bin", type=int, default=10, help='image histogram bin size')
    parser.add_argument("--in_channels_a", type=int, default=50, help='appearance embedding dimension, hist_bin*N_a when embedding histogram')
    parser.add_argument("--in_channels_t", type=int, default=20, help='transient embedding dimension, hist_bin*N_tau when embedding histogram')
    parser.add_argument("--svd_reg", default=False, action='store_true', help='use svd regularize output at training')
    parser.add_argument("--ffmlp", action='store_true', default=False, help='using NeRF with tiny-cuda-nn acceleration on mlp')
    parser.add_argument("--tcnn", action='store_true', default=False, help='using NeRF with tiny-cuda-nn acceleration, with hash encoding')
    parser.add_argument("--sh_tcnn", action='store_true', default=False, help='using NeRF with tiny-cuda-nn acceleration, with hash encoding')
    parser.add_argument("--sh_nff", action='store_true', default=False, help='Neural Feature Field (nff) with sh+tcnn+feature')
    parser.add_argument("--nerfh_nff", action='store_true', default=True, help='Neural Feature Field (nff) with pure pytorch NeRF-H')
    parser.add_argument("--nerfh_nff2", action='store_true', default=False, help='Neural Feature Field (nff) with experimental arch. currently obsolete')
    parser.add_argument("--hash_level", type=int, default=16, help='hash encoding Number of levels')
    parser.add_argument("--bound", type=float, default=25, help='set the bound value of the scene before hash encoding. Should be able to shrink sampling points value to 0-1')
    parser.add_argument("--transient_at_test", action='store_true', default=False, help='determine whether to predict transient appearance at the test time')
    parser.add_argument("--set_near_far", action='store_true', default=False, help='use customized near and far for training, must use with --near_far')
    parser.add_argument("--near_far", nargs='+', type=float, default=[0.5, 2.5], help='setting near, far params, NDC [0., 1.], no NDC [0.5, 2.5]')
    parser.add_argument("--semantic", action='store_true', default=False, help='using semantic segmentation mask to remove temporal noise at training time')
    parser.add_argument("--feature_dim", type=int, default=128, help='dimension of nff output feature, ex. 16 or 128')
    parser.add_argument("--depth", action='store_true', default=False, help='using monocular depth to improve geometry quality')

    # Neural Feature Field training options
    parser.add_argument("--out_channel_size", type=int, default=3, help='channels size of output of nerf, either 3 for rgb, or 128 for features')
    parser.add_argument("--NFF", default=False, action='store_true', help='neural feature field like GIRAFFE')
    parser.add_argument("--n_feat", type=int, default=128, help='number of features from output of NeRF, should be same as out_channel_size')
    parser.add_argument("--input_dim", type=int, default=128, help='input dimension of decoder')
    parser.add_argument("--n_blocks", type=int, default=4, help='decoder blocks for upsampling, default 4 (i.e. 15x27 -> 240x427)')
    parser.add_argument("--min_feat", type=int, default=32, help='minimum features of decoder')
    parser.add_argument("--tinyimg", action='store_true', default=False, help='render nerf img in a tiny scale image, this is a temporal compromise for direct feature matching, must FIX later')
    parser.add_argument("--tinyscale", type=float, default=4., help='determine the scale of downsizing nerf rendring, must FIX later')
    parser.add_argument("--use_fusion_res", default=False, action='store_true', help='add residual connection to the fusion block')
    parser.add_argument("--no_fusion_BN", default=False, action='store_true', help='no batchnorm for fusion block')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=64,help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', default=True, help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_only", action='store_true', help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--use_sv", action='store_true', default=False, help='Use sparse voxel volume')
    parser.add_argument("--retreat", action='store_true', default=False, help='Whether to retreat to original NeRF sampling strategy when no valid voxel along the ray')
    parser.add_argument("--use_fine_only", action='store_true', default=False, help='only use inverse CDF fine sampling for fine MLP')

    # Direct Feature Matching options
    parser.add_argument('--PoseEstimatorType', type=str, default='DFNet', choices=('PoseNet', 'MsTransformer', 'DFNet', 'NetVlad'), help='Methods for inital pose estimation')
    parser.add_argument("--sparse_feature", default=False, action='store_true', help='render NeRF based on sampled keypoints and perform direct sparse feature matching')
    # parser.add_argument("--opt_option", type=int, default=0, help='DFM option for pose only==4, \
    #                                                         Option 1: NeRF RGB + DFNet CNN, \
    #                                                         Option 2: NFF RGB + DFNet CNN, \
    #                                                         Option 3: NFF Feature, \
    #                                                         Option 4: TODO: NFF Sparse Feature')
    parser.add_argument("--lr_r", type=float, default=0.01,help='rotation update step learning rate, default 0.01 for Cambridge. Suggest to use 0.0087 (0.5deg) for 7Scenes')
    parser.add_argument("--lr_t", type=float, default=0.1,help='rotation update step learning rate, default 0.1 for Cambridge. Suggest to use 0.01 for 7Scenes')
    parser.add_argument("--opt_iter", type=int, default=50, help='optimization iterations for pose refinement, default 50')
    parser.add_argument("--no_verification_step", action='store_true', default=False, help='True to disable verification step. (beneficial for Cambrideg DFNet)')

    # legacy mesh options
    parser.add_argument("--mesh_only", action='store_true', help='do not optimize, reload weights and save mesh to a file')
    parser.add_argument("--mesh_grid_size", type=int, default=80,help='number of grid points to sample in each dimension for marching cubes')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=1, help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## legacy blender flags
    parser.add_argument("--white_bkgd", action='store_true', help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,  help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument("--no_bd_factor", action='store_true', default=False, help='do not use bd factor')

    # featruremetric supervision
    # parser.add_argument("--featuremetric", action='store_true', help='use featuremetric supervision if true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=200, help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=200, help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, help='frequency of render_poses video saving')

    # support for ngp-pl, part of useful configs for inference
    parser.add_argument("--ngp_nefes", default=False, action='store_true', help='use ngp-pl based nefes if true')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument("--NeFeS", action='store_true', default=False, 
                        help='my implementation for ngp_NeFeS, ngp-based Neural Feature Synthesizer')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')
    parser.add_argument('--loss_mode', type=int, default=0, choices=(0,1,2),
                        help='''specify to use nerfw photometric loss and feature loss, with additional CNN feature fusion loss \
                        0: 'color', 1: 'color+feature', 2: 'color+feature+fusion'
                        ''')
    return parser