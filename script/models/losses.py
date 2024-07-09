import torch
from torch import nn

class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_fine'], targets)
        if 'rgb_coarse' in inputs:
            loss += self.loss(inputs['rgb_coarse'], targets)

        return self.coef * loss

class ColorFeatureLoss(nn.Module):
    def __init__(self, coef=1, L1_loss=False):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean') # color loss
        if L1_loss: # feature loss
            print("use l1 feature loss")
            self.f_loss = nn.L1Loss(reduction='mean')
        else:
            print("use mse feature loss")
            self.f_loss = nn.MSELoss(reduction='mean')


    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_fine'], targets['rgb'])
        if 'rgb_coarse' in inputs:
            loss += self.loss(inputs['rgb_coarse'], targets['rgb'])
        
        loss_f = self.f_loss(inputs['feat_fine'], targets['feat'])

        if 'feat_coarse' in inputs:
            loss_f += self.f_loss(inputs['feat_coarse'], targets['feat'])

        return loss, loss_f

class ColorFeatureFusionLoss(nn.Module):
    def __init__(self, coef=1, L1_loss=False, cos_loss=False):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean') # color loss
        self.cos_loss = cos_loss
        if L1_loss: # feature loss
            print("use l1 feature loss")
            self.f_loss = nn.L1Loss(reduction='mean')
        elif cos_loss:
            print("use cos feature loss")
            self.f_loss = nn.CosineSimilarity(dim=1) # feature cosine similarity
        else:
            print("use mse feature loss")
            self.f_loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets, switch_on=True, color_only_switch=False):
        ''' switch_on: True: add Fusion loss, False: use Color and Feature loss only '''

        # inputs['rgb_fine'].retain_grad()
        # inputs['rgb_coarse'].retain_grad()

        loss = self.loss(inputs['rgb_fine'], targets['rgb'])
        if 'rgb_coarse' in inputs:
            loss += self.loss(inputs['rgb_coarse'], targets['rgb'])
        
        if color_only_switch==True:
            return loss

        if self.cos_loss: # cosine similarity loss
            loss_f = 1-self.f_loss(inputs['feat_fine'], targets['feat']).mean() # cosine similarity loss like in DFM
            if 'feat_coarse' in inputs:
                loss_f += 1-self.f_loss(inputs['feat_coarse'], targets['feat']).mean()
        else:
            loss_f = self.f_loss(inputs['feat_fine'], targets['feat'])
            if 'feat_coarse' in inputs:
                loss_f += self.f_loss(inputs['feat_coarse'], targets['feat'])

        if switch_on: # this is for the fusion block
            if self.cos_loss:
                loss_fusion = 1 - self.f_loss(inputs['feat_fusion'], targets['feat']).mean()
            else:
                loss_fusion = self.f_loss(inputs['feat_fusion'], targets['feat'])
            return loss, loss_f, loss_fusion
        else:
            return loss, loss_f

class NerfWLoss(nn.Module):
    """
    Equation 13 in the NeRF-W paper.
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss (1st term in equation 13)
        b_l: beta loss (2nd term in equation 13)
        s_l: sigma loss (3rd term in equation 13)
    targets # [N, 3]
    inputs['rgb_coarse'] # [N, 3]
    inputs['rgb_fine'] # [N, 3]
    inputs['beta'] # [N]
    inputs['transient_sigmas'] # [N, 2*N_Samples]
    :return:
    """
    def __init__(self, coef=1, lambda_u=0.01):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u

    def forward(self, inputs, targets, loss_mode=0):

        ret = {}
        ret['c_l'] = 0.5 * ((inputs['rgb_coarse']-targets)**2).mean()
        if 'rgb_fine' in inputs:
            if 'beta' not in inputs: # no transient head, normal MSE loss
                ret['f_l'] = 0.5 * ((inputs['rgb_fine']-targets)**2).mean()
            else:

                ret['f_l'] = ((inputs['rgb_fine']-targets)**2/(2*inputs['beta'].unsqueeze(1)**2)).mean()
                ret['b_l'] = 3 + torch.log(inputs['beta']).mean() # +3 to make it positive
                # print("ret['b_l']", ret['b_l'])
                ret['s_l'] = self.lambda_u * inputs['transient_sigmas'].mean()
                # print("ret['s_l']", ret['s_l'])

        for k, v in ret.items():
            ret[k] = self.coef * v

        loss = sum(l for l in ret.values())

        return loss

class ColorFeatureFusionNerfWLoss(nn.Module):
    def __init__(self, coef=1, L1_loss=False, lambda_u=0.01):
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u

        self.loss = NerfWLoss(coef=coef, lambda_u=lambda_u)
        if L1_loss: # feature loss
            print("use l1 feature loss")
            self.f_loss = nn.L1Loss(reduction='mean')
        else:
            print("use mse feature loss")
            self.f_loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets, switch_on=True, color_only_switch=False):
        ''' switch_on: True: add Fusion loss, False: use Color and Feature loss only '''

        # NeRFW loss for color
        loss = self.loss(inputs, targets['rgb']) # NerfWLoss

        if color_only_switch==True:
            return loss

        # feature loss
        loss_f = self.f_loss(inputs['feat_fine'], targets['feat'])
        if 'feat_coarse' in inputs:
            loss_f += self.f_loss(inputs['feat_coarse'], targets['feat'])

        # feature fusion loss
        if switch_on:
            loss_fusion = self.f_loss(inputs['feat_fusion'], targets['feat'])
            return loss, loss_f, loss_fusion
        else:
            return loss, loss_f

loss_dict = {'color': ColorLoss,
             'color_feat': ColorFeatureLoss,
             'nerfw': NerfWLoss,
             'color_feat_fusion': ColorFeatureFusionLoss,
             'color_feat_fusion_nerfw': ColorFeatureFusionNerfWLoss}

def compute_depth_loss(pred_depth, gt_depth):
    ''' From Ryan original version
    gt_depth: (DPT depth)
    pred_depth: (NeRF depth)
    lr_decay:...
    '''
    t_pred = torch.median(pred_depth)
    s_pred = torch.mean(torch.abs(pred_depth - t_pred))

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))

    pred_depth_n = (pred_depth - t_pred)/s_pred
    gt_depth_n = (gt_depth - t_gt)/s_gt

    # return torch.mean(torch.abs(pred_depth_n - gt_depth_n))
    return torch.mean(torch.pow(pred_depth_n - gt_depth_n, 2))

# def compute_depth_loss(pred_depth, gt_depth):
#     ''' From Ryan, my mod version
#     gt_depth: (DPT depth)
#     pred_depth: (NeRF depth)
#     lr_decay:...
#     '''

#     t_pred = torch.median(pred_depth)
#     s_pred = torch.mean(torch.abs(pred_depth - t_pred))

#     t_gt = torch.median(gt_depth)
#     s_gt = torch.mean(torch.abs(gt_depth - t_gt))

#     pred_depth_n = (pred_depth - t_pred)/s_pred
#     gt_depth_n = (gt_depth - t_gt)/s_gt

#     loss = torch.mean(torch.pow(pred_depth_n - gt_depth_n, 2))
#     # loss = torch.mean(torch.abs(pred_depth_n - gt_depth_n))
#     return loss

def square_error_loss(emb, sample_resolution):
    tv_x = torch.pow(emb[1:,:,:,:]-emb[:-1,:,:,:], 2).sum()
    tv_y = torch.pow(emb[:,1:,:,:]-emb[:,:-1,:,:], 2).sum()
    tv_z = torch.pow(emb[:,:,1:,:]-emb[:,:,:-1,:], 2).sum()
    return (tv_x + tv_y + tv_z)/(sample_resolution-1)

def compute_TV_loss(emb, cube_size):
    tv_x = torch.pow(emb[1:,:,:,:]-emb[:-1,:,:,:], 2).sum()
    tv_y = torch.pow(emb[:,1:,:,:]-emb[:,:-1,:,:], 2).sum()
    tv_z = torch.pow(emb[:,:,1:,:]-emb[:,:,:-1,:], 2).sum()
    return ((tv_x + tv_y + tv_z)/cube_size).sum()+1e-8

def total_variation_loss(render_kwargs_train, sample_resolution=torch.Tensor([512,256,512])):
    '''
    A total variation loss to smooth out geometry (sigmas) of NeRF
    '''
    # deltas = 0.1270 # new shopfacade params, bound=10. Magic number
    deltas = 9.5238e-02 # old hospital params, bound=8. Magic number
    # TODO: stochastic sample of the voxels V to evaluate the TV term in each optimization step (see Plenoxel paper)
    density_c = render_kwargs_train['network_fn'].density
    density_f = render_kwargs_train['network_fine'].density
    
    # Get resolution and coordinates (continuous)
    # min_vertex = torch.zeros((3,))
    # sample_resolution=torch.Tensor([512,256,512])
    cube_size = torch.floor(sample_resolution/10.0).int() # 10% of the resolution
    min_vertex = torch.stack([torch.randint(0, int(sample_resolution[i]-cube_size[i]), (1,)) for i in range(3)])
    cube_coords = torch.stack(torch.meshgrid(min_vertex[0] + torch.arange(cube_size[0]), min_vertex[1] + torch.arange(cube_size[1]), min_vertex[2] + torch.arange(cube_size[2]), indexing='ij'), dim=-1)

    # normalize sampled_cube_coords to [0,1]
    cube_coords = cube_coords/(sample_resolution-1)
    cube_coords = cube_coords.reshape((-1,3))# [N_samples, 3]

    # Inference hash encoding and density network
    out_c = density_c(cube_coords)
    out_f = density_f(cube_coords)

    sigma_c = out_c['sigma']
    sigma_f = out_f['sigma']

    alpha_c = 1-torch.exp(-deltas*sigma_c)
    alpha_f = 1-torch.exp(-deltas*sigma_f)

    # convert to 16x16x16x32
    alpha_c = alpha_c.reshape(cube_size[0],cube_size[1],cube_size[2],1)
    alpha_f = alpha_f.reshape(cube_size[0],cube_size[1],cube_size[2],1)

    loss_tv_coarse = compute_TV_loss(alpha_c,cube_size)
    loss_tv_fine = compute_TV_loss(alpha_f,cube_size)

    return loss_tv_coarse+loss_tv_fine

    # # Compute loss
    # loss_c = square_error_loss(emb_c, sample_resolution)
    # loss_f = square_error_loss(emb_f, sample_resolution)
    # return loss_c+loss_f

def L1_norm_loss(emb):
    L1_loss = torch.abs(emb).sum()
    num_values = 0
    num_values += emb.numel()
    return L1_loss/num_values

def sigma_sparsity_loss(sigmas):
    # Using Cauchy Sparsity loss on sigma values
    return torch.log(1.0 + 2*sigmas**2).sum(dim=-1)

def sigma_sparsity_loss2(sigmas, lambda_u=0.01):
    return lambda_u * sigmas.mean()

def embeddings_L1_loss(render_kwargs_train, sample_resolution=16):
    '''
    A L1 norm loss on embeddings to smooth out geometry (sigmas) of NeRF

    '''
    encoder_c = render_kwargs_train['network_fn'].encoder
    encoder_f = render_kwargs_train['network_fine'].encoder

    # Get resolution and coordinates (continuous)
    min_vertex = torch.zeros((3,))
    idx = min_vertex + torch.stack([torch.arange(sample_resolution) for _ in range(3)], dim=-1)

    cube_coords = torch.stack(torch.meshgrid(idx[:,0], idx[:,1], idx[:,2], indexing='ij'), dim=-1) # sampled world space coordinate

    # normalize sampled_cube_coords to [0,1]
    cube_coords = cube_coords/(sample_resolution-1)
    cube_coords = cube_coords.reshape((-1,3)) # [N_samples, 3]

    # Inference hash encoding and density network
    emb_c = encoder_c(cube_coords)
    emb_f = encoder_f(cube_coords)

    # convert to 16x16x16x32
    emb_c = emb_c.reshape(sample_resolution,sample_resolution,sample_resolution,32)
    emb_f = emb_f.reshape(sample_resolution,sample_resolution,sample_resolution,32)

    # Compute loss
    loss_c = L1_norm_loss(emb_c)
    loss_f = L1_norm_loss(emb_f)

    return loss_c+loss_f

def compute_smooth_loss(tgt_depth, tgt_img):
    def get_smooth_loss(disp, img):
        """
        Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        # normalize
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        disp = norm_disp

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(
            torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(
            torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    loss = get_smooth_loss(tgt_depth, tgt_img)

    return loss

def depth_loss_dpt(self, pred_depth, gt_depth, weight=None):
        """
        :param pred_depth:  (H, W)
        :param gt_depth:    (H, W)
        :param weight:      (H, W)
        :return:            scalar
        """
        
        t_pred = torch.median(pred_depth)
        s_pred = torch.mean(torch.abs(pred_depth - t_pred))
        t_gt = torch.median(gt_depth)
        s_gt = torch.mean(torch.abs(gt_depth - t_gt))
        pred_depth_n = (pred_depth - t_pred) / s_pred
        gt_depth_n = (gt_depth - t_gt) / s_gt
        if weight is not None:
            loss = F.mse_loss(pred_depth_n, gt_depth_n, reduction='none')
            loss = loss * weight
            loss = loss.sum() / (weight.sum() + 1e-8)
        else:
            loss = F.mse_loss(pred_depth_n, gt_depth_n)
        result_dict = {
                'loss': loss}
        return result_dict