import torch
import torch.nn as nn
from utils.lie_group_helper import make_c2w
from lietorch import SE3 # SO3

class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, init_c2w=None, lietorch=False):
        """
        :param num_cams: # of frames to be optimized
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        :param lietorch: True/False if True, use lietorch to compute the pose
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        self.lietorch = lietorch
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)

        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3) delta r in se(3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3) delta t in se(3)

    def forward(self, cam_id):
        r = self.r[cam_id]  # (3, ) axis-angle, lie algebra
        t = self.t[cam_id]  # (3, )

        if len(t.size()) == 2: # more than 1 query pose
            if self.lietorch:
                t_r = torch.cat([t, r], dim=1)
                c2w = SE3.exp(t_r).matrix() # input should be (6) or 1x6, [t,t,t,r,r,r]
            else:
                c2w = make_c2w(r, t)
            # learn a delta pose between init pose and target pose, if a init pose is provided
            if self.init_c2w is not None:

                c2w[:,:3,:3] = c2w[:,:3,:3] @ self.init_c2w[cam_id,:3,:3]
                c2w[:,:3,3] = c2w[:,:3,3] + self.init_c2w[cam_id,:3,3]
        
        elif len(t.size()) == 1: # only 1 query pose
            if self.lietorch:
                t_r = torch.cat([t, r], dim=0)
                c2w = SE3.exp(t_r).matrix()
            else:
                c2w = make_c2w(r, t)  # (4, 4)
            if self.init_c2w is not None:
                c2w[:3,:3] = c2w[:3,:3] @ self.init_c2w[cam_id,:3,:3]
                c2w[:3,3] = c2w[:3,3] + self.init_c2w[cam_id,:3,3]
        return c2w
