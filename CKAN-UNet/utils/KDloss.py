import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class KDloss(nn.Module):

    def __init__(self, lambda_x):
        super(KDloss, self).__init__()
        self.lambda_x = lambda_x

    def inter_fd(self, f_s, f_t):
        s_C, t_C, s_H, t_H = f_s.shape[1], f_t.shape[1], f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass

        idx_s = random.sample(range(s_C), min(s_C, t_C))
        idx_t = random.sample(range(t_C), min(s_C, t_C))

        inter_fd_loss = F.mse_loss(f_s[:, idx_s, :, :], f_t[:, idx_t, :, :].detach())
        return inter_fd_loss

    def intra_fd(self, f_s):
        sorted_s, indices_s = torch.sort(F.normalize(f_s, p=2, dim=(2, 3)).mean([0, 2, 3]), dim=0, descending=True)
        f_s = torch.index_select(f_s, 1, indices_s)
        intra_fd_loss = F.mse_loss(f_s[:, 0:f_s.shape[1] // 2, :, :], f_s[:, f_s.shape[1] // 2: f_s.shape[1], :, :])
        return intra_fd_loss

    def forward(self, feature, feature_decoder, final_up):
        f1 = feature[0]
        f2 = feature[1]
        f3 = feature[2]
        f4 = feature[3]

        f1_d = feature_decoder[0]
        f2_d = feature_decoder[1]
        f3_d = feature_decoder[2]

        final_layer = final_up

        loss = (self.intra_fd(f1) + self.intra_fd(f2) + self.intra_fd(f3) + self.intra_fd(f4)) / 4
        loss += (self.intra_fd(f1_d) + self.intra_fd(f2_d) + self.intra_fd(f3_d)) / 3

        loss += (self.inter_fd(f1_d, final_layer) + self.inter_fd(f2_d, final_layer)
                 + self.inter_fd(f3_d, final_layer) + self.inter_fd(f1, final_layer)
                 + self.inter_fd(f2, final_layer) + self.inter_fd(f3, final_layer)
                 + self.inter_fd(f4, final_layer)) / 7

        loss = loss * self.lambda_x
        return loss
