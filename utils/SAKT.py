import torch
import torch.nn as nn
from torch.nn import functional as F


class CrossBranchAlign(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.scale = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, dim, 1, 1))

        # anomaly-driven channel gate
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_normal, feat_anomaly):
        # anomaly statistics (direction only)
        mu_a = feat_anomaly.mean(dim=[2, 3], keepdim=True).detach()
        std_a = feat_anomaly.std(dim=[2, 3], keepdim=True).detach() + self.eps

        # normal statistics
        mu_n = feat_normal.mean(dim=[2, 3], keepdim=True)
        std_n = feat_normal.std(dim=[2, 3], keepdim=True) + self.eps

        # normalize normal feature
        feat_normed = (feat_normal - mu_n) / std_n

        # anomaly-guided re-scaling
        aligned = feat_normed * std_a + mu_a

        # channel-wise gate decides injection strength
        gate = self.gate(std_a)

        # soft alignment, not replacement
        out = feat_normal + gate * self.scale * (aligned - feat_normal) + self.shift
        return out


class AlignmentProjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1)
        )
        self.align = CrossBranchAlign(dim)
        self.norm = nn.GroupNorm(4, dim)

    def forward(self, feat_n, feat_a=None):
        feat_n = self.encoder(feat_n)

        if feat_a is not None:
            with torch.no_grad():
                feat_a = self.encoder(feat_a)
            feat_n = self.align(feat_n, feat_a)

        feat_n = self.norm(feat_n)
        return feat_n


class SAKTLayer(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.proj_a = AlignmentProjection(base * 4)
        self.proj_b = AlignmentProjection(base * 8)
        self.proj_c = AlignmentProjection(base * 16)

    def forward(self, features, features_noise=False):
        if features_noise is not False:
            out_noise = [
                self.proj_a(features_noise[0]),
                self.proj_b(features_noise[1]),
                self.proj_c(features_noise[2]),
            ]

            out_normal = [
                self.proj_a(features[0], features_noise[0]),
                self.proj_b(features[1], features_noise[1]),
                self.proj_c(features[2], features_noise[2]),
            ]

            return out_noise, out_normal
        else:
            return [
                self.proj_a(features[0]),
                self.proj_b(features[1]),
                self.proj_c(features[2]),
            ]


def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
    return loss


class StatisticalAlignLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, feat_n, feat_a):
        # feat: [B, C, H, W]
        mu_n = feat_n.mean(dim=[2, 3])
        mu_a = feat_a.mean(dim=[2, 3]).detach()

        std_n = feat_n.std(dim=[2, 3])
        std_a = feat_a.std(dim=[2, 3]).detach()

        loss_mean = F.mse_loss(mu_n, mu_a)
        loss_std = F.mse_loss(std_n, std_a)

        return loss_mean + loss_std


class SCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = StatisticalAlignLoss()

    def forward(self, projected_noised_feature, projected_normal_feature):
        total_loss = 0
        for a, n in zip(projected_noised_feature, projected_normal_feature):
            total_loss += self.loss_fn(n, a)
        return total_loss


if __name__ == '__main__':
    model = SAKTLayer()
    print(model)
