from packaging import version
import torch
from torch import nn
import math
from .sinkhorn import OT

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class CoNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.l2_norm = Normalize(2)

    def forward(self, feat_q, feat_k, i):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        # Therefore, we will include the negatives from the entire minibatch.
        # if self.opt.nce_includes_all_negatives_from_minibatch:
        #     batch_dim_for_bmm = 1
        # else:
        batch_dim_for_bmm = self.opt.batchSize // len(self.opt.gpu_ids)


        ot_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        ot_k = feat_k.view(batch_dim_for_bmm, -1, dim).detach()
        # pos_weight = torch.bmm(feat_c.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        # pos_weight = pos_weight.view(batchSize, 1)

        # ot_q = torch.randn(1, 256, 10)
        # ot_k = torch.randn(1, 256, 10)
        # ot_q = torch.tensor([[[0.6, 0.8],[0.3, 0.95]]])
        # ot_k = torch.tensor([[[0.3, 0.95],[0.6, 0.8]]])

        f = OT(ot_q, ot_k, eps=1.0, max_iter=50)
        f = f.permute(0, 2, 1) * self.opt.ot_weight + 1e-8
        f_max = torch.max(f, -1)[0].view(batchSize, 1)
        # f_tmp = f[0, 0, 1:]
        # print('*****', f_tmp.min(), f_tmp.max()) # tensor(0.3630) tensor(0.6370)
        # print('*****', f[0, 10, 10], f[0, 9:12, 9:12])
        # 1/0

        feat_k = feat_k.detach()
        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        if i == 4:
            l_pos = l_pos.view(batchSize, 1) + torch.log(f_max) * 0.07
        else:
            l_pos = l_pos.view(batchSize, 1)

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
        if i == 4:
            l_neg_curbatch = l_neg_curbatch + torch.log(f) * 0.07

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / 0.07

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        # print('*****cross_entro', loss.mean(), loss2)
        # 1/0

        return loss
