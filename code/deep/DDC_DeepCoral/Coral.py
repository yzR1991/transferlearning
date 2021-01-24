import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def CORAL(source, target):
    #获取tensor的行列信息
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    #covariance计算可自行推导，或参考Unsupervised Domain Adaptation by Mapped Correlation Alignment
    # source covariance
    tmp_s = torch.ones((1, ns)).to(DEVICE) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).to(DEVICE) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)

    return loss
