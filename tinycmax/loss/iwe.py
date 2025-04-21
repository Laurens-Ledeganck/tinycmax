import torch

from cuda_event_ops.cuda import trilinear_splat as trilinear_splat_cuda


def build_iwe(warped_events, base, resolution, select=None):
    # select reference times
    if select:
        start, stop = select
        warped_events = warped_events[:, :, start:stop]
        b, _, d, _ = warped_events.shape
    else:
        b, _, d, _ = warped_events.shape
        start, stop = 0, d

    # unpack events, compute contribution
    # only consider inside base
    x, y, t, t_orig, p = warped_events.unbind(-1)
    t_ref = torch.arange(start, stop, device=warped_events.device)
    t_contrib = 1 - (t_ref - t_orig).abs() / base

    # per-polarity image of warped events
    # TODO: make more efficient?
    neg, pos = p.lt(0), p.gt(0)
    neg_warped_events = torch.stack([x * neg, y * neg, t * neg, (p * neg).abs()], dim=-1)
    pos_warped_events = torch.stack([x * pos, y * pos, t * pos, (p * pos).abs()], dim=-1)

    iwe_neg = trilinear_splat_cuda(neg_warped_events.view(b, -1, 4), (d, *resolution))
    iwe_pos = trilinear_splat_cuda(pos_warped_events.view(b, -1, 4), (d, *resolution))
    iwe = torch.stack([iwe_neg, iwe_pos], dim=1)  # (b, 2, d + 1, h, w)

    # per-polarity image of warped timestamps
    neg_warped_events = torch.stack([x * neg, y * neg, t * neg, (p * neg).abs() * t_contrib], dim=-1)
    pos_warped_events = torch.stack([x * pos, y * pos, t * pos, (p * pos).abs() * t_contrib], dim=-1)

    iwt_neg = trilinear_splat_cuda(neg_warped_events.view(b, -1, 4), (d, *resolution))
    iwt_pos = trilinear_splat_cuda(pos_warped_events.view(b, -1, 4), (d, *resolution))
    iwt = torch.stack([iwt_neg, iwt_pos], dim=1)

    return iwe, iwt
