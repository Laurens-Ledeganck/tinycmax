import torch

from cuda_event_ops.torch_batch import compute_inside_mask, get_event_flow_3d, linear_warp


def format_events(events, counts, stack=False, crop=True):
    """
    Go from list of padded (b, n, 4) events with (t, y, x, p)
    to padded (b, n, 5) events with (x, y, t, pass, p)
    or (b, n, d, 5) if stack is true.
    """
    max_counts = [c.max() for c in counts]  # per batch
    if stack:
        max_counts = [max(max_counts)] * len(max_counts)  # overall
    output = []
    for i, (ev, c) in enumerate(zip(events, max_counts)):
        t, y, x, p = ev[:, :c].unbind(-1) if crop else ev.unbind(-1)  # TODO: temporary
        z = torch.ones_like(t) * i
        output.append(torch.stack([x, y, t + i, z, p], dim=-1))
    output = torch.cat(output, dim=1) if not stack else torch.stack(output, dim=2)
    return output


def linear_3d_warp(events, flows):
    """
    Linearly warps events in 3D using bilinearly-sampled flows.

    Args:
        events (torch.Tensor): A tensor of shape (b, n, 5), where each event has (x, y, t, ti, p).
        flows (torch.Tensor): A tensor of shape (b, d, h, w, 2), where each flow has (u, v).

    Returns:
        torch.Tensor: A tensor of shape (b, n, d + 1, 5) with each event (x, y, t, t_orig, p) warped to (0, d).
    """

    # get deblurring window and resolution
    b, d, h, w, _ = flows.shape
    resolution = torch.tensor([w, h], device=flows.device)

    # (b, n, 5) -> (b, d, 4, n), drop zi
    events = events[..., [0, 1, 2, 4]].view(b, d, -1, 4).transpose(2, 3)

    # sample flow that will warp events at event locations
    # ensure integer t because we don't want bilinear there
    event_flow = get_event_flow_3d(events, flows)

    # warp to the extremes (all the way backward, all the way forward)
    t0 = torch.zeros(d, device=events.device)
    t1 = torch.ones(d, device=events.device) * d
    t_ref = torch.cat([t0, t1])

    # repeat events and flow for simultaneous fw and bw warping
    events = events.repeat(1, 2, 1, 1)  # copies
    event_flow = event_flow.repeat(1, 2, 1, 1)  # copies

    # bw/fw warp events to reference time
    # all in-place
    events = linear_warp(events, event_flow, t_ref.view(1, -1, 1, 1), keep_ts=True)

    # discard events warped outside image
    mask_inside = compute_inside_mask(events[..., 0:2, :], resolution)
    events[..., 3:4, :] *= mask_inside  # mask p, can be done in-place

    # add reference time as t
    # (b, 2d, 5, n) -> (b, n, 2, 5)
    x, y, t_orig, p = events.unbind(2)
    t = torch.ones_like(t_orig) * t_ref.view(1, -1, 1)
    events = torch.stack([x, y, t, t_orig, p], dim=-1).view(b, 2, -1, 5).transpose(1, 2)

    # put zeros between 0 and d + 1
    # TODO: not nice! temporary solution
    events_0, events_t = events.unbind(2)
    events = torch.stack([events_0, *[torch.zeros_like(events_t) for _ in range(d - 1)], events_t], dim=2)

    return events
