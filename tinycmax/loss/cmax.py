from functools import partial

from dotmap import DotMap
import torch
import torch.nn as nn

from cuda_event_ops import iterative_3d_warp as iterative_3d_warp_cuda
from tinycmax.loss.cmax_utils import format_events, linear_3d_warp
from tinycmax.loss.iwe import build_iwe


class ContrastMaximization(nn.Module):
    """
    Contrast maximization loss as used in:
    - Hagenaars and Paredes-Valles et al., NeurIPS 2021
        - Linear warping to the two extreme references
    - Paredes-Valles et al., ICCV 2023
        - Iterative warping to multiple references (all bin edges in deblurring window)
    Contrast maximization is done on the warped image of average timestamps.
    """

    cls_name = "cmax"

    def __init__(self, accumulation_window, base, warp):
        super().__init__()

        self.accumulation_window = accumulation_window
        self.base = base
        self.warp = warp
        if warp == "linear":
            self.warp_fn = linear_3d_warp
        elif warp == "iterative":
            self.warp_fn = partial(iterative_3d_warp_cuda, num_warps=base)
        else:
            raise ValueError(f"Unknown warp function: {warp}")

        self.total_loss = 0
        self.passes = 0
        self.buffer = DotMap()

    def forward(self, pred, aux, target):
        self.buffer.aux += [aux]
        self.buffer.flow_maps += [pred["flow"]]
        self.passes += 1

    def prepare_backward(self):
        # format events
        events = [aux["events"] for aux in self.buffer.aux]
        counts = [aux["counts"] for aux in self.buffer.aux]
        events = format_events(events, counts, crop=self.warp != "linear")  # padded (b, n, 5)

        # stack flows
        flow_maps = torch.stack(self.buffer.flow_maps, dim=2)  # (b, 2, d, h, w)
        flow_maps = flow_maps.permute(0, 2, 3, 4, 1).contiguous()  # (b, d, h, w, 2)

        return events, flow_maps

    def compute_cmax_loss(self, events, flow_maps):
        # warp events: (b, n, 5) -> (b, n, d + 1, 5) with (x, y, t, t_orig, p)
        warped_events = self.warp_fn(events, flow_maps)

        # build iwe and iwt with (trilinear) splatting
        _, _, h, w, _ = flow_maps.shape
        iwe, iwt = build_iwe(warped_events, self.base, (h, w))  # (b, 2, d + 1, h, w)

        # TODO: needed for linear: only keep nonzero, temporary solution
        if self.warp == "linear":
            iwe = iwe[:, :, [0, -1]]
            iwt = iwt[:, :, [0, -1]]

        # split into negative and positive polarity
        iwe_neg, iwe_pos = iwe.unbind(1)
        iwt_neg, iwt_pos = iwt.unbind(1)

        # per-polarity image of warped average timestamps
        iwat_neg = iwt_neg / (iwe_neg + 1e-9)
        iwat_pos = iwt_pos / (iwe_pos + 1e-9)

        # scale by number of pixels with at least one event in iwe
        inside = (iwe_neg + iwe_pos).gt(0).float().flatten(start_dim=2).sum(2) + 1e-9

        # compute deblurring loss
        loss = iwat_neg.pow(2).flatten(start_dim=2).sum(2) + iwat_pos.pow(2).flatten(start_dim=2).sum(2)
        loss = loss / inside
        return loss

    def get_accumulated_events(self):
        events = [aux["events"] for aux in self.buffer.aux]
        counts = [aux["counts"] for aux in self.buffer.aux]
        accumulated_events = format_events(
            events, counts, stack=True, crop=self.warp != "linear"
        )  # padded (b, n, d, 5)
        accumulated_events[..., 2] = accumulated_events[..., 3]  # prevent trilinear splat
        if not accumulated_events.numel():
            return torch.zeros_like(self.buffer.flow_maps[0])
        _, _, h, w = self.buffer.flow_maps[0].shape
        accumulated_event_frame, _ = build_iwe(accumulated_events, 1, (h, w))  # (b, 2, d, h, w)
        return accumulated_event_frame.sum(2)

    def compute_iwe(self, tref):
        # get events and flow maps
        events, flow_maps = self.prepare_backward()
        if not events.numel():
            return torch.zeros_like(self.buffer.flow_maps[0])

        # warp events: (b, n, 5) -> (b, n, d + 1, 5) with (x, y, t, t_orig, p)
        warped_events = self.warp_fn(events, flow_maps)

        # build iwe and iwt with (trilinear) splatting
        _, _, h, w, _ = flow_maps.shape
        iwe, _ = build_iwe(warped_events, self.base, (h, w))  # (b, 2, d + 1, h, w)
        return iwe[:, :, tref]

    def backward(self):
        # get events and flow maps
        events, flow_maps = self.prepare_backward()

        # if no events, no loss
        # TODO: for some reason catching 0 dim in cuda doesn't work
        if not events.numel():
            return None

        # compute deblurring loss
        # mean over batch and reference times
        loss = self.compute_cmax_loss(events, flow_maps)
        self.total_loss += loss.mean()

        return self.total_loss

    def reset(self):
        self.total_loss = 0
        self.passes = 0
        self.buffer = DotMap()

    def compute_and_reset(self):
        mean_loss = self.total_loss  # already mean over passes
        self.reset()
        return {f"{self.cls_name}_loss": mean_loss}

    def visualize(self):
        visuals = DotMap()
        with torch.no_grad():
            visuals[f"{self.cls_name}_accumulated_events"] = self.get_accumulated_events()
            visuals[f"{self.cls_name}_image_warped_events_0"] = self.compute_iwe(0)
            visuals[f"{self.cls_name}_image_warped_events_t"] = self.compute_iwe(self.passes)
        return visuals


class RatioSquaredAvgTimestamps(ContrastMaximization):
    """
    RSAT loss from Hagenaars and Paredes-Valles et al., NeurIPS 2021.
    Quantifies how much warping by optical flow increases sharpness of the image of (warped) events.
    """

    cls_name = "rsat"

    def backward(self):
        # get events and flow maps
        events, flow_maps = self.prepare_backward()

        # if no events, no loss
        if not events.numel():
            return None

        # get zero flow
        zero_maps = torch.zeros_like(flow_maps)

        # compute deblurring loss
        # mean over batch, only last reference time
        loss = self.compute_cmax_loss(events, flow_maps)
        loss_unwarped = self.compute_cmax_loss(events, zero_maps)
        rsat = loss[:, -1] / (loss_unwarped[:, -1] + 1e-9)
        self.total_loss += rsat.mean()

        return self.total_loss
