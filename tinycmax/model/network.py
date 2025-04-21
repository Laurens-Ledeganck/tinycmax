import torch.nn as nn

from tinycmax.model.blocks import conv_encoder, LazyConvGru, upsample_decoder
from tinycmax.model.network_utils import NetworkWrapper


class FlowNetwork(nn.Module):
    """
    Optical flow prediction network following IDNet (Wu et al., ICRA'24).
    """

    mode = "flow"

    def __init__(
        self,
        encoder_channels,
        memory_channels,
        decoder_channels,
        activation_fn,
        final_bias,
        padding_mode,
        scaling,
    ):
        super().__init__()

        self.scaling = scaling

        self.encoder = conv_encoder(encoder_channels, activation_fn, padding_mode=padding_mode)
        self.memory = LazyConvGru(memory_channels, 3, padding_mode=padding_mode)
        self.decoder = upsample_decoder(
            decoder_channels, activation_fn, final_bias, padding_mode=padding_mode, mode=self.mode
        )

    def forward(self, input, hidden=None):
        frame = input["events"]  # .events incompatible with torch.compile?
        encoder = self.encoder(frame)
        memory = self.memory(encoder, hidden)
        flow_map = self.decoder(memory)

        flow_map *= self.scaling

        return dict(flow=flow_map), memory


class WrappedFlowNetwork(NetworkWrapper, FlowNetwork):
    pass
