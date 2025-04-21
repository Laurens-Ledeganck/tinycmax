"""
Based on Jesse's file
"""

# imports
import torch
import numpy as np
from scipy.spatial.transform import Rotation

from tinycmax.model.event_flow_models.base import BaseModel


# first attempt
class SimpleRotationModel(BaseModel):
    # normal ANN
    # TODO: make easier to modify
    # TODO: implement spiking version

    def __init__(self, n_inputs, n_outputs, n_init=0):
        super().__init__()
        self.n_init = n_init
        if self.n_init:
            n_inputs += self.n_init
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, n_outputs),
        )

    def forward(self, X, init_rot=None):
        if init_rot:
            X = torch.concat((X, init_rot))
        return self.layers(X)


# second attempt
class ConvRotationModel(BaseModel):
    # CNN
    # TODO: make easier to modify
    # TODO: implement spiking version

    def __init__(self, input_size, n_outputs, n_init=0):
        super().__init__()
        self.n_init = n_init
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(input_size[1], 16, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            # torch.nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            # torch.nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),  # 32 -> 16
        )

        n_middle = self.conv_layers(torch.randn(input_size)).reshape(input_size[0], -1).shape[1]
        if self.n_init:
            n_middle += self.n_init

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(n_middle, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_outputs),
        )

        self.layers = torch.nn.Sequential(self.conv_layers, torch.nn.Flatten(), self.linear_layers)

    def forward(self, X, init_rot=None):
        X = self.conv_layers(X)
        X = X.reshape(X.shape[0], -1)
        if init_rot is not None:
            X = torch.concat((X, init_rot), dim=-1)
        return self.linear_layers(X)


# integrating everything
class FullRotationModel(BaseModel):
    # the following was adapted from the FireNet code

    def __init__(self, unet_kwargs):
        super().__init__()

        # self.num_bins = unet_kwargs["num_bins"]
        # base_num_channels = unet_kwargs["base_num_channels"]
        # kernel_size = unet_kwargs["kernel_size"]
        # self.encoding = unet_kwargs["encoding"]
        # self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.mask = unet_kwargs["mask_output"]
        # ff_act, rec_act = unet_kwargs["activations"]
        # if type(unet_kwargs["spiking_neuron"]) is dict:
        #     for kwargs in self.kwargs:
        #         kwargs.update(unet_kwargs["spiking_neuron"])
        self.device = unet_kwargs["device"]

        self.flow_model = unet_kwargs["flow_model"]
        self.flow_model.eval()

        self.input_size = tuple([unet_kwargs["batch_size"]] + [2] + list(unet_kwargs["resolution"]))

        self.transfer_size = None
        self.n_transfers = None
        self.current_transfer = None
        self.transfer_layer = unet_kwargs["transfer_layer"]
        self.get_n_transfers()  # will update self.transfer_size and self.n_transfers

        self.include_init = unet_kwargs["include_init"]

        self.rotation_mode = unet_kwargs["rotation_mode"]
        self.rotation_type = unet_kwargs["rotation_type"]
        self.n_outputs = self.get_n_rotation_nodes()

        if unet_kwargs["model_type"] == ("conv" or "conv_model" or "ConvRotationModel"):
            self.model_type = "conv"
            self.rotation_model = ConvRotationModel(
                input_size=list(self.transfer_size),
                n_outputs=self.n_outputs,
                n_init=(self.include_init * self.get_n_rotation_nodes()),
            )
        else:
            self.model_type = "linear"
            self.rotation_model = SimpleRotationModel(
                n_inputs=self.n_transfers,
                n_outputs=self.n_outputs,
                n_init=(self.include_init * self.get_n_rotation_nodes()),
            )

    def transfer_hook(self, module, input, output):  # should I use output[0] or not?
        if self.transfer_size is None:
            self.transfer_size = output.shape
        if self.n_transfers is None:
            self.n_transfers = len(torch.flatten(output[0]))
        self.current_transfer = output

    def get_n_transfers(self):
        # first, set the hook on the transfer layer
        hook = getattr(self.flow_model, self.transfer_layer).register_forward_hook(self.transfer_hook)

        # now pass a dummy input (n_transfers will be registered)
        dummy_voxel = torch.randn(self.input_size, dtype=torch.float32).to(self.device)
        dummy_cnt = torch.randint(0, 3, self.input_size, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            self.flow_model(dummy_voxel, dummy_cnt)
            # for some reason the Git settings of this project don't allow for variables to be defined but not used

        hook.remove()
        return self.n_transfers

    def get_n_rotation_nodes(self):
        if self.rotation_type == "rotvec" or self.rotation_type.startswith("euler"):
            return 3
        elif self.rotation_type == "quat":
            return 4
        elif self.rotation_type == "matrix":
            return 9

    # @property
    # def states(self):
    #     return copy_states(self._states)

    # @states.setter
    # def states(self, states):
    #     self._states = states

    def detach_states(self):
        self.flow_model.detach_states()

    #     detached_states = []
    #     for state in self.states:
    #         if type(state) is tuple:
    #             tmp = []
    #             for hidden in state:
    #                 tmp.append(hidden.detach())
    #             detached_states.append(tuple(tmp))
    #         else:
    #             detached_states.append(state.detach())
    #     self.states = detached_states

    def reset_states(self):
        self.flow_model.reset_states()

    #     self._states = None

    # def init_cropping(self, width, height):
    #     pass

    def forward(self, event_voxel, event_cnt, init_rot=None, log=False):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 2 x H x W per-polarity event cnt and average timestamp
        :param log: log activity
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """
        event_cnt = event_cnt.to(self.device)
        event_voxel = event_voxel.to(self.device)
        if init_rot is not None:
            init_rot.to(self.device)

        # forward pass
        hook = getattr(self.flow_model, self.transfer_layer).register_forward_hook(self.transfer_hook)

        with torch.no_grad():
            flow = self.flow_model(event_voxel, event_cnt, log=log)["flow"][0]

        if self.model_type == "linear":
            transfer = torch.reshape(self.current_transfer, (self.current_transfer.shape[0], self.n_transfers))
        else:
            transfer = self.current_transfer

        if self.include_init:
            if init_rot is not None:
                r = self.rotation_model(transfer, init_rot)
            else:
                raise ValueError("Please provide a valid init_rot")
        else:
            r = self.rotation_model(transfer)

        hook.remove()

        # TODO: potentially log activity
        activity = None

        return {"flow": [flow], "activity": activity, "rotation": r}


# class RotationLoss(BaseValidationLoss):

#     def __init__(self, config, device, flow_scaling=128):
#         super().__init__(config, device, flow_scaling)
#         self.loss_fn = torch.nn.MSELoss()
#         self.y_test = None
#         self.y_pred = None

#     def prepare_loss(self, y_pred, y_test):
#         self.y_pred = y_pred
#         self.y_test = y_test
#         # print(y_pred)
#         # print(y_test)
#         # print()

#     def forward(self):
#         return self.loss_fn(self.y_pred, self.y_test)
