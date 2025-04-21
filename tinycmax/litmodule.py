from dotmap import DotMap
from lightning import LightningModule
import torch

from tinycmax.model.network_utils import recursive_clone


class Show(LightningModule):
    def __init__(self):
        super().__init__()

    def validation_step(self, batch, batch_idx):
        # unpack
        frames, targets = batch.frames, batch.targets

        # go over sequence
        log = DotMap(_dynamic=False)
        for i, frame in enumerate(frames):
            # get input, auxiliary, target
            input = DotMap(events=frame, _dynamic=False)
            target = DotMap({k: v[i] for k, v in targets.items()}, _dynamic=False)

            # add to log if visualizing/testing
            log[i] = DotMap(_dynamic=False)
            log[i].update({f"input_{k}": v for k, v in input.items()})
            log[i].update({f"target_{k}": v for k, v in target.items()})

        return log


class Train(LightningModule):
    def __init__(self, network, loss_fns, optimizer, compile_network):
        super().__init__()

        self.network = network
        self.loss_fns = loss_fns
        self.optimizer = optimizer
        self.compile_network = compile_network
        self.automatic_optimization = False  # manual because tbptt

    def setup(self, stage):
        # set visualization
        self.visualizing = any([getattr(cb, "is_visualizer", False) for cb in self.trainer.callbacks])

        # trace lazy modules if training
        if stage == "fit":
            input = DotMap(
                events=torch.zeros(self.trainer.datamodule.train_frame_shape, device=self.device), _dynamic=False
            )
            self.network.trace(input)

        # compile network
        # keep original (which shares params with compiled) for state dict loading
        if self.compile_network == "default" and not self.visualizing:
            self.compiled_network = torch.compile(self.network, fullgraph=True)  # ?% faster
        elif self.compile_network == "reduce-overhead" and not self.visualizing:
            self.compiled_network = torch.compile(self.network, fullgraph=True, mode="reduce-overhead")  # ?% faster
        else:
            self.compiled_network = self.network

        # wandb model watching
        if self.logger is not None and not self.compile_network:
            self.logger.watch(self.network, log="all", log_freq=self.trainer.log_every_n_steps * 100)

    def shared_step(self, batch, batch_idx, stage):
        # training: get optimizer because manual optimization
        if stage == "train":
            optimizer = self.optimizers()

        # unpack
        frames, auxs, targets, eofs = batch.frames, batch.auxs, batch.targets, batch.eofs

        # go over sequence
        log = DotMap(_dynamic=False)
        for i, (frame, eof) in enumerate(zip(frames, eofs)):
            # get input, auxiliary, target
            # TODO: slice all into dotmaps in collate, then feed batch everywhere
            input = DotMap(events=frame, _dynamic=False)
            aux = DotMap({k: v[i] for k, v in auxs.items()}, _dynamic=False)
            target = DotMap({k: v[i] for k, v in targets.items()}, _dynamic=False)

            # forward network
            pred = self.compiled_network(input)
            if stage == "validate":
                if self.compiled_network.state is not None:
                    self.compiled_network.state = recursive_clone(self.compiled_network.state)

            # forward loss functions
            [loss_fn(pred, aux, target) for loss_fn in self.loss_fns[stage].values()]
            if stage == "validate":
                for loss_fn in self.loss_fns[stage].values():
                    if hasattr(loss_fn, "buffer"):
                        loss_fn.buffer = recursive_clone(loss_fn.buffer)

            # add to log if visualizing
            log[i] = DotMap(_dynamic=False)
            if self.visualizing:
                log[i].update({f"input_{k}": v for k, v in input.items()})
                log[i].update({f"pred_{k}": v for k, v in pred.items()})
                log[i].update({f"target_{k}": v for k, v in target.items()})
                [
                    log[i].update({f"{k}": v for k, v in loss_fn.visualize().items()})
                    for loss_fn in self.loss_fns[stage].values()
                ]

            # go over loss functions
            # backward if enough passes
            loss = 0
            for loss_fn in self.loss_fns[stage].values():
                if loss_fn.passes == loss_fn.accumulation_window:
                    # backward loss
                    dloss = loss_fn.backward()
                    loss += dloss if dloss is not None else 0

                    # reset loss and log
                    # loss per tbptt window per batch sample
                    # default batch size (seq_len) gives same value but rounding errors
                    for name, value in loss_fn.compute_and_reset().items():
                        if stage == "train":
                            self.log(f"{stage}/{name}", value, batch_size=1, on_epoch=True, prog_bar=True)
                        elif stage == "validate":
                            self.log(f"{stage}/{name}/{batch.recording}", value, batch_size=1)
                            self.log(f"{stage}/{name}/mean", value, batch_size=1)
                        if self.visualizing:
                            log[i][name] = value.item()

            # training: backprop and optimize
            if stage == "train" and loss:
                optimizer.zero_grad()
                self.manual_backward(loss)
                self.clip_gradients(optimizer, gradient_clip_val=self.gradient_clip_val)
                optimizer.step()

                # detach network state
                self.compiled_network.detach()

            # reset if end of sequence
            if any(eof):
                self.compiled_network.reset()
                [loss_fn.reset() for loss_fn in self.loss_fns[stage].values()]

        return log if self.visualizing else None

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "validate")

    def configure_optimizers(self):
        # split gradient clipping from optimizer
        self.gradient_clip_val = self.optimizer.keywords.pop("gradient_clip_val", 0.0)
        optimizer = self.optimizer(self.network.parameters())
        return optimizer
