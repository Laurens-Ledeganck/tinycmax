from lightning.pytorch.callbacks import Callback

from tinycmax.visualization.visualizer import FileVisualizer, RerunVisualizer


class Visualizer(Callback):

    is_visualizer = True

    def on_batch_end(self, outputs):
        for output in outputs.values():
            self.visualizer.set_counter()

            # things with events
            for k in [k for k in output.keys() if "events" in k]:
                self.visualizer.event_frame(output[k][0].detach().cpu(), name=k)

            # things with flow
            for k in [k for k in output.keys() if "flow" in k and "raw" not in k]:
                self.visualizer.flow_map(output[k][0].detach().cpu(), name=k)

            # for scalar values
            for k in [k for k in output.keys() if isinstance(output[k], (int, float))]:
                self.visualizer.scalar(k, output[k])

    def on_train_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)

    def on_validation_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)


class LiveVisualizer(Visualizer):

    def __init__(self, app_id, log_dir, server, mode, compression, time_window, blueprint=None):
        self.visualizer = RerunVisualizer(app_id, log_dir, server, mode, compression, time_window, blueprint)


class FileLogger(Visualizer):

    def __init__(self, root_dir, keys, image_format, video_format, time_window):
        self.visualizer = FileVisualizer(root_dir, keys, image_format, video_format, time_window)

    def on_validation_end(self, trainer, litmodule):
        self.visualizer.save_videos()
