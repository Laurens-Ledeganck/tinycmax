from bisect import bisect_left
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import zipfile

import cv2
from dotmap import DotMap
from gdown import download_folder, download
import h5py
import hdf5plugin
from lightning import LightningDataModule
import numpy as np
from numpy.lib import recfunctions as rfn
import torch
from torch.utils.data import ConcatDataset, DataLoader
import yaml

from tinycmax.datamodule.data_utils import (
    batched,
    ConcatBatchSampler,
    InfiniteDataLoader,
    only_add_batch_dim,
    time_first_collate,
)


@dataclass
class MvsecSequence:
    root_dir: str
    recording: str
    time_window: float | int | None
    count_window: int | None
    chunk_size: int = 100
    seq_len: int | None = None
    time: tuple[float, float] | tuple[int, int] | None = None  # start, end
    crop: tuple[int, ...] | None = None  # height, width or top, left, bottom, right
    rectify: bool = False
    augmentations: list[str] | None = None
    gt: list[str] | None = None

    def __post_init__(self):
        # checks
        # TODO: implement count window
        assert not (self.time_window is not None and self.count_window is not None)
        assert self.count_window is None
        assert not (self.augmentations is not None and self.gt)  # no augmentations on gt

        # defaults
        self.root_dir = Path(self.root_dir)
        self.sensor_size = (260, 346)  # height, width

        # make paths
        recording = self.recording[:-1]  # remove trailing number
        paths = {
            "data": self.root_dir / recording / f"{self.recording}_data.hdf5",
            "gt": self.root_dir / recording / f"{self.recording}_gt.hdf5",
            "rect_map_x": self.root_dir / recording / "calib" / f"{recording}_left_x_map.txt",
            "rect_map_y": self.root_dir / recording / "calib" / f"{recording}_left_y_map.txt",
            "calibration": self.root_dir / recording / "calib" / f"camchain-imucam-{recording}.yaml",
        }
        assert all(p.exists() for p in paths.values())

        # open large h5 files only once
        self.fs = dict(
            data=h5py.File(paths["data"], "r"),
            gt=h5py.File(paths["gt"], "r"),
        )

        # forward rectification map
        # distorted -> rectified coords
        # provided map, easier than backward, but gives lines in frames due to nearest neighbor
        rect_map_x = np.loadtxt(paths["rect_map_x"])
        rect_map_y = np.loadtxt(paths["rect_map_y"])
        self.fw_rect_map = np.stack([rect_map_x, rect_map_y], axis=-1)  # x_rect, y_rect = rect_map[y, x].T

        # backward rectification/undistortion map
        # rectified/undistorted -> distorted coords
        # more work, but prevents lines in accumulated event frames
        with open(paths["calibration"], "r") as f:
            cam_to_cam = yaml.safe_load(f)
        fx, fy, cx, cy = cam_to_cam["cam0"]["intrinsics"]
        K_dist = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.K_rect = np.array(cam_to_cam["cam0"]["projection_matrix"])[:, :3]
        R_rect = np.array(cam_to_cam["cam0"]["rectification_matrix"])
        dist_coeffs = np.array(cam_to_cam["cam0"]["distortion_coeffs"])
        resolution = cam_to_cam["cam0"]["resolution"]  # xy
        rect_map_x, rect_map_y = cv2.fisheye.initUndistortRectifyMap(
            K_dist, dist_coeffs, R_rect, self.K_rect, resolution, cv2.CV_32F
        )
        self.bw_rect_map = np.stack([rect_map_x, rect_map_y], axis=-1)  # needs to be .fisheye!

        # get duration of recording
        # don't get full t because of memory usage
        # convert to us
        self.t0, self.tk = (self.fs["data"]["davis/left/events"][[0, -1], 2] * 1e6).astype(np.int64)
        if self.time is not None:
            t0, tk = self.time
            t0 = t0 + self.t0 if t0 is not None else self.t0
            tk = tk + self.t0 if tk is not None else self.tk
            self.t0, self.tk = t0, tk
        self.rec_duration = self.tk - self.t0

        # slice dataset, pre-compute crop and augmentation
        self.reset()

        # set frame shape
        self.frame_shape = (
            self.crop_corners[2] - self.crop_corners[0],
            self.crop_corners[3] - self.crop_corners[1],
        )

        # mapping from chunks to single slices
        # match seq_len if given
        self.chunk_size = self.seq_len if self.seq_len is not None else self.chunk_size
        self.chunk_map = batched(range(len(self.t_start)), self.chunk_size)

    def init_slice(self):
        # start and end time
        if self.seq_len is not None:  # randomly-sliced sequence of seq_len
            start, end = self.t0, max(self.t0, self.tk - (self.seq_len + 1) * self.time_window)  # only full
            if np.issubdtype(self.fs["events/t"], np.integer):
                t_start = np.random.randint(start, max(start + 1, end))
            elif np.issubdtype(self.fs["events/t"], np.floating):
                t_start = np.random.uniform(start, end)
            n_full_windows = self.seq_len
        else:  # full sequence
            t_start, t_end = self.t0, self.tk
            n_full_windows = max(1, int((t_end - t_start) / self.time_window))  # at least 1 window

        # window making
        # use linspace because could be floats (no rounding errors?)
        linspace = np.linspace(t_start, n_full_windows * self.time_window + t_start, n_full_windows + 1)
        self.t_start, self.t_end = linspace[:-1], linspace[1:]
        self.seq_duration = n_full_windows * self.time_window

    def init_crop(self):
        if self.crop:
            if len(self.crop) == 2:  # height, width
                h, w = self.crop
                top = np.random.randint(self.sensor_size[0] - h + 1)  # +1 because exclusive
                left = np.random.randint(self.sensor_size[1] - w + 1)
                self.crop_corners = (top, left, top + h, left + w)
            elif len(self.crop) == 4:  # top, left, bottom, right
                self.crop_corners = self.crop
        else:
            self.crop_corners = (0, 0, *self.sensor_size)

    def init_augmentation(self):
        self.augmentation = []
        if self.augmentations is not None:
            for aug in self.augmentations:
                if np.random.rand() < 0.5:
                    self.augmentation.append(aug)

    def reset(self):
        self.init_slice()  # slice up dataset
        self.init_crop()  # pre-compute crop
        self.init_augmentation()  # pre-compute augmentation

    def __len__(self):
        return len(self.chunk_map)

    def __getitem__(self, idx):
        # get new random slice, crop, augmentations
        self.reset()

        # get chunk
        chunk = self.chunk_map[idx]

        # go over slices
        frames, auxs, targets = [], DotMap(), DotMap()
        for i in chunk:
            # convert to indices (from us to s again)
            start = bisect_left(self.fs["data"]["davis/left/events"], self.t_start[i] / 1e6, key=lambda x: x[2])
            end = bisect_left(self.fs["data"]["davis/left/events"], self.t_end[i] / 1e6, key=lambda x: x[2])

            # get events as list
            t = self.fs["data"]["davis/left/events"][start:end, 2]  # float64, s
            y = self.fs["data"]["davis/left/events"][start:end, 1]  # float64
            x = self.fs["data"]["davis/left/events"][start:end, 0]  # float64
            p = self.fs["data"]["davis/left/events"][start:end, 3]  # float64 in {-1, 1}

            # rectify list: forward rectification
            if self.rectify:
                x_rect, y_rect = self.fw_rect_map[y.astype(np.int64), x.astype(np.int64)].T
            else:
                x_rect, y_rect = x, y

            # list of events to structured array
            dtype = np.dtype([("t", np.float64), ("y", np.float32), ("x", np.float32), ("p", np.int8)])
            lst = np.empty(len(t), dtype=dtype)
            lst["t"] = t
            lst["y"] = y_rect
            lst["x"] = x_rect
            lst["p"] = p

            # crop list
            top, left, bottom, right = self.crop_corners
            mask = (y_rect >= top) & (y_rect < bottom) & (x_rect >= left) & (x_rect < right)
            lst = lst[mask]
            lst["y"] -= top
            lst["x"] -= left

            # make into event count frame
            # use unrectified coordinates, convert p to {0, 1}
            y = torch.from_numpy(y.astype(np.int64))
            x = torch.from_numpy(x.astype(np.int64))
            p = torch.from_numpy(((p + 1) // 2).astype(np.int64))
            frame = torch.zeros(2, *self.sensor_size, dtype=torch.int64)  # torch is faster
            frame.index_put_((p, y, x), torch.ones_like(p), accumulate=True)

            # rectify frame: backward rectification
            # backward to prevent lines in frames
            if self.rectify:
                frame = cv2.remap(
                    frame.numpy().transpose(1, 2, 0), self.bw_rect_map, None, interpolation=cv2.INTER_NEAREST
                )
                frame = torch.from_numpy(frame.transpose(2, 0, 1))

            # crop frame
            frame = frame[..., top:bottom, left:right]

            # discard if few events or same timestamp
            if len(lst) < 10 or lst["t"][-1] == lst["t"][0]:
                lst = np.array([], dtype=lst.dtype)
                frame = torch.zeros_like(frame)

            # format list of events
            # after cropping, else normalized timestamp not correct
            # only normalize time; polarity is already in {-1, 1}
            lst["t"] = (lst["t"] - lst["t"][0]) / (lst["t"][-1] - lst["t"][0]) if len(lst) else lst["t"]

            # append
            frames.append(frame)
            auxs.events += [lst]
            auxs.counts += [len(lst)]

            # targets

            # TODO: flow (and visualization)

            # depth
            if self.gt and "depth" in self.gt:
                start = bisect_left(self.fs["gt"]["davis/left/depth_image_rect_ts"], self.t_start[i] / 1e6)
                end = bisect_left(self.fs["gt"]["davis/left/depth_image_rect_ts"], self.t_end[i] / 1e6)
                if start >= end:
                    end = start + 1  # take latest
                gt_depth_id = start + 1  # to prevent 0
                gt_depth = self.fs["gt"]["davis/left/depth_image_rect"][start:end]  # keep time dim as channel
                gt_depth = gt_depth[..., top:bottom, left:right]  # crop
                gt_depth[np.isnan(gt_depth)] = 0  # replace nans with 0
                targets.depth += [torch.from_numpy(gt_depth.astype(np.float32))]
                targets.depth_id += [torch.tensor(gt_depth_id)]

        # stack and pad
        frames = torch.stack(frames)
        max_len = max(auxs.counts)
        auxs.events = rfn.structured_to_unstructured(
            np.stack([np.pad(e, (0, max_len - len(e))) for e in auxs.events]), dtype=np.float32
        )
        auxs = DotMap({k: torch.tensor(v) for k, v in auxs.items()}, _dynamic=False)  # convert to static dotmap
        targets = DotMap({k: torch.stack(v) for k, v in targets.items()}, _dynamic=False)

        # apply augmentations; more efficient on chunks
        # not used with targets, so leave those out
        if "flip_t" in self.augmentation:
            frames = frames.flip(0)
            auxs.events[..., 0] = 1 - auxs.events[..., 0]
            auxs.events = auxs.events.flip(0)
            auxs.counts = auxs.counts.flip(0)
        if "flip_pol" in self.augmentation:
            frames = frames.flip(1)
            auxs.events[..., 3] *= -1
        if "flip_ud" in self.augmentation:
            frames = frames.flip(2)
            auxs.events[..., 1] = (bottom - top - 1) - auxs.events[..., 1]
        if "flip_lr" in self.augmentation:
            frames = frames.flip(3)
            auxs.events[..., 2] = (right - left - 2) - auxs.events[..., 2]

        # adapt camera matrices to crop and augmentations
        K_rect = self.K_rect.copy()
        K_rect[0, 2] -= left
        K_rect[1, 2] -= top
        if "vertical" in self.augmentation:
            K_rect[1, 2] = (bottom - top - 1) - K_rect[1, 2]
        if "horizontal" in self.augmentation:
            K_rect[0, 2] = (right - left - 1) - K_rect[0, 2]
        inv_K_rect = np.linalg.inv(K_rect)
        K_rect = torch.from_numpy(K_rect.astype(np.float32))
        inv_K_rect = torch.from_numpy(inv_K_rect.astype(np.float32))

        # return static dotmap
        sample = DotMap(
            frames=frames.float(),
            auxs=auxs,
            targets=targets,
            recording=self.recording,
            eofs=[i == len(self.t_start) - 1 for i in chunk],
            K_rect=K_rect,
            inv_K_rect=inv_K_rect,
            _dynamic=False,
        )

        return sample


class MvsecDataModule(LightningDataModule):

    gt = ["depth"]

    def __init__(
        self,
        root_dir,
        time_window,
        count_window,
        train_seq_len,
        train_crop,
        train_recordings,
        val_crop,
        val_recordings,
        rectify,
        augmentations,
        batch_size,
        shuffle,
        num_workers,
        download,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.time_window = time_window
        self.count_window = count_window
        self.train_seq_len = train_seq_len
        self.train_crop = train_crop
        self.train_recordings = train_recordings
        self.val_crop = val_crop
        self.val_recordings = val_recordings
        self.rectify = rectify
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.download = download

    def prepare_data(self):
        if self.download:
            # event and gt data in h5
            # parent: https://drive.google.com/drive/folders/1gDy2PwVOu_FPOsEZjojdWEB2ZHmpio8D
            urls = [
                ("indoor_flying", "https://drive.google.com/drive/folders/1CEuvvahWQntNIqXWZhXu_WknsTLm4Sum"),
                ("outdoor_day", "https://drive.google.com/drive/folders/1WUapfrd2DNQNuxPt9IqUHCcPCPKLiNvT"),
            ]

            # download if not there
            for name, url in urls:
                (self.root_dir / name).mkdir(exist_ok=True, parents=True)
                files = download_folder(url, output=str(self.root_dir / name), skip_download=True)
                for f in files:
                    id, _, local_path = f
                    if not Path(local_path).exists():
                        try:
                            download(id=id, output=local_path)
                        except Exception as e:
                            print(e)
                    if ".zip" in local_path:
                        if not (self.root_dir / name / "calib").exists():
                            with zipfile.ZipFile(local_path, "r") as f:
                                f.extractall(self.root_dir / name / "calib")
                        (self.root_dir / name / f"{name}_calib.zip").unlink()

        # all recordings
        # recordings = [
        #     ("indoor_flying1", None),
        #     ("indoor_flying2", None),
        #     ("indoor_flying3", None),
        #     ("indoor_flying4", None),
        #     ("outdoor_day1", None),
        #     ("outdoor_day2", None),
        # ]

        # train on outdoor_day2, validate on part of outdoor_day1 (default)
        train_recordings = (
            [("outdoor_day2", None)] if self.train_recordings is None else self.train_recordings
        )  # override
        val_recordings = [("outdoor_day1", (222.4, 240.4))] if self.val_recordings is None else self.val_recordings

        # store for building datasets later
        self.train_recordings = train_recordings
        self.val_recordings = val_recordings

    def setup(self, stage):
        if stage == "fit":
            train_sequence = partial(
                MvsecSequence,
                root_dir=self.root_dir,
                time_window=self.time_window,
                count_window=self.count_window,
                seq_len=self.train_seq_len,
                crop=self.train_crop,
                rectify=self.rectify,
                augmentations=self.augmentations,
            )
            train_recordings = []
            for rec in self.train_recordings:
                if isinstance(rec, str):
                    rec = (rec, None)
                r, t = rec
                seq = train_sequence(recording=r, time=t)
                train_recordings.extend([rec] * int(seq.rec_duration / seq.seq_duration))
            self.train_dataset = ConcatDataset([train_sequence(recording=r, time=t) for r, t in train_recordings])
            self.train_frame_shape = (self.batch_size, 2, *self.train_dataset.datasets[0].frame_shape)

        if stage in ["fit", "validate"]:
            val_sequence = partial(
                MvsecSequence,
                root_dir=self.root_dir,
                time_window=self.time_window,
                count_window=self.count_window,
                crop=self.val_crop,
                rectify=True,
                gt=self.gt,
            )
            for i, rec in enumerate(self.val_recordings):
                if isinstance(rec, str):
                    rec = (rec, None)
                self.val_recordings[i] = rec
            self.val_dataset = ConcatDataset([val_sequence(recording=r, time=t) for r, t in self.val_recordings])
            self.val_frame_shape = (1, 2, *self.val_dataset.datasets[0].frame_shape)

        elif stage == "test":
            raise NotImplementedError

    def train_dataloader(self):
        sampler = ConcatBatchSampler(self.train_dataset, self.batch_size, shuffle=self.shuffle)
        return InfiniteDataLoader(
            self.train_dataset, batch_sampler=sampler, num_workers=self.num_workers, collate_fn=time_first_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers // 2,
            collate_fn=only_add_batch_dim,
        )


if __name__ == "__main__":
    from hydra import initialize, compose
    from hydra.utils import instantiate

    # get config
    with initialize(config_path="../config/datamodule", version_base=None):
        config = compose(config_name="mvsec", overrides=["download=true"])

    # download data
    datamodule = instantiate(config)
    datamodule.prepare_data()
