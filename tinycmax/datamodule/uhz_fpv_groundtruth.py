"""
Work in progress. 
Run this file to modify the hdf5 files. 
"""

# ---------- dataloader for rotation data ---------- #

from torch.utils.data import ConcatDataset
from functools import partial
from dotmap import DotMap
from bisect import bisect_left
import torch
import cv2
from numpy.lib import recfunctions as rfn
from scipy.spatial.transform import Rotation

# TODO!
# from event_flow_pipeline.dataloader.h5 import H5Loader, Frames, ProgressBar, FlowMaps
H5Loader, Frames, ProgressBar, FlowMaps = None, None, None, None

from tinycmax.datamodule.uzh_fpv import UzhFpvSequence, UzhFpvDataModule


class RotationSequence(UzhFpvSequence):
    def __getitem__(self, idx):
        # get new random slice, crop, augmentations
        self.reset()

        # get chunk
        chunk = self.chunk_map[idx]

        # go over slices
        frames, auxs, targets = [], DotMap(), DotMap()
        for i in chunk:
            # convert to indices
            start = bisect_left(self.fs["events/t"], self.t_start[i])
            end = bisect_left(self.fs["events/t"], self.t_end[i])

            # get events as list
            t = self.fs["events/t"][start:end]  # uint32
            y = self.fs["events/y"][start:end]  # uint16
            x = self.fs["events/x"][start:end]  # uint16
            p = self.fs["events/p"][start:end]  # uint8 in {0, 1}

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
            # use unrectified coordinates
            y = torch.from_numpy(y.astype(np.int64))
            x = torch.from_numpy(x.astype(np.int64))
            p = torch.from_numpy(p.astype(np.int64))
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

            # format list of events: normalize time, polarity to {-1, 1}
            # after cropping, else normalized timestamp not correct
            lst["t"] = (lst["t"] - lst["t"][0]) / (lst["t"][-1] - lst["t"][0]) if len(lst) else lst["t"]
            lst["p"] = lst["p"] * 2 - 1

            # append
            frames.append(frame)
            auxs.events += [lst]
            auxs.counts += [len(lst)]

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

        # return static dotmap
        sample = DotMap(
            frames=frames.float(),
            auxs=auxs,
            targets=targets,
            recording=self.recording,
            eofs=[i == len(self.t_start) - 1 for i in chunk],
            _dynamic=False,
        )

        xs, ys, ps = None, None, None
        raise NotImplementedError

        # event_flow preprocessing
        event_cnt = self.create_cnt_encoding(xs, ys, ps)

        # prepare output
        output = {}
        output["event_cnt"] = event_cnt
        # output["event_voxel"] = event_voxel
        # output["event_mask"] = event_mask
        # output["event_list"] = event_list
        # output["event_list_pol_mask"] = event_list_pol_mask
        # if self.config["data"]["mode"] == "frames":
        #     output["frames"] = frames
        # if self.config["data"]["mode"] == "gtflow_dt1" or self.config["data"]["mode"] == "gtflow_dt4":
        #     output["gtflow"] = flowmap
        # output["dt_gt"] = torch.from_numpy(dt_gt)
        # output["dt_input"] = torch.from_numpy(dt_input)

        output["dotmap"] = sample

        # output["gt_time"] = gt_time  # laurens
        # output["gt_dt"] = gt_dt  # laurens
        # output["gt_t_init"] = t_init  # laurens
        # output["gt_translation"] = t  # laurens
        # output["gt_r_init"] = r_init  # laurens
        # output["gt_rotation"] = r  # laurens

        return output


class RotationDataModule(UzhFpvDataModule):
    def setup(self, stage):
        if stage == "fit":
            train_sequence = partial(
                RotationSequence,
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
                RotationSequence,
                root_dir=self.root_dir,
                time_window=self.time_window,
                count_window=self.count_window,
                crop=self.val_crop,
                rectify=self.rectify,
            )
            for i, rec in enumerate(self.val_recordings):
                if isinstance(rec, str):
                    rec = (rec, None)
                self.val_recordings[i] = rec
            self.val_dataset = ConcatDataset([val_sequence(recording=r, time=t) for r, t in self.val_recordings])
            self.val_frame_shape = (1, 2, *self.val_dataset.datasets[0].frame_shape)


class ModifiedH5Loader(H5Loader):

    def __init__(self, config, num_bins, round_encoding=False):
        raise NotImplementedError("This class is deprecated, please don't use.")
        super().__init__(config, num_bins, round_encoding)
        self.rotation_mode = config["loader"]["rotation_mode"]
        self.rotation_type = config["loader"]["rotation_type"]

        # self.original_getitem = super().__getitem__

    def get_start_end_times(self):
        """
        This function returns 2 tuples, start_time and end_time, resp. the first and last timestamp in each file's ground_truth data.
        """
        start_times, end_times = [], []
        for file_path in self.files:
            with h5py.File(file_path, "r") as file:
                start_times += [file["ground_truth/timestamp"][0] - file.attrs["gt0"]]
                end_times += [file["ground_truth/timestamp"][-1] - file.attrs["gt0"]]
        return start_times, end_times

    def get_gt_index(self, file, t1, t2):
        if t2 < file["ground_truth/timestamp"][0] - file.attrs["t0"]:
            # if the interval occurs before the gt data, simply provide the first row
            idx1 = 0
            idx2 = 1
        elif file["ground_truth/timestamp"][-1] - file.attrs["t0"] < t1:
            # if the interval occurs after the gt data, simply provide the last row
            idx1 = -1
            idx2 = None
        else:
            idxs = np.where(
                (t1 < (file["ground_truth/timestamp"] - file.attrs["t0"]))
                & ((file["ground_truth/timestamp"] - file.attrs["t0"]) < t2)
            )[0]
            idx1 = idxs[0] - 1 if idxs[0] != 0 else idxs[0]
            idx2 = idxs[-1] + 1 if idxs[-1] != len(file["ground_truth/timestamp"]) - 1 else idxs[-1]
        # TODO: implement error handling: what if torch.where is empty?
        return idx1, idx2

    def get_time_translation_rotation(self, file, t1, t2):
        idx1, idx2 = self.get_gt_index(file, t1, t2)

        timestamps = file["ground_truth/timestamp"][idx1:idx2]
        tx = file["ground_truth/tx"][idx1:idx2]
        ty = file["ground_truth/ty"][idx1:idx2]
        tz = file["ground_truth/tz"][idx1:idx2]
        qx = file["ground_truth/qx"][idx1:idx2]
        qy = file["ground_truth/qy"][idx1:idx2]
        qz = file["ground_truth/qz"][idx1:idx2]
        qw = file["ground_truth/qw"][idx1:idx2]

        timestamps = torch.tensor(timestamps)
        dtimestamps = timestamps[-1] - timestamps[0]
        timestamp1 = timestamps[0] - file.attrs["gt0"]

        t = np.transpose(np.vstack((tx, ty, tz)))
        t1 = t[0]
        self.translation_mode = "difference"  # TODO: add translation_mode property
        if self.translation_mode == "absolute":
            t = t[-1]
            # t = np.mean(t, axis=0)
        elif self.translation_mode == "difference":
            t = t[-1] - t[0]
        elif self.translation_mode == "zero-offset":
            t = np.mean(t, axis=0) - file.attrs["gt0"][1:4]  # TODO maybe this should also be t[-1]?
        t = torch.flatten(torch.tensor(t, dtype=torch.float32))
        t1 = torch.flatten(torch.tensor(t1, dtype=torch.float32))

        # TODO: implement zero-offset properly
        r = np.transpose(np.vstack((qx, qy, qz, qw)))
        r1 = Rotation.from_quat(r[0])
        if self.rotation_mode == "absolute":
            r = Rotation.from_quat(r[-1])
            # r = Rotation.from_quat(np.mean(r, axis=0))
        elif self.rotation_mode == "difference":
            r = Rotation.from_quat(r[-1]) * Rotation.from_quat(r[0]).inv()
        elif self.rotation_mode == "local_diff":
            r = Rotation.from_quat(r[0]).inv() * Rotation.from_quat(r[-1])
        elif self.rotation_mode == "zero-offset":
            r = (
                Rotation.from_quat(np.mean(r, axis=0)) * Rotation.from_quat(file.attrs["gt0"][4:]).inv()
            )  # TODO maybe this should also be r[-1]?

        if self.rotation_type == "quat":
            r = r.as_quat()
            r1 = r1.as_quat()
        elif self.rotation_type == "rotvec":
            r = r.as_rotvec()
            r1 = r1.as_rotvec()
        elif self.rotation_type == "matrix":
            r = r.as_matrix()
            r1 = r1.as_matrix()
        elif self.rotation_type == "euler":
            r = r.as_euler("xyz", degrees=False)
            r1 = r1.as_euler("xyz", degrees=False)
        elif self.rotation_type == "euler_deg":
            r = r.as_euler("xyz", degrees=True)
            r1 = r1.as_euler("xyz", degrees=True)

        r = torch.flatten(torch.tensor(r, dtype=torch.float32))
        r1 = torch.flatten(torch.tensor(r1, dtype=torch.float32))

        if torch.isnan(t).any() or torch.isnan(r).any():
            raise ValueError(
                f"NaN value detected in ground truth: t1 = {t1}, t2 = {t2}, idx1 = {idx1}, idx2 = {idx2}, qx = {qx}, t = {r}"
            )

        return timestamp1, dtimestamps, t1, t, r1, r

    def __getitem__(self, index):
        """
        Largely a copy of Jesse's funcion, but includes a get_translation_rotation function.
        Changes are marked with a '# laurens' comment.
        """
        while True:
            batch = index % self.config["loader"]["batch_size"]

            # trigger sequence change
            len_frames = 0
            restart = False
            if self.config["data"]["mode"] == "frames":
                len_frames = len(self.open_files_frames[batch].ts)
            elif self.config["data"]["mode"] == "gtflow_dt1" or self.config["data"]["mode"] == "gtflow_dt4":
                len_frames = len(self.open_files_flowmaps[batch].ts)
            if (
                self.config["data"]["mode"] == "frames"
                or self.config["data"]["mode"] == "gtflow_dt1"
                or self.config["data"]["mode"] == "gtflow_dt4"
            ) and int(np.ceil(self.batch_row[batch] + self.config["data"]["window"])) >= len_frames:
                restart = True

            # load events
            xs = np.zeros((0))
            ys = np.zeros((0))
            ts = np.zeros((0))
            ps = np.zeros((0))

            if not restart:
                idx0, idx1 = self.get_event_index(batch, window=self.config["data"]["window"])

                if (
                    self.config["data"]["mode"] == "frames"
                    or self.config["data"]["mode"] == "gtflow_dt1"
                    or self.config["data"]["mode"] == "gtflow_dt4"
                ) and self.config["data"]["window"] < 1.0:
                    floor_row = int(np.floor(self.batch_row[batch]))
                    ceil_row = int(np.ceil(self.batch_row[batch] + self.config["data"]["window"]))
                    if ceil_row - floor_row > 1:
                        floor_row += ceil_row - floor_row - 1

                    idx0_change = self.batch_row[batch] - floor_row
                    idx1_change = self.batch_row[batch] + self.config["data"]["window"] - floor_row

                    delta_idx = idx1 - idx0
                    idx1 = int(idx0 + idx1_change * delta_idx)
                    idx0 = int(idx0 + idx0_change * delta_idx)

                xs, ys, ts, ps = self.get_events(self.open_files[batch], idx0, idx1)

                if ts.shape[0] > 0:  # laurens
                    gt_time, gt_dt, t_init, t, r_init, r = self.get_time_translation_rotation(
                        self.open_files[batch], ts[0], ts[-1]
                    )  # laurens

            # trigger sequence change
            if (self.config["data"]["mode"] == "events" and xs.shape[0] < self.config["data"]["window"]) or (
                self.config["data"]["mode"] == "time"
                and self.batch_row[batch] + self.config["data"]["window"] >= self.batch_last_ts[batch]
            ):
                restart = True

            # handle case with very few events
            if xs.shape[0] <= 10:
                xs = np.empty([0])
                ys = np.empty([0])
                ts = np.empty([0])
                ps = np.empty([0])

                t = np.empty([0])  # laurens
                t_init = np.empty([0])  # laurens
                r = np.empty([0])  # laurens
                r_init = np.empty([0])  # laurens
                # TODO: check if this is the right approach, the code below seemed more logical but failed
                # if self.rotation_mode == "difference":  # laurens
                #     t = np.empty(len(t))  # laurens
                #     r = np.empty(len(r))  # laurens
                # elif self.rotation_mode == "absolute": # laurens
                #     t = t_init  # laurens
                #     r = r_init  # laurens

            # reset sequence if not enough input events
            if restart:
                self.new_seq = True
                self.reset_sequence(batch)
                self.batch_row[batch] = 0
                self.batch_idx[batch] = max(self.batch_idx) + 1

                self.open_files[batch].close()
                self.open_files[batch] = h5py.File(self.files[self.batch_idx[batch] % len(self.files)], "r")
                self.batch_last_ts[batch] = self.open_files[batch]["events/ts"][-1] - self.open_files[batch].attrs["t0"]

                if self.config["data"]["mode"] == "frames":
                    frames = Frames()
                    self.open_files[batch]["images"].visititems(frames)
                    self.open_files_frames[batch] = frames
                elif self.config["data"]["mode"] == "gtflow_dt1" or self.config["data"]["mode"] == "gtflow_dt4":
                    flowmaps = FlowMaps()
                    if self.config["data"]["mode"] == "gtflow_dt1":
                        self.open_files[batch]["flow_dt1"].visititems(flowmaps)
                    elif self.config["data"]["mode"] == "gtflow_dt4":
                        self.open_files[batch]["flow_dt4"].visititems(flowmaps)
                    self.open_files_flowmaps[batch] = flowmaps
                if self.config["vis"]["bars"]:
                    self.open_files_bar[batch].finish()
                    max_iters = self.get_iters(batch)
                    self.open_files_bar[batch] = ProgressBar(
                        self.files[self.batch_idx[batch] % len(self.files)].split("/")[-1], max=max_iters
                    )

                continue

            # event formatting and timestamp normalization
            dt_input = np.asarray(0.0)
            if ts.shape[0] > 0:
                dt_input = np.asarray(ts[-1] - ts[0])
            xs, ys, ts, ps = self.event_formatting(xs, ys, ts, ps)

            # data augmentation
            xs, ys, ps = self.augment_events(xs, ys, ps, batch)

            # events to tensors
            event_cnt = self.create_cnt_encoding(xs, ys, ps)
            event_mask = self.create_mask_encoding(xs, ys, ps)
            event_voxel = self.create_voxel_encoding(xs, ys, ts, ps)
            event_list = self.create_list_encoding(xs, ys, ts, ps)
            event_list_pol_mask = self.create_polarity_mask(ps)

            # hot pixel removal
            if self.config["hot_filter"]["enabled"]:
                hot_mask = self.create_hot_mask(event_cnt, batch)
                hot_mask_voxel = torch.stack([hot_mask] * self.num_bins, axis=2).permute(2, 0, 1)
                hot_mask_cnt = torch.stack([hot_mask] * 2, axis=2).permute(2, 0, 1)
                event_voxel = event_voxel * hot_mask_voxel
                event_cnt = event_cnt * hot_mask_cnt
                event_mask *= hot_mask.view((1, hot_mask.shape[0], hot_mask.shape[1]))

            # load frames when required
            if self.config["data"]["mode"] == "frames":
                curr_idx = int(np.floor(self.batch_row[batch]))
                next_idx = int(np.ceil(self.batch_row[batch] + self.config["data"]["window"]))

                frames = np.zeros((2, self.config["loader"]["resolution"][0], self.config["loader"]["resolution"][1]))
                img0 = self.open_files[batch]["images"][self.open_files_frames[batch].names[curr_idx]][:]
                img1 = self.open_files[batch]["images"][self.open_files_frames[batch].names[next_idx]][:]
                frames[0, :, :] = self.augment_frames(img0, batch)
                frames[1, :, :] = self.augment_frames(img1, batch)
                frames = torch.from_numpy(frames.astype(np.uint8))

            # load GT optical flow when required
            dt_gt = 0.0
            if self.config["data"]["mode"] == "gtflow_dt1" or self.config["data"]["mode"] == "gtflow_dt4":
                idx = int(np.ceil(self.batch_row[batch] + self.config["data"]["window"]))
                if self.config["data"]["mode"] == "gtflow_dt1":
                    flowmap = self.open_files[batch]["flow_dt1"][self.open_files_flowmaps[batch].names[idx]][:]
                elif self.config["data"]["mode"] == "gtflow_dt4":
                    flowmap = self.open_files[batch]["flow_dt4"][self.open_files_flowmaps[batch].names[idx]][:]
                flowmap = self.augment_flowmap(flowmap, batch)
                flowmap = torch.from_numpy(flowmap.copy())
                if idx > 0:
                    dt_gt = self.open_files_flowmaps[batch].ts[idx] - self.open_files_flowmaps[batch].ts[idx - 1]
            dt_gt = np.asarray(dt_gt)

            # update window
            self.batch_row[batch] += self.config["data"]["window"]

            # break while loop if everything went well
            break

        # prepare output
        output = {}
        output["event_cnt"] = event_cnt
        output["event_voxel"] = event_voxel
        output["event_mask"] = event_mask
        output["event_list"] = event_list
        output["event_list_pol_mask"] = event_list_pol_mask
        if self.config["data"]["mode"] == "frames":
            output["frames"] = frames
        if self.config["data"]["mode"] == "gtflow_dt1" or self.config["data"]["mode"] == "gtflow_dt4":
            output["gtflow"] = flowmap
        output["dt_gt"] = torch.from_numpy(dt_gt)
        output["dt_input"] = torch.from_numpy(dt_input)

        output["gt_time"] = gt_time  # laurens
        output["gt_dt"] = gt_dt  # laurens
        output["gt_t_init"] = t_init  # laurens
        output["gt_translation"] = t  # laurens
        output["gt_r_init"] = r_init  # laurens
        output["gt_rotation"] = r  # laurens

        return output


# ---------- hdf5 converter ---------- #
# be aware: not adapted to the new repository yet!


# imports
import os
import shutil
import h5py
import numpy as np
import matplotlib.pyplot as plt


def print_structure(name, obj):
    print(name)


def inspect_hd5(data_dir, file_to_read):  # inspecting the files
    with h5py.File(data_dir + "/" + file_to_read, "r") as file:
        print("Keys:", file.keys())

        # printing the structure
        file.visititems(print_structure)
        print(list(file.attrs))

        print(file.attrs["t0"])
        print(file["events/ts"][-1])
        print(file["events/ts"][-1] - file["events/ts"][0])
        print(min(file["events/xs"]), max(file["events/xs"]))
        print(min(file["events/ys"]), max(file["events/ys"]))

        # # Getting the data
        # data = list(file[a_group_key])
        # print(data)

        # print(file['events']['xs'])
        # print(file['ground_truth']['tx'])

        # plt.imshow(file['images']['image000000010'][:,:], cmap='gray')
        # plt.show()


# okay, so the h5 file consists of images and event data;
# the event data has label 'events' and has 4 subgroups: 'ps' (polarity, +1 or -1), 'ts' (time in s?), 'xs' (horizontal pixel position, 0-127), 'ys' (vertical pixel position, 0-127);
# each column has shape (500000, ) and type "|b1" (bool, ps) or "<f8" (float, ts) or "<i2" (integer, other columns);
# the image data has label 'images' and has 18 subgroups: 'image000000000' up to 'image000000017' (1E9 digits);
# each image has shape (128, 128) and type "|u1" (it is a grayscale image).


def convert_events_to_hd5(data_dir, events_file, file_to_write):  # for attempting my own conversion

    data = np.genfromtxt(data_dir + "/" + events_file, delimiter=" ", skip_header=1)  # loading the data
    columns = ["ps", "ts", "xs", "ys"]
    typs = [bool, float, np.int16, np.int16]

    with h5py.File(data_dir + "/" + file_to_write, "a") as file:  # editing the file
        events_group = file.create_group("events")
        for i in range(data.shape[1]):
            # TODO: next line hasn't been tested!
            events_group.create_dataset(
                columns[i], data=data[:50000, i].astype(typs[i])
            )  # creating a subgroup for each column, use the correct type


def convert_groundtruth_to_hd5(data_dir, ground_truth_file, file_to_write, low=0, high=50000, add_init=False):

    data = np.genfromtxt(data_dir + "/" + ground_truth_file, delimiter=" ", skip_header=1)  # loading the data
    columns = ["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]

    with h5py.File(data_dir + "/" + file_to_write, "a") as file:  # editing the file
        gt_group = file.create_group("ground_truth")
        for i in range(data.shape[1]):
            gt_group.create_dataset(
                columns[i], data=data[low:high, i].astype(float)
            )  # creating a subgroup for each column

        if add_init:
            file.attrs["gt0"] = data[columns.index("timestamp"), 0]


def modify_existing(h5_data_dir, txt_data_dir, partial_name, ground_truth_file="groundtruth.txt"):
    ts = np.genfromtxt(txt_data_dir + "/" + ground_truth_file, delimiter=" ", skip_header=1)[:, 0]

    files = os.listdir(h5_data_dir)
    files = list(filter(lambda file: partial_name in file, files))
    for i in range(len(files)):

        file_to_read = partial_name + "_" + str(i) + ".h5"
        assert file_to_read in files
        file_to_cache = partial_name[:-2] + "temp_" + str(i) + ".h5"
        file_to_write = partial_name[:-2] + "rotation_" + str(i) + ".h5"
        shutil.copy(h5_data_dir + "/" + file_to_read, txt_data_dir + "/" + file_to_cache)

        with h5py.File(h5_data_dir + "/" + file_to_read, "r") as original_file:
            if original_file["events/ts"][-1] > ts[0] and original_file.attrs["t0"] < ts[-1]:
                low = np.where((original_file.attrs["t0"] < ts) & (ts < original_file["events/ts"][-1]))[0][0]
                high = np.where((original_file.attrs["t0"] < ts) & (ts < original_file["events/ts"][-1]))[0][-1]

                if (ts[high] - ts[low]) > 0.9 * (original_file["events/ts"][-1] - original_file.attrs["t0"]):
                    shutil.copy(txt_data_dir + "/" + file_to_cache, txt_data_dir + "/" + file_to_write)

                    convert_groundtruth_to_hd5(txt_data_dir, ground_truth_file, file_to_write, low, high, add_init=True)

        os.remove(txt_data_dir + "/" + file_to_cache)


# with h5py.File(data_dir + '/' + file_to_write, 'a') as file:
#     for col, typ in zip(['ps', 'ts', 'xs', 'ys'], [bool, float, np.int16, np.int16]):
#         data = file['events'][col][:].astype(typ)
#         del file['events'][col]
#         file['events'].create_dataset(col, data=data)


# Notes on integrating in existing files:
# - use indoor_forward_3_davis_with_gt_3 up to ..._10
# - split according to ['events/ts'][-1] and attrs['t0']


if __name__ == "__main__":
    raise NotImplementedError("This code has not been adapted to the new repository yet. Please review before running.")
    new_data_dir = "datasets/data/rotation_demo"
    h5_data_dir = "datasets/data/training"
    txt_data_dir = "datasets/data/txt/"
    events_file = "events.txt"
    ground_truth_file = "groundtruth.txt"
    # file_to_write = 'test.h5'  # 'indoor_forward_3_davis_with_gt_0.h5'

    # convert_events_to_hd5(data_dir, events_file, file_to_write)  # takes ~15 mins
    # convert_groundtruth_to_hd5(data_dir, ground_truth_file, file_to_write)
    # inspect_hd5(h5_data_dir, 'indoor_forward_3_davis_with_gt_3.h5')
    for n in [5, 7, 9, 10]:
        partial_name = f"indoor_forward_{n}_davis_with_gt"
        txt_data_dir_ = txt_data_dir + partial_name
        modify_existing(h5_data_dir, txt_data_dir_, partial_name=partial_name)

    print("Done.")
