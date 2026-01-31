import os
import torch
from torch.utils.data import Dataset
import maths
import numpy as np
import torch.nn.functional as F
import random
from pathlib import Path

from sklearn.model_selection import train_test_split
class UltraPoseDataset(Dataset):
    def __init__(self, split="train", data_path=[""], window_length=30):
        super().__init__()

        self.train = split
        self.data_path = data_path
        self.window_length = window_length
        self.data = self.load_data()

    def normalize_tensor(self, tensor):
        return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))

    def load_data(self):

        input_sensor = []
        output_pose = []
        num = 0

        for folder in self.data_path:
            folder_path = Path(folder)
            if folder_path.is_dir():

                for data_files in folder_path.rglob('*.npy'):

                    fname = data_files.name
                    fdata = np.load(data_files, allow_pickle=True).item()
                    num = num + 1

                    doppler = np.transpose(fdata["doppler"], [2, 0, 1])
                    doppler = torch.abs(torch.from_numpy(doppler))
                    cir = torch.abs(torch.from_numpy(np.transpose(fdata["cir"], [1, 0])))

                    acc = torch.from_numpy(np.transpose(fdata["acc"], [1, 0, 2])) / 30 # scale the acc data
                    ori = torch.from_numpy(np.transpose(fdata["ori"], [1, 0, 2, 3]))
                    imu = torch.cat([acc.flatten(1), ori.flatten(1)], dim=1)

                    pose = torch.from_numpy(fdata["pose"])
                    pose = maths.rotation_matrix_to_r6d(pose).reshape(-1, 24, 6).reshape(-1, 24 * 6)
                    pose_split = torch.split(pose, self.window_length)[:-1]

                    idx = np.arange(0, len(pose_split), 1, dtype=int)
                    inputs = []
                    poses = []

                    if fname.find("Hand")!=-1:
                        phone_label = torch.zeros(self.window_length).unsqueeze(-1)
                    else:
                        phone_label = torch.ones(self.window_length).unsqueeze(-1)


                    for i in idx:
                        sample_i = {}
                        fmodals = []
                        sidx = i * self.window_length
                        eidx = (i + 1) * self.window_length
                        sample_i["imu"] = imu[sidx:eidx, :].float()

                        doppler[sidx:eidx, 0] = self.normalize_tensor(doppler[sidx:eidx, 0])
                        doppler[sidx:eidx, 1] = self.normalize_tensor(doppler[sidx:eidx, 1])
                        doppler[sidx:eidx, 2] = self.normalize_tensor(doppler[sidx:eidx, 2])
                        sample_i["doppler"] = (doppler[sidx:eidx]).float()
                        sample_i["cir"] = self.normalize_tensor(cir[sidx:eidx]).float()
                        sample_i["doppler"] = sample_i["doppler"].flatten(1)
                        sample_i["pose"] = pose_split[i].float()

                        fmodals.append(sample_i["imu"])
                        fmodals.append(sample_i["doppler"])
                        fmodals.append(sample_i["cir"])
                        fmodals.append(phone_label)

                        inputs.append(torch.cat(fmodals, dim=1))
                        poses.append(sample_i["pose"])

                    input_sensor.extend(torch.stack(inputs))
                    output_pose.extend(torch.stack(poses))

        self.sensor = input_sensor
        self.pose = output_pose

        data = list(zip(self.sensor, self.pose))
        train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        if self.train == "train":
            self.sensor, self.pose = zip(*train_val_data)
        else:
            self.sensor, self.pose = zip(*test_data)

        print("Dataset Size:", len(self.sensor))

    def __getitem__(self, idx):

        _input = self.sensor[idx]
        _output = self.pose[idx]

        return _input, _output

    def __len__(self):
        return len(self.sensor)


