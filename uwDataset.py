
import json
from os.path import join
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import cv2

import numpy as np

from jax.config import config

config.enable_omnistaging()







class UwDataset(Dataset):

    def __init__(self, root_dir, split_type="train", transform=None, half_res=False):
        # Load data
        self.root_dir = root_dir
        self.half_res = half_res
        json_filename = join(self.root_dir, f"transforms.json")
        with open(json_filename) as f:
            json_data = json.load(f)

        self.samples = []
        for frame in json_data["frames"]:
            img_fname = join(self.root_dir, frame["file_path"] )

            pose = np.array(frame["transform_matrix"])
            # Adjust poses so that camera front is z+
            T = np.eye(3)
            T[1, 1] = -1
            T[2, 2] = -1
            pose[:3, :3] = pose[:3, :3].dot(T)
            self.samples.append((img_fname, pose))

        # focal length
        img = self.load_image(img_fname)
        h, w = img.shape[:2]
        camera_angle_x = float(json_data["camera_angle_x"])
        self.focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

        self.transform = transform

    def load_image(self, img_fname):
        img = cv2.imread(img_fname, cv2.IMREAD_UNCHANGED)
        # Downsample
        if self.half_res:
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        img = img.astype(np.float32) / 255.0

        if img.shape[2] == 4:
            # Add arbitrary background color for image with alpha
            bkg = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            img_rgb = img[..., :3]
            img_alpha = img[..., 3:]
            img = img_rgb*img_alpha + bkg*(1.-img_alpha)
        # BGR to RGB
        img = np.array(img[...,::-1])
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_fname, pose = self.samples[idx]
        img = self.load_image(img_fname)
        sample = {"img": img, "pose": pose}
        if self.transform:
            sample = self.transform(sample)
        sample["focal"] = self.focal
        return sample