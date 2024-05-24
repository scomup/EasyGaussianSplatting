from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
from gsplat.read_write_model import *
from PIL import Image
import torch
import torchvision
from plyfile import PlyData


class Camera:
    def __init__(self, id, width, height, fx, fy, cx, cy, Rcw, tcw):
        self.id = id
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.Rcw = Rcw
        self.tcw = tcw
        self.twc = -torch.linalg.inv(Rcw) @ tcw


class GSplatDataset(Dataset):
    def __init__(self, path, device='cpu') -> None:
        super().__init__()
        self.device = device
        camera_params, image_params = read_model(Path(path, "sparse/0"), ext='.bin')
        self.cameras = []
        self.images = []
        for image_param in image_params.values():
            i = image_param.camera_id
            camera_param = camera_params[i]
            image = torchvision.io.read_image(
                str(Path(path, "images", image_param.name))).to(self.device).to(torch.float32) / 255.
            _, height, width = image.shape
            w_scale = width/camera_param.width
            h_scale = height/camera_param.height
            fx = camera_param.params[0] * w_scale
            fy = camera_param.params[1] * h_scale
            cx = camera_param.params[2] * w_scale
            cy = camera_param.params[3] * h_scale
            Rcw = torch.from_numpy(image_param.qvec2rotmat()).to(self.device).to(torch.float32)
            tcw = torch.from_numpy(image_param.tvec).to(self.device).to(torch.float32)
            camera = Camera(image_param.id, width, height, fx, fy, cx, cy, Rcw, tcw)
            self.cameras.append(camera)
            self.images.append(image)

    def __getitem__(self, index: int):
        return self.cameras[index], self.images[index]

    def __len__(self) -> int:
        return len(self.images)


if __name__ == "__main__":
    path = '/home/liu/bag/gaussian-splatting/tandt/train'
    gs_dataset = GSplatDataset(path)
    gs_dataset[0]

