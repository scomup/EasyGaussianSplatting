from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
from gsplat.read_write_model import *
from PIL import Image
import torch
import torchvision
from plyfile import PlyData


class Camera:
    def __init__(self, id, width, height, fx, fy, cx, cy, Rcw, tcw, image):
        self.id = id
        self.width = width
        self.height = height
        self.K = np.eye(3)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.Rcw = Rcw
        self.tcw = tcw
        self.twc = -torch.linalg.inv(Rcw) @ tcw
        self.image = image


class GSplatDataset(Dataset):
    def __init__(self, path, device='cpu') -> None:
        super().__init__()
        self.device = device
        cameras, images, points = read_model(Path(path, "sparse/0"), ext='.bin')
        self.cameras_info = []
        for image in images.values():
            i = image.camera_id
            camera = cameras[i]
            image_data = torchvision.io.read_image(
                str(Path(path, "images", image.name))).to(self.device).to(torch.float32) / 255.
            _, height, width = image_data.shape
            w_scale = width/camera.width
            h_scale = height/camera.height
            fx = camera.params[0] * w_scale
            fy = camera.params[1] * h_scale
            cx = camera.params[2] * w_scale
            cy = camera.params[3] * h_scale
            Rcw = torch.from_numpy(image.qvec2rotmat()).to(self.device).to(torch.float32)
            tcw = torch.from_numpy(image.tvec).to(self.device).to(torch.float32)
            cam = Camera(image.id, width, height, fx, fy, cx, cy, Rcw, tcw, image_data)
            self.cameras_info.append(cam)
        self.gs = points

    def __getitem__(self, index: int) -> Camera:
        return self.cameras_info[index]

    def __len__(self) -> int:
        return len(self.cameras_info)


if __name__ == "__main__":
    path = '/home/liu/bag/gaussian-splatting/tandt/train'
    gs_dataset = GSplatDataset(path)
    gs_dataset[0]

