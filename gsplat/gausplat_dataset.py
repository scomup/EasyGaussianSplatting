from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
from gsplat.read_write_model import *
# from read_write_model import *
from PIL import Image
import torch
import torchvision
from plyfile import PlyData
import torchvision.transforms as transforms
from PIL import Image


class Camera:
    def __init__(self, id, width, height, fx, fy, cx, cy, Rcw, tcw, path):
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
        self.path = path



class GSplatDataset(Dataset):
    def __init__(self, path, resize_rate=1, device='cuda') -> None:
        super().__init__()
        self.device = device
        self.resize_rate = resize_rate

        camera_params, image_params = read_model(Path(path, "sparse/0"), ext='.bin')
        self.cameras = []
        self.images = []
        for image_param in image_params.values():
            i = image_param.camera_id
            camera_param = camera_params[i]
            im_path = str(Path(path, "images", image_param.name))
            image = Image.open(im_path)
            if (resize_rate != 1):
                image = image.resize((image.width * self.resize_rate, image.height * self.resize_rate))

            w_scale = image.width/camera_param.width
            h_scale = image.height/camera_param.height
            fx = camera_param.params[0] * w_scale
            fy = camera_param.params[1] * h_scale
            cx = camera_param.params[2] * w_scale
            cy = camera_param.params[3] * h_scale
            Rcw = torch.from_numpy(image_param.qvec2rotmat()).to(self.device).to(torch.float32)
            tcw = torch.from_numpy(image_param.tvec).to(self.device).to(torch.float32)
            camera = Camera(image_param.id, image.width, image.height, fx, fy, cx, cy, Rcw, tcw, im_path)
            image = torchvision.transforms.functional.to_tensor(image).to(self.device).to(torch.float32)

            self.cameras.append(camera)
            self.images.append(image)
        try:
            self.gs = np.load(Path(path, "sparse/0/points3D.npy"))
        except:
            self.gs = read_points_bin_as_gau(Path(path, "sparse/0/points3D.bin"))
            np.save(Path(path, "sparse/0/points3D.npy"), self.gs)

        twcs = torch.stack([x.twc for x in self.cameras])
        cam_dist = torch.linalg.norm(twcs - torch.mean(twcs, axis=0), axis=1)
        self.sence_size = float(torch.max(cam_dist)) * 1.1

    def __getitem__(self, index: int):
        return self.cameras[index], self.images[index]

    def __len__(self) -> int:
        return len(self.images)


if __name__ == "__main__":
    path = '/home/liu/bag/gaussian-splatting/tandt/train'
    gs_dataset = GSplatDataset(path)
    gs_dataset[0]
