"""
Simple camera parameter optimization using quaternion representation
Based on the user's example
"""
import torch
import torch.nn as nn
import numpy as np


def quaternion_to_matrix(quaternions):
    """
    Convert quaternions [w,x,y,z] to rotation matrices
    Args:
        quaternions: (..., 4) tensor
    Returns:
        rotation matrices: (..., 3, 3)
    """
    # Normalize
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)
    
    w, x, y, z = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    
    # Build rotation matrix
    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], dim=-1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], dim=-1)
    ], dim=-2)
    
    return R


def matrix_to_quaternion(matrix):
    """
    Convert rotation matrix to quaternion [w, x, y, z]
    Args:
        matrix: (3, 3) or (N, 3, 3) numpy array
    Returns:
        quaternion: (4,) or (N, 4) numpy array
    """
    if matrix.ndim == 2:
        matrix = matrix[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False
    
    batch_size = matrix.shape[0]
    quat = np.zeros((batch_size, 4))
    
    for i in range(batch_size):
        R = matrix[i]
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            quat[i, 0] = 0.25 / s  # w
            quat[i, 1] = (R[2, 1] - R[1, 2]) * s  # x
            quat[i, 2] = (R[0, 2] - R[2, 0]) * s  # y
            quat[i, 3] = (R[1, 0] - R[0, 1]) * s  # z
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            quat[i, 0] = (R[2, 1] - R[1, 2]) / s  # w
            quat[i, 1] = 0.25 * s  # x
            quat[i, 2] = (R[0, 1] + R[1, 0]) / s  # y
            quat[i, 3] = (R[0, 2] + R[2, 0]) / s  # z
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            quat[i, 0] = (R[0, 2] - R[2, 0]) / s  # w
            quat[i, 1] = (R[0, 1] + R[1, 0]) / s  # x
            quat[i, 2] = 0.25 * s  # y
            quat[i, 3] = (R[1, 2] + R[2, 1]) / s  # z
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            quat[i, 0] = (R[1, 0] - R[0, 1]) / s  # w
            quat[i, 1] = (R[0, 2] + R[2, 0]) / s  # x
            quat[i, 2] = (R[1, 2] + R[2, 1]) / s  # y
            quat[i, 3] = 0.25 * s  # z
    
    if squeeze:
        quat = quat.squeeze(0)
    
    return quat


class CameraParameters(nn.Module):
    """
    Trainable camera parameters - simple quaternion + translation representation
    """
    def __init__(self, cameras, optimize_extrinsics=True, optimize_intrinsics=False):
        super().__init__()
        
        self.num_cameras = len(cameras)
        self.optimize_extrinsics = optimize_extrinsics
        self.optimize_intrinsics = optimize_intrinsics
        
        # Convert rotation matrices to quaternions [w,x,y,z]
        quats = []
        tcws = []
        for cam in cameras:
            quat = matrix_to_quaternion(cam.Rcw.cpu().numpy())
            quats.append(quat)
            tcws.append(cam.tcw.cpu().numpy())
        
        quats = np.stack(quats)
        tcws = np.stack(tcws)
        
        # Camera pose parameters
        if optimize_extrinsics:
            self.qcw = nn.Parameter(torch.from_numpy(quats).cuda().float())
            self.tcw = nn.Parameter(torch.from_numpy(tcws).cuda().float())
        else:
            self.register_buffer('qcw', torch.from_numpy(quats).cuda().float())
            self.register_buffer('tcw', torch.from_numpy(tcws).cuda().float())
        
        # Intrinsics
        fx = np.array([cam.fx for cam in cameras])
        fy = np.array([cam.fy for cam in cameras])
        cx = np.array([cam.cx for cam in cameras])
        cy = np.array([cam.cy for cam in cameras])
        
        if optimize_intrinsics:
            self.fx = nn.Parameter(torch.from_numpy(fx).cuda().float())
            self.fy = nn.Parameter(torch.from_numpy(fy).cuda().float())
            self.cx = nn.Parameter(torch.from_numpy(cx).cuda().float())
            self.cy = nn.Parameter(torch.from_numpy(cy).cuda().float())
        else:
            self.register_buffer('fx', torch.from_numpy(fx).cuda().float())
            self.register_buffer('fy', torch.from_numpy(fy).cuda().float())
            self.register_buffer('cx', torch.from_numpy(cx).cuda().float())
            self.register_buffer('cy', torch.from_numpy(cy).cuda().float())
    
    def get_camera_params(self, cam_idx):
        """Get camera parameters (with gradients if optimizing)"""
        Rcw = quaternion_to_matrix(self.qcw[cam_idx])
        tcw = self.tcw[cam_idx]
        fx = self.fx[cam_idx]
        fy = self.fy[cam_idx]
        cx = self.cx[cam_idx]
        cy = self.cy[cam_idx]
        
        return Rcw, tcw, fx, fy, cx, cy
    
    def get_optimizer_params(self):
        """Get parameter groups for optimizer"""
        param_groups = []
        
        if self.optimize_extrinsics:
            param_groups.append({'params': [self.qcw], 'lr': 0.0001, 'name': 'camera_rotation'})
            param_groups.append({'params': [self.tcw], 'lr': 0.001, 'name': 'camera_translation'})
        
        if self.optimize_intrinsics:
            param_groups.append({'params': [self.fx, self.fy], 'lr': 0.1, 'name': 'camera_focal'})
            param_groups.append({'params': [self.cx, self.cy], 'lr': 0.1, 'name': 'camera_center'})
        
        return param_groups
