"""
Simplified pose optimization with isotropic Gaussians
- Fixed positions (pws)
- No rotation (identity quaternion)
- Only optimizes: camera pose, colors, alphas, scales
- Reduces 3DGS degrees of freedom for better camera pose optimization
"""
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from gsplat.pytorch_ssim import gau_loss
from gsplat.gausplat_dataset import *
from gsplat.utils import get_alphas, get_scales
from gsplat.camera_optimizer import CameraParameters, matrix_to_quaternion
from gsplat.simple_gau import full_to_simple_params, save_simple_as_full
from gsplat.simple_pose_func import SimplePoseFunction


torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="the path of dataset")
    parser.add_argument("--gs_init", default="data/final.npy", 
                        help="initial Gaussian splatting parameters")
    args = parser.parse_args()

    if args.path:
        print("Simplified pose optimization for %s ..." % args.path)
        gs_set = GSplatDataset(args.path)
    else:
        print("No dataset path provided.")
        exit(0)

    # Load initial Gaussian parameters
    loaded = np.load(args.gs_init, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        gs_full = loaded.item()
    else:
        gs_full = loaded
    print(f"Loaded Gaussian parameters from {args.gs_init}")
    print(f"Number of Gaussians: {gs_full['pw'].shape[0]}")

    # Convert to simplified format
    pws_np = gs_full['pw']
    
    # Extract RGB from SH
    C0 = 0.28209479177387814
    sh_dc = gs_full['sh'][:, :3]
    colors_np = sh_dc * C0 + 0.5
    colors_np = np.clip(colors_np, 0, 1)
    
    # Average scales to get isotropic
    scales_np = gs_full['scale'].mean(axis=1, keepdims=True)
    alphas_np = gs_full['alpha']

    print("\nSimplified Gaussian model:")
    print(f"  Positions (pws): FIXED")
    print(f"  Rotations: FIXED (identity)")
    print(f"  High-order SH: FIXED (zero)")
    print(f"  Optimizable: colors, alphas, scales (isotropic)")

    # Initialize camera parameters
    print("\nInitializing camera parameters for optimization...")
    cam_params = CameraParameters(
        gs_set.cameras,
        optimize_extrinsics=True,
        optimize_intrinsics=False
    )

    # Create simplified training parameters
    from gsplat.utils import get_scales_raw, get_alphas_raw
    
    pws = torch.from_numpy(pws_np).type(torch.float32).to('cuda')  # FIXED
    colors = torch.from_numpy(colors_np).type(torch.float32).to('cuda').requires_grad_()
    
    scales_t = torch.from_numpy(scales_np).type(torch.float32).to('cuda')
    scales_raw = get_scales_raw(scales_t).requires_grad_()
    
    alphas_t = torch.from_numpy(alphas_np).type(torch.float32).to('cuda')
    alphas_raw = get_alphas_raw(alphas_t).requires_grad_()

    # Optimizer - only for optimizable parameters
    adam_params = [
        {'params': [colors], 'lr': 0.001, "name": "colors"},
        {'params': [alphas_raw], 'lr': 0.05, "name": "alphas_raw"},
        {'params': [scales_raw], 'lr': 0.01, "name": "scales_raw"},
    ]
    adam_params.extend(cam_params.get_optimizer_params())

    optimizer = optim.Adam(adam_params, lr=0.000, eps=1e-15)

    cam0, _ = gs_set[0]
    fig, ax = plt.subplots()
    img = ax.imshow(np.zeros(shape=(cam0.height, cam0.width, 3), dtype=np.uint8))
    txt = ax.text(50, 50, "", size=20, color='white')

    epochs = 100
    n = len(gs_set)

    print("\nStarting simplified pose optimization...")
    print(f"Epochs: {epochs}")
    print(f"Images: {n}")

    for epoch in range(epochs):
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        avg_loss = 0
        
        for i in idxs:
            cam, image_gt = gs_set[i]

            qcw = cam_params.qcw[i]
            tcw = cam_params.tcw[i]
            alphas = get_alphas(alphas_raw)
            scales = get_scales(scales_raw).squeeze()
            
            image = SimplePoseFunction.apply(
                qcw, tcw,
                pws,
                colors,
                alphas,
                scales,
                cam
            )
            
            loss = gau_loss(image, image_gt)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            avg_loss += loss.item()

            if (i == 0):
                img.set_data(np.clip(image.detach().permute(
                    1, 2, 0).to('cpu').numpy(), 0, 1))
                txt._text = "epoch %d" % epoch
                plt.pause(0.1)

        avg_loss = avg_loss / n
        print("epoch:%d avg_loss:%f" % (epoch, avg_loss))

        # Print statistics
        if epoch % 5 == 0:
            print("Camera pose changes:")
            quat_changes = torch.norm(cam_params.qcw - torch.from_numpy(
                np.stack([matrix_to_quaternion(c.Rcw.cpu().numpy()) 
                         for c in gs_set.cameras])).cuda().float(), dim=1)
            trans_changes = torch.norm(cam_params.tcw - torch.stack([c.tcw for c in gs_set.cameras]), dim=1)
            print(f"  Avg quat change: {quat_changes.mean().item():.6f}")
            print(f"  Avg trans change: {trans_changes.mean().item():.6f}")
            
            with torch.no_grad():
                alpha_vals = get_alphas(alphas_raw)
                scale_vals = get_scales(scales_raw)
                print(f"  Alpha range: [{alpha_vals.min().item():.4f}, {alpha_vals.max().item():.4f}]")
                print(f"  Scale range: [{scale_vals.min().item():.4f}, {scale_vals.max().item():.4f}]")

        # Save checkpoints
        if (epoch % 10 == 0):
            # Save in full format for viewer compatibility
            with torch.no_grad():
                simple_params = {
                    'pws': pws,
                    'colors': colors,
                    'scales_raw': scales_raw,
                    'alphas_raw': alphas_raw
                }
                fn = f"data/simple_epoch{epoch:04d}.npy"
                save_simple_as_full(fn, simple_params)

    # Save final results
    with torch.no_grad():
        simple_params = {
            'pws': pws,
            'colors': colors,
            'scales_raw': scales_raw,
            'alphas_raw': alphas_raw
        }
        save_simple_as_full('data/simple_final.npy', simple_params)

    # Save optimized camera parameters
    cam_data = {
        'qcw': cam_params.qcw.detach().cpu().numpy(),
        'tcw': cam_params.tcw.detach().cpu().numpy(),
    }
    np.save('data/camera_params_simple.npy', cam_data)
    print("\nOptimized camera parameters saved to data/camera_params_simple.npy")
    
    print("\n" + "="*60)
    print("Simplified pose optimization completed!")
    print("="*60)
