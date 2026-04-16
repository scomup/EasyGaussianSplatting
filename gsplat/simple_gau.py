"""
Simplified Gaussian Splatting with isotropic spheres
- Each Gaussian is an isotropic sphere (same radius in all directions)
- No rotation (rots fixed to identity quaternion [1,0,0,0])
- Only low-order SH (RGB color)
- Reduces degrees of freedom for better camera pose optimization
"""
import torch
import numpy as np


def simple_to_full_params(pws, colors, scales, alphas):
    """
    Convert simplified parameters to full 3DGS format
    Args:
        pws: (N, 3) positions
        colors: (N, 3) RGB colors (will be converted to SH)
        scales: (N,) or (N, 1) isotropic scales
        alphas: (N, 1) transparency
    Returns:
        Full 3DGS parameters dict
    """
    N = pws.shape[0]
    device = pws.device
    
    # Convert scales to (N,) if needed
    if scales.dim() > 1:
        scales = scales.squeeze()
    
    # Create isotropic 3D scales (same in all directions)
    scales_3d = scales.unsqueeze(1).repeat(1, 3)  # (N, 3)
    
    # Identity quaternions (no rotation)
    rots = torch.zeros((N, 4), dtype=torch.float32, device=device)
    rots[:, 0] = 1.0  # [w=1, x=0, y=0, z=0]
    
    # Convert RGB to SH (only DC component)
    # SH DC coefficient: C0 = 0.28209479177387814
    C0 = 0.28209479177387814
    sh_dc = (colors - 0.5) / C0  # (N, 3)
    
    # High-order SH coefficients (all zeros)
    sh_high = torch.zeros((N, 45), dtype=torch.float32, device=device)
    
    # Combine SH
    shs = torch.cat([sh_dc, sh_high], dim=1)  # (N, 48)
    
    return {
        'pw': pws,
        'sh': shs,
        'scale': scales_3d,
        'rot': rots,
        'alpha': alphas
    }


def full_to_simple_params(gs_params):
    """
    Extract simplified parameters from full 3DGS format
    Args:
        gs_params: dict with 'pw', 'sh', 'scale', 'rot', 'alpha'
    Returns:
        pws, colors, scales, alphas (simplified format)
    """
    pws = gs_params['pw']
    
    # Extract RGB from SH DC component
    C0 = 0.28209479177387814
    sh_dc = gs_params['sh'][:, :3]
    colors = sh_dc * C0 + 0.5
    colors = torch.clamp(colors, 0, 1)
    
    # Average the 3D scales to get isotropic scale
    scales_3d = gs_params['scale']
    scales = scales_3d.mean(dim=1, keepdim=True)  # (N, 1)
    
    alphas = gs_params['alpha']
    
    return pws, colors, scales, alphas


def create_simple_gs(pws, colors, scales, alphas):
    """
    Create simplified Gaussian parameters for training
    Args:
        pws: (N, 3) numpy array - positions (FIXED, not optimized)
        colors: (N, 3) numpy array - RGB colors [0,1]
        scales: (N,) or (N, 1) numpy array - isotropic scales
        alphas: (N,) or (N, 1) numpy array - transparency [0,1]
    Returns:
        training_params dict
    """
    # Position is FIXED (no gradient)
    pws_t = torch.from_numpy(pws).type(torch.float32).to('cuda')
    
    # Optimizable parameters
    colors_t = torch.from_numpy(colors).type(torch.float32).to('cuda').requires_grad_()
    
    # Ensure correct shape
    if scales.ndim == 1:
        scales = scales[:, np.newaxis]
    if alphas.ndim == 1:
        alphas = alphas[:, np.newaxis]
    
    # Convert to raw (inverse activation)
    from gsplat.utils import get_scales_raw, get_alphas_raw
    
    scales_t = torch.from_numpy(scales).type(torch.float32).to('cuda')
    scales_raw = get_scales_raw(scales_t).requires_grad_()
    
    alphas_t = torch.from_numpy(alphas).type(torch.float32).to('cuda')
    alphas_raw = get_alphas_raw(alphas_t).requires_grad_()
    
    params = {
        "pws": pws_t,  # FIXED - no requires_grad
        "colors": colors_t,
        "scales_raw": scales_raw,
        "alphas_raw": alphas_raw
    }
    
    return params


def get_simple_optimizer_params(params):
    """
    Get optimizer parameter groups for simplified Gaussians
    Only optimizes: colors, alphas, scales
    Position (pws) is FIXED
    """
    adam_params = [
        # pws is NOT included - completely fixed
        {'params': [params['colors']], 'lr': 0.001, "name": "colors"},  # Color
        {'params': [params['alphas_raw']], 'lr': 0.05, "name": "alphas_raw"},  # Main
        {'params': [params['scales_raw']], 'lr': 0.01, "name": "scales_raw"},  # Main
    ]
    return adam_params


def save_simple_as_full(filename, simple_params):
    """
    Save simplified parameters in full 3DGS format for viewer compatibility
    """
    from gsplat.utils import get_alphas, get_scales
    
    pws = simple_params['pws'].detach().cpu().numpy()
    colors = simple_params['colors'].detach().cpu().numpy()
    scales = get_scales(simple_params['scales_raw']).detach().cpu().numpy()
    if scales.ndim == 1:
        scales = scales[:, np.newaxis]
    alphas = get_alphas(simple_params['alphas_raw']).detach().cpu().numpy()
    
    # Convert to full format
    full_params = simple_to_full_params(
        torch.from_numpy(pws).cuda(),
        torch.from_numpy(colors).cuda(),
        torch.from_numpy(scales).cuda(),
        torch.from_numpy(alphas).cuda()
    )
    
    # Save as numpy dict
    save_dict = {
        'pw': full_params['pw'].detach().cpu().numpy(),
        'sh': full_params['sh'].detach().cpu().numpy(),
        'scale': full_params['scale'].detach().cpu().numpy(),
        'rot': full_params['rot'].detach().cpu().numpy(),
        'alpha': full_params['alpha'].detach().cpu().numpy()
    }
    
    np.save(filename, save_dict)
    print(f"Saved as full 3DGS format to {filename}")
