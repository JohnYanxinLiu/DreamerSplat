#!/usr/bin/env python3
"""
Render Gaussian Splat from camera transforms
Given a trained Gaussian Splat model and camera poses, render images
"""

import json
import numpy as np
import torch
from pathlib import Path
import argparse
from typing import Union, List
from PIL import Image
from gsplat.rendering import rasterization

def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat

def get_outputs(gsplat_model, camera):
    """Takes in a camera and returns a dictionary of outputs.

    Args:
        camera: The camera(s) for which output images are rendered. It should have
        all the needed information to compute the outputs.

    Returns:
        Outputs of model. (ie. rendered colors)
    """

    breakpoint()
    optimized_camera_to_world = camera.camera_to_worlds

    crop_ids = None

    if crop_ids is not None:
        opacities_crop = gsplat_model.opacities[crop_ids]
        means_crop = gsplat_model.means[crop_ids]
        features_dc_crop = gsplat_model.features_dc[crop_ids]
        features_rest_crop = gsplat_model.features_rest[crop_ids]
        scales_crop = gsplat_model.scales[crop_ids]
        quats_crop = gsplat_model.quats[crop_ids]
    else:
        opacities_crop = gsplat_model.opacities
        means_crop = gsplat_model.means
        features_dc_crop = gsplat_model.features_dc
        features_rest_crop = gsplat_model.features_rest
        scales_crop = gsplat_model.scales
        quats_crop = gsplat_model.quats

    colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

    viewmat = get_viewmat(optimized_camera_to_world)
    K = camera.get_intrinsics_matrices().cuda()
    W, H = int(camera.width.item()), int(camera.height.item())


    render_mode = "RGB"

    if gsplat_model.config.sh_degree > 0:
        sh_degree_to_use = min(gsplat_model.step // gsplat_model.config.sh_degree_interval, gsplat_model.config.sh_degree)
    else:
        colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
        sh_degree_to_use = None

    render, alpha, gsplat_model.info = rasterization(
        means=means_crop,
        quats=quats_crop,  # rasterization does normalization internally
        scales=torch.exp(scales_crop),
        opacities=torch.sigmoid(opacities_crop).squeeze(-1),
        colors=colors_crop,
        viewmats=viewmat,  # [1, 4, 4]
        Ks=K,  # [1, 3, 3]
        width=W,
        height=H,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        render_mode=render_mode,
        sh_degree=sh_degree_to_use,
        sparse_grad=False,
        absgrad=gsplat_model.strategy.absgrad,
        rasterize_mode=gsplat_model.config.rasterize_mode,
        # set some threshold to disregrad small gaussians for faster rendering.
        # radius_clip=3.0,
    )

    alpha = alpha[:, ...]

    background = gsplat_model._get_background_color()
    rgb = render[:, ..., :3] + (1 - alpha) * background
    rgb = torch.clamp(rgb, 0.0, 1.0)


    if render_mode == "RGB+ED":
        depth_im = render[:, ..., 3:4]
        depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
    else:
        depth_im = None
    
    if background.shape[0] == 3 and not gsplat_model.training:
        background = background.expand(H, W, 3)

    return {
        "rgb": rgb.squeeze(0),  # type: ignore
        "depth": depth_im,  # type: ignore
        "accumulation": alpha.squeeze(0),  # type: ignore
        "background": background,  # type: ignore
    }  # type: ignore

def render_splat_from_transform(
    config_path: Union[str, Path],
    camera_transform: np.ndarray,
    width: int = 960,
    height: int = 960,
    fx: float = 480.0,
    fy: float = 480.0,
    cx: float = 479.5,
    cy: float = 479.5,
    debug: bool = False,
) -> np.ndarray:
    """
    Render a Gaussian Splat from a camera transform matrix.
    
    Args:
        config_path: Path to the nerfstudio config.yml file
        camera_transform: 4x4 camera-to-world transformation matrix (OpenGL convention)
        width: Image width in pixels
        height: Image height in pixels
        fx: Focal length in x (pixels)
        fy: Focal length in y (pixels)
        cx: Principal point x coordinate
        cy: Principal point y coordinate
        debug: Print debug information about transforms
    
    Returns:
        RGB image as numpy array (H, W, 3) with values in [0, 255]
    """
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.cameras.cameras import Cameras, CameraType
    
    # Load the trained model
    config, pipeline, _, _ = eval_setup(Path(config_path))
    
    # Load the dataparser transform from the saved JSON file
    config_dir = Path(config_path).parent
    dataparser_transform_path = config_dir / "dataparser_transforms.json"
    
    if dataparser_transform_path.exists():
        with open(dataparser_transform_path, 'r') as f:
            transform_data = json.load(f)
        
        # Reconstruct the 4x4 transform matrix
        transform_3x4 = np.array(transform_data['transform'])
        scale = transform_data['scale']
        
        # Build full 4x4 matrix: first apply scale, then rotation+translation
        transform = np.eye(4)
        transform[:3, :] = transform_3x4
        
        # Apply scale to the rotation and translation parts
        transform[:3, :3] *= scale  # Scale rotation
        transform[:3, 3] *= scale   # Scale translation
        
        transform = torch.from_numpy(transform).float()
        
        if debug:
            print(f"Loaded dataparser transform from {dataparser_transform_path}")
            print(f"Scale: {scale}")
            print(f"Transform matrix:\n{transform}")
    else:
        transform = torch.eye(4)
        if debug:
            print(f"Warning: {dataparser_transform_path} not found, using identity transform")
    
    # Apply transform to the camera pose
    c2w = torch.from_numpy(camera_transform).float()
    c2w_transformed = transform @ c2w
    
    if debug:
        print(f"\nOriginal camera position: {c2w[:3, 3].numpy()}")
        print(f"Transformed camera position: {c2w_transformed[:3, 3].numpy()}")
        
        # Compare with a training camera
        train_cam = pipeline.datamanager.train_dataset.cameras[0]
        train_c2w = train_cam.camera_to_worlds
        print(f"First training camera position: {train_c2w[:3, 3].cpu().numpy()}")
    
    # Create camera object
    # Note: Nerfstudio expects (3, 4) transform, not (4, 4)
    camera = Cameras(
        camera_to_worlds=c2w_transformed[None, :3, :4],  # Shape: (1, 3, 4)
        fx=torch.tensor([fx]),
        fy=torch.tensor([fy]),
        cx=torch.tensor([cx]),
        cy=torch.tensor([cy]),
        width=torch.tensor([width]),
        height=torch.tensor([height]),
        camera_type=CameraType.PERSPECTIVE,
    ).to(pipeline.device)
    
    # Render from this camera
    with torch.no_grad():
        # outputs = pipeline.model.get_outputs_for_camera(camera)
        outputs = get_outputs(pipeline.model, camera)
    
    # Extract RGB image
    rgb = outputs["rgb"].cpu().numpy()  # (H, W, 3) in [0, 1]
    rgb = (rgb * 255).astype(np.uint8)  # Convert to [0, 255]
    
    return rgb


def render_from_transforms_json(
    config_path: Union[str, Path],
    transforms_path: Union[str, Path],
    output_dir: Union[str, Path],
    frame_indices: List[int] = None,
) -> None:
    """
    Render images for all (or selected) frames in a transforms.json file.
    
    Args:
        config_path: Path to the nerfstudio config.yml file
        transforms_path: Path to transforms.json with camera poses
        output_dir: Directory to save rendered images
        frame_indices: Optional list of frame indices to render. If None, renders all.
    """
    # Load transforms.json
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    
    # Extract camera intrinsics
    width = data.get('w', 960)
    height = data.get('h', 960)
    fx = data.get('fl_x', 480.0)
    fy = data.get('fl_y', 480.0)
    cx = data.get('cx', 479.5)
    cy = data.get('cy', 479.5)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get frames to render
    frames = data['frames']
    if frame_indices is None:
        frame_indices = range(len(frames))
    
    print(f"Rendering {len(frame_indices)} frames from Gaussian Splat...")
    print(f"Resolution: {width}x{height}, fx={fx:.2f}, fy={fy:.2f}")
    print(f"Output directory: {output_dir}")
    
    # Render each frame
    for idx in frame_indices:
        frame = frames[idx]
        c2w = np.array(frame['transform_matrix'])
        
        # Render image (pass debug flag only for first frame)
        debug = (idx == frame_indices[0]) if hasattr(render_splat_from_transform, '__defaults__') else False
        rgb = render_splat_from_transform(
            config_path, c2w, width, height, fx, fy, cx, cy, debug=debug
        )
        
        # Save image
        output_path = output_dir / f"frame_{idx:06d}.png"
        Image.fromarray(rgb).save(output_path)
        
        print(f"  Rendered frame {idx+1}/{len(frame_indices)}: {output_path.name}")
    
    print(f"\n✅ Rendering complete! Saved {len(frame_indices)} images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Render Gaussian Splat from camera transforms"
    )
    parser.add_argument("config", type=str, 
                       help="Path to nerfstudio config.yml (e.g., outputs/.../config.yml)")
    parser.add_argument("--transforms", type=str, required=True,
                       help="Path to transforms.json with camera poses")
    parser.add_argument("--output", type=str, default="splat_renders",
                       help="Output directory for rendered images")
    parser.add_argument("--frames", type=str, default=None,
                       help="Comma-separated frame indices to render (e.g., '0,5,10'). Default: all frames")
    parser.add_argument("--debug", action="store_true",
                       help="Print debug information about coordinate transforms")
    
    args = parser.parse_args()
    
    # Parse frame indices if provided
    frame_indices = None
    if args.frames:
        frame_indices = [int(x.strip()) for x in args.frames.split(',')]
    
    # Render from transforms
    render_from_transforms_json(
        args.config,
        args.transforms,
        args.output,
        frame_indices
    )


if __name__ == "__main__":
    main()
