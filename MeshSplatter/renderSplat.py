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
        outputs = pipeline.model.get_outputs_for_camera(camera)
    
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
