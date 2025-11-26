#!/usr/bin/env python3
"""
MuJoCo Renderer for NeRF datasets
Follows the structure of Replica-Dataset ReplicaSDK render.cpp
Renders images from camera poses in transforms.json (or generates default trajectory)
"""

import os
import json
import numpy as np
import mujoco
import cv2
from tqdm import tqdm
from pathlib import Path
import argparse
import tempfile
import xml.etree.ElementTree as ET



def load_camera_poses_from_transforms(trajectory_file):
    """
    Load camera poses from transforms.json file.
    Returns: (camera_poses, intrinsics_dict)
    camera_poses: list of 4x4 camera-to-world matrices
    intrinsics_dict: dict with w, h, fl_x, fl_y, cx, cy
    """
    with open(trajectory_file, 'r') as f:
        data = json.load(f)
    
    # Extract intrinsics
    intrinsics = {
        'w': data.get('w', 1280),
        'h': data.get('h', 960),
        'fl_x': data.get('fl_x', 640.0),
        'fl_y': data.get('fl_y', 640.0),
        'cx': data.get('cx', 639.5),
        'cy': data.get('cy', 479.5),
    }
    
    # Extract camera poses
    camera_poses = []
    frames = data.get('frames', [])
    
    for frame in frames:
        c2w = np.array(frame['transform_matrix'])
        camera_poses.append(c2w)
    
    print(f"Loaded {len(camera_poses)} camera poses from {trajectory_file}")
    print(f"Camera intrinsics: {intrinsics['w']}x{intrinsics['h']}, fx={intrinsics['fl_x']:.2f}, fy={intrinsics['fl_y']:.2f}")
    
    return camera_poses, intrinsics


def generate_default_trajectory(n_frames=100, center=np.array([0, 0, 0]), radius=2.0):
    """Generate a default circular trajectory if no transforms.json is provided."""
    camera_poses = []
    
    for i in range(n_frames):
        # Circular trajectory
        theta = 2 * np.pi * i / n_frames
        
        # Camera position
        eye = center + np.array([
            radius * np.cos(theta),
            radius * np.sin(theta),
            0.5  # slightly elevated
        ])
        
        # Look at center
        forward = center - eye
        forward /= np.linalg.norm(forward)
        
        # Right vector
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        
        # Recompute up
        up = np.cross(right, forward)
        
        # Build rotation matrix (camera coordinate system)
        R = np.column_stack([right, -up, -forward])
        
        # Camera-to-world transform
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = eye
        
        camera_poses.append(c2w)
    
    return camera_poses


def look_at(eye, target, up=np.array([0, 0, 1])):
    """Create rotation matrix for camera looking from eye to target."""
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        forward = np.array([0, 0, 1])
    else:
        forward /= norm
    right = np.cross(forward, up)
    norm = np.linalg.norm(right)
    if norm < 1e-8:
        right = np.array([1, 0, 0])
    else:
        right /= norm
    down = np.cross(right, forward)
    down /= np.linalg.norm(down)
    R = np.column_stack([right, down, forward])
    return R


def pose_from_eye_target_up(eye, target, up=np.array([0, 0, 1])):
    """Create camera-to-world matrix from eye, target, and up vector."""
    R = look_at(eye, target, up)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T


def spherical_to_cartesian(r, theta, phi):
    """Convert spherical coordinates to Cartesian."""
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.cos(theta)
    z = r * np.sin(theta)
    return np.array([x, y, z])


def c2w_to_mujoco_camera(c2w, lookat_distance=2.0):
    """
    Convert camera-to-world matrix to MuJoCo camera parameters.
    Uses the camera pose directly without coordinate transformation.
    
    Args:
        c2w: 4x4 camera-to-world matrix (OpenGL convention from transforms.json)
        lookat_distance: distance along forward direction to place lookat point
    
    Returns:
        (lookat, azimuth, elevation, distance) for MuJoCo camera
    """
    # Camera position
    cam_pos = c2w[:3, 3]
    
    # Camera forward direction (OpenGL: -Z axis)
    cam_forward = -c2w[:3, 2]
    cam_forward /= np.linalg.norm(cam_forward)  # Ensure normalized
    
    # Lookat point along the forward direction
    lookat = cam_pos + cam_forward * lookat_distance
    
    # Compute azimuth (rotation around Z axis)
    azimuth = np.rad2deg(np.arctan2(cam_forward[1], cam_forward[0]))
    
    # Compute elevation (angle from XY plane)
    forward_xy = np.sqrt(cam_forward[0]**2 + cam_forward[1]**2)
    if forward_xy > 1e-6:
        elevation = np.rad2deg(np.arctan2(cam_forward[2], forward_xy))
    else:
        elevation = 90.0 if cam_forward[2] > 0 else -90.0
    
    # Distance is from lookat point back to camera
    # MuJoCo computes: cam_pos = lookat - distance * direction(azimuth, elevation)
    # So distance should equal lookat_distance
    distance = lookat_distance
    
    return lookat, azimuth, elevation, distance


def mujoco_camera_to_c2w(lookat, azimuth, elevation, distance):
    """
    Convert MuJoCo camera parameters to camera-to-world matrix.
    This reconstructs what the actual camera matrix is after MuJoCo processes it.
    
    Args:
        lookat: 3D point the camera looks at
        azimuth: rotation around Z axis (degrees)
        elevation: angle from XY plane (degrees)
        distance: distance from lookat point
    
    Returns:
        4x4 camera-to-world matrix (OpenGL convention)
    """
    # Convert to radians
    az_rad = np.deg2rad(azimuth)
    el_rad = np.deg2rad(elevation)
    
    # MuJoCo camera position relative to lookat
    # The camera is positioned at distance from lookat, at given azimuth/elevation
    # In spherical coordinates: azimuth is XY angle, elevation is angle from XY plane
    cam_offset = np.array([
        distance * np.cos(el_rad) * np.cos(az_rad),
        distance * np.cos(el_rad) * np.sin(az_rad),
        distance * np.sin(el_rad)
    ])
    
    cam_pos = lookat - cam_offset
    
    # Forward direction points from camera to lookat
    forward = lookat - cam_pos
    forward /= np.linalg.norm(forward)
    
    # Up vector is world Z, but we need to make it perpendicular to forward
    world_up = np.array([0, 0, 1])
    
    # Right vector
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # Camera pointing straight up or down
        right = np.array([1, 0, 0])
    else:
        right /= right_norm
    
    # Recompute up to be perpendicular
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    
    # Build camera-to-world matrix (OpenGL convention: +X right, +Y up, -Z forward)
    c2w = np.eye(4)
    c2w[:3, 0] = right    # +X axis
    c2w[:3, 1] = up       # +Y axis
    c2w[:3, 2] = -forward # -Z axis (forward in OpenGL)
    c2w[:3, 3] = cam_pos  # camera position
    
    return c2w


def ensure_camera_in_model(xml_path, fov_deg):
    """Returns (model, data, temp_xml_path) — injects camera if needed."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    if model.ncam > 0:
        data = mujoco.MjData(model)
        return model, data, None

    print("⚠️ No cameras found in XML. Injecting a dummy camera for rendering...")
    # Parse and patch XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    
    cam = ET.SubElement(worldbody, "camera")
    cam.set("name", "auto_render")
    cam.set("pos", "0 0 1")
    cam.set("fovy", str(fov_deg))

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, dir=Path(xml_path).parent) as f:
        tree.write(f, encoding='unicode')
        tmp_path = f.name

    model = mujoco.MjModel.from_xml_path(tmp_path)
    data = mujoco.MjData(model)
    print(f"✅ Using model with injected camera (temp file: {Path(tmp_path).name})")
    return model, data, tmp_path

def main():
    parser = argparse.ArgumentParser(
        description="Render MuJoCo scene following Replica SDK render.cpp structure"
    )
    parser.add_argument("xml", type=str, help="Input MuJoCo XML file (e.g., office_0.xml)")
    parser.add_argument("--transforms", type=str, default=None, 
                       help="Optional: transforms.json with camera poses (like Replica)")
    parser.add_argument("--output", type=str, default=".", 
                       help="Output directory for images/ and transforms.json")
    parser.add_argument("--continue", dest="continue_mode", action="store_true",
                       help="Skip existing frames")
    
    # Default trajectory parameters (if no transforms.json)
    parser.add_argument("--n-frames", type=int, default=100, 
                       help="Number of frames (if no transforms.json)")
    parser.add_argument("--center", nargs=3, type=float, default=[0, 0, 0],
                       help="Orbit center for default trajectory")
    parser.add_argument("--radius", type=float, default=2.0,
                       help="Camera distance for default trajectory")
    
    # Camera parameters (will be overridden by transforms.json if provided)
    parser.add_argument("--width", type=int, default=960, help="Image width")
    parser.add_argument("--height", type=int, default=960, help="Image height")
    parser.add_argument("--fov", type=float, default=None, 
                       help="Vertical FOV in degrees (computed from fl_y if not specified)")
    parser.add_argument("--fov-scale", type=float, default=1.0,
                       help="Scale factor for FOV (use to adjust zoom if needed)")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Load camera poses from transforms.json if provided
    if args.transforms:
        camera_poses, intrinsics = load_camera_poses_from_transforms(args.transforms)
        width = intrinsics['w']
        height = intrinsics['h']
        
        # Use the same intrinsics as Replica SDK: fx = fy = width/2
        # Even though MuJoCo enforces aspect ratio in rendering,
        # we output the "canonical" intrinsics for compatibility
        fx = width / 2.0
        fy = width / 2.0  # Note: uses width, not height (matches Replica SDK)
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0
        
        # Compute FOV from Replica's focal length
        fov_y = 2 * np.rad2deg(np.arctan(height / (2 * fy)))
        if args.fov is not None:
            fov_y = args.fov  # Allow override
        
        # Apply FOV scale factor for zoom adjustment
        fov_y *= args.fov_scale
        
        print(f"Camera intrinsics (matching Replica SDK):")
        print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        print(f"  FOV: {fov_y:.4f}° (scale={args.fov_scale})")
        print(f"Note: MuJoCo enforces aspect ratio, so actual rendering may differ slightly")
        if args.fov_scale != 1.0:
            print(f"  Using FOV scale factor {args.fov_scale} to adjust zoom")
    else:
        # Generate default trajectory
        camera_poses = generate_default_trajectory(
            n_frames=args.n_frames,
            center=np.array(args.center),
            radius=args.radius
        )
        width = args.width
        height = args.height
        
        # Set FOV
        fov_y = args.fov if args.fov is not None else 60.0
        
        # Compute intrinsics from FOV
        fy = height / (2 * np.tan(np.deg2rad(fov_y) / 2))
        fx = fy  # Square pixels
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0
    
    num_frames = len(camera_poses)
    
    print(f"\n{'='*70}")
    print(f"MuJoCo Renderer (Replica SDK style)")
    print(f"{'='*70}")
    print(f"Scene: {args.xml}")
    print(f"Frames: {num_frames}")
    print(f"Resolution: {width}x{height}")
    print(f"FOV: {fov_y:.2f}°")
    print(f"Output: {output_dir.absolute()}")
    print(f"{'='*70}\n")
    
    # Load MuJoCo model
    model, data, tmp_xml = ensure_camera_in_model(args.xml, fov_y)
    
    try:
        # Configure offscreen rendering
        model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(height, model.vis.global_.offheight)
        
        # Set camera FOV
        model.cam_fovy[0] = fov_y
        
        # Prepare output frames for transforms.json
        output_frames = []
        
        # Render frames
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for i in tqdm(range(num_frames), desc="Rendering frames"):
                # Check if frame already exists in continue mode
                img_path = images_dir / f"frame_{i:06d}.png"
                if args.continue_mode and img_path.exists():
                    print(f"\rSkipping frame {i+1}/{num_frames} (already exists)... ", end='')
                    
                    # Still add to output transforms (use original pose)
                    output_frames.append({
                        "file_path": f"images/frame_{i:06d}.png",
                        "transform_matrix": camera_poses[i].tolist(),
                        "colmap_im_id": i + 1
                    })
                    continue
                
                # Get camera pose (use directly from transforms.json)
                c2w = camera_poses[i]
                
                # Extract camera position and orientation from c2w
                cam_pos = c2w[:3, 3]
                cam_rot = c2w[:3, :3]
                
                # Update the model camera position and orientation
                # MuJoCo cameras use position and quaternion
                model.cam_pos[0] = cam_pos
                
                # Convert rotation matrix to quaternion for MuJoCo
                # Using Shepperd's method for rotation matrix to quaternion
                R = cam_rot
                trace = np.trace(R)
                if trace > 0:
                    s = 0.5 / np.sqrt(trace + 1.0)
                    w = 0.25 / s
                    x = (R[2, 1] - R[1, 2]) * s
                    y = (R[0, 2] - R[2, 0]) * s
                    z = (R[1, 0] - R[0, 1]) * s
                else:
                    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                        w = (R[2, 1] - R[1, 2]) / s
                        x = 0.25 * s
                        y = (R[0, 1] + R[1, 0]) / s
                        z = (R[0, 2] + R[2, 0]) / s
                    elif R[1, 1] > R[2, 2]:
                        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                        w = (R[0, 2] - R[2, 0]) / s
                        x = (R[0, 1] + R[1, 0]) / s
                        y = 0.25 * s
                        z = (R[1, 2] + R[2, 1]) / s
                    else:
                        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                        w = (R[1, 0] - R[0, 1]) / s
                        x = (R[0, 2] + R[2, 0]) / s
                        y = (R[1, 2] + R[2, 1]) / s
                        z = 0.25 * s
                
                quat = np.array([w, x, y, z])
                quat /= np.linalg.norm(quat)  # Normalize
                model.cam_quat[0] = quat
                
                # Render using the model's camera (index 0)
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=0)  # Use camera index instead of MjvCamera
                frame = renderer.render()  # RGB uint8
                
                # Save image
                cv2.imwrite(str(img_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                # Add to output transforms (use original pose)
                output_frames.append({
                    "file_path": f"images/frame_{i:06d}.png",
                    "transform_matrix": c2w.tolist(),
                    "colmap_im_id": i + 1
                })
        
        # Write transforms.json
        transforms_output = {
            "w": int(width),
            "h": int(height),
            "fl_x": float(fx),
            "fl_y": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "camera_model": "OPENCV",
            "frames": output_frames
        }
        
        transforms_path = output_dir / "transforms.json"
        with open(transforms_path, 'w') as f:
            json.dump(transforms_output, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✅ Rendering complete!")
        print(f"Saved transforms.json to {transforms_path} with {num_frames} frames")
        print(f"Camera intrinsics in output (MuJoCo's actual values):")
        print(f"  Resolution: {int(width)}x{int(height)}")
        print(f"  Focal: fx={fx:.4f}, fy={fy:.4f}")
        print(f"  Principal point: cx={cx:.4f}, cy={cy:.4f}")
        print(f"  FOV: {fov_y:.2f}°")
        print(f"Images saved to {images_dir}/")
        print(f"{'='*70}\n")
        
    finally:
        # Clean up temp XML if created
        if tmp_xml and Path(tmp_xml).exists():
            try:
                os.unlink(tmp_xml)
            except:
                pass


if __name__ == "__main__":
    main()