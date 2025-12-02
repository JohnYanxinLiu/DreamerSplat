#!/usr/bin/env python3
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

def look_at(eye, target, up=np.array([0, 0, 1])):
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
    R = look_at(eye, target, up)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T

def spherical_to_cartesian(r, theta, phi):
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.cos(theta)
    z = r * np.sin(theta)
    return np.array([x, y, z])

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
    parser = argparse.ArgumentParser(description="Render MuJoCo to NeRF Studio format (no XML edits needed).")
    parser.add_argument("--xml", type=str, default="office_0_quantized/office_0.xml", help="Input MuJoCo XML file")
    parser.add_argument("--center", nargs=3, type=float, default=[0.88, 0.57, 0.08], help="Orbit center (x y z)")
    parser.add_argument("--radius", type=float, default=2.0, help="Camera distance from center")
    parser.add_argument("--n-views", type=int, default=100, help="Number of views to render")
    parser.add_argument("--width", type=int, default=800, help="Image width")
    parser.add_argument("--height", type=int, default=800, help="Image height")
    parser.add_argument("--fov", type=float, default=60.0, help="Vertical FOV in degrees")
    parser.add_argument("--output-dir", type=str, default="nerfstudio_dataset", help="Output directory")
    parser.add_argument("--theta-min", type=float, default=-0.5, help="Min sin(theta) for elevation")
    parser.add_argument("--theta-max", type=float, default=0.5, help="Max sin(theta)")
    parser.add_argument("--phi-start", type=float, default=0.0, help="Start azimuth (rad)")
    parser.add_argument("--phi-range", type=float, default=2 * np.pi, help="Azimuth sweep (rad)")
    args = parser.parse_args()

    # --- Setup output ---
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model (inject camera if needed) ---
    model, data, tmp_xml = ensure_camera_in_model(args.xml, args.fov)
    try:
        # Ensure offscreen buffer is large enough
        model.vis.global_.offwidth = max(args.width, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(args.height, model.vis.global_.offheight)

        # Set FOV (now safe — ncam >= 1)
        model.cam_fovy[0] = args.fov

        # Intrinsics for transforms.json
        fov_rad = np.deg2rad(args.fov)
        focal_y = args.height / (2 * np.tan(fov_rad / 2))
        focal_x = focal_y
        cx, cy = args.width / 2, args.height / 2

        center = np.array(args.center)
        phi_vals = np.linspace(args.phi_start, args.phi_start + args.phi_range, args.n_views, endpoint=False)
        sin_theta_vals = np.linspace(args.theta_min, args.theta_max, args.n_views)
        theta_vals = np.arcsin(sin_theta_vals)

        frames = []

        print(f"🎬 Rendering {args.n_views} views → {output_dir}")
        # ✅ Use context manager to avoid __del__ errors
        with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
            for i in tqdm(range(args.n_views), desc="Rendering"):
                phi = phi_vals[i]
                theta = theta_vals[i]

                eye = center + spherical_to_cartesian(args.radius, theta, phi)
                c2w = pose_from_eye_target_up(eye, center)

                # Free camera — no fovy assignment (handled by model.cam_fovy[0])
                cam = mujoco.MjvCamera()
                cam.lookat[:] = center
                cam.distance = args.radius
                cam.elevation = np.rad2deg(theta)
                cam.azimuth = np.rad2deg(phi)

                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                frame = renderer.render()  # RGB uint8

                img_path = images_dir / f"{i:04d}.png"
                cv2.imwrite(str(img_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                frames.append({
                    "file_path": f"./images/{i:04d}.png",
                    "transform_matrix": c2w.tolist()
                })

        # Save transforms.json
        transforms = {
            "camera_model": "OPENCV",
            "w": args.width,
            "h": args.height,
            "fl_x": float(focal_x),
            "fl_y": float(focal_y),
            "cx": float(cx),
            "cy": float(cy),
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "camera_angle_x": 2 * np.arctan(args.width / (2 * focal_x)),
            "camera_angle_y": 2 * np.arctan(args.height / (2 * focal_y)),
            "frames": frames
        }

        with open(output_dir / "transforms.json", "w") as f:
            json.dump(transforms, f, indent=2)

        print(f"\n✅ Success! Dataset saved to: {output_dir.absolute()}")
        print(f"\n🚀 Next: ns-train nerfacto --data {output_dir}")

    finally:
        # Clean up temp XML if created
        if tmp_xml and Path(tmp_xml).exists():
            try:
                os.unlink(tmp_xml)
            except:
                pass

if __name__ == "__main__":
    main()