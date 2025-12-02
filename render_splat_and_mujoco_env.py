from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType

import argparse
import json
import torch
import numpy as np
import cv2
from pathlib import Path

from renderSplat_utils import get_outputs
from mujoco_env import DroneXYZEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Gaussian Splat from MuJoCo camera")
    parser.add_argument(
        "--config_path",
        type=str,
        default="outputs/office_0_dataset_mujoco/splatfacto/2025-11-26_165611/config.yml"
    )
    args = parser.parse_args()

    ###########################################################################
    # Load Nerfstudio Model + Dataparser Transform
    ###########################################################################
    config, pipeline, _, _ = eval_setup(Path(args.config_path))
    config_dir = Path(args.config_path).parent
    dataparser_transform_path = config_dir / "dataparser_transforms.json"
    with open(dataparser_transform_path, "r") as f:
        transform_data = json.load(f)
    # 3x4 transform and scale
    transform_3x4 = np.array(transform_data["transform"])
    scale = transform_data["scale"]
    # Build full 4x4 matrix
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :4] = transform_3x4
    # Apply scale
    transform[:3, :4] *= scale
    transform = torch.from_numpy(transform).float()


    ###########################################################################
    # Create MuJoCo drone environment
    ###########################################################################
    env = DroneXYZEnv(xml_path="office_0_quantized_16/merged_env.xml", image_width=300, image_height=300)
    obs, _ = env.reset()

    ###########################################################################
    # Run environment + render through Nerfstudio model
    ###########################################################################
    for i in range(100):
        action = [0.0, 0.1, 0.0]
        obs, reward, terminated, truncated, info = env.step(action)

        # ----------------------------------------------------------------------
        # 1. GET CAMERA POSE (OpenGL C2W from env)
        # ----------------------------------------------------------------------
        cam_c2w = obs["cam_c2w"]               # (4,4)
        cam_c2w = torch.from_numpy(cam_c2w).float()

        # ----------------------------------------------------------------------
        # 2. APPLY NERF DATAPARSER TRANSFORM
        # ----------------------------------------------------------------------
        # Nerfstudio expects world coordinates matching training
        final_c2w = transform @ cam_c2w        # (4,4)

        # ----------------------------------------------------------------------
        # 3. BUILD CAMERA INTRINSICS
        # ----------------------------------------------------------------------
        fx = obs["fx"]
        fy = obs["fy"]
        cx = obs["cx"]
        cy = obs["cy"]

        intrinsics = torch.tensor(
            [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]],
            dtype=torch.float32
        )[None, :, :]                           # (1,3,3)

        # ----------------------------------------------------------------------
        # 4. BUILD CAMERA OBJECT FOR SPLAT RENDERING
        # ----------------------------------------------------------------------
        camera = Cameras(
            camera_to_worlds=final_c2w[None, :3, :4].to("cuda"),  # (1,3,4)
            fx=torch.tensor([fx]),
            fy=torch.tensor([fy]),
            cx=torch.tensor([cx]),
            cy=torch.tensor([cy]),
            width=torch.tensor([int(env.W)]),
            height=torch.tensor([int(env.H)]),
            camera_type=CameraType.PERSPECTIVE,
        )

        # ----------------------------------------------------------------------
        # 5. RENDER SPLAT USING THE LOADED MODEL
        # ----------------------------------------------------------------------
        with torch.no_grad():
            outputs = get_outputs(pipeline.model, camera)

        rendered_rgb = outputs["rgb"].cpu().numpy()         # (H,W,3) in [0,1]
        rendered_rgb_image = (rendered_rgb * 255).astype(np.uint8)

        mujoco_rgb = obs["image"]                              # (H,W,3) in [0,255]
        mujoco_rgb = mujoco_rgb.astype(np.uint8)

        # ----------------------------------------------------------------------
        # 6. SAVE FRAME
        # ----------------------------------------------------------------------
        cv2.imwrite(
            f"rendered_frame_{i:03d}.png",
            cv2.cvtColor(
                np.concatenate([rendered_rgb_image, mujoco_rgb], axis=1), cv2.COLOR_RGB2BGR)
        )



        print(f"[OK] Saved frame {i:03d}.png")
