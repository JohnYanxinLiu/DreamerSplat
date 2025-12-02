import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import cv2
import torch
import matplotlib.pyplot as plt

from plyfile import PlyData
from gsplat.rendering import rasterization


###############################################################################
#                         DRONE MUJOCO ENVIRONMENT
###############################################################################
###############################################################################
#                   ADD INTRINSICS + C2W TO DRONEXYZENV
###############################################################################

class DroneXYZEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, xml_path,
                 image_width=128,
                 image_height=128,
                 max_delta=0.3,
                 full_ambient=True,
                 no_z_movement=True,
                 horizon=3):

        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.horizon = horizon
        self.horizon_steps = 0

        if full_ambient:
            print("[INFO] Applying full ambient lighting...")
            self.model.vis.headlight.ambient[:]  = [1,1,1]
            self.model.vis.headlight.diffuse[:]  = [0,0,0]
            self.model.vis.headlight.specular[:] = [0,0,0]

        self.renderer = mujoco.Renderer(self.model, width=image_width, height=image_height)

        self.W = image_width
        self.H = image_height
        self.qpos0 = self.data.qpos.copy()
        self.max_delta = max_delta
        self.no_z_movement = no_z_movement

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Add intrinsics + C2W to obs space
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(image_height, image_width, 3), dtype=np.uint8),
            "cam_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "cam_rot": spaces.Box(low=-np.inf, high=np.inf, shape=(3,3), dtype=np.float32),
            "cam_c2w": spaces.Box(low=-np.inf, high=np.inf, shape=(4,4), dtype=np.float32),
            "intrinsics": spaces.Box(low=-np.inf, high=np.inf, shape=(3,3), dtype=np.float32),
            "fx": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
            "fy": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
            "cx": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
            "cy": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
        })

        # Locate drone_cam
        self.cam_id = None
        for i in range(self.model.ncam):
            if self.model.camera(i).name == "drone_cam":
                self.cam_id = i
                break
        if self.cam_id is None:
            raise RuntimeError("Camera 'drone_cam' not found in XML.")


    ###########################################################################
    # --------------------------- CAMERA INTRINSICS ---------------------------
    #
    # MuJoCo provides:
    #    fovy      (vertical field-of-view in degrees)
    #    aspect    (width / height)
    #
    # From this we compute:
    #    fx, fy using pinhole model
    #    cx, cy at image center
    #
    ###########################################################################
    def _get_intrinsics(self):
        cam_id = self.cam_id

        # Vertical field-of-view in degrees (MuJoCo)
        fovy_deg = self.model.cam_fovy[cam_id]
        fovy = np.deg2rad(fovy_deg)

        # Compute aspect ratio from actual render size
        aspect = self.W / self.H

        # Focal lengths (OpenGL pinhole)
        fy = (self.H / 2.0) / np.tan(fovy / 2.0)
        fx = fy * aspect

        # Principal point
        cx = self.W / 2.0
        cy = self.H / 2.0

        K = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1]
        ], dtype=np.float32)

        return K, fx, fy, cx, cy



    ###########################################################################
    # -------------------------- CAMERA POSE (C2W) ----------------------------
    #
    # Returns OpenGL/MuJoCo-style camera-to-world matrix.
    #
    ###########################################################################
    def _get_camera_pose_world(self):
        cam_id = self.cam_id
        body_id = self.model.cam_bodyid[cam_id]

        body_pos = self.data.xpos[body_id].copy()
        body_rot = self.data.xmat[body_id].reshape(3,3).copy()

        cam_local_pos = self.model.cam_pos[cam_id].copy()
        cam_local_rot = self.model.cam_mat0[cam_id].reshape(3,3).copy()

        cam_world_pos = body_pos + body_rot @ cam_local_pos
        cam_world_rot = body_rot @ cam_local_rot

        cam_c2w = np.eye(4, dtype=np.float32)
        cam_c2w[:3, :3] = cam_world_rot
        cam_c2w[:3, 3]  = cam_world_pos

        return cam_world_pos, cam_world_rot, cam_c2w


    ###########################################################################
    # ------------------------------ OBS RETURN -------------------------------
    ###########################################################################
    def _get_obs(self):
        # Render RGB
        self.renderer.update_scene(self.data, camera=self.cam_id)
        rgb = self.renderer.render()

        cam_pos, cam_rot, cam_c2w = self._get_camera_pose_world()
        K, fx, fy, cx, cy = self._get_intrinsics()

        return {
            "image": rgb,
            "cam_pos": cam_pos.astype(np.float32),
            "cam_rot": cam_rot.astype(np.float32),
            "cam_c2w": cam_c2w.astype(np.float32),
            "intrinsics": K,
            "fx": np.float32(fx),
            "fy": np.float32(fy),
            "cx": np.float32(cx),
            "cy": np.float32(cy),
        }


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data.qpos[:] = self.qpos0
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.horizon_steps = 0
        return self._get_obs(), {}


    def step(self, action):


        old_cam_pos = self._get_obs()['cam_c2w'][:3, 3].copy()
        delta = np.clip(action, -1, 1) * self.max_delta
        if self.no_z_movement:
            delta[2] = 0
        self.data.qpos[:3] += delta
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()

        TARGET_POS_OF_CAMERA = np.array([1.4,1.4,1.05])

        new_distance = np.linalg.norm(obs['cam_c2w'][:3, 3] - TARGET_POS_OF_CAMERA)
        old_distance = np.linalg.norm(old_cam_pos - TARGET_POS_OF_CAMERA)
        reward = old_distance - new_distance  # Reward is the reduction in distance

        self.horizon_steps += 1
        if self.horizon_steps >= self.horizon:
            self.horizon_steps = 0
            done = True
        else:
            done = False
        
        return obs, reward, done, False, {}


    def render(self):
        return self._get_obs()["image"]

    def close(self):
        self.renderer.close()

###############################################################################
#                          CAMERA UTIL
###############################################################################

def gl_perspective(fovy_deg, aspect, near, far, device):
    fovy = torch.tensor(fovy_deg * np.pi / 180.0, device=device)
    f = 1.0 / torch.tan(fovy / 2)

    P = torch.zeros((4, 4), dtype=torch.float32, device=device)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (far + near) / (near - far)
    P[2, 3] = (2 * far * near) / (near - far)
    P[3, 2] = -1.0
    return P


###############################################################################
#                         G-SPLAT GAUSSIAN LOADER
###############################################################################

_xyz = None
_scales = None
_rots = None
_colors = None
_opacity = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_gaussians_once(ply_path, coord_transform=None):
    """
    Load Gaussians from a 3DGS-style PLY (Nerfstudio format).

    NOTE: For simplicity we only use the DC color term (f_dc_0..2)
    and ignore f_rest_* SH coefficients. This keeps colors RGB and
    makes gsplat output (H, W, 3).
    
    coord_transform: Optional 4x4 transformation matrix to convert between
                     coordinate systems (e.g., Nerfstudio -> MuJoCo)
    """
    global _xyz, _scales, _rots, _colors, _opacity

    if _xyz is not None:
        return

    print(f"[gsplat] Loading Gaussians from {ply_path} ...")
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data

    # Positions
    _xyz = torch.tensor(
        np.stack([v["x"], v["y"], v["z"]], axis=1),
        dtype=torch.float32, device=_device
    )
    
    # Apply coordinate transformation if provided
    if coord_transform is not None:
        print("[gsplat] Applying coordinate transformation...")
        ones = torch.ones((_xyz.shape[0], 1), device=_device)
        xyz_homogeneous = torch.cat([_xyz, ones], dim=1)  # (N, 4)
        xyz_transformed = (coord_transform @ xyz_homogeneous.T).T  # (N, 4)
        _xyz = xyz_transformed[:, :3]

    # Scales (exponential activation for Nerfstudio PLY)
    _scales = torch.exp(torch.tensor(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1),
        dtype=torch.float32, device=_device
    ))

    # Rotations (quaternions, normalized)
    _rots = torch.tensor(
        np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1),
        dtype=torch.float32, device=_device
    )
    _rots = _rots / _rots.norm(dim=-1, keepdim=True)

    # DC color term with proper SH->RGB conversion
    dc = torch.tensor(
        np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1),
        dtype=torch.float32, device=_device
    )
    
    # SH DC scaling constant
    SH_C0 = 0.28209479177387814
    
    # Convert SH DC → RGB with sigmoid activation
    _colors = torch.sigmoid(dc * SH_C0 + 0.5)

    # Opacity (sigmoid activation)
    _opacity = torch.sigmoid(torch.tensor(
        v["opacity"],
        dtype=torch.float32, device=_device
    ))

    print(f"[gsplat] Loaded {_xyz.shape[0]} Gaussians.")
    print(f"[gsplat] Position range: X[{_xyz[:, 0].min():.3f}, {_xyz[:, 0].max():.3f}], "
          f"Y[{_xyz[:, 1].min():.3f}, {_xyz[:, 1].max():.3f}], "
          f"Z[{_xyz[:, 2].min():.3f}, {_xyz[:, 2].max():.3f}]")
    print(f"[gsplat] Color range: [{_colors.min():.3f}, {_colors.max():.3f}]")
    print(f"[gsplat] Opacity range: [{_opacity.min():.3f}, {_opacity.max():.3f}]")
    print(f"[gsplat] Scale range: [{_scales.min():.3f}, {_scales.max():.3f}]")


###############################################################################
#                           G-SPLAT RENDER
###############################################################################

def render_gaussian_with_gsplat(view, proj, K, H, W,
                                ply_path="office_0_quantized_16/office_0_splat.ply",
                                coord_transform=None):
    """
    Render Gaussians with gsplat using:
      - view:  (4,4) world→camera
      - K:     (3,3) intrinsics
      - coord_transform: Optional transformation from Gaussian space to MuJoCo space
    Returns:
      - uint8 RGB image of shape (H, W, 3)
    """
    _load_gaussians_once(ply_path, coord_transform)

    # Add batch dimension
    view = view.to(_device)[None, ...]  # (1, 4, 4)
    K = K.to(_device)[None, ...]        # (1, 3, 3)

    rendered_image, rendered_alpha, info = rasterization(
        means=_xyz,
        quats=_rots,
        scales=_scales,
        opacities=_opacity,
        colors=_colors,           # (N, 3) → output (1, H, W, 3)
        viewmats=view,
        Ks=K,
        width=W,
        height=H,
        packed=False,
        absgrad=False
    )

    # rendered_image: [1, H, W, 3], linear RGB
    linear = rendered_image[0].detach().cpu().numpy()

    # Clamp to [0,1] for display
    linear = np.clip(linear, 0.0, 1.0)

    img = (linear * 255.0).astype(np.uint8)
    return img


###############################################################################
#                    COORDINATE SYSTEM TRANSFORMATIONS
###############################################################################

def get_coord_transform(from_system, to_system, device):
    """
    Get transformation matrix between coordinate systems.
    
    Common systems:
    - 'opengl': Right-handed, Y-up, -Z forward (Nerfstudio default)
    - 'opencv': Right-handed, Y-down, Z forward
    - 'mujoco': Right-handed, Z-up, X forward
    - 'blender': Right-handed, Z-up, -Y forward
    """
    transforms = {
        # OpenGL/Nerfstudio (Y-up) -> MuJoCo (Z-up)
        'opengl_to_mujoco': torch.tensor([
            [1,  0,  0, 0],
            [0,  0,  1, 0],
            [0, -1,  0, 0],
            [0,  0,  0, 1]
        ], dtype=torch.float32, device=device),
        
        # OpenCV (Y-down) -> MuJoCo (Z-up)
        'opencv_to_mujoco': torch.tensor([
            [1,  0,  0, 0],
            [0,  0, -1, 0],
            [0,  1,  0, 0],
            [0,  0,  0, 1]
        ], dtype=torch.float32, device=device),
        
        # Blender (Z-up) -> MuJoCo (Z-up, but different forward)
        'blender_to_mujoco': torch.tensor([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0,  1, 0],
            [0,  0,  0, 1]
        ], dtype=torch.float32, device=device),
        
        # Identity (no transformation)
        'none': torch.eye(4, dtype=torch.float32, device=device),
    }
    
    key = f"{from_system}_to_{to_system}"
    if key in transforms:
        return transforms[key]
    elif from_system == to_system or from_system == 'none' or to_system == 'none':
        return transforms['none']
    else:
        raise ValueError(f"Unknown coordinate transformation: {key}")


###############################################################################
#                               MAIN LOOP
###############################################################################

if __name__ == "__main__":
    env = DroneXYZEnv("office_0_quantized_16/merged_env.xml")
    obs, _ = env.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Coordinate system options
    # Set this to None to use Gaussians as-is (no transformation)
    GAUSSIAN_COORD_SYSTEM = None  # Try: None, 'opengl', 'opencv', 'blender'
    coord_transform = None
    if GAUSSIAN_COORD_SYSTEM is not None:
        coord_transform = get_coord_transform(GAUSSIAN_COORD_SYSTEM, 'mujoco', device)
        print(f"\n[INFO] Using coordinate transformation: {GAUSSIAN_COORD_SYSTEM} -> mujoco")
    else:
        print("\n[INFO] No coordinate transformation applied")
    
    print(f"[INFO] Camera initial position: {obs['cam_pos']}")
    print(f"[INFO] Camera initial rotation:\n{obs['cam_rot']}\n")
    
    # Try both: does gsplat expect c2w or w2c?
    USE_C2W = False  # Set to True to try camera-to-world instead

    for i in range(200):
        action = np.array([0.2, 0.0, 0.0])
        obs, reward, done, trunc, info = env.step(action)

        # 1. Get MuJoCo's camera-to-world transform directly
        cam_pos = obs["cam_pos"]  # Camera position in world frame
        cam_rot = obs["cam_rot"]  # Camera rotation (camera→world)

        # Build camera-to-world matrix
        c2w = torch.eye(4, dtype=torch.float32, device=device)
        c2w[:3, :3] = torch.tensor(cam_rot, dtype=torch.float32, device=device)
        c2w[:3, 3] = torch.tensor(cam_pos, dtype=torch.float32, device=device)

        # Test: does gsplat expect c2w or w2c?
        if USE_C2W:
            view = c2w  # Camera-to-world
        else:
            view = torch.linalg.inv(c2w)  # World-to-camera (w2c)

        # 2. Intrinsics from MuJoCo camera parameters
        cam_id = env.cam_id
        W = env.W
        H = env.H

        fovy_deg = env.model.cam_fovy[cam_id]
        fovy_rad = fovy_deg * np.pi / 180.0

        fy = (H / 2.0) / np.tan(fovy_rad / 2.0)
        fx = fy
        cx = W / 2.0
        cy = H / 2.0

        K = torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=device)

        # 3. Projection matrix (mostly unused by gsplat)
        aspect = W / H
        proj = gl_perspective(fovy_deg, aspect, 0.01, 1000.0, device=device)

        # Debug: print first frame info
        if i == 0:
            print("\n[DEBUG] First frame camera matrices:")
            print("Camera-to-world (c2w):")
            print(c2w)
            print("\nUsing matrix:", "c2w" if USE_C2W else "w2c (inverted)")
            print(view)
            print(f"\nIntrinsics (K):")
            print(K)

        # 4. gsplat rendering with coordinate transformation
        gsplat_img = render_gaussian_with_gsplat(view, proj, K, H, W, 
                                                  coord_transform=coord_transform)

        # 5. Visualize both images
        plt.clf()

        plt.subplot(1, 2, 1)
        plt.title("MuJoCo")
        plt.imshow(obs["image"])
        plt.axis('off')

        plt.subplot(1, 2, 2)
        title = f"gsplat ({'c2w' if USE_C2W else 'w2c'})"
        if GAUSSIAN_COORD_SYSTEM:
            title += f"\n({GAUSSIAN_COORD_SYSTEM}→mujoco)"
        plt.title(title)
        plt.imshow(gsplat_img)
        plt.axis('off')
        
        if i % 20 == 0:
            plt.suptitle(f"Frame {i} | Cam pos: [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]")

        plt.pause(0.001)

    env.close()
    cv2.destroyAllWindows()