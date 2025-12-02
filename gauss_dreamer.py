import torch
import torch.nn as nn
import torch.nn.functional as F
from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
import json
import numpy as np
from gsplat.rendering import rasterization
from mujoco_env import DroneXYZEnv


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

class CNNEncoderWithProjection(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        latent_dim=256,
        proj_dim=64,
        n_blocks=4,
        hidden_dim=256,
    ):
        super().__init__()

        # -------------------------
        # CNN Backbone
        # -------------------------
        convs = []
        in_ch = in_channels
        out_ch = base_channels

        for _ in range(n_blocks):
            convs.append(
                nn.Conv2d(
                    in_ch, out_ch,
                    kernel_size=4, stride=2, padding=1, bias=False
                )
            )
            convs.append(nn.BatchNorm2d(out_ch))
            convs.append(nn.LeakyReLU(0.2, inplace=True))

            in_ch = out_ch
            out_ch *= 2

        self.conv = nn.Sequential(*convs)

        # -------------------------
        # FC after CNN (dynamic in_features)
        # -------------------------
        # This will infer `in_features` on the first forward pass
        self.fc = nn.LazyLinear(latent_dim)

        # -------------------------
        # MLP Projection Head
        # -------------------------
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        h = self.conv(x)            # (B, C, H', W')
        h = h.flatten(start_dim=1)  # (B, C * H' * W')
        latent = self.fc(h)         # LazyLinear infers in_features here
        proj = self.proj(latent)
        return proj



class DynamicsModel(nn.Module):
    """
    Simple MLP dynamics model that predicts next latent state given current state and action.

    Input:  (B, latent_dim + action_dim)
    Output: (B, latent_dim)
    """

    def __init__(
        self,
        img_size=256,
        action_dim=3,
        n_layers=3,
        proj_dim=64,
        gaussian_splat_configs_path="outputs/office_0_dataset_mujoco/splatfacto/2025-11-26_165611/config.yml",
        img_W = 900,
        img_H = 900,
        device='cuda',
    ):
        super().__init__()

        self.img_encoder = CNNEncoderWithProjection(
            in_channels=3,
            base_channels=64,
            latent_dim=256,
            proj_dim=proj_dim,
            n_blocks=4,
            hidden_dim=256,
        )

        self.device = device
        self.to(self.device)
        self.img_W = img_W
        self.img_H = img_H

        # MLP for dynamics
        layers = []
        in_dim = proj_dim + action_dim
        hidden_dim = 256

        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        # layers.append(nn.Linear(hidden_dim, 3)) # output hte change in position
        last = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)

        # self.change_in_camera_Mlp = nn.Sequential(*layers)
        self.change_in_camera_Mlp = nn.Sequential(*layers).to(self.device)


        # Load Gaussian Splatting Nerfstudio model
        self.gsplat_config_path = gaussian_splat_configs_path
        config, self.gsplat_pipeline, _, _ = eval_setup(Path(self.gsplat_config_path))
        config_dir = Path(self.gsplat_config_path).parent
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
        self.dataparser_transform = transform


    def get_splat_render(self, 
                         cam_c2w,
                          camera_intrinsics=None # assumes that this is in mujoco coordinates
                         ):
        
        c2w_transformed = self.dataparser_transform.to(self.device) @ cam_c2w.to(self.device)
        opacities_crop = self.gsplat_pipeline.model.opacities.to(self.device)
        means_crop = self.gsplat_pipeline.model.means.to(self.device)
        features_dc_crop = self.gsplat_pipeline.model.features_dc.to(self.device)
        features_rest_crop = self.gsplat_pipeline.model.features_rest.to(self.device)
        scales_crop = self.gsplat_pipeline.model.scales.to(self.device)
        quats_crop = self.gsplat_pipeline.model.quats.to(self.device)

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        viewmat = get_viewmat(c2w_transformed)
        K = camera_intrinsics.unsqueeze(0).to(self.device)  # [1, 3, 3]
        W, H = int(self.img_W), int(self.img_H)
        gsplat_model = self.gsplat_pipeline.model.to(self.device)

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

        return {
            "rgb": rgb[0],  # (H,W,3)
            "alpha": alpha[0],  # (H,W)
        }
        



    def forward(self, img, action, current_c2w, intrinsics):
        self.camera_intrinsics = intrinsics.to(self.device)

        img = img.to(self.device)
        action = action.to(self.device)
        current_c2w = current_c2w.to(self.device)

        latent = self.img_encoder(img)

        latent = torch.zeros((latent.shape[0], 64), device=latent.device)  # ablation --- IGNORE ---

        x = torch.cat([latent, action], dim=-1)

        delta_pos = self.change_in_camera_Mlp(x)

        # print(delta_pos)

        new_c2w = current_c2w.clone()
        new_c2w[:, :3, 3] += delta_pos

        rendered = self.get_splat_render(new_c2w)
        return rendered, new_c2w




if __name__ == "__main__":
    # simple test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    W = 300
    H = 300


    model = DynamicsModel(
        img_W = W,
        img_H = H,
        device=device
    )
    
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision

    # c2w = torch.nn.Parameter(c2w.to(device))

    target_rgb = torchvision.io.read_image("office_0_dataset_mujoco/images/frame_000001.png").float() / 255.0
    target_rgb = target_rgb.permute(1, 2, 0).to(device)

    import matplotlib.pyplot as plt


    random_trajectories_dataset = []
    num_timesteps = 10**3
    eps_length=10
    env = DroneXYZEnv(
        xml_path="office_0_quantized_16/merged_env.xml",
        image_width=W,
        image_height=H,
        max_delta = 0.1
    )

    import tqdm
    
    for _ in tqdm.tqdm(range(num_timesteps)):
        if _ % eps_length == 0:
            prev_obs, _ = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)


        random_trajectories_dataset.append({
            "obs": prev_obs,
            "action": action,
            "next_obs": obs,
        })

        prev_obs = obs

    # training
    num_trainsteps = 10**5
    batch_size = 200

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    past_losses = []
    cam_pos_differences = []

    for _ in range(num_trainsteps):
        random_timestep = random_trajectories_dataset[np.random.randint(0, len(random_trajectories_dataset))]

        obs = random_timestep["obs"]
        action = random_timestep["action"]
        next_obs = random_timestep["next_obs"]

        curr_img = torch.from_numpy(obs['image']).float() / 255  # H x W x 3
        next_img = torch.from_numpy(next_obs['image']).float() / 255  # H x W x 3
        curr_cam_c2w = torch.from_numpy(obs['cam_c2w']).float()  # 4 x 4

        intrinsics = obs['intrinsics']

        rendered, new_c2w = model(
            img=curr_img.permute(2, 0, 1)[None, ...].to(device),
            action=torch.from_numpy(action).float()[None, :].to(device),
            current_c2w=curr_cam_c2w[None, ...].to(device),
            intrinsics=torch.from_numpy(intrinsics).float().to(device),
        )

        rendered_rgb = rendered['rgb']  # H x W x 3

        rendered_rgb_ground_truth = model.get_splat_render(
            torch.from_numpy(next_obs['cam_c2w']).float()[None, ...].to(device)
        )['rgb']

        # plt.imshow(rendered_rgb.cpu().detach().numpy())
        # plt.show()

        loss = F.mse_loss(rendered_rgb, next_img.to(device))

        past_losses.append(loss.item())
        cam_pos_differences.append(torch.norm(new_c2w[0, :3, 3] - torch.from_numpy(next_obs['cam_c2w']).float().to(device)[:3, 3]).item())
        if _ % 100 == 0:
            
            print(f"Average Loss over last 100 steps: {sum(past_losses[-100:]) / 100}")
            print(f"Average Cam Pos Difference over last 100 steps: {sum(cam_pos_differences[-100:]) / 100}")

            plt.imshow(
                torch.cat([rendered_rgb.cpu().detach(), next_img.cpu(), rendered_rgb_ground_truth.cpu().detach()], dim=1).numpy()
            )
            plt.show()

        loss.backward()

        if _ % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        

