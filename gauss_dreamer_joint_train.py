import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from mujoco_env import DroneXYZEnv
from gauss_dreamer import DynamicsModel


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


###############################################################################
# Pixel encoder / actor / critic (copied and lightly adapted from pt3)
###############################################################################

class PixelEncoder(nn.Module):
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2), nn.ELU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ELU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ELU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(256 * 2 * 2, feature_dim)

    def forward(self, image):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.shape[1] != 3:
            image = image.permute(0, 3, 1, 2)
        x = self.conv(image)
        x = self.pool(x)
        x = x.reshape(image.size(0), -1)
        return self.fc(x)


class Actor(nn.Module):
    def __init__(self, encoder, feature_dim, action_dim):
        super().__init__()
        self.encoder = encoder
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
        )
        self.mean = nn.Sequential(nn.Linear(256, action_dim), nn.Tanh())
        self.std = nn.Sequential(nn.Linear(256, action_dim), nn.Tanh())

    def forward(self, image):
        z = self.encoder(image)
        h = self.net(z)
        mean = self.mean(h)
        std = torch.exp(self.std(h)) + 1e-4
        return torch.distributions.Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, encoder, feature_dim):
        super().__init__()
        self.encoder = encoder
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 1),
        )

    def forward(self, image):
        z = self.encoder(image)
        return self.net(z)


###############################################################################
# Imagination + losses (copied from pt3)
###############################################################################

def imagine_rollout(image0, c2w0, actor, dynamics, horizon):
    images = []
    actions = []
    rewards = []
    c2ws = [c2w0]

    image = image0
    c2w = c2w0

    for _ in range(horizon):
        dist = actor(image)
        action = dist.rsample()
        actions.append(action)

        next_image, reward, next_c2w = dynamics(image, action, c2w)

        images.append(next_image)
        rewards.append(reward)
        c2ws.append(next_c2w)

        image = next_image
        c2w = next_c2w

    return (
        torch.stack(images, dim=0),
        torch.stack(actions, dim=0),
        torch.stack(rewards, dim=0),
        torch.stack(c2ws, dim=0),
    )


def compute_td_targets(rewards, values, discount=0.99):
    td_targets = rewards + discount * values[1:]
    advantages = td_targets - values[:-1]
    return td_targets, advantages


def actor_loss(images, actions, advantages, actor):
    dists = actor(images[:-1])
    log_probs = dists.log_prob(actions).sum(-1, keepdim=True)
    return -(log_probs * advantages.detach()).mean()


def critic_loss(images, td_targets, critic):
    values = critic(images[:-1])
    return F.mse_loss(values, td_targets.detach())


###############################################################################
# Dynamics wrapper that uses learned MLPs + renderer
###############################################################################

class DynamicsModelWrapper(nn.Module):
    def __init__(self, renderer, c2w_change_mlp, reward_mlp, intrinsics):
        super().__init__()
        self.renderer = renderer
        self.c2w_change_mlp = c2w_change_mlp
        self.reward_mlp = reward_mlp
        self.register_buffer("intrinsics", intrinsics.float())

    def forward(self, image, action, c2w):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if c2w.dim() == 2:
            c2w = c2w.unsqueeze(0)

        delta = self.c2w_change_mlp(action)
        next_c2w = c2w.clone()
        next_c2w[:, :3, 3] += delta

        reward = self.reward_mlp(action)

        with torch.no_grad():
            rgb = self.renderer.get_splat_render(next_c2w, self.intrinsics)["rgb"]

        return rgb, reward, next_c2w


###############################################################################
# Joint training loop
###############################################################################

def train_joint(
    env,
    dynamics_renderer,
    predict_change_in_camera_mlp,
    predict_reward_mlp,
    actor,
    critic,
    optim_dynamics,
    optim_actor,
    optim_critic,
    device,
    total_env_steps=2000,
    imagination_horizon=5,
    dynamics_updates_per_step=1,
):

    # wrap dynamics for imagination
    obs, info = env.reset()
    intrinsics = torch.from_numpy(obs["intrinsics"]).float().to(device)
    dynamics = DynamicsModelWrapper(
        renderer=dynamics_renderer,
        c2w_change_mlp=predict_change_in_camera_mlp,
        reward_mlp=predict_reward_mlp,
        intrinsics=intrinsics,
    ).to(device)

    # initialize real state
    obs, info = env.reset()
    real_image = torch.tensor(obs["image"] / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    real_c2w = torch.tensor(obs["cam_c2w"], dtype=torch.float32, device=device).unsqueeze(0)

    episode_reward = 0.0

    for step in trange(1, total_env_steps + 1):
        # select action from actor (no gradient)
        with torch.no_grad():
            action = actor(real_image).mean
        action_np = action.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        episode_reward += reward

        # --- Dynamics supervised update using the observed transition ---
        # prepare tensors
        action_t = torch.from_numpy(action_np).float().to(device).unsqueeze(0)
        target_c2w = torch.from_numpy(next_obs["cam_c2w"]).float().to(device).unsqueeze(0)
        target_reward = torch.tensor([[reward]], dtype=torch.float32, device=device)

        for _ in range(dynamics_updates_per_step):
            pred_delta = predict_change_in_camera_mlp(action_t)  # (1,3)
            pred_reward = predict_reward_mlp(action_t)  # (1,1)

            pred_c2w = real_c2w.clone()
            pred_c2w[:, :3, 3] = pred_c2w[:, :3, 3] + pred_delta

            loss_c2w = F.mse_loss(pred_c2w, target_c2w)
            loss_reward = F.mse_loss(pred_reward.squeeze(), target_reward.squeeze())
            loss_dyn = loss_c2w + 10.0 * loss_reward

            optim_dynamics.zero_grad()
            loss_dyn.backward()
            optim_dynamics.step()

        # --- Actor-Critic updates using imagination from current real state ---
        stats = train_step_single(
            real_image, real_c2w, dynamics, actor, critic, optim_actor, optim_critic, horizon=imagination_horizon
        )

        # move environment state forward
        real_image = torch.tensor(next_obs["image"] / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        real_c2w = torch.tensor(next_obs["cam_c2w"], dtype=torch.float32, device=device).unsqueeze(0)

        if terminated or truncated:
            print(f"Episode finished at step {step} reward={episode_reward:.3f}")
            obs, info = env.reset()
            real_image = torch.tensor(obs["image"] / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
            real_c2w = torch.tensor(obs["cam_c2w"], dtype=torch.float32, device=device).unsqueeze(0)
            episode_reward = 0.0

        if step % 100 == 0:
            print(f"[step {step}] dyn_loss={loss_dyn.item():.4f} actor_loss={stats['actor_loss']:.4f} critic_loss={stats['critic_loss']:.4f}")

    # return final objects
    return predict_change_in_camera_mlp, predict_reward_mlp, actor, critic


def train_step_single(
    real_image,
    real_c2w,
    dynamics,
    actor,
    critic,
    optim_actor,
    optim_critic,
    horizon=5,
):
    # Critic update (rollout with detached actor)
    with torch.no_grad():
        images_critic, actions_critic, rewards_critic, c2ws_critic = imagine_rollout(
            real_image, real_c2w, actor, dynamics, horizon
        )

    values = critic(images_critic)
    td_targets, advantages = compute_td_targets(rewards_critic, values)

    c_loss = critic_loss(images_critic, td_targets, critic)
    optim_critic.zero_grad()
    c_loss.backward()
    optim_critic.step()

    # Actor update (fresh rollout)
    images_actor, actions_actor, rewards_actor, c2ws_actor = imagine_rollout(
        real_image, real_c2w, actor, dynamics, horizon
    )
    with torch.no_grad():
        values = critic(images_actor)
        td_targets, advantages = compute_td_targets(rewards_actor, values)

    a_loss = actor_loss(images_actor, actions_actor, advantages, actor)
    optim_actor.zero_grad()
    a_loss.backward()
    optim_actor.step()

    return {"actor_loss": a_loss.item(), "critic_loss": c_loss.item()}


###############################################################################
# Main: assemble everything and run
###############################################################################

if __name__ == "__main__":
    seed_everything(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    W = 300
    H = 300

    env = DroneXYZEnv(
        xml_path="warehouse/merged_env.xml",
        image_width=W,
        image_height=H,
        max_delta=0.3,
        horizon=100,
    )

    # Dynamics MLPs (same architecture as pt2)
    predict_change_in_camera_mlp = nn.Sequential(
        nn.Linear(3, 128).to(device),
        nn.ReLU().to(device),
        nn.Linear(128, 3).to(device),
    ).to(device)

    predict_reward_mlp = nn.Sequential(
        nn.Linear(3, 128).to(device),
        nn.ReLU().to(device),
        nn.Linear(128, 1).to(device),
    ).to(device)

    # zero-initialize last layer of change predictor
    with torch.no_grad():
        last = predict_change_in_camera_mlp[-1]
        last.weight.zero_()
        last.bias.zero_()

    optim_dynamics = torch.optim.Adam(
        list(predict_change_in_camera_mlp.parameters()) + list(predict_reward_mlp.parameters()), lr=1e-5
    )

    # renderer (Gaussian splat renderer) reused for imagination
    renderer = DynamicsModel(img_W=W, img_H=H, device=device)

    # actor / critic
    encoder_actor = PixelEncoder(in_channels=3, feature_dim=256).to(device)
    encoder_critic = PixelEncoder(in_channels=3, feature_dim=256).to(device)
    actor = Actor(encoder_actor, feature_dim=256, action_dim=3).to(device)
    critic = Critic(encoder_critic, feature_dim=256).to(device)

    optim_actor = torch.optim.Adam(actor.parameters(), lr=2e-6)
    optim_critic = torch.optim.Adam(critic.parameters(), lr=5e-6)

    # run joint training
    predict_change_in_camera_mlp, predict_reward_mlp, actor, critic = train_joint(
        env=env,
        dynamics_renderer=renderer,
        predict_change_in_camera_mlp=predict_change_in_camera_mlp,
        predict_reward_mlp=predict_reward_mlp,
        actor=actor,
        critic=critic,
        optim_dynamics=optim_dynamics,
        optim_actor=optim_actor,
        optim_critic=optim_critic,
        device=device,
        total_env_steps=2000,
        imagination_horizon=8,
        dynamics_updates_per_step=1,
    )

    # save final checkpoint
    torch.save({
        "predict_change_in_camera_mlp": predict_change_in_camera_mlp,
        "predict_reward_mlp": predict_reward_mlp,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
    }, "gauss_dreamer_joint_checkpoint.pth")

    print("Saved gauss_dreamer_joint_checkpoint.pth")
