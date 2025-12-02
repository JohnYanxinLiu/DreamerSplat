import torch
import torch.nn as nn
import torch.nn.functional as F
from mujoco_env import DroneXYZEnv
from gauss_dreamer import DynamicsModel as GaussianSplatRenderer
import numpy as np
from gauss_dreamer_pt2 import seed_everything
import imageio
import os

###############################################################################
#   PIXEL ENCODER    (shared across actor & critic)
#   Resolution-agnostic due to AdaptiveAvgPool2d((2,2))
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
        self.pool = nn.AdaptiveAvgPool2d((2, 2))  # makes it size-agnostic
        self.fc = nn.Linear(256 * 2 * 2, feature_dim)

    def forward(self, image):
        if image.dim() == 3:
            image = image.unsqueeze(0)  # add batch dim
        if image.shape[1] != 3:
            # rearrange to (B,C,H,W)
            image = image.permute(0, 3, 1, 2)
            
        x = self.conv(image)
        x = self.pool(x)                          # → (B,256,2,2)
        x = x.reshape(image.size(0), -1)
        return self.fc(x)


###############################################################################
#   ACTOR: image → action distribution
###############################################################################

class Actor(nn.Module):
    def __init__(self, encoder, feature_dim, action_dim):
        super().__init__()
        self.encoder = encoder
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
        )
        self.mean = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.std = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, image):
        z = self.encoder(image)
        h = self.net(z)
        mean = self.mean(h)
        std  = torch.exp(self.std(h)) + 1e-4
        return torch.distributions.Normal(mean, std)


###############################################################################
#   CRITIC: image → value
###############################################################################

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



import torch
import torch.nn.functional as F

###############################################################################
#   FIXED IMAGINATION: Include initial state
###############################################################################

def imagine_rollout(image0, c2w0, actor, dynamics, horizon):
    """
    Returns:
        images: [horizon+1, B, C, H, W] - includes initial image
        actions: [horizon, B, action_dim]
        rewards: [horizon, B, 1]
        c2ws: [horizon+1, B, 4, 4]
    """
    images = []  # Include initial state
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
        torch.stack(images,  dim=0),  # [horizon+1, ...]
        torch.stack(actions, dim=0),  # [horizon, ...]
        torch.stack(rewards, dim=0),  # [horizon, ...]
        torch.stack(c2ws,    dim=0),  # [horizon+1, ...]
    )


###############################################################################
#   1-STEP TD TARGETS
###############################################################################

def compute_td_targets(rewards, values, discount=0.99):
    """
    Compute 1-step TD targets and advantages.
    
    Args:
        rewards: [T, B, 1]
        values: [T+1, B, 1] - includes bootstrap value
    
    Returns:
        td_targets: [T, B, 1] - r_t + γV(s_{t+1})
        advantages: [T, B, 1] - r_t + γV(s_{t+1}) - V(s_t)
    """
    # TD target: r_t + γV(s_{t+1})
    td_targets = rewards + discount * values[1:]
    
    # Advantage: TD_target - V(s_t) = TD error
    advantages = td_targets - values[:-1]
    
    return td_targets, advantages


###############################################################################
#   FIXED ACTOR & CRITIC LOSSES
###############################################################################

def actor_loss(images, actions, advantages, actor):
    """
    images: [T+1, B, C, H, W]
    actions: [T, B, action_dim]
    advantages: [T, B, 1]
    """
    # Evaluate actor on first T states (exclude final state)
    dists = actor(images[:-1])  # [T, B, action_dim]
    log_probs = dists.log_prob(actions).sum(-1, keepdim=True)  # [T, B, 1]
    return -(log_probs * advantages.detach()).mean()


def critic_loss(images, td_targets, critic):
    """
    images: [T+1, B, C, H, W]
    td_targets: [T, B, 1]
    """
    # Evaluate critic on first T states
    values = critic(images[:-1])  # [T, B, 1]
    return F.mse_loss(values, td_targets.detach())


###############################################################################
#   FIXED TRAIN STEP
###############################################################################

def train_step(
    real_image, real_c2w,
    dynamics,
    actor, critic,
    optim_actor, optim_critic,
    horizon=15,
):
    """
    Fixed version with 1-step TD and separate rollouts.
    """
    # ===== CRITIC UPDATE =====
    # Do a rollout with detached actor to avoid graph issues
    with torch.no_grad():
        images_critic, actions_critic, rewards_critic, c2ws_critic = imagine_rollout(
            real_image, real_c2w, actor, dynamics, horizon
        )
    
    # Compute values for all states (including bootstrap)
    values = critic(images_critic)  # [horizon+1, B, 1]
    
    # Compute TD targets
    td_targets, advantages = compute_td_targets(rewards_critic, values)  # [horizon, B, 1]
    
    # Critic loss
    c_loss = critic_loss(images_critic, td_targets, critic)
    optim_critic.zero_grad()
    c_loss.backward()
    optim_critic.step()

    # ===== ACTOR UPDATE =====
    # Fresh rollout for actor (needed because we need gradients through actor)
    images_actor, actions_actor, rewards_actor, c2ws_actor = imagine_rollout(
        real_image, real_c2w, actor, dynamics, horizon
    )
    
    # Use updated critic to compute advantages
    with torch.no_grad():
        values = critic(images_actor)  # [horizon+1, B, 1]
        td_targets, advantages = compute_td_targets(rewards_actor, values)  # [horizon, B, 1]
    
    # Actor loss
    a_loss = actor_loss(images_actor, actions_actor, advantages, actor)
    optim_actor.zero_grad()
    a_loss.backward()
    optim_actor.step()

    return {"actor_loss": a_loss.item(), "critic_loss": c_loss.item()}



###############################################################################
#   TRAIN LOOP
###############################################################################

def train_dreamer(
    env,
    actor,
    critic,
    dynamics,
    optim_actor,
    optim_critic,
    num_steps=100000,
    imagination_horizon=3,
    print_every=100,
    reward_alpha=0.01,
):
    device = next(actor.parameters()).device

    obs, info = env.reset()

    real_image = torch.tensor(
        obs["image"] / 255.0, dtype=torch.float32, device=device
    ).permute(2,0,1).unsqueeze(0)

    real_c2w = torch.tensor(
        obs["cam_c2w"], dtype=torch.float32, device=device
    ).unsqueeze(0)

    episode_reward = 0.0
    running_reward = 0.0

    for step in range(1, num_steps + 1):

        stats = train_step(
            real_image,
            real_c2w,
            dynamics,
            actor,
            critic,
            optim_actor,
            optim_critic,
            horizon=imagination_horizon,
        )

        # real env step
        with torch.no_grad():
            action = actor(real_image).mean
        action_np = action.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action_np)

        episode_reward += reward
        running_reward = (1 - reward_alpha) * running_reward + reward_alpha * reward

        real_image = torch.tensor(
            next_obs["image"] / 255.0, dtype=torch.float32, device=device
        ).permute(2,0,1).unsqueeze(0)

        real_c2w = torch.tensor(
            next_obs["cam_c2w"], dtype=torch.float32, device=device
        ).unsqueeze(0)

        if terminated or truncated:
            last_episode_reward = episode_reward
            episode_reward = 0.0
            obs, info = env.reset()
            real_image = torch.tensor(
                obs["image"] / 255.0, dtype=torch.float32, device=device
            ).permute(2,0,1).unsqueeze(0)
            real_c2w = torch.tensor(
                obs["cam_c2w"], dtype=torch.float32, device=device
            ).unsqueeze(0)
        else:
            last_episode_reward = None

        if step % print_every == 0:
            msg = (
                f"[step {step}] "
                f"actor_loss={stats['actor_loss']:.4f} "
                f"critic_loss={stats['critic_loss']:.4f} "
                f"avg_reward={running_reward:.3f}"
            )
            if last_episode_reward is not None:
                msg += f"  episode_reward={last_episode_reward:.3f}"
            print(msg)


###############################################################################
#   DYNAMICS MODEL WRAPPER
#   **Uses YOUR EXACT MLP models from checkpoint**
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

        # predict delta translation
        delta = self.c2w_change_mlp(action)  # (B,3)

        next_c2w = c2w.clone()
        next_c2w[:, :3, 3] += delta

        reward = self.reward_mlp(action)  # (B,1)

        with torch.no_grad():
            # render next image
            rgb = self.renderer.get_splat_render(
                next_c2w,
                self.intrinsics,
            )["rgb"]

        # return exactly: next_image, reward, next_c2w
        return rgb, reward, next_c2w


###############################################################################
#   MAIN
###############################################################################

if __name__ == "__main__":
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    W = 300
    H = 300

    env = DroneXYZEnv(
        xml_path="office_0_quantized_16/merged_env.xml",
        image_width=W,
        image_height=H,
        max_delta=0.3,
    )

    start_obs, info = env.reset()

    # Load ENTIRE objects (your choice)
    world_dynamics = torch.load(
        "gauss_dreamer_pt2_dynamics_model.pth",
        map_location=device
    )

    # These are FULL MLP MODULES, as you insisted
    c2w_change_mlp = world_dynamics["predict_change_in_camera_mlp"].to(device)
    reward_mlp     = world_dynamics["predict_reward_mlp"].to(device)

    renderer = GaussianSplatRenderer(
        img_size=W,
        img_W=W,
        img_H=H,
        device=device,
    )

    dynamics = DynamicsModelWrapper(
        renderer=renderer,
        c2w_change_mlp=c2w_change_mlp,
        reward_mlp=reward_mlp,
        intrinsics=torch.from_numpy(start_obs["intrinsics"]),
    ).to(device)

    # Build policy + value networks
    encoder_actor = PixelEncoder(in_channels=3, feature_dim=256).to(device)
    encoder_critic = PixelEncoder(in_channels=3, feature_dim=256).to(device)
    actor   = Actor(encoder_actor, feature_dim=256, action_dim=3).to(device)
    critic  = Critic(encoder_critic, feature_dim=256).to(device)

    optim_actor  = torch.optim.Adam(actor.parameters(),  lr=1e-4)
    optim_critic = torch.optim.Adam(critic.parameters(), lr=1e-4)

    # Run Dreamer training
    train_dreamer(
        env=env,
        actor=actor,
        critic=critic,
        dynamics=dynamics,
        optim_actor=optim_actor,
        optim_critic=optim_critic,
        num_steps=20000,
        imagination_horizon=15,
        print_every=50,
    )
