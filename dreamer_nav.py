import os
import math
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib
import torchvision.utils as vutils
import datetime
from torch.utils.tensorboard import SummaryWriter
from mujoco_env import DroneXYZEnv  # same env as your current file


# -------------------------
# Utils
# -------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor_image_uint8(obs_img_uint8, device):
    # obs_img_uint8: (H, W, C) uint8
    img = torch.from_numpy(obs_img_uint8).to(device=device, dtype=torch.float32) / 255.0
    img = img.permute(2, 0, 1)  # C,H,W
    return img


def td_lambda(rewards, continues, values, lam, discount):
    # rewards:   [B, T, 1]
    # continues: [B, T, 1]
    # values:    [B, T+1, 1]
    B, T, _ = rewards.shape
    returns = torch.zeros_like(rewards)
    next_value = values[:, -1]  # [B,1]
    for t in reversed(range(T)):
        target = rewards[:, t] + continues[:, t] * (1 - lam) * values[:, t + 1]
        next_value = target + continues[:, t] * lam * next_value
        returns[:, t] = next_value
    return returns  # [B,T,1]


# -------------------------
# Replay Buffer (episodic, sequence sampling)
# -------------------------
class Episode:
    def __init__(self):
        self.obs: List[np.ndarray] = []        # each (H,W,C) uint8
        self.actions: List[np.ndarray] = []    # each (A,)
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def __len__(self):
        return len(self.rewards)


class ReplayBuffer:
    def __init__(self, capacity_episodes: int = 1000):
        self.capacity = capacity_episodes
        self.episodes: List[Episode] = []

    def start_episode(self):
        self.episodes.append(Episode())
        # Evict if over capacity
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    def add(self, obs_img_uint8: np.ndarray, action: np.ndarray, reward: float, done: bool):
        assert len(self.episodes) > 0, "Call start_episode() before add()"
        ep = self.episodes[-1]
        ep.obs.append(obs_img_uint8)
        ep.actions.append(action.astype(np.float32))
        ep.rewards.append(float(reward))
        ep.dones.append(bool(done))

    def sample(self, batch_size: int, seq_len: int, device: torch.device, resize_to=(64, 64)):
        # choose episodes with enough length
        eligible = [i for i, ep in enumerate(self.episodes) if len(ep) >= seq_len]
        assert len(eligible) > 0, "Not enough data to sample."
        H, W = resize_to
        obs_batch = torch.zeros((batch_size, seq_len, 3, H, W), dtype=torch.float32, device=device)
        actions_batch = []
        rewards_batch = []
        dones_batch = []
        for b in range(batch_size):
            ei = random.choice(eligible)
            ep = self.episodes[ei]
            start = random.randint(0, len(ep) - seq_len)
            end = start + seq_len
            # images
            imgs = []
            for t in range(start, end):
                img = ep.obs[t]  # (H0,W0,C)
                # simple bilinear resize via torch (convert to tensor then interpolate)
                img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                img_t = torch.nn.functional.interpolate(img_t, size=(H, W), mode="bilinear", align_corners=False)
                imgs.append(img_t.squeeze(0))
            obs_batch[b] = torch.stack(imgs, dim=0)
            # actions
            actions = np.stack(ep.actions[start:end], axis=0)  # (seq_len, A)
            rewards = np.array(ep.rewards[start:end], dtype=np.float32).reshape(seq_len, 1)
            dones = np.array(ep.dones[start:end], dtype=np.float32).reshape(seq_len, 1)
            actions_batch.append(torch.from_numpy(actions))
            rewards_batch.append(torch.from_numpy(rewards))
            dones_batch.append(torch.from_numpy(dones))
        actions_batch = torch.stack(actions_batch, dim=0).to(device)
        rewards_batch = torch.stack(rewards_batch, dim=0).to(device)
        dones_batch = torch.stack(dones_batch, dim=0).to(device)
        return obs_batch, actions_batch, rewards_batch, dones_batch


# -------------------------
# World Model (Dreamer v1-ish)
# -------------------------

class ConvEncoder(nn.Module):
    def __init__(self, depth=32, input_shape=(3, 64, 64), activation=nn.ReLU):
        super().__init__()
        self.depth = depth
        self.input_shape = input_shape
        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=depth * 1,
                kernel_size=4,
                stride=2,
                padding="valid",
            ),
            activation(),
            nn.Conv2d(
                in_channels=depth * 1,
                out_channels=depth * 2,
                kernel_size=4,
                stride=2,
                padding="valid",
            ),
            activation(),
            nn.Conv2d(
                in_channels=depth * 2,
                out_channels=depth * 4,
                kernel_size=4,
                stride=2,
                padding="valid",
            ),
            activation(),
            nn.Conv2d(
                in_channels=depth * 4,
                out_channels=depth * 8,
                kernel_size=4,
                stride=2,
                padding="valid",
            ),
            activation(),
        )
        # weight init identical
        def initialize_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
        self.conv_layer.apply(initialize_weights)

    def forward(self, x):
        # match models.py batching behavior
        batch_shape = x.shape[:-len(self.input_shape)]
        if not batch_shape:
            batch_shape = (1,)
        x = x.reshape(-1, *self.input_shape)
        out = self.conv_layer(x)
        return out.reshape(*batch_shape, -1)

class ConvDecoder(nn.Module):
    """Decode latent dynamic (observation model) identical to utils/models.py"""
    def __init__(self, stochastic_size, deterministic_size, depth=32, out_shape=(3, 64, 64), activation=nn.ReLU):
        super().__init__()
        self.out_shape = out_shape
        self.net = nn.Sequential(
            nn.Linear(deterministic_size + stochastic_size, depth * 32),
            nn.Unflatten(1, (depth * 32, 1)),
            nn.Unflatten(2, (1, 1)),
            nn.ConvTranspose2d(
                depth * 32,
                depth * 4,
                kernel_size=5,
                stride=2,
            ),
            activation(),
            nn.ConvTranspose2d(
                depth * 4,
                depth * 2,
                kernel_size=5,
                stride=2,
            ),
            activation(),
            nn.ConvTranspose2d(
                depth * 2,
                depth * 1,
                kernel_size=5 + 1,
                stride=2,
            ),
            activation(),
            nn.ConvTranspose2d(
                depth * 1,
                out_shape[0],
                kernel_size=5 + 1,
                stride=2,
            ),
        )
        # same init
        def initialize_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
        self.net.apply(initialize_weights)

    def forward(self, posterior, deterministic, mps_flatten=False):
        # posterior,deter: allow [B,T,*] or [B,*]
        x = torch.cat((posterior, deterministic), -1)
        batch_shape = x.shape[:-1]
        if not batch_shape:
            batch_shape = (1,)
        x = x.reshape(-1, x.shape[-1])
        if mps_flatten:
            batch_shape = (-1,)
        mean = self.net(x).reshape(*batch_shape, *self.out_shape)
        dist = torch.distributions.Normal(mean, 1)
        return torch.distributions.Independent(dist, len(self.out_shape))

class RSSM(nn.Module):
    def __init__(self, stoch_size=30, deter_size=200, embed_size=1024, action_dim=3):
        super().__init__()
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.action_dim = action_dim

        self.rnn = nn.GRUCell(stoch_size + action_dim, deter_size)

        self.prior_mean = nn.Linear(deter_size, stoch_size)
        self.prior_std = nn.Linear(deter_size, stoch_size)

        self.post_mean = nn.Linear(deter_size + embed_size, stoch_size)
        self.post_std = nn.Linear(deter_size + embed_size, stoch_size)

    def init_state(self, B, device):
        deter = torch.zeros(B, self.deter_size, device=device)
        stoch = torch.zeros(B, self.stoch_size, device=device)
        return stoch, deter

    def recurrent(self, prev_stoch, action, prev_deter):
        x = torch.cat([prev_stoch, action], dim=-1)
        deter = self.rnn(x, prev_deter)
        return deter

    def transition(self, deter):
        mean = self.prior_mean(deter)
        std = F.softplus(self.prior_std(deter)) + 1e-3
        dist = torch.distributions.Normal(mean, std)
        stoch = dist.rsample()
        return torch.distributions.Independent(dist, 1), stoch

    def representation(self, embed, deter):
        x = torch.cat([deter, embed], dim=-1)
        mean = self.post_mean(x)
        std = F.softplus(self.post_std(x)) + 1e-3
        dist = torch.distributions.Normal(mean, std)
        stoch = dist.rsample()
        return torch.distributions.Independent(dist, 1), stoch


class RewardNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 1),
        )
    def forward(self, s, h):  # [B,T,Z], [B,T,H]
        x = torch.cat([s, h], dim=-1)
        B, T, D = x.shape
        out = self.net(x.reshape(B * T, D))
        out = out.reshape(B, T, 1)
        return out


class ContinueNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 1),
        )
    def forward(self, s, h):
        x = torch.cat([s, h], dim=-1)
        B, T, D = x.shape
        logits = self.net(x.reshape(B * T, D)).reshape(B, T, 1)
        dist = torch.distributions.Bernoulli(logits=logits)
        return logits, torch.distributions.Independent(dist, 1)


# -------------------------
# Actor / Critic on latent
# -------------------------
class Actor(nn.Module):
    def __init__(self, in_dim, action_dim, min_std=0.1):
        super().__init__()
        self.min_std = min_std
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
        )
        self.mu = nn.Linear(256, action_dim)
        self.std = nn.Linear(256, action_dim)

    def forward(self, s, h):  # [B,N,Z],[B,N,H] or [B,Z],[B,H]
        x = torch.cat([s, h], dim=-1)
        x = self.net(x)
        mu = torch.tanh(self.mu(x))
        std = F.softplus(self.std(x)) + self.min_std
        return torch.distributions.Independent(torch.distributions.Normal(mu, std), 1)


class Critic(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 1),
        )
    def forward(self, s, h):
        x = torch.cat([s, h], dim=-1)
        return self.net(x)


# -------------------------
# Dreamer Agent
# -------------------------
class DreamerV1:
    def _save_frame_grid(self, imgs, fname):
        # imgs: Tensor [N,3,H,W], values in [0,1]
        pathlib.Path("viz").mkdir(parents=True, exist_ok=True)
        vutils.save_image(imgs.clamp(0,1), os.path.join("viz", fname), nrow=imgs.size(0))

    @torch.no_grad()
    def visualize_decoder(self, obs_batch, posts, deters, horizon_decode=15, step=0):
        # obs_batch: [B,T,3,H,W] in [0,1]
        B, T, C, H, W = obs_batch.shape
        b0 = 0  # visualize first batch item
        self._save_frame_grid(obs_batch[b0], f"gt_seq_step_{step:06d}.png")

        # 1) Reconstructions of observed frames (t=1..T-1)
        posts_b = posts[b0:b0+1].float()      # ensure FP32 for decoder weights
        deters_b = deters[b0:b0+1].float()
        recon_dist = self.decoder(posts_b, deters_b)            # [1,T-1,3,H,W]
        recon_mean = recon_dist.base_dist.mean.squeeze(0)       # [T-1,3,H,W]
        target = obs_batch[b0, 1:]                              # [T-1,3,H,W]
        self._save_frame_grid(target, f"gt_target_step_{step:06d}.png")

        # Arrange as row: target | recon
        recon_row = torch.cat([target, recon_mean], dim=0)      # [2*(T-1),3,H,W]
        self._save_frame_grid(recon_row, f"recon_step_{step:06d}.png")

        # 2) Imagined future frames from last posterior state
        s_last = posts[b0, -1].float()    # [Z], cast to FP32
        h_last = deters[b0, -1].float()   # [H]
        s_list, h_list = [], []
        # keep batch dim for GRUCell
        s = s_last.unsqueeze(0)  # [1,Z]
        h = h_last.unsqueeze(0)  # [1,H]
        for t in range(horizon_decode):
            dist_a = self.actor(s, h)        # Independent Normal over actions
            a = dist_a.mean                  # deterministic action [1,A]
            h = self.rssm.recurrent(s, a, h) # [1,H]
            _, s = self.rssm.transition(h)   # [1,Z]
            s_list.append(s.squeeze(0))      # [Z]
            h_list.append(h.squeeze(0))      # [H]
        s_im = torch.stack(s_list, dim=0).unsqueeze(0).float()  # [1,H,Z]
        h_im = torch.stack(h_list, dim=0).unsqueeze(0).float()  # [1,H,H]
        im_dist = self.decoder(s_im, h_im)                      # [1,H,3,H,W]
        im_mean = im_dist.base_dist.mean.squeeze(0)             # [H,3,H,W]
        self._save_frame_grid(im_mean, f"imagine_step_{step:06d}.png")

    def __init__(self, action_dim, img_size=(64, 64), device=None, cfg=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        logdir = os.path.join("runs", f"dreamer_nav_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(logdir, exist_ok=True)
        self.tb_writer = SummaryWriter(logdir)
        self.logdir = logdir
        print(f"TensorBoard logging to: {self.logdir}")
        self._global_step = 0

        # Config
        default_cfg = dict(
            seed=42,
            total_iter=500,
            save_freq=200,
            eval_freq=5,
            collect_iter=10,
            data_init_ep=3,
            data_interact_ep=1,
            batch_size=16,
            seq_len=40,
            horizon=15,
            free_nats=3.0,
            kl_scale=1.0,
            discount=0.99,
            lambda_=0.95,
            actor_lr=4e-5,
            critic_lr=4e-5,
            model_lr=3e-4,
            clip_grad=100.0,
            continue_loss=True,
            log_every=50,
        )
        if cfg:
            default_cfg.update(cfg)
        self.cfg = default_cfg
        seed_everything(self.cfg["seed"])

        self.img_size = img_size
        self.action_dim = action_dim

        # World model
        self.encoder = ConvEncoder(depth=32, input_shape=(3, 64, 64), activation=nn.ReLU).to(self.device)
        self.rssm = RSSM(stoch_size=30, deter_size=200, embed_size=1024, action_dim=action_dim).to(self.device)
        self.decoder = ConvDecoder(stochastic_size=30, deterministic_size=200, depth=32, out_shape=(3, 64, 64), activation=nn.ReLU).to(self.device)
        self.reward = RewardNet(in_dim=30 + 200).to(self.device)
        if self.cfg["continue_loss"]:
            self.cont_net = ContinueNet(in_dim=30 + 200).to(self.device)
        else:
            self.cont_net = None

        self.model_params = list(self.encoder.parameters()) + list(self.rssm.parameters()) + list(self.decoder.parameters()) + list(self.reward.parameters())
        if self.cont_net:
            self.model_params += list(self.cont_net.parameters())

        # Behavior
        beh_in = 30 + 200
        self.actor = Actor(in_dim=beh_in, action_dim=action_dim).to(self.device)
        self.critic = Critic(in_dim=beh_in).to(self.device)

        # Optims
        self.opt_model = torch.optim.Adam(self.model_params, lr=self.cfg["model_lr"])
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.cfg["actor_lr"])
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.cfg["critic_lr"])

        # AMP
        self.use_amp = self.device.type == "cuda"
        self.scaler_model = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.scaler_actor = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.scaler_critic = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Buffer
        self.buffer = ReplayBuffer(capacity_episodes=1000)

    @torch.no_grad()
    def evaluate(self, env: DroneXYZEnv, n_episodes=3, max_steps=1000):
        returns = []
        for ep in range(n_episodes):
            obs, info = env.reset()
            stoch, deter = self.rssm.init_state(1, self.device)
            self.last_action = None
            done = False
            ep_ret = 0.0
            steps = 0
            while not done and steps < max_steps:
                img_t = to_tensor_image_uint8(obs["image"], self.device).unsqueeze(0)
                # act without exploration (deterministic mean)
                stoch, deter, action = self.act(stoch, deter, img_t)
                act_np = action.squeeze(0).cpu().numpy().astype(np.float32)
                next_obs, reward, terminated, truncated, info = env.step(act_np)
                done = terminated or truncated
                ep_ret += float(reward)
                obs = next_obs
                steps += 1
            returns.append(ep_ret)
        avg_ret = float(np.mean(returns))
        std_ret = float(np.std(returns)) if len(returns) > 1 else 0.0
        print(f"[EVAL] episodes={n_episodes} avg_return={avg_ret:.3f} std={std_ret:.3f}")
        # Log to TensorBoard
        if hasattr(self, "tb_writer"):
            # Use a global_step-like counter if available; fallback to 0
            step = getattr(self, "_global_step", 0)
            self.tb_writer.add_scalar("eval/avg_return", avg_ret, step)
            self.tb_writer.add_scalar("eval/std_return", std_ret, step)
            self.tb_writer.flush()
        return avg_ret, std_ret

    # -------------------------
    # World model learning
    # -------------------------
    def dynamic_learning(self, obs, actions, rewards, dones):
        # obs: [B,T,C,H,W], actions: [B,T,A], rewards: [B,T,1], dones: [B,T,1]
        B, T, C, H, W = obs.shape
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            embeds = self.encoder(obs)  # [B,T,E]

            stoch = torch.zeros(B, self.rssm.stoch_size, device=self.device)
            deter = torch.zeros(B, self.rssm.deter_size, device=self.device)

            priors_m, priors_s, posts_m, posts_s = [], [], [], []
            posts, deters = [], []

            for t in range(1, T):
                deter = self.rssm.recurrent(stoch, actions[:, t - 1], deter)
                prior_dist, prior = self.rssm.transition(deter)
                post_dist, stoch = self.rssm.representation(embeds[:, t], deter)

                priors_m.append(prior_dist.base_dist.mean)
                priors_s.append(prior_dist.base_dist.stddev)
                posts_m.append(post_dist.base_dist.mean)
                posts_s.append(post_dist.base_dist.stddev)
                posts.append(stoch)
                deters.append(deter)

            posts = torch.stack(posts, 1)           # [B,T-1,Z]
            deters = torch.stack(deters, 1)         # [B,T-1,H]
            priors_m = torch.stack(priors_m, 1)
            priors_s = torch.stack(priors_s, 1)
            posts_m = torch.stack(posts_m, 1)
            posts_s = torch.stack(posts_s, 1)

            # Reconstruction
            recon_dist = self.decoder(posts, deters)  # Normal over pixels
            target = obs[:, 1:]  # [B,T-1,C,H,W]
            recon_loss = recon_dist.log_prob(target).mean()

            # Reward
            pred_r = self.reward(posts, deters)  # [B,T-1,1]
            # std must match dtype to avoid AMP Half vs Float mismatch
            r_dist = torch.distributions.Independent(
                torch.distributions.Normal(pred_r, torch.ones_like(pred_r)),
                1,
            )
            rew_loss = r_dist.log_prob(rewards[:, 1:]).mean()

            # Continue
            if self.cont_net:
                cont_logits, cont_dist = self.cont_net(posts, deters)
                cont_target = (1.0 - dones[:, 1:]) * self.cfg["discount"]
                cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)
            else:
                cont_loss = torch.zeros((), device=self.device)

            # KL
            prior_dist = torch.distributions.Independent(torch.distributions.Normal(priors_m, priors_s), 1)
            post_dist = torch.distributions.Independent(torch.distributions.Normal(posts_m, posts_s), 1)
            kl = torch.distributions.kl.kl_divergence(post_dist, prior_dist).mean()
            free_nats = torch.as_tensor(self.cfg["free_nats"], device=self.device)
            kl_loss = torch.maximum(kl, free_nats)

            total = self.cfg["kl_scale"] * kl_loss - recon_loss - rew_loss + cont_loss

        self.opt_model.zero_grad(set_to_none=True)
        self.scaler_model.scale(total).backward()
        nn.utils.clip_grad_norm_(self.model_params, self.cfg["clip_grad"])
        self.scaler_model.step(self.opt_model)
        self.scaler_model.update()

        stats = dict(
            kl=kl_loss.detach().item(),
            recon=-recon_loss.detach().item(),
            reward_ll=-rew_loss.detach().item(),
            cont=cont_loss.detach().item() if self.cont_net else 0.0,
            model_total=total.detach().item(),
        )
        return posts.detach(), deters.detach(), stats

    # -------------------------
    # Behavior learning via imagination
    # -------------------------
    def imagine(self, s0, h0, horizon):
        # s0,h0: [B, Z/H]
        B = s0.size(0)
        s_traj = []
        h_traj = []
        s = s0
        h = h0
        for t in range(horizon):
            dist_a = self.actor(s, h)
            a = dist_a.rsample()
            h = self.rssm.recurrent(s, a, h)
            _, s = self.rssm.transition(h)
            s_traj.append(s)
            h_traj.append(h)
        s_traj = torch.stack(s_traj, 1)  # [B,HZ,Z]
        h_traj = torch.stack(h_traj, 1)  # [B,HZ,H]
        return s_traj, h_traj

    def behavioral_learning(self, posts, deters):
        B, Tm1, Z = posts.shape
        s0 = posts.reshape(B * Tm1, Z)
        h0 = deters.reshape(B * Tm1, deters.size(-1))

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            s_traj, h_traj = self.imagine(s0, h0, self.cfg["horizon"])  # [BTm1,H,?]
            pred_r = self.reward(s_traj, h_traj)                         # [BTm1,H,1]
            # Match dtype under autocast
            pred_r = pred_r.float()
            r_dist = torch.distributions.Independent(
                torch.distributions.Normal(pred_r, torch.ones_like(pred_r)),
                1,
            )
            rewards = r_dist.mode

            if self.cont_net:
                logits, cont_dist = self.cont_net(s_traj, h_traj)
                continues = cont_dist.mean                
            else:
                continues = torch.full_like(rewards, self.cfg["discount"]).float()

            # Values (first pass for actor objective)
            v_all = self.critic(s_traj, h_traj).float()                 # [BTm1,H,1]
            last_v = self.critic(s_traj[:, -1], h_traj[:, -1]).unsqueeze(1).float()
            v_boot = torch.cat([v_all, last_v], dim=1)    
            returns = td_lambda(rewards, continues, v_boot, self.cfg["lambda_"], self.cfg["discount"])
            discount_prefix = torch.cumprod(
                torch.cat([torch.ones_like(continues[:, :1]).float(), continues[:, :-1].float()], dim=1),
                 dim=1,
            )

            actor_loss = -(discount_prefix.float() * returns.float()).mean()

        # Actor update
        self.opt_actor.zero_grad(set_to_none=True)
        self.scaler_actor.scale(actor_loss).backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg["clip_grad"])
        self.scaler_actor.step(self.opt_actor)
        self.scaler_actor.update()

        # Recompute critic forward after actor backward to avoid double-backward on freed graph
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            v_all_crit = self.critic(s_traj.detach(), h_traj.detach())  # detach imagination features
            critic_loss = F.mse_loss(v_all_crit, returns.detach())

        # Critic update
        self.opt_critic.zero_grad(set_to_none=True)
        self.scaler_critic.scale(critic_loss).backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg["clip_grad"])
        self.scaler_critic.step(self.opt_critic)
        self.scaler_critic.update()

        return dict(actor_loss=actor_loss.detach().item(), critic_loss=critic_loss.detach().item())

    # -------------------------
    # Acting in env (posterior update)
    # -------------------------
    @torch.no_grad()
    def act(self, prev_stoch, prev_deter, obs_img):
        # obs_img: [1,3,H,W] float
        embed = self.encoder(obs_img)  # [1,E]
        deter = self.rssm.recurrent(prev_stoch, self.last_action, prev_deter) if self.last_action is not None else prev_deter
        post_dist, stoch = self.rssm.representation(embed, deter)
        dist_a = self.actor(stoch, deter)
        action = dist_a.mean.clamp(-1.0, 1.0)  # mean action
        return stoch, deter, action

    # -------------------------
    # Training loop (collect/train)
    # -------------------------
    def train(self, env: DroneXYZEnv):
        device = self.device
        H, W = self.img_size
        # Running reward tracking
        running_reward = 0.0
        last_episode_reward = None
        # Prefill
        for _ in range(self.cfg["data_init_ep"]):
            self.buffer.start_episode()
            obs, info = env.reset()
            done = False
            while not done:
                # random action in [-1,1]
                a = np.random.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)
                next_obs, reward, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                # store
                img = next_obs["image"]
                # resize/store raw uint8 from env step; we store the next image so each step stores "state after action"
                self.buffer.add(img, a, reward, done)
        # Training
        global_step = 0
        for it in range(1, self.cfg["total_iter"] + 1):
            # Train steps
            for _ in range(self.cfg["collect_iter"]):
                obs_b, act_b, rew_b, done_b = self.buffer.sample(
                    batch_size=self.cfg["batch_size"],
                    seq_len=self.cfg["seq_len"],
                    device=device,
                    resize_to=self.img_size,
                )
                posts, deters, mstats = self.dynamic_learning(obs_b, act_b, rew_b, done_b)
                bstats = self.behavioral_learning(posts, deters)
                global_step += 1
                self._global_step = global_step

                self.tb_writer.add_scalar("model/kl", mstats["kl"], global_step)
                self.tb_writer.add_scalar("model/recon_ll", mstats["recon"], global_step)
                self.tb_writer.add_scalar("model/reward_ll", mstats["reward_ll"], global_step)
                self.tb_writer.add_scalar("model/continue_ll", mstats["cont"], global_step)
                self.tb_writer.add_scalar("model/total", mstats["model_total"], global_step)
                self.tb_writer.add_scalar("behavior/actor_loss", bstats["actor_loss"], global_step)
                self.tb_writer.add_scalar("behavior/critic_loss", bstats["critic_loss"], global_step)
                # self.tb_writer.add_scalar("reward/running_reward", running_reward, global_step)
                # if last_episode_reward is not None:
                #     self.tb_writer.add_scalar("reward/episode_reward", last_episode_reward, global_step)
                # Ensure data is written to disk frequently
                if global_step % 10 == 0:
                    self.tb_writer.flush()

                if global_step % self.cfg["log_every"] == 0:
                    print(
                        f"[it {it:04d} | step {global_step:06d}] "
                        f"KL={mstats['kl']:.3f} ReconLL={mstats['recon']:.3f} "
                        f"RewLL={mstats['reward_ll']:.3f} Cont={mstats['cont']:.3f} "
                        f"ModelTot={mstats['model_total']:.3f} "
                        f"A={bstats['actor_loss']:.3f} C={bstats['critic_loss']:.3f}"
                    )
                    self.visualize_decoder(obs_b, posts, deters, horizon_decode=10, step=global_step)

            # Periodic evaluation without exploration
            if (it % self.cfg.get("eval_freq", 50)) == 0:
                self.evaluate(env, n_episodes=self.cfg.get("eval_episodes", 3), max_steps=self.cfg.get("eval_max_steps", 1000))

            # Collect new data with current policy (1 episode)
            for _ in range(self.cfg["data_interact_ep"]):
                self.buffer.start_episode()
                obs, info = env.reset()
                # init latent
                stoch, deter = self.rssm.init_state(1, device)
                self.last_action = None
                done = False
                while not done:
                    img_t = to_tensor_image_uint8(obs["image"], device).unsqueeze(0)
                    stoch, deter, action = self.act(stoch, deter, img_t)
                    # exploration noise
                    noise = 0.3 * torch.randn_like(action)
                    act_np = (action + noise).clamp(-1, 1).squeeze(0).cpu().numpy().astype(np.float32)
                    next_obs, reward, terminated, truncated, info = env.step(act_np)
                    done = terminated or truncated
                    self.buffer.add(next_obs["image"], act_np, reward, done)
                    obs = next_obs
        self.tb_writer.close()

        print("Training finished.")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use smaller images for speed; Dreamer decodes to 64x64
    W = 64
    H = 64

    env = DroneXYZEnv(
        xml_path="office_0_quantized_16/merged_env.xml",
        image_width=W,
        image_height=H,
        max_delta=0.3,
    )

    agent = DreamerV1(
        action_dim=3,
        img_size=(H, W),
        device=device,
        cfg=dict(
            total_iter=300,      # adjust as needed
            collect_iter=10,
            data_init_ep=3,
            batch_size=16,
            seq_len=3,
            horizon=15,
            log_every=50,
        ),
    )

    agent.train(env)