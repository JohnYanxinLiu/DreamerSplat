import cv2
from tqdm import tqdm
from gauss_dreamer import DynamicsModel
import torch
import matplotlib.pyplot as plt
from mujoco_env import DroneXYZEnv

import torch
import numpy as np
import random
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # CuDNN
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # (Optional) for PyTorch 2+
    # torch.use_deterministic_algorithms(True)




if __name__ == "__main__":
    seed_everything(0)

    # simple test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    W = 300
    H = 300


    model = DynamicsModel(
        img_W = W,
        img_H = H,
        device=device
    )

    env = DroneXYZEnv(
        xml_path="office_0_quantized_16/merged_env.xml",
        image_width=W,
        image_height=H,
        max_delta = 0.3
    )

    predict_change_in_camera_mlp = torch.nn.Sequential(
        torch.nn.Linear(3, 128).to(device),
        torch.nn.ReLU().to(device),
        torch.nn.Linear(128, 3).to(device)
    ).to(device)


    predict_reward_mlp = torch.nn.Sequential(
        torch.nn.Linear(3, 128).to(device),
        torch.nn.ReLU().to(device),
        torch.nn.Linear(128, 1).to(device)
    ).to(device)

    # --- Zero-initialize the last layer ---
    with torch.no_grad():
        last = predict_change_in_camera_mlp[-1]
        last.weight.zero_()
        last.bias.zero_()

    # --- Optimizer for both MLPs ---
    optimizer = torch.optim.Adam(
        list(predict_change_in_camera_mlp.parameters()) + list(predict_reward_mlp.parameters()), lr=1e-3
    )

    num_epochs = 200
    record_interval = 10000
    for _epoch in tqdm(range(num_epochs)):
        if _epoch % record_interval == 0:
            losses = []
        images = []
        rendered_images = []
        actions = []
        obs, _ = env.reset()
        for iters in range(10):
            current_c2w = torch.from_numpy(obs['cam_c2w']).float().to(device)
            current_img = torch.from_numpy(obs['image']).float().to(device) / 255.0

            action_np = env.action_space.sample()
            action = torch.from_numpy(action_np).float().to(device)
            next_obs, reward, terminated, truncated, info = env.step(action_np)

            change_in_c2w = predict_change_in_camera_mlp(action)

            predicted_reward = predict_reward_mlp(action)


            new_c2w = current_c2w.clone()
            new_c2w[:3, 3] = new_c2w[:3, 3] + change_in_c2w

            next_img = torch.from_numpy(next_obs['image']).float().to(device) / 255.0
            rendered = model.get_splat_render(
                new_c2w[None, ...],
                camera_intrinsics=torch.from_numpy(next_obs['intrinsics']).float().to(device)
            )
            rendered_rgb = rendered['rgb']  # H x W x 3

            c2w_loss = torch.nn.functional.mse_loss(
                new_c2w,
                torch.from_numpy(next_obs['cam_c2w']).float().to(device)
            )

            reward_loss = torch.nn.functional.mse_loss(
                predicted_reward.squeeze(),
                torch.tensor(reward).float().to(device)
            )

            loss = c2w_loss + 10 * reward_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            images.append(current_img.cpu().numpy())
            actions.append(action.cpu().numpy())

            rendered_images.append(rendered_rgb.cpu().detach().numpy())

            obs = next_obs
        if (_epoch+1) % record_interval == 0:
            print(f"Epoch {_epoch}, Average Loss: {sum(losses) / len(losses)}")
            fig, axs = plt.subplots(2, 10, figsize=(20, 4))
            for i in range(10):

                axs[0, i].imshow(images[i])
                axs[0, i].axis('off')
                axs[0, i].set_title(str(actions[i].round(2)))
                axs[1, i].imshow(rendered_images[i])
                axs[1, i].axis('off')

            plt.show()



    actions = [
        (0.0, 0.1, 0.0) for i in range(100)
    ]
    import numpy as np

    gt_images = []
    rendered_images = []
    rewards = []
    predicted_rewards = []
    cam_pos = []
    obs, _ = env.reset()
    current_c2w = torch.from_numpy(obs['cam_c2w']).float().to(device)
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(np.array(action))

        rewards.append(reward)

        gt_images.append(obs['image'] / 255.0)

    for action in actions:
        action_torch = torch.from_numpy(np.array(action)).float().to(device)

        with torch.no_grad():
            change_in_c2w = predict_change_in_camera_mlp(action_torch)
        current_c2w[:3, 3] = current_c2w[:3, 3] + change_in_c2w

        rendered = model.get_splat_render(
            current_c2w[None, ...],
            camera_intrinsics=torch.from_numpy(obs['intrinsics']).float().to(device)
        )
        rendered_rgb = rendered['rgb']  # H x W x 3

        rendered_images.append(rendered_rgb.cpu().detach().numpy())

        predicted_reward = predict_reward_mlp(action_torch)
        predicted_rewards.append(predicted_reward.item())

        cam_pos.append(current_c2w[:3, 3].cpu().numpy())

    # save gifs with concatenated images
    import imageio

    gif_images = []
    for i in range(len(actions)):
        concat_img = np.concatenate([gt_images[i], rendered_images[i]], axis=1)
        gif_images.append((concat_img * 255).astype(np.uint8))
        # write the camera position on the image
        # You can use OpenCV or PIL to draw text on the image if needed

        opencv_img = cv2.cvtColor(gif_images[-1], cv2.COLOR_RGB2BGR)
        position_text = f"Cam Pos: {cam_pos[i][0]:.2f}, {cam_pos[i][1]:.2f}, {cam_pos[i][2]:.2f}"
        cv2.putText(opencv_img, position_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        reward_text = f"Reward: {rewards[i]:.2f}"
        cv2.putText(opencv_img, reward_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        reward_text = f"Predicted Reward: {predicted_rewards[i]:.2f}"
        cv2.putText(opencv_img, reward_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        gif_images[-1] = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)

    imageio.mimsave('comparison.gif', gif_images, duration=0.1)
    print("Saved comparison.gif")




    actions = [
        (0.1, 0.1, 0.0) for i in range(100)
    ]
    import numpy as np

    gt_images = []
    rendered_images = []
    rewards = []
    predicted_rewards = []
    cam_pos = []
    obs, _ = env.reset()
    current_c2w = torch.from_numpy(obs['cam_c2w']).float().to(device)
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(np.array(action))

        rewards.append(reward)

        gt_images.append(obs['image'] / 255.0)

    for action in actions:
        action_torch = torch.from_numpy(np.array(action)).float().to(device)

        with torch.no_grad():
            change_in_c2w = predict_change_in_camera_mlp(action_torch)
        current_c2w[:3, 3] = current_c2w[:3, 3] + change_in_c2w

        rendered = model.get_splat_render(
            current_c2w[None, ...],
            camera_intrinsics=torch.from_numpy(obs['intrinsics']).float().to(device)
        )
        rendered_rgb = rendered['rgb']  # H x W x 3

        rendered_images.append(rendered_rgb.cpu().detach().numpy())

        predicted_reward = predict_reward_mlp(action_torch)
        predicted_rewards.append(predicted_reward.item())

        cam_pos.append(current_c2w[:3, 3].cpu().numpy())

    # save gifs with concatenated images
    import imageio

    gif_images = []
    for i in range(len(actions)):
        concat_img = np.concatenate([gt_images[i], rendered_images[i]], axis=1)
        gif_images.append((concat_img * 255).astype(np.uint8))
        # write the camera position on the image
        # You can use OpenCV or PIL to draw text on the image if needed

        opencv_img = cv2.cvtColor(gif_images[-1], cv2.COLOR_RGB2BGR)
        position_text = f"Cam Pos: {cam_pos[i][0]:.2f}, {cam_pos[i][1]:.2f}, {cam_pos[i][2]:.2f}"
        cv2.putText(opencv_img, position_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        reward_text = f"Reward: {rewards[i]:.2f}"
        cv2.putText(opencv_img, reward_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        reward_text = f"Predicted Reward: {predicted_rewards[i]:.2f}"
        cv2.putText(opencv_img, reward_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        gif_images[-1] = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)

    imageio.mimsave('comparison_2.gif', gif_images, duration=0.1)
    print("Saved comparison_2.gif")
    


    actions = [
        (0.1, -0.1, 0.0) for i in range(100)
    ]
    import numpy as np

    gt_images = []
    rendered_images = []
    rewards = []
    predicted_rewards = []
    cam_pos = []
    obs, _ = env.reset()
    current_c2w = torch.from_numpy(obs['cam_c2w']).float().to(device)
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(np.array(action))

        rewards.append(reward)

        gt_images.append(obs['image'] / 255.0)

    for action in actions:
        action_torch = torch.from_numpy(np.array(action)).float().to(device)

        with torch.no_grad():
            change_in_c2w = predict_change_in_camera_mlp(action_torch)
        current_c2w[:3, 3] = current_c2w[:3, 3] + change_in_c2w

        rendered = model.get_splat_render(
            current_c2w[None, ...],
            camera_intrinsics=torch.from_numpy(obs['intrinsics']).float().to(device)
        )
        rendered_rgb = rendered['rgb']  # H x W x 3

        rendered_images.append(rendered_rgb.cpu().detach().numpy())

        predicted_reward = predict_reward_mlp(action_torch)
        predicted_rewards.append(predicted_reward.item())

        cam_pos.append(current_c2w[:3, 3].cpu().numpy())

    # save gifs with concatenated images
    import imageio

    gif_images = []
    for i in range(len(actions)):
        concat_img = np.concatenate([gt_images[i], rendered_images[i]], axis=1)
        gif_images.append((concat_img * 255).astype(np.uint8))
        # write the camera position on the image
        # You can use OpenCV or PIL to draw text on the image if needed

        opencv_img = cv2.cvtColor(gif_images[-1], cv2.COLOR_RGB2BGR)
        position_text = f"Cam Pos: {cam_pos[i][0]:.2f}, {cam_pos[i][1]:.2f}, {cam_pos[i][2]:.2f}"
        cv2.putText(opencv_img, position_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        reward_text = f"Reward: {rewards[i]:.2f}"
        cv2.putText(opencv_img, reward_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        reward_text = f"Predicted Reward: {predicted_rewards[i]:.2f}"
        cv2.putText(opencv_img, reward_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        gif_images[-1] = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)

    imageio.mimsave('comparison_3.gif', gif_images, duration=0.1)
    print("Saved comparison_3.gif")



    actions = [
        env.action_space.sample() for i in range(100)
    ]
    import numpy as np

    gt_images = []
    rendered_images = []
    rewards = []
    predicted_rewards = []
    cam_pos = []
    obs, _ = env.reset()
    current_c2w = torch.from_numpy(obs['cam_c2w']).float().to(device)
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(np.array(action))

        rewards.append(reward)

        gt_images.append(obs['image'] / 255.0)

    for action in actions:
        action_torch = torch.from_numpy(np.array(action)).float().to(device)

        with torch.no_grad():
            change_in_c2w = predict_change_in_camera_mlp(action_torch)
        current_c2w[:3, 3] = current_c2w[:3, 3] + change_in_c2w

        rendered = model.get_splat_render(
            current_c2w[None, ...],
            camera_intrinsics=torch.from_numpy(obs['intrinsics']).float().to(device)
        )
        rendered_rgb = rendered['rgb']  # H x W x 3

        rendered_images.append(rendered_rgb.cpu().detach().numpy())

        predicted_reward = predict_reward_mlp(action_torch)
        predicted_rewards.append(predicted_reward.item())

        cam_pos.append(current_c2w[:3, 3].cpu().numpy())

    # save gifs with concatenated images
    import imageio

    gif_images = []
    for i in range(len(actions)):
        concat_img = np.concatenate([gt_images[i], rendered_images[i]], axis=1)
        gif_images.append((concat_img * 255).astype(np.uint8))
        # write the camera position on the image
        # You can use OpenCV or PIL to draw text on the image if needed

        opencv_img = cv2.cvtColor(gif_images[-1], cv2.COLOR_RGB2BGR)
        position_text = f"Cam Pos: {cam_pos[i][0]:.2f}, {cam_pos[i][1]:.2f}, {cam_pos[i][2]:.2f}"
        cv2.putText(opencv_img, position_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        reward_text = f"Reward: {rewards[i]:.2f}"
        cv2.putText(opencv_img, reward_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        reward_text = f"Predicted Reward: {predicted_rewards[i]:.2f}"
        cv2.putText(opencv_img, reward_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        gif_images[-1] = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)

    imageio.mimsave('comparison_4.gif', gif_images, duration=0.1)
    print("Saved comparison_4.gif")


    torch.save({
        'predict_change_in_camera_mlp': predict_change_in_camera_mlp,
        'predict_reward_mlp': predict_reward_mlp,
    }, 'gauss_dreamer_pt2_dynamics_model.pth')