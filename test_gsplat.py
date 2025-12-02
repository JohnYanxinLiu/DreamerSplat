import torch
import numpy as np
from plyfile import PlyData
from gsplat.rendering import rasterization   # 👈 CHANGED
import matplotlib.pyplot as plt


def load_ply(path, device):
    ply = PlyData.read(path)
    v = ply["vertex"].data

    xyz = torch.tensor(
        np.stack([v["x"], v["y"], v["z"]], axis=1),
        dtype=torch.float32, device=device, requires_grad=True
    )

    scales = torch.tensor(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1),
        dtype=torch.float32, device=device, requires_grad=True
    )

    rots = torch.tensor(
        np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1),
        dtype=torch.float32, device=device, requires_grad=True
    )

    colors = torch.tensor(
        np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1),
        dtype=torch.float32, device=device, requires_grad=True
    )

    opacity = torch.tensor(
        v["opacity"],
        dtype=torch.float32, device=device, requires_grad=True
    )

    return xyz, scales, rots, colors, opacity


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xyz, scales, rots, colors, opacity = load_ply(
        "office_0_quantized_16/office_0_splat.ply", device
    )

    H, W = 512, 512

    # Camera intrinsics (3x3)
    K = torch.tensor([
        [700., 0.,   W/2],
        [0.,   700., H/2],
        [0.,   0.,   1.],
    ], dtype=torch.float32, device=device)

    # Camera pose (4x4 world-to-camera)
    viewmat = torch.eye(4, dtype=torch.float32, device=device)

    # =====================================================
    # RASTERIZATION CALL
    # =====================================================
    rendered_image, rendered_alpha, info = rasterization(
        means=xyz,
        quats=rots,              # quaternions for rotation
        scales=scales,
        opacities=opacity,
        colors=colors,
        viewmats=viewmat[None, ...],  # Shape: [1, 4, 4] (batch dimension)
        Ks=K[None, ...],              # Shape: [1, 3, 3] (batch dimension)
        width=W,
        height=H,
        packed=False,            # Returns dense [B, H, W, C] tensors
        absgrad=False            # Use standard gradients
    )

    # Output shapes:
    # rendered_image: [1, H, W, 3]
    # rendered_alpha: [1, H, W, 1]
    
    # Remove batch dimension for visualization
    img = rendered_image[0]  # [H, W, 3]
    alpha = rendered_alpha[0, ..., 0]  # [H, W]

    # Visualize
    img_np = img.detach().cpu().numpy()
    alpha_np = alpha.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Rendered Image")
    axes[0].axis("off")
    
    axes[1].imshow(alpha_np, cmap='gray')
    axes[1].set_title("Alpha Channel")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig('rendered_output.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Check gradients
    loss = rendered_image.mean()
    loss.backward()
    print("Gradients computed successfully!")
    print(f"xyz.grad is not None: {xyz.grad is not None}")
    print(f"colors.grad is not None: {colors.grad is not None}")
    print(f"Rendered {len(xyz)} Gaussians")
    print(f"Info keys: {info.keys()}")


if __name__ == "__main__":
    main()