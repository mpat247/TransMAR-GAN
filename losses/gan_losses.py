
import torch
import torch.nn.functional as F

# real_logits, fake_logits are lists: [D¹(real), D½(real), D¼(real)], each (B,1,H',W')

def hinge_d_loss(real_logits, fake_logits):
    loss_real = 0.0
    loss_fake = 0.0
    for r_logit, f_logit in zip(real_logits, fake_logits):
        loss_real += F.relu(1.0 - r_logit).mean()
        loss_fake += F.relu(1.0 + f_logit).mean()
    return loss_real + loss_fake


def hinge_g_loss(fake_logits):
    loss = 0.0
    for f_logit in fake_logits:
        loss += -f_logit.mean()
    return loss



def feature_matching_loss(real_features, fake_features):
    """
    Feature matching loss across all scales and layers.

    real_features, fake_features:
        lists over scales -> lists over layers -> Tensor

    Each Tensor has shape (B, C_l, H_l, W_l)

    L_FM = sum_{s, l} E || D_l^s(real) - D_l^s(fake) ||_1

    We approximate (1 / N_l^s) * ||.||_1 with mean absolute value.
    """
    total = 0.0
    for fr_scale, ff_scale in zip(real_features, fake_features):
        for fr, ff in zip(fr_scale, ff_scale):
            total += torch.mean(torch.abs(fr - ff))
    return total


# ═══════════════════════════════════════════════════════════════
# METAL-AWARE LOSS FUNCTIONS (Equations 3, 4, 6)
# ═══════════════════════════════════════════════════════════════

def extract_metal_mask(ct, threshold=0.6):
    """
    Extract binary metal mask M from corrupted CT image.
    
    Args:
        ct: torch tensor [B, C, H, W] in normalized space [-1, 1]
        threshold: Intensity threshold for metal detection (data is in [-1,1] range,
                   metal artifacts appear as bright spots typically > 0.6)
    
    Returns:
        M: binary mask [B, 1, H, W] with 1 at metal pixels
    """
    # Data is already normalized to [-1, 1], threshold directly
    # Metal artifacts appear as high-intensity values
    M = (ct > threshold).float()
    
    return M


def dilate_mask(mask, radius=5):
    """
    Morphologically dilate binary mask using max pooling.
    
    Args:
        mask: binary mask [B, 1, H, W]
        radius: dilation radius in pixels
    
    Returns:
        B: dilated mask [B, 1, H, W]
    """
    kernel_size = 2 * radius + 1
    padding = radius
    
    # Use max pooling for dilation
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    
    return dilated


def compute_metal_aware_loss(fake, real_cpu, ct, beta=1.0, radius=5, w_max=3.0, threshold=0.6):
    """
    Compute metal-aware weighted L1 reconstruction loss.
    L_mw_rec = || w * (xK - x) ||_1
    
    Args:
        fake: generator output (xK) [B, C, H, W]
        real_cpu: ground truth clean image (x) [B, C, H, W]
        ct: corrupted input [B, C, H, W]
        beta: weight factor for metal-aware weighting
        radius: dilation radius
        w_max: maximum weight value (clipping)
        threshold: intensity threshold for metal detection (for data in [-1,1] range)
    
    Returns:
        loss: scalar tensor
    """
    # Extract metal mask M
    M = extract_metal_mask(ct, threshold=threshold)
    
    # Dilate to get band B
    B = dilate_mask(M, radius=radius)
    
    # Compute weighting map w = 1 + beta * B, clipped to w_max
    w = 1.0 + beta * B
    w = torch.clamp(w, max=w_max)
    
    # Compute weighted L1 loss
    diff = fake - real_cpu
    weighted = w * diff
    loss = torch.mean(torch.abs(weighted))
    
    return loss


def compute_weight_map(ct, beta=1.0, radius=5, w_max=3.0, threshold=0.6):
    """
    Compute metal-aware weighting map w = 1 + beta * B.
    
    Args:
        ct: corrupted input [B, C, H, W]
        beta: weight factor
        radius: dilation radius
        w_max: maximum weight value (clipping)
        threshold: intensity threshold for metal detection (for data in [-1,1] range)
    
    Returns:
        w: weighting map [B, 1, H, W]
    """
    # Extract metal mask M
    M = extract_metal_mask(ct, threshold=threshold)
    
    # Dilate to get band B
    B = dilate_mask(M, radius=radius)
    
    # Compute weighting map w = 1 + beta * B, clipped to w_max
    w = 1.0 + beta * B
    w = torch.clamp(w, max=w_max)
    
    return w


def compute_image_gradients(x):
    """
    Compute spatial gradients using finite differences.
    
    Args:
        x: image tensor [B, C, H, W]
    
    Returns:
        grad_x: horizontal gradient [B, C, H, W]
        grad_y: vertical gradient [B, C, H, W]
    """
    # Compute gradients
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]  # [B, C, H, W-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]  # [B, C, H-1, W]
    
    # Pad back to original size
    grad_x = F.pad(dx, (0, 1, 0, 0), mode='replicate')  # [B, C, H, W]
    grad_y = F.pad(dy, (0, 0, 0, 1), mode='replicate')  # [B, C, H, W]
    
    return grad_x, grad_y


def compute_metal_aware_edge_loss(fake, real_cpu, w):
    """
    Compute metal-aware edge loss (Equation 4).
    L_mw_edge = || w * (∇xK - ∇x) ||_1
    
    Args:
        fake: generator output (xK) [B, C, H, W]
        real_cpu: ground truth clean image (x) [B, C, H, W]
        w: weighting map [B, 1, H, W]
    
    Returns:
        loss: scalar tensor
    """
    # Compute gradients
    grad_fake_x, grad_fake_y = compute_image_gradients(fake)
    grad_real_x, grad_real_y = compute_image_gradients(real_cpu)
    
    # Compute gradient differences
    grad_diff_x = grad_fake_x - grad_real_x
    grad_diff_y = grad_fake_y - grad_real_y
    
    # Total gradient magnitude difference
    grad_diff = torch.abs(grad_diff_x) + torch.abs(grad_diff_y)
    
    # Apply spatial weighting
    weighted = w * grad_diff
    
    return weighted.mean()


def metal_consistency_loss(fake, real, M):
    """
    Compute metal-consistency loss (L_metal).
    L_metal = || M ⊙ (x̂ − x) ||_1
    
    This enforces accuracy specifically inside the metal region.
    Different from metal-aware loss which uses dilated band B.
    
    Args:
        fake: generator output (x̂) [B, C, H, W]
        real: ground truth clean image (x) [B, C, H, W]
        M: binary metal mask [B, 1, H, W] (1 at metal pixels)
    
    Returns:
        loss: scalar tensor
    """
    diff = fake - real
    masked_diff = M * diff
    return torch.mean(torch.abs(masked_diff))


def physics_loss_syn(fake, real_cpu, M, projector):
    """
    Compute physics-consistency loss (Equation 6) for synthetic data.
    L_phys_syn = || (1 - Mp) * (P(x_hat) - P(x_clean)) ||_1
    
    Args:
        fake: generator output (x_hat) [B, C, H, W]
        real_cpu: ground truth clean image (x_clean) [B, C, H, W]
        M: metal mask in image domain [B, C, H, W]
        projector: TorchRadon forward projector
    
    Returns:
        loss: scalar tensor
    """
    # Forward project fake and clean images to sinogram domain
    # TorchRadon expects [B, H, W] input, so squeeze channel dim if C=1
    fake_2d = fake.squeeze(1) if fake.shape[1] == 1 else fake
    real_2d = real_cpu.squeeze(1) if real_cpu.shape[1] == 1 else real_cpu
    M_2d = M.squeeze(1) if M.shape[1] == 1 else M
    
    proj_fake = projector.forward(fake_2d)    # [B, num_angles, detector_size]
    proj_clean = projector.forward(real_2d)   # [B, num_angles, detector_size]
    
    # Project metal mask to get metal trace in sinogram domain
    M_proj = projector.forward(M_2d)          # [B, num_angles, detector_size]
    Mp = (M_proj > 0).float()                 # Binary mask in projection domain
    
    # Compute masked difference (only non-metal rays contribute)
    diff = torch.abs(proj_fake - proj_clean)
    masked_diff = (1.0 - Mp) * diff
    
    return masked_diff.mean()
