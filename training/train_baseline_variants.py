import os
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
from torch_radon import Radon



# Import customized dataset
from data.datasets import MARTrainDataset, SpineWebTrainDataset  # Assuming MARTrainDataset is available in Dataset module
from models.generator.ngswin import NGswin  # Assuming NGswin is the desired model to be used in Generator

# Import metal-aware loss functions
from losses.gan_losses import (
    extract_metal_mask,
    dilate_mask,
    compute_metal_aware_loss,
    compute_weight_map,
    compute_image_gradients,
    compute_metal_aware_edge_loss,
    physics_loss_syn
)



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ═══════════════════════════════════════════════════════════════
# DATASET SELECTION - SET TO True FOR SPINEWEB FINE-TUNING
# ═══════════════════════════════════════════════════════════════
USE_SPINEWEB = True  # Set to True for fine-tuning on SpineWeb

# Dataset Paths
path = "/home/Drive-D/SynDeepLesion/"  # Updated dataset path
PATCH_SIZE = 64
BATCH_SIZE = 4
torch.backends.cudnn.benchmark = True  # Enable optimized convolutions
# Load Dataset
def safe_load_dataset(dataset_class, *args, **kwargs):
    while True:
        try:
            dataset = dataset_class(*args, **kwargs)
            return dataset
        except FileNotFoundError as e:
            print(f"File not found: {e}. Skipping to the next available file.")
            continue



# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(False)

# Number of workers for dataloader
workers = 2

# Batch size during training                 
batch_size = 4

# Spatial size of training images
image_size = 64

# Number of channels in the training images (1 for grayscale CT images)
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 100 if not USE_SPINEWEB else 25  # Fewer epochs for fine-tuning

# Learning rate for optimizers
lr = 0.0001 if not USE_SPINEWEB else 0.00001  # Lower learning rate for fine-tuning

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Metal-aware edge loss weight (Equation 4)
lambda_edge = 0.2

# Physics-consistency loss weight (Equation 6)
lambda_phys = 0.1

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT / RESULTS SETUP
# ═══════════════════════════════════════════════════════════════
run_root = "./finetune_results" if USE_SPINEWEB else "./train_results"
os.makedirs(run_root, exist_ok=True)
run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
run_dir = os.path.join(run_root, run_id)
os.makedirs(run_dir, exist_ok=True)
os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(run_dir, "test_examples"), exist_ok=True)

print(f"[Run] Results will be saved under {run_dir}")

# TensorBoard writer
writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))


# ═══════════════════════════════════════════════════════════════
# LOAD DATASET BASED ON SELECTION
# ═══════════════════════════════════════════════════════════════
if USE_SPINEWEB:
    # SpineWeb dataset for fine-tuning (train split)
    artifact_dir = "/home/Drive-D/UWSpine_adn/train/synthesized_metal_transfer/"
    clean_dir = "/home/Drive-D/UWSpine_adn/train/no_metal/"

    print("═══════════════════════════════════════════════════════════════")
    print("LOADING SPINEWEB DATASET FOR FINE-TUNING (TRAIN)")
    print("═══════════════════════════════════════════════════════════════")

    train_dataset = SpineWebTrainDataset(
        artifact_dir=artifact_dir,
        clean_dir=clean_dir,
        patchSize=PATCH_SIZE,
        paired=True,  # Set to False for unpaired training
        hu_range=(-1000, 2000)
    )

    # SpineWeb test dataset (assumes parallel test folders)
    spineweb_test_artifact = "/home/Drive-D/UWSpine_adn/test/synthesized_metal_transfer/"
    spineweb_test_clean = "/home/Drive-D/UWSpine_adn/test/no_metal/"

    if os.path.isdir(spineweb_test_artifact) and os.path.isdir(spineweb_test_clean):
        print("[Data] Detected SpineWeb test folders; using a test dataloader with the same dataset class.")
        spineweb_test_dataset = SpineWebTrainDataset(
            artifact_dir=spineweb_test_artifact,
            clean_dir=spineweb_test_clean,
            patchSize=PATCH_SIZE,
            paired=True,
            hu_range=(-1000, 2000),
        )
        test_loader = DataLoader(spineweb_test_dataset, batch_size=1, shuffle=False, num_workers=workers)
    else:
        print("[Data] SpineWeb test folders not found; skipping test evaluation.")
        test_loader = None
else:
    # Original SynDeepLesion dataset
    print("═══════════════════════════════════════════════════════════════")
    print("LOADING SYNDEEPLESION DATASET")
    print("═══════════════════════════════════════════════════════════════")
    
    train_mask = np.load(os.path.join(path, 'trainmask.npy'))
    
    # Safely load datasets
    train_dataset = safe_load_dataset(MARTrainDataset, path, patchSize=PATCH_SIZE, length=BATCH_SIZE * 4000, mask=train_mask)
#val_dataset = safe_load_dataset(MARValDataset, path, mask=train_mask)

# Use TestDataset for testing
#test_dataset = safe_load_dataset(TestDataset, path, mask=train_mask)

datasets = {'train': train_dataset}
dataloader = DataLoader(datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)
#val_loader = DataLoader(datasets['val'], batch_size=1, shuffle=False)
#test_loader = DataLoader(datasets['test'], batch_size=1, shuffle=False)




# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
      # Generator: Using NGswin as the generator model
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = NGswin()  # Assuming NGswin takes 3-channel input images
        
    def forward(self, input):
        return self.main(input)
   

# Create the generator
print("[Init] Building Generator (NGswin)...")
netG = Generator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    print(f"[Init] Using DataParallel for Generator on {ngpu} GPUs")
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
print("[Init] Initializing Generator weights...")
netG.apply(weights_init)

# Print the model
print("[Init] Generator architecture:")
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


# Create the discriminator (DCGAN-style, matches checkpoint)
print("[Init] Building Discriminator (DCGAN-style)...")
netD = Discriminator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    print(f"[Init] Using DataParallel for Discriminator on {ngpu} GPUs")
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
print("[Init] Initializing Discriminator weights...")
netD.apply(weights_init)

# Print the model
print("[Init] Discriminator architecture:")
print(netD)

# Loss function and optimizers
criterion = nn.BCELoss()
real_label = 1.
fake_label = 0.
# For real labels, target should be 1 (or close to real labels)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training checkpoint setup
checkpoint_dir = './training_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Select checkpoint for fine-tuning or regular training
if USE_SPINEWEB:
    # For fine-tuning, load the best checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'bestcheckpoint.pth')
    print(f"Fine-tuning mode: Loading checkpoint from {checkpoint_path}")
else:
    # Regular training, use latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')

# Load Checkpoint if exists
start_epoch = 0
if os.path.isfile(checkpoint_path):
    print(f"[Checkpoint] Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1 if not USE_SPINEWEB else 0
    if USE_SPINEWEB:
        print(f"[Checkpoint] Loaded pre-trained weights for fine-tuning. Resetting epoch to 0 (original epoch in checkpoint: {checkpoint.get('epoch', 'N/A')}).")
    else:
        print(f"[Checkpoint] Resuming training from epoch {start_epoch}.")
else:
    if USE_SPINEWEB:
        print(f"WARNING: No checkpoint found at {checkpoint_path}")
        print("Fine-tuning requires a pre-trained model!")
    print('No checkpoint found, training from scratch')

print("Starting Training Loop...")
print(f"Training for {num_epochs} epochs with learning rate {lr}")
print(f"Using {'SpineWeb' if USE_SPINEWEB else 'SynDeepLesion'} dataset")
print("═══════════════════════════════════════════════════════════════")

# ═══════════════════════════════════════════════════════════════
# Initialize TorchRadon projector for physics-consistency loss
# ═══════════════════════════════════════════════════════════════
img_height = PATCH_SIZE
img_width = PATCH_SIZE
num_angles = 180
angles = torch.linspace(0, np.pi, num_angles, device=device)
projector = Radon(img_height, angles).to(device)
print(f"[Init] TorchRadon projector initialized: {img_height}x{img_width}, {num_angles} angles")

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
generator_update_steps = 2

total_epochs = start_epoch + num_epochs

# Training Loop
for epoch in range(start_epoch, start_epoch + num_epochs):
    print(f"\n[Epoch {epoch+1}/{total_epochs}] Starting epoch...")

    epoch_bar = tqdm(
        enumerate(dataloader, 0),
        total=len(dataloader),
        desc=f"Epoch {epoch+1}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )

    for i, data in epoch_bar:
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with real batch
        if i % 50 == 0:
            print(f"  [Batch {i+1}/{len(dataloader)}] Updating Discriminator...")

        netD.zero_grad()
        real_cpu = data[1].to(device)

        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        # Discriminator takes only the image as input
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train with fake batch
        ct = data[0].to(device)
        fake = netG(ct)  # Generate fake images using low dose CT images
        label.fill_(fake_label)
        output_fake = netD(fake.detach()).view(-1)

        # Calculate loss for fake images
        errD_fake = criterion(output_fake, label)
        errD_fake.backward()
        D_G_z1 = output_fake.mean().item()

        # Update D
        errD = errD_real + errD_fake
        optimizerD.step()

        # (2) Update G network multiple times
        ###########################
        for step in range(generator_update_steps):  # Generator prioritization
            netG.zero_grad()
            fake = netG(ct)
            label.fill_(real_label)  # Fake labels are real for generator cost
            output = netD(fake).view(-1)
            
            # Calculate G's loss
            adv_loss = criterion(output, label)
            rec_loss_mw = compute_metal_aware_loss(fake, real_cpu, ct)  # Equation (3): Metal-aware weighted L1
            
            # Compute metal-aware edge loss (Equation 4)
            w = compute_weight_map(ct)
            edge_loss_mw = compute_metal_aware_edge_loss(fake, real_cpu, w)
            
            # Compute physics-consistency loss (Equation 6)
            M = extract_metal_mask(ct, threshold=2000.0)
            phys_loss = physics_loss_syn(fake, real_cpu, M, projector)
            
            # Total generator loss
            errG = adv_loss + 0.5 * rec_loss_mw + lambda_edge * edge_loss_mw + lambda_phys * phys_loss
            errG.backward()

            D_G_z2 = output.mean().item()
            optimizerG.step()

        # Update tqdm bar with latest stats
        epoch_bar.set_postfix({
            'Loss_D': f"{errD.item():.4f}",
            'Loss_G': f"{errG.item():.4f}",
            'D(x)': f"{D_x:.4f}",
            'D(Gz1)': f"{D_G_z1:.4f}",
            'D(Gz2)': f"{D_G_z2:.4f}",
        })

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, start_epoch + num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed images
        if (iters % 500 == 0) or ((epoch == start_epoch + num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(ct).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        
        iters += 1

    # Save model checkpoint every epoch (original path)
    save_checkpoint_path = checkpoint_path
    if USE_SPINEWEB:
        # Save fine-tuned checkpoints with different name under training_checkpoints
        save_checkpoint_path = os.path.join(checkpoint_dir, f'finetuned_spineweb_epoch_{epoch+1}.pth')

    checkpoint_payload = {
        'epoch': epoch,
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'lossG': errG.item(),
        'lossD': errD.item(),
    }

    torch.save(checkpoint_payload, save_checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1} to {save_checkpoint_path}")

    # Also save a copy of the checkpoint into the run directory for this experiment
    run_ckpt_path = os.path.join(run_dir, "checkpoints", f'epoch_{epoch+1}.pth')
    torch.save(checkpoint_payload, run_ckpt_path)


# Save training config for this run
config = {
    'USE_SPINEWEB': USE_SPINEWEB,
    'num_epochs': num_epochs,
    'learning_rate': lr,
    'batch_size': BATCH_SIZE,
    'patch_size': PATCH_SIZE,
    'checkpoint_path': checkpoint_path,
}
with open(os.path.join(run_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)


# Plotting the loss curves for Generator and Discriminator and save to run folder
fig = plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
fig.tight_layout()
fig.savefig(os.path.join(run_dir, 'loss_curves.png'))
plt.close(fig)


# Real vs Fake visualization from last epoch, saved to run folder
if len(img_list) > 0:
    real_batch = next(iter(dataloader))
    fig_rf = plt.figure(figsize=(12, 6))

    # Real Images
    ax1 = fig_rf.add_subplot(1, 2, 1)
    ax1.axis('off')
    ax1.set_title('Real Images')
    ax1.imshow(np.transpose(
        vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),
        (1, 2, 0)
    ))

    # Fake Images
    ax2 = fig_rf.add_subplot(1, 2, 2)
    ax2.axis('off')
    ax2.set_title('Fake Images (last epoch)')
    ax2.imshow(np.transpose(img_list[-1], (1, 2, 0)))

    fig_rf.tight_layout()
    fig_rf.savefig(os.path.join(run_dir, 'real_vs_fake.png'))
    plt.close(fig_rf)


# ──────────────────────────────────────────────────────────────
# Post-training SpineWeb test evaluation (if test_loader exists)
# ──────────────────────────────────────────────────────────────
test_metrics = {}
if USE_SPINEWEB and 'test_loader' in globals() and test_loader is not None:
    print("[Test] Running SpineWeb test evaluation...")
    netG.eval()
    mse_list = []
    psnr_list = []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            # SpineWebTrainDataset returns (artifact, clean)
            ct_artifact = batch[0].to(device)
            clean_target = batch[1].to(device)

            pred = netG(ct_artifact)

            # MSE in normalized space
            mse = F.mse_loss(pred, clean_target, reduction='mean').item()
            mse_list.append(mse)

            # PSNR assuming input in [-1,1]; dynamic range is 2
            max_val = 1.0
            psnr = 10 * np.log10((2 * max_val) ** 2 / (mse + 1e-8))
            psnr_list.append(psnr)

            # Save a few qualitative examples
            if idx < 10:
                # Denormalize from [-1,1] to [0,1]
                def denorm(x):
                    return (x + 1.0) / 2.0

                ct_vis = denorm(ct_artifact.cpu())
                clean_vis = denorm(clean_target.cpu())
                pred_vis = denorm(pred.cpu())

                grid = vutils.make_grid(
                    torch.cat([ct_vis, clean_vis, pred_vis], dim=0),
                    nrow=ct_artifact.size(0),
                    padding=5,
                    normalize=True,
                )
                save_path = os.path.join(run_dir, 'test_examples', f'sample_{idx:03d}.png')
                vutils.save_image(grid, save_path)

    mean_mse = float(np.mean(mse_list)) if mse_list else None
    mean_psnr = float(np.mean(psnr_list)) if psnr_list else None

    test_metrics = {
        'mean_mse': mean_mse,
        'mean_psnr': mean_psnr,
        'num_samples': len(mse_list),
    }

    print(f"[Test] Done. mean MSE={mean_mse:.6f}  mean PSNR={mean_psnr:.2f} dB over {len(mse_list)} samples")

    # Log to TensorBoard
    if mean_mse is not None:
        writer.add_scalar('Test/MSE', mean_mse, total_epochs)
    if mean_psnr is not None:
        writer.add_scalar('Test/PSNR', mean_psnr, total_epochs)

    with open(os.path.join(run_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
else:
    print("[Test] No SpineWeb test loader available; skipping test evaluation.")


# Close TensorBoard writer
writer.close()
