import os
#os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from models.discriminator.ms_patchgan import MultiScaleDiscriminator
from losses.gan_losses import hinge_d_loss, hinge_g_loss, feature_matching_loss

from data.datasets import MARTrainDataset  # Assuming MARTrainDataset is available in Dataset module
from models.generator.ngswin import NGswin  # Assuming NGswin is the desired model to be used in Generator

from torchvision.utils import save_image

#save outputs
samples_dir = "./samples"
plots_dir   = "./plots"
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(plots_dir,   exist_ok=True)


def to01(x):
    return (x.clamp_(-1, 1) + 1) / 2

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
num_epochs = 100


# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


train_mask = np.load(os.path.join(path, 'trainmask.npy'))

# Safely load datasets
train_dataset = safe_load_dataset(MARTrainDataset, path, patchSize=PATCH_SIZE, length=BATCH_SIZE * 4000, mask=train_mask)

datasets = {'train': train_dataset}
dataloader = DataLoader(datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)

fixed_batch = next(iter(dataloader))

fixed_ct = fixed_batch[0].to(device)  # network input (LI)
fixed_gt = fixed_batch[1].to(device)  # ground truth



def weights_init(m):
    
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
        
        
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = NGswin()  # Assuming NGswin takes 3-channel input images
        
    def forward(self, input):
        return self.main(input)
   

# --- build models ---
netG = Generator(ngpu).to(device)  # your existing Generator (NGswin-based)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG)


# Create the generator
#netG = Generator(ngpu).to(device)
# Handle multi-GPU if desired
#if (device.type == 'cuda') and (ngpu > 1):
#    netG = nn.DataParallel(netG, list(range(ngpu)))


netG.apply(weights_init)

print(netG)

netD = MultiScaleDiscriminator(
    in_channels=2,   # concat[ct_corr, real/fake]
    base_channels=64,
    num_layers=5,
    num_scales=3,
    use_sn=True
).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD)


       
# Create the Discriminator
#netD = Discriminator(ngpu).to(device)


# Apply the weights_init function to randomly initialize all weights
netD.apply(weights_init)

print(netD)

# Loss function and optimizers
#criterion = nn.BCELoss()
#real_label = 1.
#fake_label = 0.
# For real labels, target should be 1 (or close to real labels)


# --- TTUR ---
lrG = 1e-4
lrD = 2e-4
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(0.5, 0.999))

# --- weights ---
lambda_adv = 0.1
lambda_fm  = 10.0
lambda_rec = 0.5


# Training checkpoint setup
checkpoint_dir = './PatchGAN_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')

# Load Checkpoint if exists
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(checkpoint['netG_state_dict'], strict=False)
    print("Loaded G weights; training with new MS-PatchGAN D.")
   # netD.load_state_dict(checkpoint['netD_state_dict'])
   # optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    #optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
   # print('Checkpoint loaded!')
    

#if os.path.isfile(checkpoint_path):
    #ckpt = torch.load(checkpoint_path, map_location=device)
    #netG.load_state_dict(ckpt['netG_state_dict'], strict=False)  # strict=False in case channels changed
    #print("Loaded G weights; training with new MS-PatchGAN D.")


print("Starting Training Loop...")
# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
generator_update_steps = 2

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):

        ct   = data[0].to(device)   
        real = data[1].to(device)   

        # 1) Update D
       
        netD.zero_grad()

        with torch.no_grad():
            fake = netG(ct)

        real_pair = torch.cat([ct, real], dim=1)
        fake_pair = torch.cat([ct, fake], dim=1)

        # multiscale outputs: list of logits per scale
        real_logits, _ = netD(real_pair, return_features=True)
        fake_logits, _ = netD(fake_pair, return_features=True)

        loss_D = hinge_d_loss(real_logits, fake_logits)

        loss_D.backward()
        optimizerD.step()

        # 2) Update G
        
        netG.zero_grad()

        fake = netG(ct)
        fake_pair = torch.cat([ct, fake], dim=1)

        fake_logits, fake_feats = netD(fake_pair, return_features=True)

       
        with torch.no_grad():
            _, real_feats = netD(real_pair, return_features=True)

        loss_G_adv = hinge_g_loss(fake_logits)
        loss_FM    = feature_matching_loss(real_feats, fake_feats)
        loss_rec   = torch.mean((fake - real) ** 2)

        loss_G = (
            lambda_adv * loss_G_adv +
            lambda_fm  * loss_FM +
            lambda_rec * loss_rec
        )

        loss_G.backward()
        optimizerG.step()

        # Logging
     
        if i % 50 == 0:
            print(
                f"[{epoch}/{num_epochs}] [{i}/{len(dataloader)}] "
                f"D: {loss_D.item():.4f} | "
                f"G: {loss_G.item():.4f} "
                f"(adv {loss_G_adv.item():.4f}, fm {loss_FM.item():.4f}, rec {loss_rec.item():.4f})"
            )

    torch.save({
        'epoch': epoch + 1,
        'netG': netG.state_dict(),
        'netD': netD.state_dict(),
        'optG': optimizerG.state_dict(),
        'optD': optimizerD.state_dict(),
    }, checkpoint_path)

    print(f"Checkpoint saved at epoch {epoch+1}")


