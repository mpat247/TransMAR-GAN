import os
#os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime
import torch.nn.functional as F

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torch import nn


import time
from progress.bar import IncrementalBar  # Assuming you're using IncrementalBar for progress tracking



# Import customized dataset
from data.datasets import MARTrainDataset  # Assuming MARTrainDataset is available in Dataset module
from models.generator.ngswin import NGswin  # Assuming NGswin is the desired model to be used in Generator



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset Paths
path = "/home/Drive-D/SynDeepLesion/"  # Updated dataset path
PATCH_SIZE = 64
BATCH_SIZE = 4

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

# Learning rate for optimizers
lr = 0.0001

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


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
netG = Generator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
netG.apply(weights_init)

# Print the model
print(netG)


        
class BasicBlock(nn.Module):
    """Basic block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
        self.isn = None
        if norm:
            self.isn = nn.InstanceNorm2d(outplanes)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        fx = self.conv(x)
        
        if self.isn is not None:
            fx = self.isn(fx)
            
        fx = self.lrelu(fx)
        return fx
    
    
class ConditionalDiscriminator(nn.Module):
    """Conditional Discriminator"""
    def __init__(self,):
        super().__init__()
        self.block1 = BasicBlock(2, 64, norm=False)
        self.block2 = BasicBlock(64, 128)
        self.block3 = BasicBlock(128, 256)
        self.block4 = BasicBlock(256, 512)
        self.block5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        
    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        # blocks forward
        fx = self.block1(x)
        fx = self.block2(fx)
        fx = self.block3(fx)
        fx = self.block4(fx)
        fx = self.block5(fx)
        
        return fx 
        
# Create the Discriminator
netD = ConditionalDiscriminator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
netD.apply(weights_init)

# Print the model
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
checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')

# Load Checkpoint if exists
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    print('Checkpoint loaded!')

print("Starting Training Loop...")
# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0




# Training loop
for epoch in range(num_epochs):
    ge_loss = 0.
    de_loss = 0.
    start = time.time()
    bar = IncrementalBar(f'[Epoch {epoch+1}/{num_epochs}]', max=len(dataloader))
    
    # Set models to training mode
    netG.train()
    netD.train()
    
    for i, data in enumerate(dataloader, 0):
        # Input data
        ct = data[0].to(device)        # Input CT patches
        real = data[1].to(device)      # Ground truth CT patches
        

        # --------- Generator's Loss ---------
        fake = netG(ct)
        fake_pred = netD(fake, ct) # Discriminator prediction for generated images
        g_loss_adv = criterion(fake_pred, fake) # Adversarial loss
        g_loss_rec = nn.MSELoss()(fake, real)  # Reconstruction loss
        g_loss = g_loss_adv + 0.5 * g_loss_rec  # Combined generator loss

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # --------- Discriminator's Loss ---------
        fake_detached = fake.detach()  # Detach fake images to avoid affecting generator gradients
        fake_pred = netD(fake_detached, ct)  # Discriminator prediction for fake images
        real_pred = netD(real, ct)           # Discriminator prediction for real images

        d_loss_fake = criterion(fake_pred, x)  # Loss for fake predictions
        d_loss_real = criterion(real_pred, x)   # Loss for real predictions
        d_loss = (d_loss_fake + d_loss_real) / 2  # Average discriminator loss

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Accumulate batch losses
        ge_loss += g_loss.item()
        de_loss += d_loss.item()
        bar.next()
    
    bar.finish()
    
    # Calculate epoch losses
    g_loss_epoch = ge_loss / len(dataloader)
    d_loss_epoch = de_loss / len(dataloader)

 
    
    avg_ssim = ssim_score / (num_batches * BATCH_SIZE)
    avg_psnr = psnr_score / (num_batches * BATCH_SIZE)

    # --------- Log Metrics ---------
    logger.add_scalar('generator_loss', g_loss_epoch, epoch + 1)
    logger.add_scalar('discriminator_loss', d_loss_epoch, epoch + 1)
    logger.add_scalar('SSIM', avg_ssim, epoch + 1)
    logger.add_scalar('PSNR', avg_psnr, epoch + 1)
    logger.save_weights(netG.state_dict(), 'generator')
    logger.save_weights(netD.state_dict(), 'discriminator')

    print(f"[Epoch {epoch+1}/{num_epochs}] [G loss: {g_loss_epoch:.3f}] [D loss: {d_loss_epoch:.3f}] "
          f"[SSIM: {avg_ssim:.3f}] [PSNR: {avg_psnr:.3f}] ETA: {time.time() - start:.3f}s")
    
    # Return to training mode
    netG.train()
    netD.train()

    # Save model checkpoint every epoch
    torch.save({
        'epoch': epoch,
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'lossG': errG.item(),
        'lossD': errD.item(),
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1}")

