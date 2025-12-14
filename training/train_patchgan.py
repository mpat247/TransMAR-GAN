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
from models.discriminator.conditional_patchgan import ConditionalDiscriminator, BasicBlock
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import matplotlib.animation as animation



# Import customized dataset
from data.datasets import MARTrainDataset  # Assuming MARTrainDataset is available in Dataset module
from models.generator.ngswin import NGswin  # Assuming NGswin is the desired model to be used in Generator
# --- imports ---
from models.discriminator.ms_patchgan import MultiScaleDiscriminator

    


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
   




# --- build models ---
netG = Generator(ngpu).to(device)  # your existing Generator (NGswin-based)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG)




# Create the generator
#netG = Generator(ngpu).to(device)
# Handle multi-GPU if desired
#if (device.type == 'cuda') and (ngpu > 1):
#    netG = nn.DataParallel(netG, list(range(ngpu)))


# Apply the weights_init function to randomly initialize all weights
netG.apply(weights_init)

# Print the model
print(netG)


 netD = MultiScaleDiscriminator(in_ch_cond=1, in_ch_img=1, base_ch=64, use_sn=True).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD)


       
# Create the Discriminator
#netD = Discriminator(ngpu).to(device)

# Handle multi-GPU if desired
#if (device.type == 'cuda') and (ngpu > 1):
   # netD = nn.DataParallel(netD, list(range(ngpu)))

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
generator_update_steps = 2

# Training Loop
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with real batch
        
        
        netD.zero_grad()
        real_cpu = data[1].to(device)
        print(real_cpu.shape)
        
       
        b_size = real_cpu.size(0)
        # target=data[1].to(device)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        #label_real = torch.full_like(output_real, real_label, dtype=torch.float, device=device)
        
        ## Train with fake batch
        ct = data[0].to(device)
        fake = netG(ct)  # Generate fake images using low dose CT images
        label.fill_(fake_label)
        output_fake = netD(fake.detach()).view(-1)
        #b_size = ct.size(0)
       
        print(f"NGswin output shape: {fake.shape}")
        
        print(f"Discriminator input shape: {fake.detach().shape}")

        # Create label tensor dynamically with matching size
        #label_fake = torch.full_like(output_fake, fake_label, dtype=torch.float, device=device)

        # Calculate loss for fake images
        errD_fake = criterion(output_fake, label)
        errD_fake.backward()
        D_G_z1 = output_fake.mean().item()

        # Update D
        errD = errD_real + errD_fake
        optimizerD.step()

        # (2) Update G network multiple times
        ###########################
        for _ in range(generator_update_steps):  # Generator prioritization
            netG.zero_grad()
            fake = netG(ct)
            label.fill_(real_label)  # Fake labels are real for generator cost
            output = netD(fake).view(-1)
            
            # Calculate G's loss
            adv_loss = criterion(output, label)
            rec_loss = nn.MSELoss()(fake, real_cpu)  # MAR-specific loss
            errG = adv_loss + 0.5 * rec_loss  # Adjust weighting as needed
            errG.backward()
            
            D_G_z2 = output.mean().item()
            optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed images
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(ct).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        
        iters += 1

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

# Plotting the loss curves for Generator and Discriminator
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Visualization of G's progression
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)


# Real Images vs. Fake Images
real_batch = next(iter(dataloader))
plt.figure(figsize=(15, 15))
# Plot Real Images
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))


# Plot Fake Images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()
