import os.path
import os
import os.path
import argparse
import numpy as np
import torch
from CLINIC_metal.preprocess_clinic.preprocessing_clinic import clinic_input_data
#from network.indudonet import InDuDoNet
import nibabel
import time
import torch
import torch.nn as nn
import numpy as np
from models.generator.ngswin import NGswin

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description="YU_Test")
parser.add_argument("--model_dir", type=str, default="models", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="/home/grad/smalruba/Documents/DCGAN/CLINIC_metal/test/", help='path to training data')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--save_path", type=str, default="results/CLINIC_metal/", help='path to training data')
parser.add_argument('--num_channel', type=int, default=32, help='the number of dual channels')
parser.add_argument('--T', type=int, default=4, help='the number of ResBlocks in every ProxNet')
parser.add_argument('--S', type=int, default=10, help='the number of total iterative stages')
parser.add_argument('--eta1', type=float, default=1, help='initialization for stepsize eta1')
parser.add_argument('--eta2', type=float, default=5, help='initialization for stepsize eta2')
parser.add_argument('--alpha', type=float, default=0.5, help='initialization for weight factor')
opt = parser.parse_args()
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")
Pred_nii = opt.save_path +'/X_mar/'
mkdir(Pred_nii)

def image_get_minmax():
    return 0.0, 1.0
def proj_get_minmax():
    return 0.0, 4.0

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max) 
    data = (data - data_min) / (data_max - data_min)
    #data = data * 255.0
    data = (data * 2) - 1
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    return data

def test_image(allXma, allXLI, allM, vol_idx, slice_idx):
    Xma = allXma[vol_idx][...,slice_idx]
    XLI = allXLI[vol_idx][...,slice_idx]
    M = allM[vol_idx][...,slice_idx]
    #Sma = allSma[vol_idx][...,slice_idx]
    #SLI = allSLI[vol_idx][...,slice_idx]
    #Tr = allTr[vol_idx][...,slice_idx]
    Xma = normalize(Xma, image_get_minmax())  # *255
    XLI = normalize(XLI, image_get_minmax())
    #Sma = normalize(Sma, proj_get_minmax())
    #SLI = normalize(SLI, proj_get_minmax())
    Xma = (Xma * 2) - 1  # now in [-1, 1]
    XLI = (XLI * 2) - 1  # now in [-1, 1]
    #Xma = (Xma * 2) - 1  # now in [-1, 1]
    #Tr = 1-Tr.astype(np.float32)
    #Tr = np.expand_dims(np.transpose(np.expand_dims(Tr, 2), (2, 0, 1)),0) # 1*1*h*w
    Mask = M.astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)),0)
    return torch.Tensor(Xma).cuda(), torch.Tensor(XLI).cuda(), torch.Tensor(Mask).cuda()


# Generator: Using NGswin as the generator model
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = NGswin()  # Assuming NGswin takes 3-channel input images

    def forward(self, input):
        return self.main(input)



def main():
    print('Loading model ...\n')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_path = './training_checkpoints/checkpoint.pth'
    netG = Generator(ngpu=1).to(device)  # Ensure Generator class is defined
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        print("Checkpoint loaded successfully!")
    else:
        print("No checkpoint found. Exiting.")
        exit()
    
    # Print keys to debug
    #print("Checkpoint keys:", checkpoint.keys())

    # Try to load only the relevant state_dict
    #if 'netG_state_dict' in checkpoint:
       # netG.load_state_dict(checkpoint['netG_state_dict'], strict=False)
   # elif 'model_state_dict' in checkpoint:
        #netG.load_state_dict(checkpoint['model_state_dict'], strict=False)
    #else:
        #netG.load_state_dict(checkpoint)  # Fallback if it's just weights

    netG.eval()
    
    print('--------------load---------------all----------------nii-------------')
    allXma, allXLI, allM, allSma, allSLI, allTr, allaffine, allfilename = clinic_input_data(opt.data_path)
    breakpoint()
    index = 0  # pick the first patient
    slice_num = 50  # pick a mid-slice or any relevant slice

    Xma_slice = allXma[index][:,:,slice_num]
    XLI_slice = allXLI[index][:,:,slice_num]
    M_slice = allM[index][:,:,slice_num]
    #Sma_slice = allSma[index][:,:,slice_num]
    #SLI_slice = allSLI[index][:,:,slice_num]
    #Tr_slice = allTr[index][:,:,slice_num]

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    axs[0, 0].imshow(Xma_slice, cmap='gray')
    axs[0, 0].set_title('Metal-affected CT (Xma)')
    axs[0, 1].imshow(XLI_slice, cmap='gray')
    axs[0, 1].set_title('LI-corrected CT (XLI)')
    axs[0, 2].imshow(M_slice, cmap='gray')
    axs[0, 2].set_title('Metal Mask (M)')

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


    breakpoint()

    print('--------------test---------------all----------------nii-------------')
    for vol_idx in range(len(allXma)):
        print('test %d th volume.......' % vol_idx)
        num_s = allXma[vol_idx].shape[2]
        breakpoint()
        pre_Xout = np.zeros_like(allXma[vol_idx])
        pre_name = allfilename[vol_idx]
        
        for slice_idx in range(num_s):
            Xma, XLI, M = test_image(allXma, allXLI, allM, vol_idx, slice_idx)
           
            with torch.no_grad():
                if opt.use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()
                ListX= netG(XLI)
            
            Xout = (ListX + 1) / 2
            pre_Xout[..., slice_idx] = Xout.data.cpu().numpy().squeeze()
        
        nibabel.save(nibabel.Nifti1Image(pre_Xout, allaffine[vol_idx]), os.path.join(Pred_nii, pre_name))

if __name__ == "__main__":
    main()
