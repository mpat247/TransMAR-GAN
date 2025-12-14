import os
import os.path
import numpy as np
import random
import h5py
import torch
import torch.utils.data as udata
from numpy.random import RandomState
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import glob

def image_get_minmax():
    return 0.0, 1.0


def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data * 2.0 - 1.0
    data = data.astype(np.float32)
    data = np.transpose(np.expand_dims(data, 2), (2, 0, 1))
    return data


def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    def _augment(img):
        if hflip: img = img[:, ::-1]
        if vflip: img = img[::-1, :]
        return img
    return [_augment(a) for a in args]

class MARTrainDataset(udata.Dataset):
    def __init__(self, dir, patchSize, length, mask):
        super().__init__()
        self.dir = dir
        self.train_mask = mask
        self.patch_size = patchSize
        self.sample_num = length
        
        # Scan for actual gt.h5 files instead of using txt
        import glob
        gt_pattern = os.path.join(self.dir, 'train_640geo', '*', '*', 'gt.h5')
        gt_files = glob.glob(gt_pattern)
        
        # Convert to relative paths like txt format
        self.mat_files = []
        for f in gt_files:
            rel_path = f.replace(os.path.join(self.dir, 'train_640geo/'), '')
            self.mat_files.append(rel_path + '\n')
        
        self.file_num = len(self.mat_files)
        print(f"[Dataset] Found {self.file_num} actual training images")
        
        if self.file_num == 0:
            raise ValueError(f"No gt.h5 files found in {self.dir}/train_640geo/")

        self.rand_state = RandomState(66)
        self.start = 0
        self.end = self.file_num  # Use all found files for training
        self.sample_num = length
    
    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        while True:
            try:
                gt_dir = self.mat_files[idx % self.end]
                random_mask = np.random.randint(0, 79)  # 80 total metal masks for training
                file_dir = gt_dir[:-6]
                data_file = file_dir + str(random_mask) + '.h5'
                abs_dir = os.path.join(self.dir, 'train_640geo/', data_file)

                # Check for file existence
                if not os.path.isfile(abs_dir):
                    raise FileNotFoundError(f"File not found: {abs_dir}")

                # Attempt to open the HDF5 file
                with h5py.File(abs_dir, 'r') as file:
                    Xma = file['ma_CT'][()]
                    XLI = file['LI_CT'][()]

                gt_absdir = os.path.join(self.dir, 'train_640geo/', gt_dir[:-1])
                if not os.path.isfile(gt_absdir):
                    raise FileNotFoundError(f"Ground truth file not found: {gt_absdir}")

                with h5py.File(gt_absdir, 'r') as gt_file:
                    Xgt = gt_file['image'][()]

                # Get the metal mask and resize it
                #M512 = self.train_mask[:, :, random_mask]
                #M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
                
                Xgtclip = np.clip(Xgt,0,1)
                B = Xgtclip
                Xmaclip = np.clip(Xma, 0, 1)
                O = Xmaclip
                XLIclip = np.clip(XLI, 0,1)
                LI = XLIclip
                
               
                B = (Xgtclip * 2) - 1
                O = (Xmaclip * 2) - 1
                LI = (XLIclip * 2) - 1
                
                O, row, col = self.crop(O)
        
                B = B[row: row + self.patch_size, col: col + self.patch_size]
        
                LI = LI[row: row + self.patch_size, col: col + self.patch_size]
                
                #M = M[row: row + self.patch_size, col: col + self.patch_size]
                
                O = O.astype(np.float32)
                LI = LI.astype(np.float32)
                B = B.astype(np.float32)
                
                #Mask = M.astype(np.float32)
                
                O, B, LI = augment(O, B, LI)
                
                #B =  np.stack([B] * 3, axis=-1)

                #O = np.stack([O] * 3, axis=-1)
              
                #LI = np.stack([LI] * 3, axis=-1)
                
                #O = np.transpose(O , (2, 0, 1)).astype(np.float32)
                #B = np.transpose(B , (2, 0, 1)).astype(np.float32)
                #LI = np.transpose(LI , (2, 0, 1)).astype(np.float32)
        
        
                O = np.transpose(np.expand_dims(O, 2), (2, 0, 1))
                B = np.transpose(np.expand_dims(B, 2), (2, 0, 1))
                LI = np.transpose(np.expand_dims(LI, 2), (2, 0, 1))
            
                
                #Mask = np.transpose(np.expand_dims(Mask, 2), (2, 0, 1))
                #Mask = np.transpose(np.expand_dims(Mask, 2), (2, 0, 1))
                #non_Mask = 1 - Mask #non_metal region
                return torch.from_numpy(O.copy()), torch.from_numpy(B.copy()), torch.from_numpy(LI.copy())
                
            except (OSError, FileNotFoundError) as e:
                # Skip corrupted or missing files (shouldn't happen with scanned files)
                idx = (idx + 1) % self.end  # Move to the next file

    def crop(self, img):
        h, w = img.shape
        p_h, p_w = self.patch_size, self.patch_size
        if h == p_h:
            r = 0
            c = 0
            O = img
        else:
            r = self.rand_state.randint(0, h - p_h)
            c = self.rand_state.randint(0, w - p_w)
            O = img[r: r + p_h, c: c + p_w]
        return O, r, c

class MARValDataset(udata.Dataset):
    def __init__(self, dir, mask):
        super().__init__()
        self.dir = dir
        self.train_mask = mask
        
        # Scan for actual gt.h5 files in train_640geo
        import glob
        gt_pattern = os.path.join(self.dir, 'train_640geo', '*', '*', 'gt.h5')
        all_files = glob.glob(gt_pattern)
        
        # Use last 10% for validation
        split_idx = int(len(all_files) * 0.9)
        val_files = all_files[split_idx:]
        
        # Convert to relative paths
        self.mat_files = []
        for f in val_files:
            rel_path = f.replace(os.path.join(self.dir, 'train_640geo/'), '')
            self.mat_files.append(rel_path + '\n')
        
        self.file_num = len(self.mat_files)
        print(f"[Dataset] Found {self.file_num} validation images")
        
        self.rand_state = RandomState(66)
        self.sample_num = self.file_num
        
    def __len__(self):
        return int(self.sample_num)

    def __getitem__(self, idx):
        while True:
            try:
                gt_dir = self.mat_files[idx % self.file_num]
                random_mask = np.random.randint(0, 79)
                
                file_dir = gt_dir[:-6]
                data_file = file_dir + str(random_mask) + '.h5'
                abs_dir = os.path.join(self.dir, 'train_640geo/', data_file)

                if not os.path.isfile(abs_dir):
                    raise FileNotFoundError(f"File not found: {abs_dir}")

                with h5py.File(abs_dir, 'r') as file:
                    Xma = file['ma_CT'][()]
                    XLI = file['LI_CT'][()]

                gt_absdir = os.path.join(self.dir, 'train_640geo/', gt_dir[:-1])
                if not os.path.isfile(gt_absdir):
                    raise FileNotFoundError(f"Ground truth file not found: {gt_absdir}")

                with h5py.File(gt_absdir, 'r') as gt_file:
                    Xgt = gt_file['image'][()]

                # Clip and normalize
                Xgtclip = np.clip(Xgt, 0, 1)
                Xmaclip = np.clip(Xma, 0, 1)
                XLIclip = np.clip(XLI, 0, 1)
                
                Xgtclip = (Xgtclip * 2) - 1
                Xmaclip = (Xmaclip * 2) - 1
                XLIclip = (XLIclip * 2) - 1
                
                # Transpose properly
                O = np.transpose(np.expand_dims(Xmaclip, 2), (2, 0, 1)).astype(np.float32)
                B = np.transpose(np.expand_dims(Xgtclip, 2), (2, 0, 1)).astype(np.float32)
                LI = np.transpose(np.expand_dims(XLIclip, 2), (2, 0, 1)).astype(np.float32)

                return torch.from_numpy(O.copy()), torch.from_numpy(B.copy()), torch.from_numpy(LI.copy())

            except (OSError, FileNotFoundError) as e:
                idx = (idx + 1) % self.file_num
                
                
       
class TestDataset(udata.Dataset):
    def __init__(self, dir, mask):
        super().__init__()
        self.dir = dir
        self.test_mask = mask
        self.txtdir = os.path.join(self.dir, 'test_640geo_dir.txt')
        self.mat_files = open(self.txtdir, 'r').readlines()
        self.file_num = len(self.mat_files)
        self.sample_num = self.file_num * 10   # every clean CT image for testing is paired with 10 metal masks.
        self.rand_state = RandomState(66)

    def __len__(self):
        return int(self.sample_num)

    def __getitem__(self, idx):
        gt_dir = self.mat_files[idx % int(self.file_num)]
        random_mask = random.randint(0, 9)
        file_dir = gt_dir[:-6]
        data_file = file_dir + str(random_mask) + '.h5'
        abs_dir = os.path.join(self.dir, 'test_640geo/', data_file)
        gt_absdir = os.path.join(self.dir, 'test_640geo/', gt_dir[:-1])
        gt_file = h5py.File(gt_absdir, 'r')
        Xgt = gt_file['image'][()]
        gt_file.close()
        file = h5py.File(abs_dir, 'r')
        Xma = file['ma_CT'][()]
        XLI = file['LI_CT'][()]
        file.close()
        M512 = self.test_mask[:, :, random_mask]
        M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
       
        Xgtclip = np.clip(Xgt, 0, 1)
        Xmaclip = np.clip(Xma, 0, 1)
        XLIclip = np.clip(XLI, 0, 1)
        
        B = (Xgtclip * 2) - 1
        O = (Xmaclip * 2) - 1
        LI = (XLIclip * 2) - 1
       
        #B =  np.stack([Xgtclip] * 3, axis=-1)
                
        #O = np.stack([Xmaclip] * 3, axis=-1)
              
        #LI = np.stack([XLIclip] * 3, axis=-1)
        
        # Convert to torch tensors
        
        O = np.transpose(np.expand_dims(O, 2), (2, 0, 1))
        B = np.transpose(np.expand_dims(B, 2), (2, 0, 1))
        LI = np.transpose(np.expand_dims(LI, 2), (2, 0, 1))
        Mask = np.transpose(np.expand_dims(M, 2), (2, 0, 1))
        #O = np.transpose(O , (2, 0, 1)).astype(np.float32)
        #B = np.transpose(B , (2, 0, 1)).astype(np.float32)
        #LI = np.transpose(LI , (2, 0, 1)).astype(np.float32)
        
        #Mask = M.astype(np.float32)
        Mask = np.transpose(np.expand_dims(M, 2), (2, 0, 1))
        O = O.astype(np.float32)
        LI = LI.astype(np.float32)
        B = B.astype(np.float32)
        Mask = M.astype(np.float32)
        non_Mask = 1 - Mask
        
        # Return: (ma_CT, Ground_Truth, LI_CT) - consistent with MARTrainDataset
        return torch.from_numpy(O.copy()), torch.from_numpy(B.copy()), torch.from_numpy(LI.copy())


class SpineWebTrainDataset(udata.Dataset):
    def __init__(self, artifact_dir, clean_dir, patchSize, paired=True, hu_range=(-1000, 2000)):
        super().__init__()
        self.artifact_dir = artifact_dir
        self.clean_dir = clean_dir
        self.patch_size = patchSize
        self.paired = paired
        self.hu_range = hu_range

        # Scan directories for .npy files and build basename-matched pairs
        artifact_files = []
        if os.path.exists(artifact_dir):
            for fn in sorted(os.listdir(artifact_dir)):
                if fn.endswith('.npy'):
                    artifact_files.append(os.path.join(artifact_dir, fn))

        clean_files = []
        if os.path.exists(clean_dir):
            for fn in sorted(os.listdir(clean_dir)):
                if fn.endswith('.npy'):
                    clean_files.append(os.path.join(clean_dir, fn))

        if len(artifact_files) == 0:
            raise ValueError(f"No .npy files found in {artifact_dir}")
        if len(clean_files) == 0:
            raise ValueError(f"No .npy files found in {clean_dir}")

        # Match by basename like DatasetSyntheticTransfer
        artifact_map = {os.path.basename(p): p for p in artifact_files}
        clean_map = {os.path.basename(p): p for p in clean_files}

        common_basenames = sorted(set(artifact_map.keys()) & set(clean_map.keys()))
        if not common_basenames:
            raise ValueError(
                f"No matching artifact/clean .npy basenames between {artifact_dir} and {clean_dir}"
            )

        self.artifact_files = [artifact_map[b] for b in common_basenames]
        self.clean_files = [clean_map[b] for b in common_basenames]

        self.sample_num = len(self.artifact_files)
        self.rand_state = RandomState(66)

        print(
            f"SpineWeb Train Dataset: {len(self.artifact_files)} paired images "
            f"(from {artifact_dir} and {clean_dir})"
        )
    
    def __len__(self):
        return self.sample_num
    
    def __getitem__(self, idx):
        while True:
            try:
                # Load artifact image (metal_artifact)
                artifact_file = self.artifact_files[idx % self.sample_num]
                artifact = np.load(artifact_file).astype(np.float32)

                # Load clean image (no_metal) with matched basename
                if self.paired:
                    clean_file = self.clean_files[idx % self.sample_num]
                else:
                    clean_file = self.clean_files[np.random.randint(0, len(self.clean_files))]

                clean = np.load(clean_file).astype(np.float32)
                
                # HU normalization - CRITICAL STEP
                hu_min, hu_max = self.hu_range
                
                # Clip to valid CT HU range
                artifact = np.clip(artifact, hu_min, hu_max)
                clean = np.clip(clean, hu_min, hu_max)
                
                # Normalize to [0, 1]
                artifact = (artifact - hu_min) / (hu_max - hu_min)
                clean = (clean - hu_min) / (hu_max - hu_min)
                
                # Scale to [-1, 1] (matches SynDeepLesion preprocessing)
                artifact = artifact * 2.0 - 1.0
                clean = clean * 2.0 - 1.0
                
                # Random crop to patch_size
                artifact_crop, row, col = self.crop(artifact)
                clean_crop = clean[row: row + self.patch_size, col: col + self.patch_size]
                
                # Convert to float32
                artifact_crop = artifact_crop.astype(np.float32)
                clean_crop = clean_crop.astype(np.float32)
                
                # Data augmentation
                artifact_crop, clean_crop = augment(artifact_crop, clean_crop)
                
                # Add channel dimension [H, W] -> [1, H, W]
                artifact_crop = np.transpose(np.expand_dims(artifact_crop, 2), (2, 0, 1))
                clean_crop = np.transpose(np.expand_dims(clean_crop, 2), (2, 0, 1))
                
                # Return format matching MARTrainDataset
                return torch.from_numpy(artifact_crop.copy()), torch.from_numpy(clean_crop.copy()), torch.from_numpy(artifact_crop.copy())
                
            except (OSError, FileNotFoundError) as e:
                # Skip corrupted or missing files
                print(f"Skipping file due to error: {e}")
                idx = (idx + 1) % self.sample_num
    
    def crop(self, img):
        h, w = img.shape
        p_h, p_w = self.patch_size, self.patch_size
        if h == p_h:
            r = 0
            c = 0
            O = img
        else:
            r = self.rand_state.randint(0, h - p_h)
            c = self.rand_state.randint(0, w - p_w)
            O = img[r: r + p_h, c: c + p_w]
        return O, r, c


class SpineWebTestDataset(udata.Dataset):
    def __init__(self, artifact_dir, clean_dir, hu_range=(-1000, 2000)):
        super().__init__()
        self.artifact_dir = artifact_dir
        self.clean_dir = clean_dir
        self.hu_range = hu_range

        # Scan directories for .npy files and build basename-matched pairs
        artifact_files = []
        if os.path.exists(artifact_dir):
            for fn in sorted(os.listdir(artifact_dir)):
                if fn.endswith('.npy'):
                    artifact_files.append(os.path.join(artifact_dir, fn))

        clean_files = []
        if os.path.exists(clean_dir):
            for fn in sorted(os.listdir(clean_dir)):
                if fn.endswith('.npy'):
                    clean_files.append(os.path.join(clean_dir, fn))

        if len(artifact_files) == 0:
            raise ValueError(f"No .npy files found in {artifact_dir}")
        if len(clean_files) == 0:
            raise ValueError(f"No .npy files found in {clean_dir}")

        artifact_map = {os.path.basename(p): p for p in artifact_files}
        clean_map = {os.path.basename(p): p for p in clean_files}

        common_basenames = sorted(set(artifact_map.keys()) & set(clean_map.keys()))
        if not common_basenames:
            raise ValueError(
                f"No matching artifact/clean .npy basenames between {artifact_dir} and {clean_dir}"
            )

        self.artifact_files = [artifact_map[b] for b in common_basenames]
        self.clean_files = [clean_map[b] for b in common_basenames]

        self.sample_num = len(self.artifact_files)

        print(
            f"SpineWeb Test Dataset: {len(self.artifact_files)} paired images "
            f"(from {artifact_dir} and {clean_dir})"
        )
    
    def __len__(self):
        return self.sample_num
    
    def __getitem__(self, idx):
        # Load artifact image (paired by basename)
        artifact_file = self.artifact_files[idx]
        artifact = np.load(artifact_file).astype(np.float32)
        
        # Load corresponding clean image (paired for testing)
        clean_file = self.clean_files[idx]
        clean = np.load(clean_file).astype(np.float32)
        
        # HU normalization
        hu_min, hu_max = self.hu_range
        
        # Clip to valid CT HU range
        artifact = np.clip(artifact, hu_min, hu_max)
        clean = np.clip(clean, hu_min, hu_max)
        
        # Normalize to [0, 1]
        artifact = (artifact - hu_min) / (hu_max - hu_min)
        clean = (clean - hu_min) / (hu_max - hu_min)
        
        # Scale to [-1, 1]
        artifact = artifact * 2.0 - 1.0
        clean = clean * 2.0 - 1.0
        
        # Convert to float32
        artifact = artifact.astype(np.float32)
        clean = clean.astype(np.float32)
        
        # Add channel dimension [H, W] -> [1, H, W]
        artifact = np.transpose(np.expand_dims(artifact, 2), (2, 0, 1))
        clean = np.transpose(np.expand_dims(clean, 2), (2, 0, 1))
        
        return torch.from_numpy(artifact.copy()), torch.from_numpy(clean.copy()), torch.from_numpy(artifact.copy())
