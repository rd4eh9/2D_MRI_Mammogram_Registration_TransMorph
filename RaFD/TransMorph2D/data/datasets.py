import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt

import numpy as np


class RaFDDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_gray, y_gray = pkload(path)
        #x_gray, y_gray = x_gray[None, ...], y_gray[None, ...]
        # Ensure 2D images
        # --- Ensure grayscale images are 3D: [C, H, W] ---
        if x_gray.ndim == 2:
            x_gray = x_gray[None, ...]  # add channel dim
        if y_gray.ndim == 2:
            y_gray = y_gray[None, ...]
        #x_gray = x_gray if x_gray.ndim == 2 else x_gray.squeeze()
        #y_gray = y_gray if y_gray.ndim == 2 else y_gray.squeeze()

        # --- Convert to torch.Tensor and float ---
        if not isinstance(x_gray, torch.Tensor):
            x_gray = torch.from_numpy(x_gray).float()
        else:
            x_gray = x_gray.float()

        if not isinstance(y_gray, torch.Tensor):
            y_gray = torch.from_numpy(y_gray).float()
        else:
            y_gray = y_gray.float()

        # Apply transforms
        if self.transforms is not None:
            x_gray, y_gray = self.transforms([x_gray, y_gray])
        #plt.figure()
        #plt.imshow(x_gray[0], cmap='gray')
        #plt.show()

        #x = np.ascontiguousarray(x_gray)
        #y = np.ascontiguousarray(y_gray)

        # Ensure channel dimension: [1, H, W]
        '''
        if x_gray.ndim == 2:
            x_gray = x_gray.unsqueeze(0)
        if y_gray.ndim == 2:
            y_gray = y_gray.unsqueeze(0)
        
        if not isinstance(x_gray, torch.Tensor):
            x_gray = torch.from_numpy(x_gray.astype(np.float32))
        if not isinstance(y_gray, torch.Tensor):
            y_gray = torch.from_numpy(y_gray.astype(np.float32))
        '''
        # Stack along channel dimension: [2, H, W]
        x_in = torch.cat([x_gray, y_gray], dim=0)
        #assert x_in.ndim == 3, f"x_in must be 3D, got {x_in.shape}"
        #assert x_in.shape[0] == 2, f"Expected 2 channels, got {x_in.shape[0]}"

        # Target: [1, H, W]
        y_target = y_gray

        return x_in, y_target

        return x, y

    def __len__(self):
        return len(self.paths)


class RaFDInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_gray, y_gray = pkload(path)

        # --- Ensure 3D grayscale: [C, H, W] ---
        if x_gray.ndim == 2:
            x_gray = x_gray[None, ...]
        if y_gray.ndim == 2:
            y_gray = y_gray[None, ...]

        # Apply transforms if defined
        if self.transforms is not None:
            x_gray, y_gray = self.transforms([x_gray, y_gray])

        # --- Convert to torch.Tensor and float ---
        if not isinstance(x_gray, torch.Tensor):
            x_gray = torch.from_numpy(x_gray).float()
        else:
            x_gray = x_gray.float()

        if not isinstance(y_gray, torch.Tensor):
            y_gray = torch.from_numpy(y_gray).float()
        else:
            y_gray = y_gray.float()

        # --- Ensure original images x, y are tensors with channel dim ---
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).unsqueeze(0).contiguous().float()
        else:
            x = x.unsqueeze(0).contiguous().float()

        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).unsqueeze(0).contiguous().float()
        else:
            y = y.unsqueeze(0).contiguous().float()
        '''
        # Convert to torch.Tensor
        x_gray = torch.from_numpy(np.ascontiguousarray(x_gray.astype(np.float32)))
        y_gray = torch.from_numpy(np.ascontiguousarray(y_gray.astype(np.float32)))

        # Original images (optional, for visualization)
        x = x.unsqueeze(0).contiguous().float()  # add channel dim, contiguous, convert to float32
        y = y.unsqueeze(0).contiguous().float()  # same for y
        '''
        #x = torch.from_numpy(np.ascontiguousarray(x[None, ...].astype(np.float32)))
        #y = torch.from_numpy(np.ascontiguousarray(y[None, ...].astype(np.float32)))

        return x, y, x_gray, y_gray
        '''
        x, y = x[None, ...], y[None, ...]
        x_gray, y_gray = x_gray[None, ...], y_gray[None, ...]
        x_gray = np.ascontiguousarray(x_gray.astype(np.float32))
        y_gray = np.ascontiguousarray(y_gray.astype(np.float32))
        x = np.ascontiguousarray(x.astype(np.float32))
        y = np.ascontiguousarray(y.astype(np.float32))
        x_gray, y_gray = torch.from_numpy(x_gray), torch.from_numpy(y_gray)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        '''


    def __len__(self):
        return len(self.paths)