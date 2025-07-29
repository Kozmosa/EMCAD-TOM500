import os
import random
import h5py
import numpy as np
import torch
import cv2
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import torch.nn.functional as F


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            # 使用PyTorch的插值函数替代scipy.zoom，速度更快
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
            label_tensor = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()
            
            image_tensor = F.interpolate(image_tensor, size=self.output_size, mode='bilinear', align_corners=False)
            label_tensor = F.interpolate(label_tensor, size=self.output_size, mode='nearest')
            
            image = image_tensor.squeeze().numpy()
            label = label_tensor.squeeze().numpy()
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class TOM500_dataset_optimized(Dataset):
    def __init__(self, base_dir, list_dir, split, nclass=9, transform=None, cache_data=False):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.nclass = nclass
        self.cache_data = cache_data
        self.cached_data = {}
        
        # 预加载小数据集到内存中以加速训练
        if cache_data and len(self.sample_list) < 1000:  # 只有数据集较小时才缓存
            print(f"Caching {len(self.sample_list)} samples to memory...")
            for idx in range(len(self.sample_list)):
                self._load_data(idx, cache=True)
            print("Data cached successfully!")

    def __len__(self):
        return len(self.sample_list)
    
    def _load_data(self, idx, cache=False):
        if cache and idx in self.cached_data:
            return self.cached_data[idx]
            
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        result = {'image': image, 'label': label, 'case_name': self.sample_list[idx].strip('\n')}
        
        if cache:
            self.cached_data[idx] = result
            
        return result

    def __getitem__(self, idx):
        sample = self._load_data(idx, cache=self.cache_data)
        
        # 创建副本以避免修改缓存的数据
        if self.cache_data:
            sample = {
                'image': sample['image'].copy(),
                'label': sample['label'].copy(),
                'case_name': sample['case_name']
            }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
