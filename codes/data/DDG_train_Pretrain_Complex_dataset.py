import random
import numpy as np
from PIL import Image as Image
import torch
import torch.utils.data as data
# import data.util as util
# import data.bsrgan_degradation_transfer as bsrgan_degradation
import os
from torchvision import transforms
import sys
sys.path.append('/zs/DegAE_DegradationAutoencoder')
from codes.utils.file_client import FileClient
from codes.utils.image_utils import crop_img
# imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP
from codes.utils.img_util import imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP
from codes.utils.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
import cv2
import torch.nn.functional as F

class DDGTrainDataset(data.Dataset):
    def __init__(self, opt):
        super(DDGTrainDataset, self).__init__()
        self.opt = opt
        self.dataroot_LQ1 = opt['dataroot_LQ1']
        self.dataroot_HQ1 = opt['dataroot_HQ1']
        self.dataroot_LQ2 = opt['dataroot_LQ2']
        self.dataroot_HQ2 = opt['dataroot_HQ2']
        self.LQ1_list = sorted(os.listdir(self.dataroot_LQ1))
        self.HQ1_list = sorted(os.listdir(self.dataroot_HQ1))
        self.LQ2_list = sorted(os.listdir(self.dataroot_LQ2))
        self.HQ2_list = sorted(os.listdir(self.dataroot_HQ2))
        self.transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        LQ1_path = os.path.join(self.dataroot_LQ1, self.LQ1_list[index])
        HQ1_path = os.path.join(self.dataroot_HQ1, self.HQ1_list[index])
        LQ2_path = os.path.join(self.dataroot_LQ2, self.LQ2_list[index])
        HQ2_path = os.path.join(self.dataroot_HQ2, self.HQ2_list[index])

        LQ1 = self.transform(Image.open(LQ1_path))
        HQ1 = self.transform(Image.open(HQ1_path))
        LQ2 = self.transform(Image.open(LQ2_path))
        HQ2 = self.transform(Image.open(HQ2_path))

        input_deg_type = 1  # 输入退化类型
        target_deg_type = 2  # 目标退化类型

        return {
            'LQ1': LQ1,
            'LQ2': LQ2,
            'HQ1': HQ1,
            'HQ2': HQ2,
            'input_deg_type': input_deg_type,
            'target_deg_type': target_deg_type,
            'img_name': self.HQ1_list[index]
        }
            

    def __len__(self):
        return len(self.LQ1_list)


class MultiDDGTrainDataset2(data.Dataset):
    def __init__(self, opt):
        super(MultiDDGTrainDataset2, self).__init__()
        self.opt = opt
        self.dataroot = opt['dataroot']
        self.deg_list = ['fog', 'rain', 'snow', 'brightness', 'motion_blur', 'compression', 'gauss_noise']
        self.degdict = {
            'fog':0, 'rain':1, 'snow':2, 'brightness':3, 'motion_blur':4, 'compression':5, 'gauss_noise':6
        }
        self.path = {}
        for deg in self.deg_list:
            self.path[deg] = [os.path.join(self.dataroot, deg + '1'), os.path.join(self.dataroot, deg + '2')]

        self.dataroot_HQ1 = os.path.join(self.dataroot, 'HQ1')
        self.dataroot_HQ2 = os.path.join(self.dataroot, 'HQ2')
        self.HQ1_list = []
        self.HQ2_list = []
        self.LQ1_list = []
        self.LQ2_list = []
        self.deg1_list = []
        self.deg2_list = []
        for deg, paths in self.path.items():
            LQ1_namelist = sorted(os.listdir(paths[0]))
            LQ2_namelist = sorted(os.listdir(paths[1]))
            # print(paths[0], paths[1])
            for name1, name2 in zip(LQ1_namelist, LQ2_namelist):
                path1 = os.path.join(paths[0], name1)
                path2 = os.path.join(paths[1], name2)
                self.LQ1_list.append(path1)
                self.LQ2_list.append(path2)
                self.deg1_list.append(self.degdict[deg])
                self.deg2_list.append(self.degdict[deg])
            self.HQ1_list.extend(sorted(os.listdir(self.dataroot_HQ1)))
            self.HQ2_list.extend(sorted(os.listdir(self.dataroot_HQ2)))

        print(len(self.LQ1_list), len(self.HQ1_list))
        
        self.transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        LQ1_path = self.LQ1_list[index]
        HQ1_path = os.path.join(self.dataroot_HQ1, self.HQ1_list[index])
        LQ2_path = self.LQ2_list[index]
        HQ2_path = os.path.join(self.dataroot_HQ2, self.HQ2_list[index])

        LQ1 = self.transform(Image.open(LQ1_path))
        HQ1 = self.transform(Image.open(HQ1_path))
        LQ2 = self.transform(Image.open(LQ2_path))
        HQ2 = self.transform(Image.open(HQ2_path))
        deg1 = self.deg1_list[index]
        deg2 = self.deg2_list[index]
        deg_type = ''
        if index <= 120:
            deg_type = self.deg_list[(index // 20)]

        return {
            'LQ1': LQ1,
            'LQ2': LQ2,
            'HQ1': HQ1,
            'HQ2': HQ2,
            'deg1': deg1,
            'deg2': deg2,
            'input_deg_type': deg_type,
            'target_deg_type': deg_type,
            'img_name': self.HQ1_list[index]
        }


    def __len__(self):
        return len(self.LQ1_list)
    
class MultiDDGTrainDataset3(data.Dataset):
    def __init__(self, opt):
        super(MultiDDGTrainDataset3, self).__init__()
        self.opt = opt
        dataroot = "/zs/Data/AllInOneDataset"
        type = opt['dir']
        gtdataroot = [os.path.join(os.path.join(dataroot, name), f"{type}/gt") for name in sorted(os.listdir(dataroot))]
        lqdateroot = [os.path.join(os.path.join(dataroot, name), f"{type}/input") for name in sorted(os.listdir(dataroot))]
        self.gt_imgs_paths = []
        self.lq_imgs_paths = []
        for gt in sorted(gtdataroot):
            gt_imgs = [os.path.join(gt, name) for name in sorted(os.listdir(gt))]
            self.gt_imgs_paths.extend(gt_imgs)

        for lq in sorted(lqdateroot):
            lq_imgs = [os.path.join(lq, name) for name in sorted(os.listdir(lq))]
            self.lq_imgs_paths.extend(lq_imgs)

        self.HQ1_list = self.gt_imgs_paths
        self.HQ2_list = self.gt_imgs_paths
        self.LQ1_list = self.lq_imgs_paths 
        self.LQ2_list = self.lq_imgs_paths 
        
        self.transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.classes = {
            'fog':0, 'rain':1, 'snow':2, 'lowlight':3, 'blur':4, 'jpeg':5, 'noise':6
        }

    def __getitem__(self, index):
        LQ1_path = self.LQ1_list[index]
        HQ1_path = self.HQ1_list[index]
        LQ2_path = self.LQ2_list[index]
        HQ2_path = self.HQ2_list[index]

        LQ1 = self.transform(Image.open(LQ1_path))
        HQ1 = self.transform(Image.open(HQ1_path))
        LQ2 = self.transform(Image.open(LQ2_path))
        HQ2 = self.transform(Image.open(HQ2_path))
        deg_type = LQ1_path.split('/')[4]
        deg = self.classes[deg_type]
        deg1 = deg
        deg2 = deg

        HQ1, LQ1 = padding(HQ1, LQ1, 256)
        HQ1, LQ1 = paired_random_crop(HQ1, LQ1, 256, 1)
        HQ2, LQ2 = padding(HQ2, LQ2, 256)
        HQ2, LQ2 = paired_random_crop(HQ2, LQ2, 256, 1)

        return {
            'LQ1': LQ1,
            'LQ2': LQ2,
            'HQ1': HQ1,
            'HQ2': HQ2,
            'deg1': deg1,
            'deg2': deg2,
            'input_deg_type': deg_type,
            'target_deg_type': deg_type,
            'img_name': self.HQ1_list[index]
        }


    def __len__(self):
        return len(self.LQ1_list)
    

class MultiDDGTrainDataset(data.Dataset):
    def __init__(self, opt):
        super(MultiDDGTrainDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        dataroot = "/zs/Data/AllInOneDataset"
        type = opt['dir']
        gtdataroot = [os.path.join(os.path.join(dataroot, name), f"{type}/gt") for name in sorted(os.listdir(dataroot))]
        lqdateroot = [os.path.join(os.path.join(dataroot, name), f"{type}/input") for name in sorted(os.listdir(dataroot))]
        self.gt_imgs_paths = []
        self.lq_imgs_paths = []
        for gt in sorted(gtdataroot):
            gt_imgs = [os.path.join(gt, name) for name in sorted(os.listdir(gt))]
            self.gt_imgs_paths.extend(gt_imgs)

        for lq in sorted(lqdateroot):
            lq_imgs = [os.path.join(lq, name) for name in sorted(os.listdir(lq))]
            self.lq_imgs_paths.extend(lq_imgs)
    
        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

        self.classes = {
            'fog':0, 'rain':1, 'snow':2, 'lowlight':3, 'blur':4, 'jpeg':5, 'noise':6
        }

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = 1
        index = index % len(self.lq_imgs_paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.gt_imgs_paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.lq_imgs_paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
        # else:
        #     img_gt = cv2.resize(img_gt, (256, 256))
        #     img_lq = cv2.resize(img_lq, (256, 256))
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        
        deg_type = lq_path.split('/')[4]
        deg = self.classes[deg_type]

        return {
            'LQ1': img_lq,
            'LQ2': img_lq,
            'HQ1': img_gt,
            'HQ2': img_gt,
            'deg1': deg,
            'deg2': deg,
            'input_deg_type': deg_type,
            'target_deg_type': deg_type,
            'img_name': self.lq_imgs_paths[index]
        }
    

    def __len__(self):
        return len(self.lq_imgs_paths)
  
if __name__ == "__main__":
    opt = {
        'dataroot' : '/zs/Data/recdeg/train',
        "io_backend":{"type": "disk"},
        "dir": "minitest",
        "phase":"test",
        "geometric_augs": "true",
        "gt_size" : 256
    }
    data = MultiDDGTrainDataset(opt)
    d = data[51]
    print(len(data), d['LQ1'].shape, d['HQ1'].shape, d['deg1'], d['input_deg_type'], d['target_deg_type'])