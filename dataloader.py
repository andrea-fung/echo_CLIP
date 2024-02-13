#import os
from os.path import join
from random import randint
from typing import List, Dict, Union#, Optional, Callable, Iterable
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose
#from torchvision.transforms._transforms_video import RandomResizedCropVideo, RandomHorizontalFlipVideo
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip
import warnings
from random import lognormvariate
from random import seed
import torch.nn as nn
import random
from utils import load_as_data, preprocess_as_data, fix_leakage

seed(42)
torch.random.manual_seed(42)
np.random.seed(42)

img_path_dataset = '/data/workspace/andrea/as_tom_annotations-all.csv'
tab_path_dataset = '/data/workspace/andrea/finetuned_df.csv'
dataset_root = r"/data/workspace/andrea/as_tom" #r"/workspace/as_tom"
cine_loader = 'mat_loader'

# filter out pytorch user warnings for upsampling behaviour
warnings.filterwarnings("ignore", category=UserWarning)

# for now, this only gets the matlab array loader
# Dict is a lookup table and can be expanded to add mpeg loader, etc
# returns a function
def get_loader(loader_name):
    loader_lookup_table = {'mat_loader': mat_loader}
    return loader_lookup_table[loader_name]

def mat_loader(path):
    mat = loadmat(path)
    if 'cine' in mat.keys():    
        return loadmat(path)['cine']
    if 'cropped' in mat.keys():    
        return loadmat(path)['cropped']


label_schemes: Dict[str, Dict[str, Union[int, float]]] = {
    'binary': {'normal': 0.0, 'mild': 1.0, 'moderate': 1.0, 'severe': 1.0},
    'all': {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3},
    'not_severe': {'normal': 0, 'mild': 0, 'moderate': 0, 'severe': 1},
    'as_only': {'mild': 0, 'moderate': 1, 'severe': 2},
    'mild_moderate': {'mild': 0, 'moderate': 1},
    'moderate_severe': {'moderate': 0, 'severe': 1}
}
class_labels: Dict[str, List[str]] = {
    'binary': ['Normal', 'AS'],
    'all': ['Normal', 'Mild', 'Moderate', 'Severe'],
    'not_severe': ['Not Severe', 'Severe'],
    'as_only': ['mild', 'moderate', 'severe'],
    'mild_moderate': ['mild', 'moderate'],
    'moderate_severe': ['moderate', 'severe']
}

    
def get_video_dataloader(args, split, mode):
    '''
    Uses the configuration dictionary to instantiate AS dataloaders

    Parameters
    ----------
    config : Configuration dictionary
        follows the format of get_config.py
    split : string, 'train'/'val'/'test' for which section to obtain
    mode : string, 'train'/'val'/'test' for setting augmentation/metadata ops

    Returns
    -------
    Training, validation or test dataloader with data arranged according to
    pre-determined splits

    '''
    
    if mode=='train':
        flip=args.flip_rate 
        tra = True
        show_info = False
    elif mode=='val':
        flip = 0.0
        tra = False
        show_info = False
    elif mode=='test':
        flip = 0.0
        tra = False
        show_info = True
        
    fr = 16
    bsize = 1
    
    # read in the data directory CSV as a pandas dataframe
    raw_dataset = pd.read_csv(img_path_dataset)
    dataset = pd.read_csv(img_path_dataset)
        
    # append dataset root to each path in the dataframe
    dataset['path'] = dataset['path'].map(lambda x: join(dataset_root, x))
    view = args.view
        
    if view in ('plax', 'psax'):
        dataset = dataset[dataset['view'] == view]
    elif view != 'all':
        raise ValueError(f'View should be plax, psax or all, got {view}')
       
    # remove unnecessary columns in 'as_label' based on label scheme
    label_scheme_name = args.label_scheme_name
    scheme = label_schemes[label_scheme_name]
    dataset = dataset[dataset['as_label'].isin( scheme.keys() )]

    #load tabular dataset
    tab_train, tab_val, tab_test = load_as_data(csv_path = tab_path_dataset,
                                                drop_cols = args.drop_cols,
                                                num_ex = None,
                                                scale_feats = args.scale_feats)
                                                

    #perform imputation 
    train_set, val_set, test_set, all_cols = preprocess_as_data(tab_train, tab_val, tab_test, args.categorical_cols)
    
    # Take train/test/val
    if split in ('train', 'val', 'test'):
        dataset = dataset[dataset['split'] == split]
        if split=='train':
            tab_dataset = train_set
        elif split=='val':
            tab_dataset = val_set
        elif split=='test':
            tab_dataset = test_set
    else:
        raise ValueError(f'View should be train/val/test, got {split}')
    
    #Fix data leakage 
    if split == 'train':
        dataset = fix_leakage(df=raw_dataset, df_subset=dataset, split=split)
    elif split == 'val':
        dataset = fix_leakage(df=raw_dataset, df_subset=dataset, split=split)
    elif split == 'test':
        dataset = fix_leakage(df=raw_dataset, df_subset=dataset, split=split)
        
    dset = AorticStenosisDataset(video=True,
                                img_path_dataset=dataset, 
                                tab_dataset=tab_dataset,
                                split=split,
                                transform=tra,
                                normalize=True,
                                frames=fr,
                                return_info=show_info,
                                flip_rate=flip,
                                label_scheme = scheme)
    
    if mode=='train':
        if args.sampler == 'AS':
            sampler_AS = dset.class_samplers()
            loader = DataLoader(dset, batch_size=bsize, sampler=sampler_AS, num_workers=args.num_workers)
        else: # random sampling
            loader = DataLoader(dset, batch_size=bsize, shuffle=True, num_workers=args.num_workers)
    else:
        loader = DataLoader(dset, batch_size=bsize, shuffle=True, num_workers=args.num_workers)
    return loader

def get_img_dataloader(args, split, mode='train'):
    if mode=='train':
        flip=args.flip_rate 
        tra = True
        show_info = False
    elif mode=='val':
        flip = 0.0
        tra = False
        show_info = False
    elif mode=='test':
        flip = 0.0
        tra = False
        show_info = True
        
    fr = 16
    bsize = args.batch_size
    
    # read in the data directory CSV as a pandas dataframe
    raw_dataset = pd.read_csv(img_path_dataset)
    dataset = pd.read_csv(img_path_dataset)
        
    # append dataset root to each path in the dataframe
    dataset['path'] = dataset['path'].map(lambda x: join(dataset_root, x))
    view = args.view
        
    if view in ('plax', 'psax'):
        dataset = dataset[dataset['view'] == view]
    elif view != 'all':
        raise ValueError(f'View should be plax, psax or all, got {view}')
       
    # remove unnecessary columns in 'as_label' based on label scheme
    label_scheme_name = args.label_scheme_name
    scheme = label_schemes[label_scheme_name]
    dataset = dataset[dataset['as_label'].isin( scheme.keys() )]

    #load tabular dataset
    tab_train, tab_val, tab_test = load_as_data(csv_path = tab_path_dataset,
                                                drop_cols = args.drop_cols,
                                                num_ex = None,
                                                scale_feats = args.scale_feats)
                                                

    #perform imputation 
    train_set, val_set, test_set, all_cols = preprocess_as_data(tab_train, tab_val, tab_test, args.categorical_cols)
    
    # Take train/test/val
    if split in ('train', 'val', 'test'):
        dataset = dataset[dataset['split'] == split]
        if split=='train':
            tab_dataset = train_set
        elif split=='val':
            tab_dataset = val_set
        elif split=='test':
            tab_dataset = test_set
    else:
        raise ValueError(f'View should be train/val/test, got {split}')
    
    #Fix data leakage 
    if split == 'train':
        dataset = fix_leakage(df=raw_dataset, df_subset=dataset, split=split)
    elif split == 'val':
        dataset = fix_leakage(df=raw_dataset, df_subset=dataset, split=split)
    elif split == 'test':
        dataset = fix_leakage(df=raw_dataset, df_subset=dataset, split=split)
    
    dset = AorticStenosisDataset(video=False,
                                img_path_dataset=dataset, 
                                tab_dataset=tab_dataset,
                                split=split,
                                transform=tra,
                                normalize=True,
                                frames=fr,
                                return_info=show_info,
                                flip_rate=flip,
                                label_scheme = scheme)
    
    if mode=='train':
        if args.sampler == 'AS':
            sampler_AS = dset.class_samplers()
            loader = DataLoader(dset, batch_size=bsize, sampler=sampler_AS, num_workers=args.num_workers)
        else: # random sampling
            loader = DataLoader(dset, batch_size=bsize, shuffle=True, num_workers=args.num_workers)
    else:
        loader = DataLoader(dset, batch_size=bsize, shuffle=True, num_workers=args.num_workers)
    return loader


class AorticStenosisDataset(Dataset):
    def __init__(self, 
                 video,
                 label_scheme,
                 img_path_dataset,
                 tab_dataset,
                 split: str = 'train',
                 transform: bool = True, normalize: bool = True, 
                 frames: int = 16, resolution: int = 224,
                 cine_loader: str = 'mat_loader', return_info: bool = False, 
                 contrastive_method: str = 'CE',
                 flip_rate: float = 0.3, min_crop_ratio: float = 0.8, 
                 hr_mean: float = 4.237, hr_std: float = 0.1885,
                 **kwargs):

        self.video = video
        self.return_info = return_info
        self.hr_mean = hr_mean
        self.hr_srd = hr_std
        self.scheme = label_scheme
        self.cine_loader = get_loader(cine_loader)
        self.dataset = img_path_dataset
        self.tab_dataset = tab_dataset
        self.frames = frames
        self.resolution = (resolution, resolution)
        self.split = split
        self.transform = None
        self.transform_contrastive = None
        if transform:
            self.transform = Compose(
                [RandomResizedCrop(size=self.resolution, scale=(min_crop_ratio, 1)),
                 RandomHorizontalFlip(p=flip_rate)]
            )
            
        self.normalize = normalize
        self.contrstive = contrastive_method

    def class_samplers(self):
        # returns WeightedRandomSamplers
        # based on the frequency of the class occurring
        
        # storing labels as a dictionary will be in a future update
        labels_AS = np.array(self.dataset['as_label'])  
        labels_AS = np.array([self.scheme[t] for t in labels_AS])
        class_sample_count_AS = np.array([len(np.where(labels_AS == t)[0]) 
                                          for t in np.unique(labels_AS)])
        weight_AS = 1. / class_sample_count_AS
        if len(weight_AS) != 4:
            weight_AS = np.insert(weight_AS,0,0)
        samples_weight_AS = np.array([weight_AS[t] for t in labels_AS])
        samples_weight_AS = torch.from_numpy(samples_weight_AS).double()
        sampler_AS = WeightedRandomSampler(samples_weight_AS, len(samples_weight_AS))
        
        return sampler_AS
        

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def get_random_interval(vid, length):
        length = int(length)
        start = randint(0, max(0, len(vid) - length))
        return vid[start:start + length]
    
    # expands one channel to 3 color channels, useful for some pretrained nets
    @staticmethod
    def gray_to_gray3(in_tensor):
        # in_tensor is 1xTxHxW
        return in_tensor.expand(-1, 3, -1, -1)
    
    # normalizes pixels based on pre-computed mean/std values
    @staticmethod
    def bin_to_norm(in_tensor):
        # in_tensor is 1xTxHxW
        m = 0.099
        std = 0.171
        return (in_tensor-m)/std
    
    def tab_to_text(self, tab_numpy):
        #text = ["%.2f" % num for num in tab_numpy.tolist()]
        text = ','.join(['{:.2f}'.format(num) for num in tab_numpy])
        return text

    def __getitem__(self, item):
        data_info = self.dataset.iloc[item]

        #get associated tabular data based on echo ID
        study_num = data_info['Echo ID#']
        tab_info = self.tab_dataset.loc[int(study_num)]
        #tab_info = torch.tensor(tab_info.values, dtype=torch.float32)

        #turn tabular data into text
        tab_info = self.tab_to_text(tab_info.values)

        cine_original = self.cine_loader(data_info['path'])
            
        window_length = 60000 / (lognormvariate(self.hr_mean, self.hr_srd) * data_info['frame_time'])
        cine = self.get_random_interval(cine_original, window_length)

        if self.video:
            cine = resize(cine, (32, *self.resolution)) 
        else:
            frame_choice = np.random.randint(0, cine.shape[0], 1)
            cine = cine[frame_choice, :, :]
            cine = resize(cine, (1, *self.resolution)) 

        cine = torch.tensor(cine).unsqueeze(1) #[f, c, h, w]
        
        # storing labels as a dictionary will be in a future update        
        labels_AS = torch.tensor(self.scheme[data_info['as_label']])

        if self.transform:
            if self.contrstive == 'CE' or self.contrstive == 'Linear':
                cine = self.transform(cine)
            else:
                cine_org = self.transform(cine)
                cine_aug = self.transform_contrastive(cine)
                cine = cine_org
                if random.random() < 0.4:
                    upsample = nn.Upsample(size=(16,224, 224), mode='nearest')
                    cine_aug = cine_aug[:, :,  0:180, 40:180].unsqueeze(1)
                    cine_aug = upsample(cine_aug).squeeze(1)   
                
        if self.normalize:
            cine = self.bin_to_norm(cine)
            
        cine = self.gray_to_gray3(cine)
        cine = cine.float()
        
        ret = (cine, tab_info, labels_AS)
        
        return ret