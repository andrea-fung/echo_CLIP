#import os
from os.path import join
from random import randint
from typing import List, Dict, Union#, Optional, Callable, Iterable
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
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

cine_loader = 'mat_loader'

# filter out pytorch user warnings for upsampling behaviour
warnings.filterwarnings("ignore", category=UserWarning)

# for now, this only gets the matlab array loader
# Dict is a lookup table and can be expanded to add mpeg loader, etc
# returns a function

def get_loader(loader_name):
    loader_lookup_table = {'mat_loader': mat_loader, 'png_loader': png_loader}
    return loader_lookup_table[loader_name]

def mat_loader(path):
    mat = loadmat(path)
    if 'cine' in mat.keys():    
        return loadmat(path)['cine']
    if 'cropped' in mat.keys():    
        return loadmat(path)['cropped']

def png_loader(path):
    return plt.imread(path)

tufts_label_schemes: Dict[str, Dict[str, Union[int, float]]] = {
    'binary': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 1, 'moderate_AS': 1, 'severe_AS': 1},
    'mild_mod': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 1, 'moderate_AS': 1, 'severe_AS': 2},
    'mod_severe': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 1, 'moderate_AS': 2, 'severe_AS': 2},
    'four_class': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 1, 'moderate_AS': 2, 'severe_AS': 3},
    'five_class': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 2, 'moderate_AS': 3, 'severe_AS': 4},
}

view_scheme = {'PLAX':0, 'PSAX':1, 'A2C':2, 'A4C':3, 'A4CorA2CorOther':4}
view_schemes: Dict[str, Dict[str, Union[int, float]]] = {
    'three_class': {'PLAX':0, 'PSAX':1, 'A2C':2, 'A4C':2, 'A4CorA2CorOther':2},
    'four_class': {'PLAX':0, 'PSAX':1, 'A2C':2, 'A4C':2, 'A4CorA2CorOther':3},
    'five_class': {'PLAX':0, 'PSAX':1, 'A2C':2, 'A4C':3, 'A4CorA2CorOther':4},
}

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

    
def get_video_dataloader(config, split, mode):
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
        flip=config['flip_rate'] 
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
    raw_dataset = pd.read_csv(config['img_path_dataset'])
    dataset = raw_dataset.copy()
        
    # append dataset root to each path in the dataframe
    dataset['path'] = dataset['path'].map(lambda x: join(config['dataset_root'], x))
    view = config['view']
        
    if view in ('plax', 'psax'):
        dataset = dataset[dataset['view'] == view]
    elif view != 'all':
        raise ValueError(f'View should be plax, psax or all, got {view}')
       
    # remove unnecessary columns in 'as_label' based on label scheme
    label_scheme_name = config['label_scheme_name']
    scheme = label_schemes[label_scheme_name]
    dataset = dataset[dataset['as_label'].isin( scheme.keys() )]

    #load tabular dataset
    #AS label INCLUDED in text data 
    train_set, val_set, test_set = load_as_data(csv_path = config['report_path_dataset'])
        
    # Take train/test/val
    if split in ('train', 'val', 'test'):
        dataset = dataset[dataset['split'] == split]
        if split=='train':
            report_dataset = train_set
        elif split=='val':
            report_dataset = val_set
        elif split=='test':
            report_dataset = test_set
    else:
        raise ValueError(f'View should be train/val/test, got {split}')
    
    #Fix data leakage 
    if split == 'train':
        dataset = fix_leakage(df=raw_dataset, df_subset=dataset, split=split)
    elif split == 'val':
        dataset = fix_leakage(df=raw_dataset, df_subset=dataset, split=split)
    elif split == 'test':
        dataset = fix_leakage(df=raw_dataset, df_subset=dataset, split=split)
        
    dset = AorticStenosisDataset(config=config,
                                video=config["video_input"],
                                img_path_dataset=dataset, 
                                report_dataset=report_dataset,
                                type_of_training=config["type_of_training"],
                                split=split,
                                transform=tra,
                                normalize=True,
                                frames=fr,
                                return_info=show_info,
                                flip_rate=flip,
                                label_scheme = scheme)
    
    if mode=='train':
        if config['sampler'] == 'AS':
            sampler_AS = dset.class_samplers()
            loader = DataLoader(dset, batch_size=bsize, sampler=sampler_AS, num_workers=config['num_workers'])
        else: # random sampling
            loader = DataLoader(dset, batch_size=bsize, shuffle=True, num_workers=config['num_workers'])
    else:
        loader = DataLoader(dset, batch_size=bsize, shuffle=False, num_workers=config['num_workers'])
    return loader

# def get_img_dataloader(config, split, mode='train'):
#     if mode=='train':
#         flip=config['flip_rate'] 
#         tra = True
#         show_info = False
#     elif mode=='val':
#         flip = 0.0
#         tra = False
#         show_info = False
#     elif mode=='test':
#         flip = 0.0
#         tra = False
#         show_info = True
        
#     fr = 16
#     bsize = config['batch_size']
    
#     # read in the data directory CSV as a pandas dataframe
#     raw_dataset = pd.read_csv(config['img_path_dataset'])
#     dataset = raw_dataset.copy()
        
#     # append dataset root to each path in the dataframe
#     dataset['path'] = dataset['path'].map(lambda x: join(config['dataset_root'], x))
#     view = config['view']
        
#     if view in ('plax', 'psax'):
#         dataset = dataset[dataset['view'] == view]
#     elif view != 'all':
#         raise ValueError(f'View should be plax, psax or all, got {view}')
    
#     # remove unnecessary columns in 'as_label' based on label scheme
#     label_scheme_name = config['label_scheme_name']
#     scheme = label_schemes[label_scheme_name]

#     #load tabular dataset
#     tab_train, tab_val, tab_test = load_as_data(csv_path = config['tab_path_dataset'],
#                                                 drop_cols = config['drop_cols'],
#                                                 num_ex = None,
#                                                 scale_feats = config['scale_feats'])
                                                

#     #perform imputation 
#     if config['tab_preprocess']:
#         train_set, val_set, test_set, all_cols = preprocess_as_data(tab_train, tab_val, tab_test, config['categorical_cols'])
#     else: 
#         train_set = tab_train.drop("as_label", axis=1)
#         val_set = tab_val.drop("as_label", axis=1)
#         test_set = tab_test.drop("as_label", axis=1)
#         all_cols = train_set.columns.to_list()
#     # Take train/test/val
#     if split in ('train', 'val', 'test'):
#         dataset = dataset[dataset['split'] == split]
#         if split=='train':
#             tab_dataset = train_set
#         elif split=='val':
#             tab_dataset = val_set
#         elif split=='test':
#             tab_dataset = test_set
#     else:
#         raise ValueError(f'View should be train/val/test, got {split}')
    
#     #Fix data leakage 
#     if split == 'train':
#         dataset = fix_leakage(df=raw_dataset, df_subset=dataset, split=split)
#     elif split == 'val':
#         dataset = fix_leakage(df=raw_dataset, df_subset=dataset, split=split)
#     elif split == 'test':
#         dataset = fix_leakage(df=raw_dataset, df_subset=dataset, split=split)
    
#     dset = AorticStenosisDataset(args=config,
#                                 video=False,
#                                 img_path_dataset=dataset, 
#                                 tab_dataset=tab_dataset,
#                                 tab_cols=all_cols,
#                                 split=split,
#                                 transform=tra,
#                                 normalize=True,
#                                 frames=fr,
#                                 return_info=show_info,
#                                 flip_rate=flip,
#                                 label_scheme = scheme)
    
#     if mode=='train':
#         if config['sampler'] == 'AS':
#             sampler_AS = dset.class_samplers()
#             loader = DataLoader(dset, batch_size=bsize, sampler=sampler_AS, num_workers=config['num_workers'])
#         else: # random sampling
#             loader = DataLoader(dset, batch_size=bsize, shuffle=True, num_workers=config['num_workers'])
#     else:
#         loader = DataLoader(dset, batch_size=bsize, shuffle=False, num_workers=config['num_workers'])
#     return loader

def get_tufts_dataloader(config, split, mode='train'):
    
    if mode=='train':
        flip = 0.5
        tra = True
        bsize = config['batch_size'] 
        patient_info = False
    elif mode=='val':
        flip = 0.0
        tra = False
        bsize = config['batch_size'] 
        patient_info = False
    elif mode=='test':
        flip = 0.0
        tra = False
        bsize = 1
        patient_info = True
        
    dset = TMEDDataset(args=config,
                       dataset_root=config['tufts_droot'], 
                        split=split,
                        view=config['view'],
                        transform=tra,
                        normalize=True,
                        flip_rate=flip,
                        patient_info = patient_info, 
                        label_scheme_name=config['tufts_label_scheme_name'],
                        view_scheme_name=config['view_scheme_name']
                         )
    
    if mode=='train':
        loader = DataLoader(dset, batch_size=bsize, shuffle=True)
    else:
        loader = DataLoader(dset, batch_size=bsize, shuffle=False)
    return loader


class AorticStenosisDataset(Dataset):
    def __init__(self, 
                 config,
                 video,
                 label_scheme,
                 img_path_dataset,
                 report_dataset,
                 type_of_training,
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
        self.report_dataset = report_dataset
        self.type_of_training = type_of_training
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

    def __getitem__(self, item):
        data_info = self.dataset.iloc[item]

        #get associated tabular data based on echo ID
        study_num = data_info['Echo ID#']
        # storing labels as a dictionary will be in a future update        
        labels_AS = torch.tensor(self.scheme[data_info['as_label']])

        
        if self.type_of_training == "pretraining":
            report_info = self.report_dataset.loc[int(study_num)]
            report_text = report_info['report_text']

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
        
        if self.type_of_training == "pretraining":
            ret = (cine, report_text, labels_AS, study_num)
        else: #finetuning
            ret = (cine, labels_AS, study_num)
        
        return ret

class TMEDDataset(Dataset):
    def __init__(self, 
                 config,
                 dataset_root: str = '~/as',
                 view: str = 'PLAX', # PLAX/PSAX/PLAXPSAX/no_other/all
                 split: str = 'all', # train/val/test/'all'
                 transform: bool = True, 
                 normalize: bool = False, 
                 resolution: int = 224,
                 image_loader: str = 'png_loader', 
                 flip_rate: float = 0.5,  
                 patient_info: bool = False,
                 label_scheme_name: str = 'all', # see above
                 view_scheme_name: str = 'three_class',
                 **kwargs):
        # navigation for linux environment
        self.dataset_root = dataset_root
        
        # read in the data directory CSV as a pandas dataframe
        dataset = pd.read_csv(join(config['tufts_droot'], config['tufts_csv_name'])) 
        # append dataset root to each path in the dataframe
        dataset['path'] = dataset.apply(self.get_data_path_rowwise, axis=1)
        
        if view in ('PLAX', 'PSAX'):
            dataset = dataset[dataset['view_label'] == view]
        elif view == 'plaxpsax':
            dataset = dataset[dataset['view_label'].isin(['PLAX', 'PSAX'])]
        elif view == 'no_other':
            dataset = dataset[dataset['view_label'] != 'A4CorA2CorOther']
        elif view != 'all':
            raise ValueError(f'View should be PLAX/PSAX/PLAXPSAX/no_other/all, got {view}')
       
        # remove unnecessary columns in 'as_label' based on label scheme
        self.scheme = tufts_label_schemes[label_scheme_name]
        self.scheme_view = view_schemes[view_scheme_name]
        dataset = dataset[dataset['diagnosis_label'].isin( self.scheme.keys() )]

        self.image_loader = get_loader(image_loader)
        self.patient_info = patient_info

        # Take train/test/val
        if split in ('train', 'val', 'test'):
            dataset = dataset[dataset['diagnosis_classifier_split'] == split]
        elif split != 'all':
            raise ValueError(f'View should be train/val/test/all, got {split}')

        self.dataset = dataset
        self.resolution = (resolution, resolution)

        self.transform = None
        if transform:
            self.transform = Compose(
                [RandomResizedCrop(size=self.resolution, scale=(0.8, 1)),
                 RandomHorizontalFlip(p=flip_rate)]
            )
        self.normalize = normalize
        
    def __len__(self) -> int:
        return len(self.dataset)

    # get a dataset path from the TMED2 CSV row
    def get_data_path_rowwise(self, pdrow):
        path = join(self.dataset_root, pdrow['SourceFolder'], pdrow['query_key'])
        return path

    def get_random_interval(self, vid, length):
        length = int(length)
        start = randint(0, max(0, len(vid) - length))
        return vid[start:start + length]
    
    # expands one channel to 3 color channels, useful for some pretrained nets
    def gray_to_gray3(self, in_tensor):
        # in_tensor is 1xTxHxW
        return in_tensor.expand(3, -1, -1)
    
    # normalizes pixels based on pre-computed mean/std values
    def bin_to_norm(self, in_tensor):
        # in_tensor is 1xTxHxW
        m = 0.061
        std = 0.140
        return (in_tensor-m)/std
    
    def _get_image(self, data_info):
        '''
        General method to get an image and apply tensor transformation to it

        Parameters
        ----------
        data_info : ID of the item to retrieve (based on the position in the dataset)
            DESCRIPTION.

        Returns
        -------
        ret : size 3xTxHxW tensor representing image
            if return_info is true, also returns metadata

        '''

        img_original = self.image_loader(data_info['path'])
        
        img = resize(img_original, self.resolution) # HxW
        x = torch.tensor(img).unsqueeze(0) # 1xHxW
        
        y_view = torch.tensor(self.scheme_view[data_info['view_label']])
        y_AS = torch.tensor(self.scheme[data_info['diagnosis_label']])

        if self.transform:
            x = self.transform(x)
        if self.normalize:
            x = self.bin_to_norm(x)

        x = self.gray_to_gray3(x)
        x = x.float() # 3xHxW
        
        ret = {'x':x, 'y_AS':y_AS, 'y_view':y_view}
        
        if self.patient_info:
            p_id = data_info['query_key'].split('_')[0]
            ret = {'x':x, 'y_AS':y_AS, 'y_view':y_view, 'p_id': p_id}
        
        return ret

    def __getitem__(self, item):
        data_info = self.dataset.iloc[item]
        return self._get_image(data_info)
    
    def tensorize_single_image(self, img_path):
        """
        Creates a video tensor that is consistent with config specifications
    
        Parameters
        ----------
        img_path : String
            the path to the image, the function will find matches of the path substring
    
        Returns
        -------
        see get_item_from_info
    
        """
        # look for a path in the dataset resembling the video path
        matches_boolean = self.dataset['path'].str.contains(img_path)
        found_entries=self.dataset[matches_boolean]
        if len(found_entries) == 0:
            raise ValueError('Found 0 matches for requested substring ' + img_path)
        elif len(found_entries) > 1:
            warnings.warn('Found multiple matches, returning first result')
        data_info = found_entries.iloc[0]
        return self._get_item_from_info(data_info)