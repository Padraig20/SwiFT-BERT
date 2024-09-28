# 4D_fMRI_Transformer
import os
import torch
from torch.utils.data import Dataset

# import augmentations #commented out because of cv errors
import numpy as np
import torchio as tio
import nibabel as nb
import random

import glob

class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__()      
        self.register_args(**kwargs)
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride = max(round(self.stride_between_seq * self.sample_duration),1)
        self.data = self._set_data(self.root, self.subject_dict)

        transforms_dict = {
            tio.RandomMotion(): 0.25,
            tio.RandomBlur(): 0.25,
            tio.RandomNoise(): 0.25,
            tio.RandomGamma(): 0.25,
        } # Using 3 and 1 as probabilities would have the same effect
        self.transform = tio.Compose([
            tio.RandomAffine(),
            tio.OneOf(transforms_dict),
        ])
    
    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs
    
    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None): 
        y = []
        if self.shuffle_time_sequence: # shuffle whole sequences
            load_fnames = [f'frame_{frame}.pt' for frame in random.sample(list(range(0,num_frames)),sample_duration//self.stride_within_seq)]
        else:
            load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration,self.stride_within_seq)]
            
        if self.with_voxel_norm:
            load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
                
        for fname in load_fnames:
            img_path = os.path.join(subject_path, fname)
            y_i = torch.load(img_path).unsqueeze(0)
            y.append(y_i)
        y = torch.cat(y, dim=4)
        return y

    def __len__(self):
        return  len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError("Required function")

    def _set_data(self, root, subject_dict):
        raise NotImplementedError("Required function")

class ABCD(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            # subject_name = subject[4:]
            
            subject_path = os.path.join(img_root, 'sub-'+subject_name)

            num_frames = len(glob.glob(os.path.join(subject_path,'frame_*'))) # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.sample_duration, num_frames, target, sex) #? stride or sample_duration?
                data.append(data_tuple)
                        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]

        # resting or task
        y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)
        if self.downstream_task == 'tfMRI_3D':
            target = torch.tensor(nb.load(target).get_fdata()) # 3D tensor
            background_value = target.flatten()[0]
            target = torch.nn.functional.pad(target, (6, -5, -11, -10, -1, -2), value=background_value) # 96, 96, 96

        background_value = y.flatten()[0]
        y = y.permute(0,4,1,2,3)
        if self.input_type == 'rest':
            # ABCD rest image shape: 79, 97, 85
            # latest version might be 96,96,95
            y = torch.nn.functional.pad(y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
        elif self.input_type == 'task':
            # ABCD task image shape: 96, 96, 95
            # background value = 0
            # minmax scaled in brain (0~1)
            y = torch.nn.functional.pad(y, (0, 1, 0, 0, 0, 0), value=background_value) # adjust this padding level according to your data
        y = y.permute(0,2,3,4,1)

        if self.time_as_channel:
            y = y.permute(0,4,1,2,3).squeeze()

        return {
            "fmri_sequence": y,
            "subject_name": subject_name,
            "target": target,
            "TR": start_frame,
            "sex": sex,
        } 
        

class UKB(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        if self.use_ic:
            for i, subject_name in enumerate(subject_dict):
                sex, target = subject_dict[subject_name]
                subject20227 = str(subject_name)+f'_20227_2_0_features_{self.sequence_length}_comps.npy'
                subject_path = os.path.join(self.input_features_path, subject20227)
                input_mask_path = self.input_mask_path
                start_frame = 0
                data_tuple = (i, subject_name, subject_path, start_frame, target, sex, input_mask_path) 
                data.append(data_tuple)
            # train dataset
            # for regression tasks
            if self.train: 
                self.target_values = np.array([tup[4] for tup in data]).reshape(-1, 1)
        
        else:
            img_root = os.path.join(root, 'img')

            for i, subject_name in enumerate(subject_dict):
                sex, target = subject_dict[subject_name]
                subject20227 = str(subject_name)+'_20227_2_0' # only uses release 2
                subject_path = os.path.join(img_root, subject20227)
                num_frames = len(glob.glob(os.path.join(subject_path,'frame_*'))) # voxel mean & std
                session_duration = num_frames - self.sample_duration + 1

                for start_frame in range(0, session_duration, self.stride):
                    data_tuple = (i, subject_name, subject_path, start_frame, self.sample_duration, num_frames, target, sex)
                    data.append(data_tuple)

            # train dataset
            # for regression tasks
            if self.train: 
                self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        if self.use_ic:
            _, subject_name, subject_path, start_frame, target, sex, input_mask_path = self.data[index]
            features = np.load(subject_path) # ic * features
            mask = nb.load(input_mask_path).get_fdata() 
            non_zero_indices = np.nonzero(mask.flatten())[0]
            num_ic = features.shape[0]
            final_matrix = np.zeros((num_ic,) + mask.shape)
            flat_final_matrix = final_matrix.reshape(num_ic, -1)
            flat_final_matrix[:, non_zero_indices] = features
            final_matrix = flat_final_matrix.reshape((num_ic,) + mask.shape)
            y = torch.tensor(final_matrix)
            
            # padding
            background_value = y.flatten()[0]
            y = torch.nn.functional.pad(y, (3, 2, -7, -6, 3, 2), value=background_value) # ic, 96, 96, 96
            y = y.permute(1,2,3,0).unsqueeze(0).half() # 1, 96, 96, 96, ic
            
            if self.downstream_task == 'tfMRI_3D':
                target = torch.tensor(nb.load(target).get_fdata()) # 3D tensor
                background_value = target.flatten()[0]
                target = torch.nn.functional.pad(target, (3, 2, -7, -6, 3, 2), value=background_value)
            
            if self.time_as_channel:
                y = y.permute(0,4,1,2,3).squeeze()
            
        else:
            _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            if self.downstream_task == 'tfMRI_3D':
                target = torch.tensor(nb.load(target).get_fdata()) # 3D tensor
                background_value = target.flatten()[0]
                target = torch.nn.functional.pad(target, (3, 2, -7, -6, 3, 2), value=background_value) # 96, 96, 96

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3) 
            y = torch.nn.functional.pad(y, (3, 9, 0, 0, 10, 8), value=background_value) # adjust this padding level according to your data 
            y = y.permute(0,2,3,4,1) 

            if self.time_as_channel:
                y = y.permute(0,4,1,2,3).squeeze()

        return {
                    "fmri_sequence": y,
                    "subject_name": subject_name,
                    "target": target,
                    "TR": start_frame,
                    "sex": sex,
                } 
        
class HBN(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        
        img_root = os.path.join(root, 'img') 

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, subject_name)
            num_frames = len(glob.glob(os.path.join(subject_path,'frame_*'))) # voxel mean & std
            #num_frames = 250 # TODO remove this line, just for testing
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                if self.train and self.sliding_window_overlap:
                    start_frame = max(0, start_frame - self.sliding_window_overlap)
                data_tuple = (i, subject_name, subject_path, start_frame, self.sample_duration, num_frames, target[start_frame:min(start_frame+self.sample_duration,num_frames)], sex)
                if self.train:
                    print(start_frame, min(start_frame+self.sample_duration,num_frames), target[start_frame:min(start_frame+self.sample_duration,num_frames)])
                data.append(data_tuple)

        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

        background_value = y.flatten()[0]
        y = y.permute(0,4,1,2,3) 
        y = torch.nn.functional.pad(y, (7, 8, 0, 1, 7, 8), value=background_value) # adjust this padding level according to your data 
        y = y.permute(0,2,3,4,1) 

        if self.time_as_channel:
            y = y.permute(0,4,1,2,3).squeeze()

        return {
                    "fmri_sequence": y,
                    "subject_name": subject_name,
                    "target": target,
                    "TR": start_frame,
                    "sex": sex,
                } 
    
class Dummy(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, total_samples=100000) # 100000000 for swifun
        

    def _set_data(self, root, subject_dict):
        data = []
        for k in range(0,self.total_samples):
            data.append((k, 'subj'+ str(k), 'path'+ str(k), self.stride))
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([val for val in range(len(data))]).reshape(-1, 1)
            
        return data

    def __len__(self):
        return self.total_samples

    def __getitem__(self,idx):
        _, subj, _, sequence_length = self.data[idx]
        y = torch.randn(( 1, 96, 96, 96, sequence_length),dtype=torch.float16) #self.y[seq_idx]
        sex = torch.randint(0,2,(1,)).float()
        target = torch.randint(0,2,(1,)).float()
        
        return {
                "fmri_sequence": y,
                "subject_name": subj,
                "target": target,
                "TR": 0,
                "sex": sex,
            } 