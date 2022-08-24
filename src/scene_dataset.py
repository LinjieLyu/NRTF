import os
import torch
import numpy as np

import utils as utils


class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 data_dir,
                 img_res,
            
                 num_envmap=1, 
                 n_images=64,
                 cam_file=None
                 ):

        self.instance_dir =data_dir
        
        self.total_pixels = img_res[0] * img_res[1]
        print(img_res[0],img_res[1])
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        
        self.num_envmap=num_envmap
        
        self.rgb_images = torch.zeros([n_images,num_envmap,img_res[0],img_res[1],3])
        for e in range(num_envmap):
                    
            for i in range(n_images):
               
                
                image_dir = '{}/image/envmap{}/{}.exr'.format(self.instance_dir,e,i)
                rgb = utils.load_rgb(image_dir,img_res)
            
                          
                self.rgb_images[i,e,:,:,:]=rgb
    
               
        self.n_images = n_images
                                 
        self.object_masks = []
        

        for i in range(n_images):
            
            mask_dir = '{}/mask/{}.png'.format(self.instance_dir,i)
            object_mask = utils.load_mask(mask_dir,img_res)            
            
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).bool())
            
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)              
        
       
        
        self.Intri_all = camera_dict['Intri']
        self.Extri_all = camera_dict['Extri']
        self.fov_all = camera_dict['FOV']
             
    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):        
        sample = {
            "object_mask": self.object_masks[idx],
            "intri": self.Intri_all[idx],
            "extri": self.Extri_all[idx],
            "fov": self.fov_all[idx]
     
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }
        
        return idx, sample, ground_truth
    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

   
