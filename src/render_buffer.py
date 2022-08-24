import torch
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')
from mitsuba.core import Thread, Vector3f
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render_torch, write_bitmap
from mitsuba.core.xml import load_string
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from scene_dataset import SceneDataset

import imageio
import skimage
from skimage.transform import rescale, resize, downscale_local_mean
import random

import argparse
import random
import os

import pyredner
print(torch.__version__)

device=torch.device("cuda:0")


parser = argparse.ArgumentParser()


parser.add_argument('--data_dir', type=str, 
                    help='Path to data.')

parser.add_argument('--scene',default=None, type=str, 
                    help='Scene name.')


parser.add_argument('--output_dir', type=str, 
                    help='Output dir.')


parser.add_argument('--W', type=int, default=800,
                    help='Image width.')

parser.add_argument('--H', type=int, default=600,
                    help='Image height.')

parser.add_argument('--n_images', type=int, default=64,
                    help='Image numbers.')


args =parser.parse_args()

n_images=args.n_images
img_res=[args.H,args.W]

scene_dir=os.path.join(args.data_dir,args.scene)


input_cameras=np.load(os.path.join(scene_dir,'cameras.npz'))
olat_cameras=np.load(os.path.join(scene_dir,'olat_cameras.npz'))
test_cameras=np.load(os.path.join(scene_dir,'test_cameras.npz'))

Extri=np.concatenate((input_cameras['Extri'], olat_cameras['Extri'], test_cameras['Extri']), axis=0)
FOV=np.concatenate((input_cameras['FOV'], olat_cameras['FOV'], test_cameras['FOV']), axis=0)


mesh_dir=os.path.join(scene_dir,'mesh/mesh.obj')


total_cam_num=Extri.shape[0]
test_cam_start=input_cameras['Extri'].shape[0]+olat_cameras['Extri'].shape[0]


def make_scene(index=0): 

    
    
    Rotation=Extri[index]
    fov=FOV[index]
    
    Rotation[:,1]*=-1
    Rotation[:,0]*=-1
    Rotation_list=Rotation.reshape(-1).tolist()
    R_m=" ".join(str(x) for x in Rotation_list)
      
    pos_scene=load_string("""
        <?xml version="1.0"?>
        <scene version="2.2.1">
            <integrator type="aov">
                <string name="aovs" value="nn:position,nn:sh_normal"/>
                <integrator type="path" name="my_image"/>
            </integrator>
                                                   
    
            <sensor type="perspective">
                <transform name="to_world">
                     <matrix value="{matrix_values}"/>
                
                     </transform>
                    <float name="fov" value="{fov}"/>   
                    
                <sampler type="independent">
                    <integer name="sample_count" value="1"/>
                    
                </sampler>
                
                <film type="hdrfilm">
                    <integer name="width" value="{W}"/>
                    <integer name="height" value="{H}"/>
                    <rfilter type="box"/>
                </film>
            </sensor>
            
            <shape type="obj">
               
                <string name="filename" value="{mesh_dir}"/>   
                            
            </shape>
            
        </scene>
    """.format(matrix_values=R_m,fov=fov,W=args.W,H=args.H,mesh_dir=mesh_dir ))
    

  
    
    pos_params= traverse(pos_scene)     
    pos_img=render_torch(pos_scene, params=pos_params, unbiased=True,malloc_trim=True,
                                  spp=1).to(device).detach() 
        
    mask=pos_img[:,:,-1].float().view(-1)>0
    positions=pos_img[:,:,3:6].view(-1,3)[mask,:]
    normals=pos_img[:,:,6:9].view(-1,3)[mask,:]
    

       
    cam_loc=torch.tensor(Rotation[:3,3]).to(device).view([1,3])
    ray_dir=(cam_loc-positions)
    ray_dir_normed=ray_dir/(ray_dir.norm(2,dim=1).unsqueeze(-1))
    
    
  
    
    pos_input=torch.cat([positions,ray_dir_normed,normals], dim=-1)
 
    return pos_input,mask

output_dir=os.path.join(args.output_dir,args.scene)
if not os.path.exists(output_dir):
    os.makedirs(os.path.join(output_dir))
            
            
for c in range(total_cam_num):
    pos_input,mask=make_scene(c)
    ckpt=os.path.join(output_dir,'buffer')
    if not os.path.exists(ckpt):
            os.makedirs(ckpt)
    torch.save({'total_cam_num':total_cam_num,
                'test_cam_start':test_cam_start,
                'olat_cam_start':n_images,
                'pos_input': pos_input,
                'mask': mask,                             
            }, '{}/{}.pt'.format(ckpt,c))

##### OLAT environment map rendering ########
for i in range(512):
    
    img=torch.zeros([512,3])
    img[i,:]=100
    pyredner.imwrite(img.view(16,32,3).cpu(),'{}/olat_env/{}.exr'.format(output_dir,i),gamma=1)

