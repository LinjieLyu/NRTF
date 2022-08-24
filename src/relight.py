import torch

import numpy as np
import time
from torch.utils.data import Dataset, DataLoader

import argparse
import random
import os

from scene_dataset import SceneDataset


import pyredner





from model import PRTNetwork1

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from hash_encoding import HashEmbedder, SHEncoder

device=torch.device("cuda:0")

pyredner.set_print_timing(False)
pyredner.set_use_gpu( True )

parser = argparse.ArgumentParser()


parser.add_argument('--data_dir', type=str, 
                    help='Path to data.')

parser.add_argument('--scene',default=None, type=str, 
                    help='Scene name.')


parser.add_argument('--output_dir', type=str, 
                    help='output dir.')


parser.add_argument('--n_images', type=int, default=64,
                    help='Image numbers.')

parser.add_argument('--W', type=int, default=800,
                    help='Image width.')

parser.add_argument('--H', type=int, default=600,
                    help='Image height.')

args =parser.parse_args()

n_images=args.n_images
H=args.H
W=args.W

scene_dir=os.path.join(args.data_dir,args.scene)
output_dir=os.path.join(args.output_dir,args.scene)

train_dataset=SceneDataset(data_dir=scene_dir,img_res=[H,W],num_envmap=1,n_images=n_images)
dataloader=DataLoader(train_dataset, batch_size=1, shuffle=True)

mesh_dir=os.path.join(scene_dir,'mesh/mesh.obj')      



######### initial hash encoding ##############
objects=pyredner.load_obj(mesh_dir,return_objects=True)
vertices=objects[0].vertices

total_cam_num=torch.load('{}/buffer/0.pt'.format(args.output_dir))['total_cam_num']
test_cam_start=torch.load('{}/buffer/0.pt'.format(args.output_dir))['test_cam_start']

for mesh in objects:
    vertices=torch.cat([vertices,mesh.vertices],0)


for i in range(total_cam_num):
    xyz=torch.load('{}/buffer/{}.pt'.format(args.output_dir,i))['pos_input'].to(device)[:,:3]

    vertices=torch.cat([vertices,xyz],0)
    
bounding_box=[torch.min(vertices,0)[0]-1e-3,torch.max(vertices,0)[0]+1e-3]

print('bounding_box:',bounding_box)

def get_hashembedder(bounding_box=[-1,1]):    
    embed = HashEmbedder(bounding_box=bounding_box,                    
                          n_features_per_level=16,
                          base_resolution=16.,
                          n_levels=19,
                          finest_resolution=256.,
                          sparse=True,
                          vertices=vertices
                        
                        )
    out_dim = embed.out_dim
  
    return embed, out_dim
 
def get_SHembedder():
  
    embed = SHEncoder()
    out_dim = embed.out_dim
  
    return embed, out_dim   


embedview, view_ch = get_SHembedder()

embed_fn, pts_ch = get_hashembedder(bounding_box)  
embed_fn=embed_fn.to(device)


embedding_params = list(embed_fn.parameters())
input_features=pts_ch+ view_ch +view_ch 

renderer=PRTNetwork1(W=512,D=7,skips=[3],din=input_features,dout=3,activation='relu').to(device)
grad_vars = list(renderer.parameters())


ckpt = torch.load('{}/joint/checkpoint/latest.pt'.format(output_dir))
renderer.load_state_dict(ckpt['network_fn_state_dict'])

embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])

######## incoming direction ################
envmap_height=16

def uv_to_dir(u,v):
    theta=(u+0.5)*np.pi/envmap_height
    phi=(v+0.5)*2*np.pi/(envmap_height*2)
    return np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
    


uni_in_dirs=torch.ones([envmap_height,envmap_height*2,3])
for i in range(envmap_height):
    for j in range(envmap_height*2):
        x,y,z=uv_to_dir(i,j)
        uni_in_dirs[i,j,0]=x
        uni_in_dirs[i,j,1]=y
        uni_in_dirs[i,j,2]=z

uni_in_dirs=uni_in_dirs.view(-1,3).float().to(device)
uni_in_dirs_=embedview(uni_in_dirs)


Envmaps_opt=ckpt['Envmaps']
Test_envmaps=pyredner.imread('{}/envmap/test.exr'.format(scene_dir),gamma=1)[:,:,:3].to(device)



###### Evaluation #########

renderer.eval()
torch.cuda.empty_cache()

for I_idx in range(test_cam_start,total_cam_num):
    rendered_img=torch.ones([H*W,3]).to(device)
    
      
    pos_input=torch.load('{}/buffer/{}.pt'.format(output_dir,I_idx))['pos_input'].to(device).float()  
    mask1=torch.load('{}/buffer/{}.pt'.format(output_dir,I_idx))['mask'].to(device)
    
    
    points=pos_input[:,:3]
    out_dirs=pos_input[:,3:6]
    normals=pos_input[:,6:9]
    
    
    out_dirs_ = embedview(out_dirs)   
    points_=embed_fn(points)
    cam_input = torch.cat([points_, out_dirs_,normals], dim=-1)
    
    Radiance=[]
    
    
    
    for pos,outd in zip(torch.split(points_,1000),torch.split(out_dirs_,1000)):
        radiance=renderer(pos,outd,uni_in_dirs_).unsqueeze(0)/100       #1*N*M*3    
        rendered_pixels=torch.sum(radiance*Test_envmaps.view(1,1,-1,3),dim=-2).view(-1,3) 
        Radiance.append(rendered_pixels.detach())    
        
              
    
    rendered_img[mask1,:]=torch.cat(Radiance,dim=0)
    
    
    
     
      
    pyredner.imwrite(rendered_img.view([H,W,3]).cpu(),'{}/relight/{}.png'.format(output_dir,I_idx))
    

