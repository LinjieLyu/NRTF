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


 

# ###################optimizer and loss ###########################################
lrate=5e-4
class MultipleOptimizer(object):
    def __init__(self, op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


class MultipleScheduler(object):
    def __init__(self, multioptimizers,step,gamma):
        self.optimizers = multioptimizers.optimizers
        self.schedulers=[]
   
        for op in self.optimizers:
            
            scheduler = torch.optim.lr_scheduler.StepLR(op, step, gamma=gamma)
            
         
            self.schedulers.append(scheduler)
            
    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()
        
    

            
optimizer = MultipleOptimizer([torch.optim.SparseAdam(embedding_params,lr=lrate, betas=(0.9, 0.99), eps= 1e-15),
                                           torch.optim.Adam(grad_vars,lr=lrate, betas=(0.9, 0.99), eps= 1e-15)])

scheduler=MultipleScheduler(optimizer, 30000, gamma=0.33)

def tone_loss(x,y):
    dif=x-y
    # mapping=torch.abs(x.detach())+1e-3
    mapping=x.detach().clamp(1e-3,1)
    loss=dif/mapping
    return torch.norm(loss,p=1.8)


l2_loss =lambda x, y : torch.sum((x - y) ** 2) # torch.nn.MSELoss()
l1_loss=torch.nn.L1Loss(reduction='sum')



print("Starting optimization:")
iterations=150001

monitor=os.path.join(output_dir,'events/OLAT')
if not os.path.exists(monitor):
    os.makedirs(monitor)
writer = SummaryWriter(log_dir=monitor)




olat_envmap=1*torch.ones([1,1,3]).to(device)
batch_size=20
for it in range(iterations):
    

    renderer.train()
    start=time.time()

    
##########  training  ######################       

    ###### shaffle indices ######     
    I_idx=random.randint(0,total_cam_num-1) 
    pos_input=torch.load('{}/buffer/{}.pt'.format(output_dir,I_idx))['pos_input'].to(device).float()  
    mask=torch.load('{}/buffer/{}.pt'.format(output_dir,I_idx))['mask'].to(device)
    
    points=pos_input[:,:3]
    out_dirs=pos_input[:,3:6]
    normals=pos_input[:,6:9]
    
    
    out_dirs_ = embedview(out_dirs)
    points_=embed_fn(points)
    
    Loss=0
    optimizer.zero_grad()  
    

    
    ###### OLAT Loss ###### 
    for b in range(batch_size):  
       
        
    
        e_index=random.randint(0,511)    
              
        in_dirs=uni_in_dirs[e_index].view(-1,3)
        in_dir_=embedview(in_dirs)
        
        
     
        
    
        olat_gt=torch.load('{}/olat_data/cam{}light{}.pt'.format(output_dir,I_idx,e_index)).to(device) 
    
        olat_gt_m=olat_gt.view([-1,3])[mask,:]
        olat_radiance=renderer(points_,out_dirs_,in_dir_)  
        olat_pixels=torch.sum(olat_radiance*olat_envmap,dim=-2)
        Loss+=tone_loss(olat_pixels,olat_gt_m)/float(points_.shape[0])
        
    Loss.backward()
    

    optimizer.step()
    scheduler.step()
    end=time.time()
    print("Loss at iter {}_{}:".format(it,e_index) , Loss.item(),"time:",end-start)
    
    writer.add_scalar('Loss/OLAT', Loss.item(), it)    


        
               
    if it % 500==0:
        renderer.eval()
        
        rendered_img=torch.ones([H*W,3]).to(device)
       
        gt_img=torch.ones([H*W,3]).to(device)
        
        
        
        olat_img=torch.zeros([H*W,3]).to(device)    
        olat_img[mask,:]=olat_pixels.detach() 
        olat_img=torch.cat([olat_img.view([H,W,3]),olat_gt.view([H,W,3])],dim=1)
        

        
         
        pyredner.imwrite(olat_img.cpu(),'{}/OLAT/{}.png'.format(output_dir,it))  
        
    if it % 100==0:
        ckpt=os.path.join(output_dir,'OLAT/checkpoint')
        if not os.path.exists(ckpt):
            os.makedirs(ckpt)
            
        torch.save({
                    'global_step': it,
                    'network_fn_state_dict': renderer.state_dict(),              
                    'embed_fn_state_dict': embed_fn.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                }, '{}/{}.pt'.format(ckpt,it)) 
        
        torch.save({
                    'global_step': it,
                    'network_fn_state_dict': renderer.state_dict(),              
                    'embed_fn_state_dict': embed_fn.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                }, '{}/latest.pt'.format(ckpt)) 
        
        renderer.train()

    


