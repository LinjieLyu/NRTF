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

parser.add_argument('--batch_size', type=int, default=3000,
                    help='Image height.')

args =parser.parse_args()

n_images=args.n_images
H=args.H
W=args.W
batch_size=args.batch_size

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


ckpt = torch.load('{}/OLAT/checkpoint/latest.pt'.format(output_dir))
renderer.load_state_dict(ckpt['network_fn_state_dict'])

embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])




Envmaps_ini=pyredner.imread('{}/material/envmap/3000.exr'.format(output_dir),gamma=1)[:,:,:3].to(device).unsqueeze(0)
Envmaps=Envmaps_ini.detach()
Envmaps.requires_grad=True




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
lrate=1e-4
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
                                           torch.optim.Adam(grad_vars,lr=lrate, betas=(0.9, 0.99), eps= 1e-15),
                                           torch.optim.Adam([Envmaps], lr=2e-4)])

scheduler=MultipleScheduler(optimizer, 30000, gamma=0.33)

def tone_loss(x,y):
    dif=x-y
    # mapping=torch.abs(x.detach())+1e-3
    mapping=x.detach().clamp(1e-3,1)
    loss=dif/mapping
    return torch.norm(loss,p=1)


l2_loss =lambda x, y : torch.sum((x - y) ** 2) # torch.nn.MSELoss()
l1_loss=torch.nn.L1Loss(reduction='sum')



print("Starting optimization:")
iterations=150001

monitor=os.path.join(output_dir,'events/joint')
if not os.path.exists(monitor):
    os.makedirs(monitor)
writer = SummaryWriter(log_dir=monitor)



uni_in_dirs_=embedview(uni_in_dirs)
olat_envmap=1*torch.ones([1,1,3]).to(device)

for e in range(iterations):
    

    renderer.train()
    start=time.time()

    
##########  training  ######################       

    I_idx, model_input, ground_truth=next(iter(dataloader)) 
    I_idx=I_idx.item()
    e_idx=0  
    envmap_img=Envmaps[0,:,:,:]
    envmap_img_ini=Envmaps_ini[e_idx,:,:,:]
    
    rgbs=ground_truth['rgb'][0][e_idx].to(device)
    
    pos_input=torch.load('{}/buffer/{}.pt'.format(output_dir,I_idx))
    mask=torch.load('{}/buffer/{}.pt'.format(output_dir,I_idx))  
    
    masked_gt=rgbs.view([1,-1,3])[:,mask,:]
    
    points=pos_input[:,:3].detach()
    out_dirs=pos_input[:,3:6].detach()
    normals=pos_input[:,6:9]
    
    
    out_dirs_ = embedview(out_dirs)   
    
    indices = torch.split(torch.randperm(len(points)),batch_size)
    


    index=indices[0]
    
    
    
    optimizer.zero_grad()
 
    
    points_=embed_fn(points)
 
    pos=points_[index,:]
    outd=out_dirs_[index,:]
    m_gt=masked_gt[:,index,:].detach()
    

    # ###### RGB Loss ######
  
    radiance=renderer(pos,outd,uni_in_dirs_).unsqueeze(0)/100       #1*N*M*3    
    rendered_pixels=torch.sum(radiance*envmap_img.view(1,1,-1,3),dim=-2)  
    
    
    
    rgb_loss=(tone_loss(rendered_pixels,m_gt))/float(pos.shape[0])
 
 
    ###### OLAT Loss ######                       
    e_index=random.randint(0,511)    
              
    in_dirs=uni_in_dirs[e_index].view(-1,3)
    in_dir_=embedview(in_dirs)
    
     
    olat_gt=torch.load('{}/olat_data/cam{}light{}.pt'.format(output_dir,I_idx,e_index)).to(device)

 
    olat_ref=olat_gt.view([-1,3])[mask,:]
    
   
    olat_radiance=renderer(points_,out_dirs_,in_dir_)   
    olat_pixels=torch.sum(olat_radiance*olat_envmap,dim=-2)
           
    olat_loss=0.1*tone_loss(olat_pixels,olat_ref)/float(points.shape[0])

    env_loss=1e-5*l2_loss(envmap_img,envmap_img_ini)
    
    if e%20==0 & i==0:
       
        dxe=envmap_img-torch.roll(envmap_img,1,1)
        dye=envmap_img-torch.roll(envmap_img,1,0)
        env_loss+=1e-6*(torch.norm(dxe,p=1)+torch.norm(dye,p=1))
        
    Loss=rgb_loss+env_loss+olat_loss
       
    Loss.backward()

   
    optimizer.step()
    scheduler.step()
    
    
    
    end=time.time()
    
    print("Loss at iter {}_{}:".format(e,i) , Loss.item(),rgb_loss.item(),olat_loss.item(),env_loss.item(),"time:",end-start)#
   
    writer.add_scalar('Loss/rgb', rgb_loss.item(), e)
    writer.add_scalar('Loss/OLAT', olat_loss.item(), e)
    writer.add_scalar('Loss/envmap', env_loss.item(), e)
       
    Envmaps.data=Envmaps.data.clamp(0,100) 
    

    

    
    if e % 500==0:
        renderer.eval()
        torch.cuda.empty_cache()
        
        rendered_img=torch.ones([H*W,3]).to(device)
        gt_img=torch.ones([H*W,3]).to(device)
          
        I_idx=random.randint(0,total_cam_num-1) 
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
            rendered_pixels=torch.sum(radiance*envmap_img.view(1,1,-1,3),dim=-2).view(-1,3) 
            Radiance.append(rendered_pixels.detach())    
            
        olat_img=torch.zeros([H*W,3]).to(device)    
        olat_img[mask,:]=olat_pixels.detach() 
        olat_img=torch.cat([olat_img.view([H,W,3]),olat_gt.view([H,W,3])],dim=1)
              
        
        rendered_img[mask1,:]=torch.cat(Radiance,dim=0)
  
        
        
         
        pyredner.imwrite(olat_img.cpu(),'{}/joint/OLAT/{}.png'.format(output_dir,e))     
        pyredner.imwrite(rendered_img.view([H,W,3]).cpu(),'{}/joint/val/{}.png'.format(output_dir,e))
        pyredner.imwrite(envmap_img.cpu(),'{}/joint/val/envmaps/est_env{}.exr'.format(output_dir,e_idx),gamma=1)    
    if e % 100==0:
        ckpt=os.path.join(output_dir,'joint/checkpoint')
        if not os.path.exists(ckpt):
            os.makedirs(ckpt)
            
        torch.save({
                    'global_step': e,
                    'network_fn_state_dict': renderer.state_dict(),
                  
                    'Envmaps':Envmaps,
                    'embed_fn_state_dict': embed_fn.state_dict(),
                 
                }, '{}/{}.pt'.format(ckpt,e))
        torch.save({
                    'global_step': e,
                    'network_fn_state_dict': renderer.state_dict(),
                  
                    'Envmaps':Envmaps,
                    'embed_fn_state_dict': embed_fn.state_dict(),
                 
                }, '{}/latest.pt'.format(ckpt))
        
       
        renderer.train()