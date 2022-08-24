import torch
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render_torch, write_bitmap
from mitsuba.core.xml import load_string

import time
from torch.utils.data import  DataLoader
from torch.utils.tensorboard import SummaryWriter
from scene_dataset import SceneDataset

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
                    help='output dir.')


parser.add_argument('--n_images', type=int, default=64,
                    help='Image numbers.')

parser.add_argument('--W', type=int, default=800,
                    help='Image width.')

parser.add_argument('--H', type=int, default=600,
                    help='Image height.')

args =parser.parse_args()

n_images=args.n_images
img_res=[args.H,args.W]


scene_dir=os.path.join(args.data_dir,args.scene)
output_dir=os.path.join(args.output_dir,args.scene,'material')


train_dataset=SceneDataset(data_dir=scene_dir,img_res=img_res,num_envmap=1,n_images=n_images)
dataloader=DataLoader(train_dataset, batch_size=1, shuffle=True)

mesh_dir=os.path.join(scene_dir,'mesh/mesh.obj')
env_dir=os.path.join(scene_dir,'mitsuba','envmap.png')
tex_dir=os.path.join(scene_dir,'mitsuba','texture.png')



print('dataset loaded')

def make_scene(I_idx): 
    I_idx, model_input, ground_truth=train_dataset.__getitem__(I_idx)  
    
    rgb=ground_truth['rgb'][0].to(device)
    mask=model_input['object_mask'].to(device)
    Rotation=model_input['extri'].copy()
    fov=model_input['fov']
    
    Rotation[:,1]*=-1
    Rotation[:,0]*=-1
    Rotation=Rotation.reshape(-1).tolist()
    R_m=" ".join(str(x) for x in Rotation)
    
    scene=load_string("""
        <?xml version="1.0"?>
        <scene version="2.2.1">
            <integrator type="path">
                           <integer name="max_depth" value="4"/>
                           
                       </integrator>
                      
            
            <emitter type="envmap">
            <string name="filename" value="{env_dir}"/>
                <transform name="to_world">
     			<matrix value="0 0 1 0 1 0 0 0 0 1 0 0 0 0 0 1"/>
     			
    		    </transform>
            </emitter>
    
            <sensor type="perspective">
                <transform name="to_world">
                    <matrix value="{matrix_values}"/>
                
                     </transform>
                    <float name="fov" value="{fov}"/>                    
                <sampler type="independent">
                    <integer name="sample_count" value="8"/>
                    
                </sampler>
                
                <film type="hdrfilm">
                    <integer name="width" value="{W}"/>
                    <integer name="height" value="{H}"/>
                    <rfilter type="box"/>
                </film>
            </sensor>
            
            <shape type="obj">
               
                <string name="filename" value="{mesh_dir}"/> 
   
                
              <bsdf type="blendbsdf">
                  <float name="weight" value="0.5"/> 
                      <bsdf type="roughconductor">
                             <float name="alpha" value="0.5"/>
                </bsdf>
                    
                    <bsdf type="diffuse">
                        <texture type="bitmap" name="reflectance">
                            <string name="filename" value="{tex_dir}"/>
                        </texture>
                    </bsdf>      
                </bsdf>     
            </shape>
            
            
        </scene>
    """.format(matrix_values=R_m,fov=fov,W=args.W,H=args.H,mesh_dir=mesh_dir,tex_dir=tex_dir,env_dir=env_dir))
    
    # # Find differentiable scene parameters
    params = traverse(scene)
    # print(params)
    params.keep([
                 'EnvironmentMapEmitter.data',
                  'OBJMesh.bsdf.weight.value',                                                      
                  'OBJMesh.bsdf.bsdf_1.reflectance.data', 
                  'OBJMesh.bsdf.bsdf_0.alpha.value'
                  
                  ])
        
    return I_idx,scene,params,rgb,mask

      
I_idx,scene,params,rgb,mask =make_scene(0)



params['EnvironmentMapEmitter.data']*=0.
params['EnvironmentMapEmitter.data']+=0.2
params.update()

# # Which parameters should be exposed to the PyTorch optimizer?
params_torch = params.torch()


Envmaps=torch.ones([16,32,4],device=device)    
Envmaps[:,:,:3]*=0.1
Envmaps.requires_grad=True

weight=0.5*torch.ones([1],device=device)
weight.requires_grad=True
weight_backup=weight.detach().clone()

alpha=0.5*torch.ones([1],device=device)
alpha_backup=alpha.detach().clone()
alpha.requires_grad=True

params_torch['OBJMesh.bsdf.bsdf_0.alpha.value'].requires_grad=True

E_opt = torch.optim.SGD([Envmaps], lr=10)

T_opt = torch.optim.Adam([params_torch['OBJMesh.bsdf.bsdf_1.reflectance.data']
                          ], lr=1e-1,betas=(0.5, 0.6))

W_opt = torch.optim.Adam([
            
                {'params': [alpha]},
                {'params': [weight], 'lr': 1e-4}
            ], lr=1e-3,betas=(0.5, 0.6))



E_scheduler=torch.optim.lr_scheduler.StepLR(E_opt, 1000, gamma=0.2)
T_scheduler=torch.optim.lr_scheduler.StepLR(T_opt, 1000, gamma=0.2)
W_scheduler=torch.optim.lr_scheduler.StepLR(W_opt, 1000, gamma=0.5)

objective = torch.nn.MSELoss()
l1_loss=torch.nn.L1Loss()

print(params_torch)





for it in range(3001):
    
    time_a = time.time()
    E_opt.zero_grad()
    T_opt.zero_grad()

    W_opt.zero_grad()

    
    
    I_idx=random.randint(0,n_images-1)  
    I_idx,scene,params,rgb,mask=make_scene(I_idx)    

    

    gt=rgb
    masked_gt=gt.view(-1,3)#[mask,:]  
    params_torch['EnvironmentMapEmitter.data']=Envmaps.view(-1)  
    params_torch['OBJMesh.bsdf.bsdf_0.alpha.value']=alpha
    params_torch['OBJMesh.bsdf.weight.value']=weight
    image = render_torch(scene, params=params, unbiased=True,
                          spp=8, **params_torch)
 
    masked_image=image.view(-1,3)#[mask,:]   
    ob_val=objective(masked_image, masked_gt)+0.5*l1_loss(masked_image, masked_gt)
    
    img=torch.cat([image.detach(),gt],dim=-2)   
    
    if it%20==0:
        texture=params_torch['OBJMesh.bsdf.bsdf_1.reflectance.data'].view(512,512,3)
        dx=texture-torch.roll(texture,1,1)
        dy=texture-torch.roll(texture,1,0)
        ob_val+=1e-7*(torch.norm(dx,p=1)+torch.norm(dy,p=1))
        
        
        envmap=Envmaps.view(16,32,4)[:,:,:3]
        dxe=envmap-torch.roll(envmap,1,1)
        dye=envmap-torch.roll(envmap,1,0)
        ob_val+=1e-6*(torch.norm(dxe,p=1)+torch.norm(dye,p=1))
        
        
    # Back-propagate errors to input parameters
    ob_val.backward()
    torch.nn.utils.clip_grad_norm_([alpha], 1e-2)
 
    # Optimizer: take a gradient step
    E_opt.step()
    T_opt.step()
    W_opt.step()

    E_scheduler.step() 
    T_scheduler.step() 
    W_scheduler.step()
   
    time_b = time.time()
    print('Iteration {}:'.format(it),ob_val.item(),'time:',time_b-time_a)
   
    params_torch['OBJMesh.bsdf.bsdf_1.reflectance.data'].data=params_torch['OBJMesh.bsdf.bsdf_1.reflectance.data'].data.clamp(0,1)
    
    if torch.isnan(alpha).item()  :
        print(torch.isnan(alpha).item(),alpha_backup.item())
        alpha=alpha_backup.clone()
        alpha.requires_grad=True
   
        W_opt =torch.optim.Adam([
            
                {'params': [alpha]},
                {'params': [weight], 'lr': 1e-4}
            ], lr=1e-3,betas=(0.5, 0.6))
        W_scheduler=torch.optim.lr_scheduler.StepLR(W_opt, 1000, gamma=0.5,last_epoch=it)
    
        print(torch.isnan(alpha).item())
        
    elif torch.isnan(weight).item():
        print(torch.isnan(weight).item(),weight_backup.item())
        weight=weight_backup.clone()
        weight.requires_grad=True
    
        W_opt = torch.optim.Adam([
            
                {'params': [alpha]},
                {'params': [weight], 'lr': 1e-4}
            ], lr=1e-3,betas=(0.5, 0.6))
        W_scheduler=torch.optim.lr_scheduler.StepLR(W_opt, 1000, gamma=0.5,last_epoch=it)
    
        print(torch.isnan(weight).item())
    else:   
        alpha.data=alpha.data.clamp(1e-8,1-1e-8)
        alpha_backup=alpha.detach().clone()
       
        weight.data=weight.data.clamp(1e-8,1-1e-8)
        weight_backup=weight.detach().clone()
   
    
    print(alpha.item(),weight.item())
    Envmaps.data=Envmaps.data.clamp(0,1000)  
    
    
    
    
    if it%100==0:   
        pyredner.imwrite(img.cpu(),'{}/opt_{}.png'.format(output_dir,it))  
        
        pyredner.imwrite(params_torch['OBJMesh.bsdf.bsdf_1.reflectance.data'].view(512,512,3).cpu(),'{}/diffuse/{}.exr'.format(output_dir,it),gamma=1)  
        pyredner.imwrite(Envmaps.view(16,32,4)[:,:,:3].cpu(),'{}/envmap/{}.exr'.format(output_dir,it),gamma=1)
        
        ckpt=os.path.join(output_dir,'checkpoint')
        if not os.path.exists(ckpt):
            os.makedirs(ckpt)
        torch.save({
                    'global_step': it,
                    'alpha': alpha.item(),              
                    'weight': weight.item(),
                    'parmas_torch': params_torch,
                    'envmap':Envmaps
                }, '{}/{}.pt'.format(ckpt,it)) 
  
        
        

    