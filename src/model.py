import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F

import re
import math

             
                 
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires,input_dims):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

def gaussian_mapping(x,B_gauss):
    x_proj=torch.matmul(2*np.pi*x,B_gauss)
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    

########################
# Initialization methods
           
def weight_init(m,actfunc='sine'):
    if actfunc=='sine':
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                # See supplement Sec. 1.5 for discussion of factor 30
                m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
                
    elif actfunc=='relu':
            if hasattr(m, 'weight'):
                nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def first_layer_init(m,actfunc='sine'):
    if actfunc=='sine':
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
                m.weight.uniform_(-1 / num_input, 1 / num_input)
    elif actfunc=='relu':
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')



class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

                     
class PRTNetwork1(nn.Module):
    def __init__(
            self,           
            W=64,
            D=8,
            skips=[4],
            din=9,
            dout=3,
            activation='relu'
            
    
            
            
    ):
        super().__init__()
        
        
       
                
        
             
        self.in_features=din
        self.out_features=dout
         
        self.W=W
    
        self.skips=skips
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.in_features,W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.in_features, W) for i in range(D-1)])
        
        
        for i, l in enumerate(self.pts_linears):
            if i==0:
                first_layer_init(self.pts_linears[i],activation)
            else:    
                weight_init(self.pts_linears[i],activation)
            
        
        self.output_linear = nn.Linear(W, self.out_features)
        
        if activation=='sine':
            self.activation=Sine()
     
        elif activation=='relu':
            self.activation=nn.ReLU(inplace=True)
        
             
 
        
    def forward(self,pos,dout,din):
        
   
        N=pos.shape[0]
        M=din.shape[0]  
        
        pos=pos.unsqueeze(1).expand([-1,M,-1]) 
        dout=dout.unsqueeze(1).expand([-1,M,-1]) 
        din=din.unsqueeze(0).expand([N,-1,-1])
        
          
        all_input=torch.cat([pos, dout,din], dim=-1).view(N*M,-1)
        
        h=all_input
   
        for i, l in enumerate(self.pts_linears):
            
            h = self.pts_linears[i](h)
            h = self.activation(h)
        
            if i in self.skips:
                h = torch.cat([all_input, h], -1)
        
                  
        output=self.output_linear(h)    

        return output.view(N,M,self.out_features)
    
