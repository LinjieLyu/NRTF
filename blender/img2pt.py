import pyredner
import torch
import os
import argparse

jobid = os.getenv('SLURM_ARRAY_TASK_ID')

I=int(jobid)

parser = argparse.ArgumentParser()

parser.add_argument('--scene',default=None, type=str, 
                    help='Scene name.')


parser.add_argument('--output_dir', type=str, 
                    help='Output dir.')

args =parser.parse_args()

olat_dir=os.path.join(args.output_dir,args.scene,'olat_data')

for c in range(264):
    
    img_file=os.path.join(olat_dir,'cam{}light{}.exr'.format(c,I))
    olat_file=os.path.join(olat_dir,'cam{}light{}.pt'.format(c,I))
    
    if os.path.exists(img_file):
        
        olat=pyredner.imread(img_file,gamma=1)[:,:,:3]
        torch.save(olat,olat_file)
        
        os.remove(img_file)
        

