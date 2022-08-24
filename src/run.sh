#!/bin/bash

data_root="/HPS/VisibilityLearning/work/Programs/SDFRelight/release/data"
output_root='/HPS/VisibilityLearning/work/Programs/SDFRelight/release/output'



# material estimation
python mitsuba_material.py --data_dir $data_root --output_dir $output_root/material 
# render view buffer
python render_buffer.py --data_dir $data_root  --output_dir $output_root
# render olat synthesis
## TODO

# train 
python train_olat.py --data_dir $data_root  --output_dir $output_root
python train_joint.py --data_dir $data_root  --output_dir $output_root

# test and relight
## TODO