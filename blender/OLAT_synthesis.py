import bpy
import os 
import numpy as np
from pathlib import Path
import sys

sys.stderr.write("blah blah")
C = bpy.context
scn = C.scene

jobid = os.getenv('SLURM_ARRAY_TASK_ID')
index=int(jobid)
# Get the environment node tree of the current scene
node_tree = scn.world.node_tree
tree_nodes = node_tree.nodes

Env_node=tree_nodes['Environment Texture']

# Load and assign the image to the node property



for c in range(264):
        
        bpy.context.scene.camera = bpy.data.objects["Camera.{}".format(c)]
        print('Set camera %s' % bpy.context.scene.camera.name )
        
        my_file=Path( "outputdir/scene/olat_data/cam{}light{}.exr".format(c,index))
        if not my_file.is_file():    
            envmap=bpy.data.images.load("outputdir/scene/olat_env/{}.exr".format(index))
            Env_node.image = envmap
            

                        
            file = "outputdir/scene/olat_data/cam{}light{}.exr".format(c,index)
            bpy.context.scene.render.filepath = file
            bpy.ops.render.render( write_still=True )
        else:
            continue    

                