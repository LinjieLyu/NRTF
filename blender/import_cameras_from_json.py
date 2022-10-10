import bpy
import math
import random
from mathutils import Vector
import os
from math import degrees,radians
import numpy as np

import json

f = open('./transforms_train.json')
 

data = json.load(f)
 
fov=data['camera_angle_x']
c=0
for cam in data['frames']:
    wmat=cam['transform_matrix']
    camera_data = bpy.data.cameras.new(name='Camera.%d' % c)
    camera = bpy.data.objects.new('Camera.%d' % c, camera_data)
    bpy.context.scene.collection.objects.link(camera)

    # Set the new camera as active
    for i in range(4):
        for j in range(4):
            camera.matrix_world[i][j] = wmat[i][j]
#    camera.data.lens_unit = 'FOV'
    camera.data.angle = fov
    c=c+1