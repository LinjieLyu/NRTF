import bpy
import math
import random
from mathutils import Vector
import os
from math import degrees,radians
import numpy as np

n_images=64

cam_file = './cameras.npz'

camera_dict = np.load(cam_file)

Intri_all = camera_dict['Intri']
Extri_all = camera_dict['Extri']
        
        

scale=1

for c in range(n_images):
    K=Intri_all[c]
    RT=Extri_all[c]
    
    
    RT[:,1]=-1*RT[:,1]
    RT[:,2]=-1*RT[:,2]
    camera_data = bpy.data.cameras.new(name='Camera.%d' % c)
    camera = bpy.data.objects.new('Camera.%d' % c, camera_data)
    bpy.context.scene.collection.objects.link(camera)
    
    scene = bpy.context.scene
    sensor_width_in_mm = K[1,1]*K[0,2] / (K[0,0]*K[1,2])
    sensor_height_in_mm = 1  # doesn't matter
    resolution_x_in_px = K[0,2]*2  # principal point assumed at the center
    resolution_y_in_px = K[1,2]*2  # principal point assumed at the center

    s_u = resolution_x_in_px / sensor_width_in_mm
    s_v = resolution_y_in_px / sensor_height_in_mm
    # TODO include aspect ratio
    f_in_mm = K[0,0] / s_u
    # recover original resolution
    if c==0:
        scene.render.resolution_x = resolution_x_in_px / scale
        scene.render.resolution_y = resolution_y_in_px / scale
        scene.render.resolution_percentage = scale * 100
        
    # Set the new camera as active
    for i in range(4):
        for j in range(4):
            camera.matrix_world[i][j] = RT[i,j]
#    camera.data.lens_unit = 'FOV'
    camera.data.type = 'PERSP'
    camera.data.lens = f_in_mm 
    camera.data.lens_unit = 'MILLIMETERS'
    camera.data.sensor_width  = sensor_width_in_mm
    
