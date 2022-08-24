import bpy
import math
import random
from mathutils import Vector
import os
from math import degrees
import numpy as np


cam_file = 'Z:/VisibilityLearning/work/Programs/SDFRelight/mesh/scene_data/scan4/cameras.npz'

camera_dict = np.load(cam_file)

Intri_all = camera_dict['Intri']
Extri_all = camera_dict['Extri']
        
K=Intri_all[0]        

scale=1


for c in range(200):
    
    
    radius = random.uniform(1., 2.)
    
    theta =  math.pi * random.uniform(0., 1.)
    phi=2 * math.pi * random.uniform(0., 1.)
    
    # Randomly place the camera on a circle around the object at the same height as the main camera
    new_camera_pos = Vector((radius * math.sin(theta)* math.cos(phi), radius * math.sin(theta)* math.sin(phi), radius * math.cos(theta)))
    
    
    
    theta1 =  math.pi * random.uniform(0., 1.)
    phi1=2 * math.pi * random.uniform(0., 1.)
    
    # Randomly place the camera on a circle around the object at the same height as the main camera
    new_track_pos = Vector((0.1 * math.sin(theta1)* math.cos(phi1), 0.1 * math.sin(theta1)* math.sin(phi1), 0.1* math.cos(theta1)))
    
    
    
    direction=new_track_pos-new_camera_pos
    rot_quat = direction.to_track_quat('-Z', 'Y')
    
    rotation_euler = rot_quat.to_euler()
    
    R= rotation_euler.to_matrix()
    
    
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
    for i in range(3):
        camera.matrix_world[i][3] = new_camera_pos[i]
        for j in range(3):
            camera.matrix_world[i][j] = R[i][j]
#    camera.data.lens_unit = 'FOV'
    camera.data.type = 'PERSP'
    camera.data.lens = f_in_mm 
    camera.data.lens_unit = 'MILLIMETERS'
    camera.data.sensor_width  = sensor_width_in_mm



