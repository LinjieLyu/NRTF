import bpy
import bpy_extras
from mathutils import Matrix
from mathutils import Vector
import numpy as np
from math import degrees

#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    assert scene.render.resolution_percentage == 100
    # assume angles describe the horizontal field of view
    assert camd.sensor_fit != 'VERTICAL'

    f_in_mm = camd.lens
    sensor_width_in_mm = camd.sensor_width

    w = scene.render.resolution_x
    h = scene.render.resolution_y

    pixel_aspect = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x

    f_x = f_in_mm / sensor_width_in_mm * w
    f_y = f_x * pixel_aspect
        

    # Parameters of intrinsic calibration matrix K
    c_x = w * (0.5 - camd.shift_x)
    c_y = (h/2) + (camd.shift_y * w)
    
    K = Matrix(
        ((   f_x, 0,       c_x,0),
        (    0  , f_y,     c_y,0),
        (    0  , 0,        1 ,0),
        (    0  , 0,        0 ,1)))
        
    return K

# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

#    # Transpose since the rotation is object rotation, 
#    # and we want coordinate rotation
#    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
#    # T_world2bcam = -1*R_world2bcam * location
#    #
#    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix()#.transposed()

#    # Convert camera location to translation vector used in coordinate changes
#    # T_world2bcam = -1*R_world2bcam*cam.location
#    # Use location from matrix_world to account for constraints:     
    T_world2bcam =  location #-1*R_world2bcam @

#    # Build the coordinate transform matrix from world to computer vision camera
#    # NOTE: Use * instead of @ here for older versions of Blender
#    # TODO: detect Blender version
#  
    print(cam.matrix_world)
    R_world2cv = R_world2bcam @ R_bcam2cv #R_bcam2cv@
    T_world2cv = T_world2bcam
    
#    # put into 4x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],),
        (    0  , 0,        0 ,1)
         ))
         
    

    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K@RT, K, RT

# ----------------------------------------------------------
# Alternate 3D coordinates to 2D pixel coordinate projection code
# adapted from https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex?lq=1
# to have the y axes pointing up and origin at the top-left corner
def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))


scene = bpy.context.scene
render=bpy.context.scene.render

Intri=[]
Extri=[]
FOV=[]
for ob in scene.objects:
    if ob.type == 'CAMERA':
        cam = ob
        index=ob.name.split('.')[-1]
        if index=='Camera':
            index='000'
        index=(int)(index)
        print('Set camera %s' % index )
        P, K, RT = get_3x4_P_matrix_from_blender(ob)      
      
        K=np.matrix(K)
        RT=np.matrix(RT)
        P=np.matrix(P)
        

        m = np.matrix(ob.matrix_world)
        fov = degrees(ob.data.angle)
        print(fov)
        
        Intri.append(K)
        Extri.append(RT)
        FOV.append(fov)
        

np.savez('C:/Users/User/Desktop/blender/cameras.npz', Intri=Intri, Extri=Extri,FOV=FOV)



