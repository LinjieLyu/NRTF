# How to export or import camera models
See export_cameras.py and import_cameras.py

# How to sample random OLAT camera poses
See randomcamera.py

# UV mapping

* Import the extracted mesh from NeuS to a new blender scene.
* (Optional)Add smooth modifier. Tune the surface smoothness until you are satisfied.
* Export the mesh with uv coordinates. In transform, select Y Forward and Z up.

# OLAT synthesis
See the OLAT_example.blend  
* You should have done material estimation with Mitsuba2.  
* Replace the mesh, texture, and material parameters in shading. Be careful, when you import the mesh, in transform, select Y Forward and Z up.
* Frac in Mix shader equals the optimized weight in Mitsuba2.
* Roughness in Glossy BSDF equals the alpha of rough conductor material in Mitsuba2.

To generate the synthetic OLAT data, you probably need some dense CPU (or GPU) servers for parallel computation, or it will take quite a long time.
We provide a slurm example.
```
#SBATCH --array=0-447

cd /software/blender-2.93.8-linux-x64
./blender [output dir]/OLAT_example.blend  --background --python [output dir]/OLAT_synthesis.py 1> nul

python img2pt.py --scene $scene --output_dir $output_root
```
If your synthetic OLAT data is noisy, check if you add a denoiser in blender. Please refer to OLAT_example.blend and OLAT_synthesis.py for more details.