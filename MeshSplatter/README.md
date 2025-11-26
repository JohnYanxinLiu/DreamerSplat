


Create a conda environment with the nerfstudio module with:  

```conda env create -f environment.yml```

To clone the Replica-Dataset fork and its submodules:  
```git submodule init --recursive```

Download Replica Dataset environments and build tools:

```
cd Replica-Dataset/environments
./download.sh .
cd ..
./build.sh
cd ..
```

Firstly, ensure that the environment can be imported into mujoco:  
1. Reduce the color complexity of the ply mesh by quantizing the colors:  
```python quantize_ply_colors.py Replica-Dataset/environments/office_0/mesh.ply office_0_quantized_smooth.ply <# rgb quantization buckets> 0```
2. Convert the mesh into a mujoco-compatible environment (our solution was a colored submesh for every color in the environment):  
```python ply_to_mjcf_submeshes.py office_0_quantized_smooth.ply office_0_quantized office_0```

The flow of the Mesh Splatter is (example using office_0 environment):
1. Create a gaussian splatting dataset 
```mkdir office_0_dataset```
2. Generate a sparse point cloud   
```python ply_to_splat.py MeshSplatter/Replica-Dataset/environments/office_0/mesh.ply office_0_dataset/sparse_pc.ply 15000```
3. Generate the transforms.json which holds a list of camera poses through the interactive viewer  
```./Replica-Dataset/build/ReplicaSDK/ReplicaViewer Replica-Dataset/environments/office_0/mesh.ply Replica-Dataset/environments/office_0/textures Replica-Dataset/environments/office_0/glass.sur office_0_dataset/transforms.json```
4. Generate the dataset using the views provided in the transforms.json using the ReplicaRenderer  
```./Replica-Dataset/build/ReplicaSDK/ReplicaRenderer Replica-Dataset/environments/office_0/mesh.ply   Replica-Dataset/environments/office_0/textures Replica-Dataset/environments/office_0/glass.sur office_0_dataset/transforms.json office_0_dataset```
5. Train the gaussian splat with the point cloud initialized  
```ns-train splatfacto --data office_0_dataset/```