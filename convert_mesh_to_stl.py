
import os
import trimesh

mesh_root_dir = "../DA2/data/simplified"
mesh_save_dir = "../DA2/data/meshes"
#convert meshes in directory to to stl


for filename in os.listdir(mesh_root_dir):
    mesh_file_path = os.path.join(mesh_root_dir, filename)
    if os.path.isfile(mesh_file_path):
        mesh = trimesh.load_mesh(mesh_file_path)
        mesh.export(mesh_save_dir + "/" + filename[:-3] + "stl")
        print(filename + " converted to stl")