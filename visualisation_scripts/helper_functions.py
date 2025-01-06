import os.path as osp
import torch
import numpy as np
import trimesh
import random
import copy
from scipy.spatial.transform import Rotation
from imageio import get_writer, imread
from matplotlib import cm
import h5py
from autolab_core import RigidTransform
import os
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import json

def random_rotation_matrix():
    # random quaternion
    # quat = np.random.random(4)
    # quat = [0,0,1,1]
    # normalize to represent rotation
    quat = quat / np.linalg.norm(quat)
    # express as transformation matrix with no translaton
    rot = np.eye(4)
    rot[:-1,:-1] = Rotation.from_quat(quat).as_matrix()
    return rot

def count_objects_in_pc(filepath, mesh_root_dir, n_pts=10000):
    # Function to count the number of objects in a pointcloud
    # This is done by clustering the pointcloud and counting the number of clusters
    # The number of clusters is the number of objects
    obj_mesh = load_mesh(filepath, mesh_root_dir)
    obj_pc = obj_mesh.sample(n_pts)
    obj_pc = obj_pc - np.mean(obj_pc, axis=0)
    clustering = DBSCAN(eps=0.2, min_samples=500).fit(obj_pc)
    print(len(np.unique(clustering.labels_)))
    trimesh_show([obj_pc])
    return len(np.unique(clustering.labels_))

def show_obj(filepath, mesh_root_dir, n_pts=10000):
    #Show the object
    #Save the mesh to a file based on a keypress
    obj_mesh = load_mesh(filepath, mesh_root_dir)
    trimesh.Scene([obj_mesh]).show()
    


# Function to visualize pointcloud with trimesh
def trimesh_show(np_pcd_list, color_list=None, rand_color=False, show=True):
    #colormap = matplotlib.colormaps.get_cmap(len(np_pcd_list))
    colormap = cm.get_cmap('brg', len(np_pcd_list))
    # colormap = cm.get_cmap('gist_ncar_r', len(np_pcd_list))
    colors = [
        (np.asarray(colormap(val)) * 255).astype(np.int32) for val in np.linspace(0.05, 0.95, num=len(np_pcd_list))
    ]
    if color_list is None:
        if rand_color:
            color_list = []
            for i in range(len(np_pcd_list)):
                color_list.append((np.random.rand(3) * 255).astype(np.int32).tolist() + [255])
        else:
            color_list = colors
    
    tpcd_list = []
    for i, pcd in enumerate(np_pcd_list):
        tpcd = trimesh.PointCloud(pcd)
        tpcd.colors = np.tile(color_list[i], (tpcd.vertices.shape[0], 1))

        tpcd_list.append(tpcd)
    
    scene = trimesh.Scene()
    scene.add_geometry(tpcd_list)
    if show:
        scene.show() 

    return scene

# Function to introduce blobs (missing points) in order to simultae partial object view
def remove_blobs(pointcloud, n_blobs, blob_size):
    """
    Remove coherent blobs from the pointcloud to simulate partial view.

    :param pointcloud: The input pointcloud as a numpy array of shape (N, 3).
    :param n_blobs: Number of blobs to remove.
    :param blob_size: Number of points to remove in each blob.
    :return: The pointcloud with blobs removed.
    """
    # Ensure blob_size is not larger than the number of points
    blob_size = min(blob_size, len(pointcloud))

    # Copy the pointcloud to avoid modifying the original
    modified_pointcloud = np.copy(pointcloud)

    for _ in range(n_blobs):
        # Randomly choose a point to be the center of the blob
        blob_center_idx = np.random.randint(len(modified_pointcloud))
        blob_center = modified_pointcloud[blob_center_idx]

        # Compute distances from the blob center to all points
        distances = np.linalg.norm(modified_pointcloud - blob_center, axis=1)

        # Find the indices of the closest points to form the blob
        blob_indices = np.argsort(distances)[:blob_size]

        # Remove the points belonging to the blob
        modified_pointcloud = np.delete(modified_pointcloud, blob_indices, axis=0)

    return modified_pointcloud

# from DA2 repo (DA2_tools/utils.py)



# Load an object mesh from h5py file
def load_mesh(filename, mesh_root_dir, scale=None):
    """Load a mesh from a JSON or HDF5 file from the grasp dataset. The mesh will be scaled accordingly.

    Args:
        filename (str): JSON or HDF5 file name.
        scale (float, optional): If specified, use this as scale instead of value from the file. Defaults to None.

    Returns:
        trimesh.Trimesh: Mesh of the loaded object.
    """
    if filename.endswith(".json"):
        data = json.load(open(filename, "r"))
        mesh_fname = data["object"].decode('utf-8')
        mesh_scale = data["object_scale"] if scale is None else scale
    elif filename.endswith(".h5"):
        data = h5py.File(filename, "r")
        mesh_fname = data["object/file"][()].decode('utf-8')
        mesh_scale = data["object/scale"][()] if scale is None else scale
    else:
        raise RuntimeError("Unknown file ending:", filename)

    obj_mesh = trimesh.load(os.path.join(mesh_root_dir, mesh_fname))
    # obj_mesh = trimesh.load(os.path.join(mesh_root_dir, mesh_fname.removesuffix(".obj") + ".stl"))

    obj_mesh.apply_transform(RigidTransform(np.eye(3), -obj_mesh.centroid).matrix)
    obj_mesh = obj_mesh.apply_scale(mesh_scale)

    return obj_mesh

def save_to_json(filename, save_directory, grasps, metric, scale = None):
    data = h5py.File(filename, "r")
    object_name = data["object/file"][()].decode('utf-8')
    object_name = object_name.removesuffix(".obj") + ".stl"
    object_name = "meshes/" + object_name
    object_scale = data["object/scale"][()] if scale is None else scale
    object_scale = float(object_scale)
    #Save everything in a json file
    grasps = grasps.tolist()
    #check type of metric
    if type(metric) == np.ndarray:
        metric = metric.tolist()
    json_dict = {"transforms": grasps, "object": object_name, "object_scale": object_scale, "metric": metric}
    with open(save_directory + os.path.basename(filename).removesuffix(".h5") + ".json" , 'w') as f:
        json.dump(json_dict, f)

# Function to load grasps from h5py file, as 4x4 gripper transform matrices + quality metrics
def load_dual_grasps(filename):
    """Load transformations and qualities of grasps from a JSON file from the dataset.

    Args:
        filename (str): HDF5 or JSON file name.

    Returns:
        np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
        np.ndarray: List of binary values indicating grasp success in simulation.
    """
    data = h5py.File(filename, "r")
    T = np.array(data["grasps/transforms"])
    print(T.shape)
    #Scale the grasps to the correct size
    # scale =  0.5
    # S = np.diag([scale, scale, scale, 1])
    # for i in range(len(T)):
    #     T[i][0] = S.dot(T[i][0])
    #     T[i][1] = S.dot(T[i][1])


    # q = np.array(data["grasps/qualities/gazebo/physical_qualities"])
    f = np.array(data["grasps/qualities/Force_closure"])
    d = np.array(data["grasps/qualities/Dexterity"])
    t = np.array(data["grasps/qualities/Torque_optimization"])

    return T, f, d, t


# Function to render the mesh of the gripper (using robotiq here)
def create_robotiq_marker(color=[0, 0, 255], tube_radius=0.001, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 0],
            [4.10000000e-02, -7.27595772e-12, 0.067500],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 0],
            [-4.100000e-02, -7.27595772e-12, 0.067500],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002, sections=sections, segment=[[0, 0, -0.067500/2], [0, 0, 0]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[[-0.085/2, 0, 0], [0.085/2, 0, 0]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    # z axis to x axis
    R = np.array([[1,0,0],[0,0,1],[0,-1,0]]).reshape(3,3)
    t =  np.array([0, 0, 0]).reshape(3,1)
    #
    T = np.r_[np.c_[R, t], [[0, 0, 0, 1]]]
    tmp.apply_transform(T)

    return tmp

# Function to combine the three quality metrics into one
# Grasps with high quality in all three metrics are preferred
def combine_metrics(qua_for, qua_dex, qua_tor):
    qua_combined = qua_dex + qua_for + qua_tor
    # qua_combined = (1 * qua_for) + (1.75 * qua_dex) +  (0.25 * qua_tor)
    # print(qua_combined)
    return qua_combined

def filter_grasps_based_on_metric_partial(filepath,
                                         metric = "dex", # for, dex, tor, combined
                                         threshold = 0.5,
                                         n_pts = 100000, # pointcloud subsampling
                                         n_blobs = 3, # pointcloud masking for partial
                                         blob_size=5000, # pointcloud masking for partial
                                         dist_thresh=0.02, # threshold for filtering out grasps at missing regions
                                         n_grasps = 1, # number of grasps to show
                                         mesh_root_dir = "DA2/data/simplified"
):
    # helper
    def countX(lst, x):
        count = 0
        for ele in lst:
            if (ele == x).all() :
                count = count + 1
        return count

    # mesh to partial pointcloud, as above
    def mesh_to_partial_pc():
        obj_mesh = load_mesh(filepath, mesh_root_dir)
        # obj_mesh = obj_mesh.apply_transform(random_rotation_matrix())
        # trimesh.Scene([obj_mesh]).show()
        obj_pc = obj_mesh.sample(n_pts)
        obj_pc = obj_pc - np.mean(obj_pc, axis=0)


        # mask out blobs to simulate partial view
        obj_pc_partial = remove_blobs(np.array(obj_pc), n_blobs=n_blobs, blob_size=blob_size)
        return obj_pc_partial, obj_mesh

    # check if grasp is in missing region
    def check_grasp_visible(t1, t2, pc, dist_thresh):

        # Function to compute distance of query points to closest point in pointcloud
        def compute_minimum_distances(pointcloud, query_points):
            tree = cKDTree(pointcloud)
            distances, _ = tree.query(query_points, k=1)
            return distances
        
        # grasp centers
        # print(t1)
        center1 = t1[:3, -1]
        center2 = t2[:3, -1]
        # print(center1)
        # distance from closest point
        distances = compute_minimum_distances(pc, np.stack([center1, center2], axis=0))
        # print(distances)
        if distances[0] <= dist_thresh and distances[1] <= dist_thresh:
            return True
        return False
        
    grasp_transforms, qua_for, qua_dex, qua_tor = load_dual_grasps(filepath)
    qua_combined = combine_metrics(qua_for, qua_dex, qua_tor)
    
    if metric == "for":
        qua_metric = qua_for
    elif metric == "tor":
        qua_metric = qua_tor
    elif metric == "dex":
        qua_metric = qua_dex
    elif metric == "combined":
        qua_metric = qua_combined
        qua_thresh = threshold + threshold + threshold
    else:
        raise ValueError(f"Unknown metric {metric}. Choose one of [for, tor, dex, combined]")
    
    # print(metric)
    # print(qua_metric.max())
    
    qua_thresh = threshold

    #Load the pointcloud and meshes
    obj_pc, obj_mesh = mesh_to_partial_pc()
    # #Check if there are multiple objects in the same file
    # print(count_objects_in_pc(filepath, mesh_root_dir))
    if len(np.where( (qua_metric >= qua_thresh))[0]) > n_grasps: # if we've got more "good" grasps than what we want
        # we have to sample n_grasps of them
        #Select the grasp with the highest quality combined metric (not randomly)
        # grasps_select = grasp_transforms[np.random.choice(np.where(qua_metric >= qua_thresh)[0], n_grasps)]
        best_grasps = []
        grasps_metric = []
        while len(best_grasps) < n_grasps:
            # best_grasps.append(qua_combined.argmax())
            #Check if the grasp with the highest quality metric is visible, if so add it to the list and remove it from the list of grasps to check
            #If it is not visible, remove it from the list of grasps to check and select the next best grasp
            
            # if check_grasp_visible(grasp_transforms[index][0], grasp_transforms[index][1], obj_pc, dist_thresh):
            index = qua_metric.argmax()
            if qua_metric[index] < qua_thresh:
                print("no grasps left with good quality metric")
                break
            best_grasps.append(index)
            grasps_metric.append(qua_metric[index][0])
            qua_metric[index] = 0
            # else:
            #     qua_metric[index] = 0
                # print("Grasp in missing region, skipping")
            if sum(qua_metric) == 0:
                print("No more visible grasps available, stopping.")
                break

        grasps_select = grasp_transforms[best_grasps]
    else:
        print("here")
        #Select all grasps with a quality metric above the threshold
        grasps_select = grasp_transforms[np.where(qua_metric >= qua_thresh)[0]]
        grasps_metric = qua_metric[np.where(qua_metric >= qua_thresh)[0]]
    

   
    database = []
    wave = n_grasps//3
    if wave == 0:
        wave = 1
    #Successful grasps are marked with a marker
    #Succesfull grasps to save are the transforms of the grasps
    successful_grasps = []
    successful_grasps_to_save = grasps_select
    marker = []
    for i, (t1, t2) in enumerate(grasps_select):
        # t1[:3, :3] = np.eye(3)
        # t2[:3, :3] = np.eye(3)
        t1= np.eye(4)
        t2 = np.eye(4)
        current_t1 = countX(database, t1)
        current_t2 = countX(database, t2)
        color = i/wave*255
        code1 = color if color<=255 else 0
        code2 = color%255 if color>255 and color<=510 else 0
        code3 = color%510 if color>510 and color<=765 else 0
        successful_grasps.append((create_robotiq_marker(color=[code1, code2, code3]).apply_transform(t1), create_robotiq_marker(color=[code1, code2, code3]).apply_transform(t2)))

        trans1 = t1.dot(np.array([0,-0.067500/2-0.02*current_t1,0,1]).reshape(-1,1))[0:3]
        trans2 = t2.dot(np.array([0,-0.067500/2-0.02*current_t2,0,1]).reshape(-1,1))[0:3]

        tmp1 = trimesh.creation.icosphere(radius = 0.01).apply_transform(RigidTransform(np.eye(3), trans1).matrix)
        tmp1.visual.face_colors = [code1, code2, code3]
        tmp2 = trimesh.creation.icosphere(radius = 0.01).apply_transform(RigidTransform(np.eye(3), trans2).matrix)
        tmp2.visual.face_colors = [code1, code2, code3]
        marker.append(copy.deepcopy(tmp1))
        marker.append(copy.deepcopy(tmp2))
        database.append(t1)
        database.append(t2)
    
    return obj_pc, obj_mesh, successful_grasps, grasps_metric, marker, successful_grasps_to_save


def filter_negative_grasps(filepath,
                                         metric = "dex", # for, dex, tor, combined
                                         n_pts = 100000, # pointcloud subsampling
                                         n_blobs = 3, # pointcloud masking for partial
                                         blob_size=5000, # pointcloud masking for partial
                                         dist_thresh=0.02, # threshold for filtering out grasps at missing regions
                                         n_grasps = 1, # number of grasps to show
                                         mesh_root_dir = "DA2/data/simplified"
):
    # helper
    def countX(lst, x):
        count = 0
        for ele in lst:
            if (ele == x).all() :
                count = count + 1
        return count

    # mesh to partial pointcloud, as above
    def mesh_to_partial_pc():
        obj_mesh = load_mesh(filepath, mesh_root_dir)
        # obj_mesh = obj_mesh.apply_transform(random_rotation_matrix())
        # trimesh.Scene([obj_mesh]).show()
        obj_pc = obj_mesh.sample(n_pts)
        obj_pc = obj_pc - np.mean(obj_pc, axis=0)


        # mask out blobs to simulate partial view
        obj_pc_partial = remove_blobs(np.array(obj_pc), n_blobs=n_blobs, blob_size=blob_size)
        return obj_pc_partial, obj_mesh

    # check if grasp is in missing region
    def check_grasp_visible(t1, t2, pc, dist_thresh):

        # Function to compute distance of query points to closest point in pointcloud
        def compute_minimum_distances(pointcloud, query_points):
            tree = cKDTree(pointcloud)
            distances, _ = tree.query(query_points, k=1)
            return distances
        
        # grasp centers
        # print(t1)
        center1 = t1[:3, -1]
        center2 = t2[:3, -1]
        # print(center1)
        # distance from closest point
        distances = compute_minimum_distances(pc, np.stack([center1, center2], axis=0))
        # print(distances)
        if distances[0] <= dist_thresh and distances[1] <= dist_thresh:
            return True
        return False
        
    grasp_transforms, qua_for, qua_dex, qua_tor = load_dual_grasps(filepath)
    qua_combined = combine_metrics(qua_for, qua_dex, qua_tor)
    
    if metric == "for":
        qua_metric = qua_for
    elif metric == "tor":
        qua_metric = qua_tor
    elif metric == "dex":
        qua_metric = qua_dex
    elif metric == "combined":
        qua_metric = qua_combined
    else:
        raise ValueError(f"Unknown metric {metric}. Choose one of [for, tor, dex, combined]")
    
    # print(metric)
    
    #Load the pointcloud and meshes
    obj_pc, obj_mesh = mesh_to_partial_pc()
    # #Check if there are multiple objects in the same file
    # print(count_objects_in_pc(filepath, mesh_root_dir))
    neg_metric = []
    neg_grasps = []
    while len(neg_grasps) < n_grasps:
        # best_grasps.append(qua_combined.argmax())
        #Check if the grasp with the highest quality metric is visible, if so add it to the list and remove it from the list of grasps to check
        #If it is not visible, remove it from the list of grasps to check and select the next best grasp
        # if check_grasp_visible(grasp_transforms[qua_metric.argmin()][0], grasp_transforms[qua_metric.argmin()][1], obj_pc, dist_thresh):
        index = qua_metric.argmin()
        neg_grasps.append(index)
        neg_metric.append(qua_metric[index][0])
        qua_metric[index] = 3
        # else:
        #     qua_metric[qua_metric.argmin()] = 3
        #     # print("Grasp in missing region, skipping")
        if sum(qua_metric) == (len(qua_metric) * 3):
            print("No more visible grasps available, stopping.")
            break

    grasps_select = grasp_transforms[neg_grasps]
    

   

    database = []
    wave = n_grasps//3
    if wave == 0:
        wave = 1
    #Successful grasps are marked with a marker
    #Succesfull grasps to save are the transforms of the grasps
    negative_grasps = []
    negative_grasps_to_save = grasps_select
    marker = []
    for i, (t1, t2) in enumerate(grasps_select):
        
        current_t1 = countX(database, t1)
        current_t2 = countX(database, t2)
        color = i/wave*255
        code1 = color if color<=255 else 0
        code2 = color%255 if color>255 and color<=510 else 0
        code3 = color%510 if color>510 and color<=765 else 0
        negative_grasps.append((create_robotiq_marker(color=[code1, code2, code3]).apply_transform(t1), create_robotiq_marker(color=[code1, code2, code3]).apply_transform(t2)))

        trans1 = t1.dot(np.array([0,-0.067500/2-0.02*current_t1,0,1]).reshape(-1,1))[0:3]
        trans2 = t2.dot(np.array([0,-0.067500/2-0.02*current_t2,0,1]).reshape(-1,1))[0:3]

        tmp1 = trimesh.creation.icosphere(radius = 0.01).apply_transform(RigidTransform(np.eye(3), trans1).matrix)
        tmp1.visual.face_colors = [code1, code2, code3]
        tmp2 = trimesh.creation.icosphere(radius = 0.01).apply_transform(RigidTransform(np.eye(3), trans2).matrix)
        tmp2.visual.face_colors = [code1, code2, code3]
        marker.append(copy.deepcopy(tmp1))
        marker.append(copy.deepcopy(tmp2))
        database.append(t1)
        database.append(t2)
    
    return obj_pc, obj_mesh, negative_grasps, neg_metric, marker, negative_grasps_to_save

def show_scene_with_grasps(n_grasps, metric, qua_thresh, obj_pc, obj_mesh, successful_grasps, marker, mode="individual"):
    # visualize sucesful grasps        
    print(f'Showing {n_grasps} grasps with {metric} metric above {qua_thresh}')
    # if type(obj_pc) != None:
        # scene = trimesh_show([obj_pc], show=False)
        # orientation = trimesh.creation.axis(axis_length=0.1)
        # scene.add_geometry(orientation)
        # if mode == "individual":
        #     for succesfull_grasp in successful_grasps:
        #         scene.add_geometry(succesfull_grasp)
        #         scene.show()
        # else:
        #     scene.add_geometry(successful_grasps + marker)
        # scene.add_geometry(successful_grasps)
    # trimesh.Scene([obj_mesh] + successful_grasps + marker).show()
    orientation = trimesh.creation.axis(axis_length=0.1)
    trimesh.Scene(successful_grasps[0][0]).show()
