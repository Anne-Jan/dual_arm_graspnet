from __future__ import print_function
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh
from utils import utils, sample


def get_color_plasma_org(x):
    return tuple([x for i, x in enumerate(plt.cm.plasma(x)) if i < 3])


def get_color_plasma(x):
    return tuple([float(1 - x), float(x), float(0)])


def plot_mesh(ax, mesh, color=(0.5, 0.5, 0.5)):
    """Helper function to plot a trimesh object in matplotlib"""
    if isinstance(mesh, trimesh.Trimesh):
        mesh_vertices = mesh.vertices
        mesh_faces = mesh.faces
        mesh_collection = Poly3DCollection(mesh_vertices[mesh_faces], facecolor=color, linewidths=0.1, alpha=0.5)
        ax.add_collection3d(mesh_collection)


# Set limits for the axes manually to achieve a 1:1:1 aspect ratio
def set_axes_equal(ax):
    """Set equal scaling for all axes in a 3D plot."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    # Get the maximum range across all axes
    max_range = max(x_range, y_range, z_range)

    # Center the plot and set equal limits
    x_mid = 0.5 * (x_limits[0] + x_limits[1])
    y_mid = 0.5 * (y_limits[0] + y_limits[1])
    z_mid = 0.5 * (z_limits[0] + z_limits[1])

    ax.set_xlim3d([x_mid - 0.5 * max_range, x_mid + 0.5 * max_range])
    ax.set_ylim3d([y_mid - 0.5 * max_range, y_mid + 0.5 * max_range])
    ax.set_zlim3d([z_mid - 0.5 * max_range, z_mid + 0.5 * max_range])


def draw_scene(pc,
               grasps=[],
               grasp_scores=None,
               grasp_color=None,
               gripper_color=(0, 1, 0),
               mesh=None,
               show_gripper_mesh=False,
               grasps_selection=None,
               visualize_diverse_grasps=False,
               min_seperation_distance=0.03,
               pc_color=None,
               plasma_coloring=False,
               target_cps=None,
               save_path='./demo/scene.png'):
    """
    Draws the 3D scene for the object and the scene using matplotlib and trimesh.
    Saves the plot as a file (e.g., PNG) for visualization in a headless environment.

    Args:
      pc: point cloud of the object
      grasps: list of 4x4 numpy array indicating the transformation of the grasps.
      grasp_scores: grasps will be colored based on the scores.
      grasp_color: if it is a tuple, sets the color for all the grasps.
      mesh: If not None, shows the mesh of the object (trimesh).
      show_gripper_mesh: If True, shows the gripper mesh for each grasp.
      grasps_selection: if provided, filters the grasps.
      visualize_diverse_grasps: declutters the grasps for better visualization.
      pc_color: colors of point cloud (n x 3 array).
      plasma_coloring: If True, sets the plasma colormap for the pc.
      target_cps: target control points.
      save_path: path where the plot is saved as a .png file.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot point cloud
    if pc_color is not None:
        # Ensure pc_color is in the correct range [0, 1]
        if pc_color.max() > 1.0:  # If colors are in [0, 255], normalize them
            pc_color = pc_color / 255.0

        # Ensure the color array is correctly shaped
        if pc_color.shape[1] == 3:  # If RGB, add alpha channel
            pc_color = np.hstack([pc_color, np.ones((pc_color.shape[0], 1))])

    if pc is not None:
        if pc_color is None:
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], color=(0.1, 0.1, 1), s=1)
        else:
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc_color, s=1)
    
    # Plot mesh if provided
    if mesh is not None:
        if isinstance(mesh, list):
            for elem in mesh:
                plot_mesh(ax, elem)
        else:
            plot_mesh(ax, mesh)

    # Transform and plot grasps
    grasp_pc = np.squeeze(utils.get_control_point_tensor(1, False), 0)
    mid_point = 0.5 * (grasp_pc[2, :] + grasp_pc[3, :])
    
    modified_grasp_pc = []
    modified_grasp_pc.append(grasp_pc[0])
    modified_grasp_pc.append(mid_point)
    modified_grasp_pc.append(grasp_pc[2])
    modified_grasp_pc.append(grasp_pc[4])
    modified_grasp_pc.append(grasp_pc[2])
    modified_grasp_pc.append(grasp_pc[3])
    modified_grasp_pc.append(grasp_pc[5])

    grasp_pc = np.asarray(modified_grasp_pc)

    def transform_grasp_pc(g):
        output = np.matmul(grasp_pc, g[:3, :3].T)
        output += np.expand_dims(g[:3, 3], 0)
        return output

    if grasp_scores is not None:
        indexes = np.argsort(-np.asarray(grasp_scores))
    else:
        indexes = range(len(grasps))

    selected_grasps_so_far = []
    removed = 0

    for ii in range(len(grasps)):
        i = indexes[ii]
        if grasps_selection is not None:
            if grasps_selection[i] == False:
                continue

        g = grasps[i]
        is_diverse = True
        for prevg in selected_grasps_so_far:
            distance = np.linalg.norm(prevg[:3, 3] - g[:3, 3])
            if distance < min_seperation_distance:
                is_diverse = False
                break

        if visualize_diverse_grasps:
            if not is_diverse:
                removed += 1
                continue
            selected_grasps_so_far.append(g)

        # Transform the grasp and plot
        pts = transform_grasp_pc(g)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=gripper_color)

        if show_gripper_mesh:
            gripper_mesh = trimesh.load_mesh('gripper_models/panda_gripper.obj')
            gripper_mesh.apply_transform(g)
            plot_mesh(ax, gripper_mesh, color=gripper_color)

    # Plot target control points if provided
    if target_cps is not None:
        target_cps = np.array(target_cps)
        for i in range(len(target_cps)):
            ax.scatter(target_cps[i, :, 0], target_cps[i, :, 1], target_cps[i, :, 2], color='r', s=10)

    # Set labels and equal scaling
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    #ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    ax.view_init(elev=30, azim=60)

    # Save the plot as a .png file
    plt.savefig(save_path)
    plt.close(fig)

    print(f'Scene saved to {save_path}.')
    print('Removed {} similar grasps.'.format(removed))

