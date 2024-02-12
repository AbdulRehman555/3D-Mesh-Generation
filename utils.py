import os
import torch
import trimesh
import pytorch3d
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from plyfile import PlyData, PlyElement


def plot_pointcloud(
    vertices,
    alpha=.5,
    title=None,
    max_points=10_000,
    xlim=(-1, 1),
    ylim=(-1, 1),
    zlim=(-1, 1)
    ):
    """Plot a pointcloud tensor of shape (N, coordinates)
    """
    vertices = vertices.cpu()

    assert len(vertices.shape) == 2
    N, dim = vertices.shape
    assert dim==2 or dim==3

    if N > max_points:
        vertices = np.random.default_rng().choice(vertices, max_points, replace=False)
    fig = plt.figure(figsize=(6,6))
    if dim == 2:
        ax = fig.add_subplot(111)
    elif dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel("z")
        ax.set_zlim(zlim)
        ax.view_init(elev=120., azim=270)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # ax.set_axis_off()

    ax.scatter(*vertices.T, alpha=alpha, marker=',', lw=.5, s=1, color='black')
    plt.show(fig)
    

def generate_camera_locations(center: torch.Tensor, radius: float, num_points: int) -> torch.Tensor:
    """ Generate camera locations on the circumference of a circle along the
        xz-plane.

    Args:
        center: location of the center of the circle of shape (3,).
                We need to sample points on the circumference of the circle.
        radius: radius of the circle.
        num_points: number of points
    Returns:
        camera_locations: location of the cameras of shape (num_points, 3)
    """
    theta = torch.linspace(0, 2 * torch.pi, num_points, dtype=center.dtype, device=center.device)
    x = center[0] + radius * torch.cos(theta)
    z = center[2] + radius * torch.sin(theta)

    camera_locations = torch.stack([z, torch.zeros_like(x), x], dim=1)
    # print(camera_locations)

    return camera_locations


def get_look_at_views(points: torch.Tensor, look_at_points: torch.Tensor):
    """ Compute the world2cam rotation 'R' and translation 'T' using the camera
        locations 'points' and the look_at_points. Use the look_at_view_transform
        to get the R and T.

    Args:
        points: location of the cameras of shape (..., 3)
        look_at_points: location where the cameras are pointed at of shape (..., 3)
    Returns:
        R: rotation matrix for the world2cam matrix
        T: translation for the world2cam matrix
    """

    R, T = pytorch3d.renderer.look_at_view_transform(at=look_at_points, eye=points)
    return R.to(points.device), T.to(points.device)


def get_normalized_pixel_coordinates_pt3d(
    y_resolution: int,
    x_resolution: int,
    device: torch.device = torch.device('cpu')
):
    """For an image with y_resolution and x_resolution, return a tensor of pixel coordinates
    normalized to lie in [0, 1], with the origin (0, 0) in the bottom left corner,
    the x-axis pointing right, and the y-axis pointing up. The top right corner
    being at (1, 1).

    Returns:
        xy_pix: a meshgrid of values from [0, 1] of shape 
                (y_resolution, x_resolution, 2)
    """
    xs = torch.linspace(1, 0, steps=x_resolution)  # Inverted the order for x-coordinates
    ys = torch.linspace(1, 0, steps=y_resolution)  # Inverted the order for y-coordinates
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    return torch.cat([x.unsqueeze(dim=2), y.unsqueeze(dim=2)], dim=2).to(device)


def clean_mesh(vertices: torch.Tensor, faces: torch.Tensor, edge_threshold: float = 0.1, min_triangles_connected: int = -1, fill_holes: bool = True) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Performs the following steps to clean the mesh:

    1. edge_threshold_filter
    2. remove_duplicated_vertices, remove_duplicated_triangles, remove_degenerate_triangles
    3. remove small connected components
    4. remove_unreferenced_vertices
    5. fill_holes

    :param vertices: (3, N) torch.Tensor of type torch.float32
    :param faces: (3, M) torch.Tensor of type torch.long
    :param colors: (3, N) torch.Tensor of type torch.float32 in range (0...1) giving RGB colors per vertex
    :param edge_threshold: maximum length per edge (otherwise removes that face). If <=0, will not do this filtering
    :param min_triangles_connected: minimum number of triangles in a connected component (otherwise removes those faces). If <=0, will not do this filtering
    :param fill_holes: If true, will perform trimesh fill_holes step, otherwise not.

    :return: (vertices, faces, colors) tuple as torch.Tensors of similar shape and type
    """
    if edge_threshold > 0:
        # remove long edges
        faces = edge_threshold_filter(vertices, faces, edge_threshold)

    # cleanup via open3d
    mesh = torch_to_o3d_mesh(vertices, faces) #, colors)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    if min_triangles_connected > 0:
        # remove small components via open3d
        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_triangles_connected
        mesh.remove_triangles_by_mask(triangles_to_remove)

    # cleanup via open3d
    mesh.remove_unreferenced_vertices()

    if fill_holes:
        # misc cleanups via trimesh
        mesh = o3d_to_trimesh(mesh)
        mesh.process()
        mesh.fill_holes()
        return mesh
        # return trimesh_to_torch(mesh, v=vertices, f=faces) #, c=colors)
    # else:
    #     return o3d_mesh_to_torch(mesh, v=vertices, f=faces) #, c=colors)
    
    
def torch_to_o3d_mesh(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.T.cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(faces.T.cpu().numpy())
    # mesh.vertex_colors = o3d.utility.Vector3dVector(colors.T.cpu().numpy())
    return mesh


def o3d_to_trimesh(mesh: o3d.geometry.TriangleMesh):
    return trimesh.base.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=(np.asarray(mesh.vertex_colors).clip(0, 1) * 255).astype(np.uint8),
        process=False)


def edge_threshold_filter(vertices, faces, edge_threshold=0.1):
    """
    Only keep faces where all edges are smaller than edge_threshold.
    Will remove stretch artifacts that are caused by inconsistent depth at object borders

    :param vertices: (3, N) torch.Tensor of type torch.float32
    :param faces: (3, M) torch.Tensor of type torch.long
    :param edge_threshold: maximum length per edge (otherwise removes that face).

    :return: filtered faces
    """

    p0, p1, p2 = vertices[:, faces[0]], vertices[:, faces[1]], vertices[:, faces[2]]
    d01 = torch.linalg.vector_norm(p0 - p1, dim=0)
    d02 = torch.linalg.vector_norm(p0 - p2, dim=0)
    d12 = torch.linalg.vector_norm(p1 - p2, dim=0)

    mask_small_edge = (d01 < edge_threshold) * (d02 < edge_threshold) * (d12 < edge_threshold)
    faces = faces[:, mask_small_edge]

    return faces


def trimesh_to_torch(mesh: trimesh.base.Trimesh, v=None, f=None, c=None):
    vertices = torch.from_numpy(np.asarray(mesh.vertices)).T
    if v is not None:
        vertices = vertices.to(v)
    faces = torch.from_numpy(np.asarray(mesh.faces)).T
    if f is not None:
        faces = faces.to(f)
    colors = torch.from_numpy(np.asarray(mesh.visual.vertex_colors, dtype=float) / 255).T[:3]
    if c is not None:
        colors = colors.to(c)
    return vertices, faces, colors


def save_mesh(mesh, filename):
    try:
        mesh.export(filename)
    except Exception as e:
        print(f"Error during export: {e}")
        
        
def get_mesh(world_space_points, depth, H, W):
    # define vertex_ids for triangulation
    '''
    00---01
    |    |
    10---11
    '''
    vertex_ids = torch.arange(H*W).reshape(H, W).to(depth.device)
    vertex_00 = remapped_vertex_00 = vertex_ids[:H-1, :W-1]
    vertex_01 = remapped_vertex_01 = (remapped_vertex_00 + 1)
    vertex_10 = remapped_vertex_10 = (remapped_vertex_00 + W)
    vertex_11 = remapped_vertex_11 = (remapped_vertex_00 + W + 1)


    # triangulation: upper-left and lower-right triangles from image structure
    faces_upper_left_triangle = torch.stack(
        [remapped_vertex_00.flatten(), remapped_vertex_10.flatten(), remapped_vertex_01.flatten()],  # counter-clockwise orientation
        dim=0
    )
    faces_lower_right_triangle = torch.stack(
        [remapped_vertex_10.flatten(), remapped_vertex_11.flatten(), remapped_vertex_01.flatten()],  # counter-clockwise orientation
        dim=0
    )

    # filter faces with -1 vertices and combine
    mask_upper_left = torch.all(faces_upper_left_triangle >= 0, dim=0)
    faces_upper_left_triangle = faces_upper_left_triangle[:, mask_upper_left]
    mask_lower_right = torch.all(faces_lower_right_triangle >= 0, dim=0)
    faces_lower_right_triangle = faces_lower_right_triangle[:, mask_lower_right]
    faces = torch.cat([faces_upper_left_triangle, faces_lower_right_triangle], dim=1)
    
    # clean mesh
    mesh = clean_mesh(
        vertices=world_space_points,
        faces=faces,
    )
    
    return mesh


