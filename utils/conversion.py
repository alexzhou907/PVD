
from skimage import measure
import numpy as np


def get_mesh(tsdf_vol, color_vol, threshold=0, vol_max=.5, vol_min=-.5):
    """Compute a mesh from the voxel volume using marching cubes.
    """
    vol_origin = vol_min
    voxel_size = (vol_max - vol_min) / tsdf_vol.shape[-1]

    # Marching cubes
    verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=threshold)
    verts_ind = np.round(verts).astype(int)
    verts = verts * voxel_size + vol_origin  # voxel grid coordinates to world coordinates

    # Get vertex colors
    if color_vol is None:
        return verts, faces, norms
    colors = color_vol[:, verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]].T

    return verts, faces, norms, colors


def get_point_cloud(tsdf_vol, color_vol, vol_max=0.5, vol_min=-0.5):
    vol_origin = vol_min
    voxel_size = (vol_max - vol_min) / tsdf_vol.shape[-1]
    # Marching cubes
    verts = measure.marching_cubes_lewiner(tsdf_vol, level=0)[0]
    verts_ind = np.round(verts).astype(int)
    verts = verts * voxel_size + vol_origin

    # Get vertex colors
    colors = color_vol[:, verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]].T

    pc = np.hstack([verts, colors])

    return pc

def sparse_to_dense_voxel(coords, feats, res):
    coords = coords.astype('int64', copy=False)
    a = np.zeros((res, res, res), dtype=feats.dtype)

    a[coords[:,0],coords[:,1],coords[:,2] ] = feats[:,0].astype(a.dtype, copy=False)

    return a