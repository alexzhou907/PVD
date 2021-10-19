import sys

sys.path.append('..')
import argparse
import os
import numpy as np
import trimesh
import glob
from joblib import Parallel, delayed
import re
from utils.mitsuba_renderer import write_to_xml_batch
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotate(vertices, faces):
    '''
    vertices: [numpoints, 3]
    '''
    N = rotation_matrix([1, 0, 0], 3* np.pi / 4).transpose()
    # M = rotation_matrix([0, 1, 0], -np.pi / 2).transpose()


    v, f = vertices.dot(N), faces
    return v, f

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh
def process_one(shape_dir, cat):
    pc_paths = glob.glob(os.path.join(shape_dir, "*.obj"))
    pc_paths = sorted(pc_paths)

    xml_paths = [] #[re.sub('.ply', '.xml', os.path.basename(pth)) for pth in pc_paths]

    gen_pcs = []
    for path in pc_paths:
        sample_mesh = trimesh.load(path, force='mesh')
        v, f = rotate(sample_mesh.vertices,sample_mesh.faces)
        mesh = trimesh.Trimesh(v, f)
        sample_pts = trimesh.sample.sample_surface(mesh, 2048)[0]
        gen_pcs.append(sample_pts)
        xml_paths.append(re.sub('.obj', '.xml', os.path.basename(path)))



    gen_pcs = np.stack(gen_pcs, axis=0)
    write_to_xml_batch(os.path.dirname(pc_paths[0]), gen_pcs, xml_paths, cat=cat)


def process(args):
    shape_names = [n for n in sorted(os.listdir(args.src)) if
                   os.path.isdir(os.path.join(args.src, n)) and not n.startswith('x')]

    all_shape_dir = [os.path.join(args.src, name) for name in shape_names]

    Parallel(n_jobs=10, verbose=2)(delayed(process_one)(path) for path in all_shape_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--cat", type=str)
    args = parser.parse_args()

    process_one(args.src, args.cat)


if __name__ == '__main__':
    main()