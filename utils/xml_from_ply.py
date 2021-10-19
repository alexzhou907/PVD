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


def process_one(shape_dir):
    pc_paths = glob.glob(os.path.join(shape_dir, "fake*.ply"))
    pc_paths = sorted(pc_paths)

    xml_paths = [re.sub('.ply', '.xml', os.path.basename(pth)) for pth in pc_paths]

    gen_pcs = []
    for path in pc_paths:
        sample_pts = trimesh.load(path)
        sample_pts = np.array(sample_pts.vertices)
        gen_pcs.append(sample_pts)

    raw_pc = np.array(trimesh.load(os.path.join(shape_dir, "raw.ply")).vertices)
    raw_pc = np.concatenate([raw_pc, np.tile(raw_pc[0:1], (gen_pcs[0].shape[0]-raw_pc.shape[0],1))])

    gen_pcs.append(raw_pc)
    gen_pcs = np.stack(gen_pcs, axis=0)
    xml_paths.append('raw.xml')

    write_to_xml_batch(os.path.dirname(pc_paths[0]), gen_pcs, xml_paths)


def process(args):
    shape_names = [n for n in sorted(os.listdir(args.src)) if
                   os.path.isdir(os.path.join(args.src, n)) and not n.startswith('x')]

    all_shape_dir = [os.path.join(args.src, name) for name in shape_names]

    Parallel(n_jobs=10, verbose=2)(delayed(process_one)(path) for path in all_shape_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    args = parser.parse_args()

    process_one(args)


if __name__ == '__main__':
    main()