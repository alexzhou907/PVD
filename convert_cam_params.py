
from glob import glob
import re
import argparse
import numpy as np
from pathlib import Path
import os

def raw_camparam_from_xml(path, pose="lookAt"):
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    elm = tree.find("./sensor/transform/" + pose)
    camparam = elm.attrib
    origin = np.fromstring(camparam['origin'], dtype=np.float32, sep=',')
    target = np.fromstring(camparam['target'], dtype=np.float32, sep=',')
    up = np.fromstring(camparam['up'], dtype=np.float32, sep=',')
    height = int(
        tree.find("./sensor/film/integer[@name='height']").attrib['value'])
    width = int(
        tree.find("./sensor/film/integer[@name='width']").attrib['value'])

    camparam = dict()
    camparam['origin'] = origin
    camparam['up'] = up
    camparam['target'] = target
    camparam['height'] = height
    camparam['width'] = width
    return camparam

def get_cam_pos(origin, target, up):
    inward = origin - target
    right = np.cross(up, inward)
    up = np.cross(inward, right)
    rx = np.cross(up, inward)
    ry = np.array(up)
    rz = np.array(inward)
    rx /= np.linalg.norm(rx)
    ry /= np.linalg.norm(ry)
    rz /= np.linalg.norm(rz)

    rot = np.stack([
        rx,
        ry,
        -rz
    ], axis=0)


    aff = np.concatenate([
        np.eye(3), -origin[:,None]
    ], axis=1)


    ext = np.matmul(rot, aff)

    result = np.concatenate(
        [ext, np.array([[0,0,0,1]])], axis=0
    )



    return result



def convert_cam_params_all_views(datapoint_dir, dataroot, camera_param_dir):
    depths = sorted(glob(os.path.join(datapoint_dir, '*depth.png')))
    cam_ext = ['_'.join(re.sub(dataroot.strip('/'), camera_param_dir.strip('/'), f).split('_')[:-1])+'.xml' for f in depths]


    for i, (f, pth) in enumerate(zip(cam_ext, depths)):
        if not os.path.exists(f):
            continue
        params=raw_camparam_from_xml(f)
        origin, target, up, width, height = params['origin'], params['target'], params['up'],\
                                            params['width'], params['height']

        ext_matrix = get_cam_pos(origin, target, up)

        #####
        diag = (0.036 ** 2 + 0.024 ** 2) ** 0.5
        focal_length = 0.05
        res = [480, 480]
        h_relative = (res[1] / res[0])
        sensor_width = np.sqrt(diag ** 2 / (1 + h_relative ** 2))
        pix_size = sensor_width / res[0]

        K = np.array([
            [focal_length / pix_size, 0, (sensor_width / pix_size - 1) / 2],
            [0, -focal_length / pix_size, (sensor_width * (res[1] / res[0]) / pix_size - 1) / 2],
            [0, 0, 1]
        ])

        np.savez(pth.split('depth.png')[0]+ 'cam_params.npz', extr=ext_matrix, intr=K)


def main(opt):
    dataroot_dir = Path(opt.dataroot)

    leaf_subdirs = []

    for dirpath, dirnames, filenames in os.walk(dataroot_dir):
        if (not dirnames) and opt.mitsuba_xml_root not in dirpath:
            leaf_subdirs.append(dirpath)



    for k, dir_ in enumerate(leaf_subdirs):
        print('Processing dir {}/{}: {}'.format(k, len(leaf_subdirs), dir_))

        convert_cam_params_all_views(dir_, opt.dataroot, opt.mitsuba_xml_root)





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataroot', type=str, default='GenReData/')
    args.add_argument('--mitsuba_xml_root', type=str, default='GenReData/genre-xml_v2')

    opt = args.parse_args()

    main(opt)
