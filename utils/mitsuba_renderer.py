import numpy as np
from pathlib import Path
import os


def standardize_bbox(pcl, points_per_object, scale=None):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices]  # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    if scale is None:
        scale = np.amax(maxs - mins)
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result


xml_head = \
    """
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="100"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="{}"/>

            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="256"/>
                <integer name="height" value="256"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>

        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
        </bsdf>

    """

xml_ball_segment = \
    """
        <shape type="sphere">
            <float name="radius" value="{}"/>
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

xml_tail = \
    """
        <shape type="rectangle">
            <bsdf type="diffuse">
                <rgb name="reflectance" value="1"/>
            </bsdf>
            <transform name="toWorld">
                <scale x="100" y="100" z="1"/>
                <translate x="0" y="0" z="{}"/>
            </transform>
        </shape>
    	<shape type="sphere">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="2,0,18" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
            	<rgb name="radiance" value="5"/>
            </emitter>
        </shape>
        <shape type="sphere">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="-30,0,18" target="-100,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
            	<rgb name="radiance" value="5"/>
            </emitter>
        </shape>
    </scene>
    """


def colormap_fn(x, y, z):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


color_dict = {'r': [163, 102, 96], 'g': [20, 130, 3],
              'o': [145, 128, 47], 'b': [91, 102, 112], 'p':[133,111,139], 'br':[111,92,81]}

color_map = {'airplane': 'r', 'chair': 'o', 'car': 'b', 'table': 'p', 'lamp':'br'}
fov_map = {'airplane': 12, 'chair': 16, 'car':15,  'table': 13, 'lamp':13}
radius_map = {'airplane': 0.02, 'chair': 0.035, 'car': 0.01, 'table':0.035, 'lamp':0.035}

def write_to_xml_batch(dir, pcl_batch, filenames=None, color_batch=None, cat='airplane'):
    default_color = color_map[cat]
    Path(dir).mkdir(parents=True, exist_ok=True)
    if filenames is not None:
        assert len(filenames) == pcl_batch.shape[0]
    # mins = np.amin(pcl_batch, axis=(0,1))
    # maxs = np.amax(pcl_batch, axis=(0,1))
    # scale = 1; print(np.amax(maxs - mins))

    for k, pcl in enumerate(pcl_batch):
        xml_segments = [xml_head.format(fov_map[cat])]
        pcl = standardize_bbox(pcl, pcl.shape[0])
        pcl = pcl[:, [2, 0, 1]]
        pcl[:, 0] *= -1
        pcl[:, 2] += 0.0125
        for i in range(pcl.shape[0]):
            if color_batch is not None:
                color = color_batch[k, i]
            else:
                color = np.array(color_dict[default_color]) / 255
            # color = colormap_fn(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5-0.0125)
            xml_segments.append(xml_ball_segment.format(radius_map[cat], pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
        xml_segments.append(
            xml_tail.format(pcl[:, 2].min()))

        xml_content = str.join('', xml_segments)

        if filenames is None:
            fn = 'sample_{}.xml'.format(k)
        else:
            fn = filenames[k]
        with open(os.path.join(dir, fn), 'w') as f:
            f.write(xml_content)
