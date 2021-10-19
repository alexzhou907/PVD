import numpy as np
from pathlib import Path
import os


def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices]  # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
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
                <lookat origin="{},{},{}" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="20"/>

            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="480"/>
                <integer name="height" value="480"/>
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


color_dict = {'r': [163, 102, 96], 'p': [133,111,139], 'g': [20, 130, 3],
              'o': [145, 128, 47], 'b': [91, 102, 112]}

color_map = {'airplane': 'r', 'chair': 'o', 'car': 'b', 'table': 'p'}
fov_map = {'airplane': 12, 'chair': 15, 'car':12, 'table':12}
radius_map = {'airplane': 0.0175, 'chair': 0.035, 'car': 0.025, 'table': 0.02}

def write_to_xml_batch(dir, pcl_batch, color_batch=None, cat='airplane', elev=15, azim=45, radius=np.sqrt(18)):
    elev_rad = elev * np.pi / 180
    azim_rad = azim * np.pi / 180

    x = radius * np.cos(elev_rad)*np.cos(azim_rad)
    y = radius * np.cos(elev_rad)*np.sin(azim_rad)
    z = radius * np.sin(elev_rad)

    default_color = color_map[cat]
    Path(dir).mkdir(parents=True, exist_ok=True)
    for k, pcl in enumerate(pcl_batch):
        xml_segments = [xml_head.format(x,y,z)]
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
            xml_segments.append(xml_ball_segment.format(0.0175, pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
        xml_segments.append(
            xml_tail.format(pcl[:, 2].min()))

        xml_content = str.join('', xml_segments)

        with open(os.path.join(dir, 'sample_{}.xml'.format(k)), 'w') as f:
            f.write(xml_content)

def write_to_xml(file, pcl, cat='airplane', elev=15, azim=45, radius=np.sqrt(18)):
    assert pcl.ndim == 2
    elev_rad = elev * np.pi / 180
    azim_rad = azim * np.pi / 180

    x = radius * np.cos(elev_rad)*np.cos(azim_rad)
    y = radius * np.cos(elev_rad)*np.sin(azim_rad)
    z = radius * np.sin(elev_rad)

    default_color = color_map[cat]

    xml_segments = [xml_head.format(x,y,z)]
    pcl = standardize_bbox(pcl, pcl.shape[0])
    pcl = pcl[:, [2, 0, 1]]
    pcl[:, 0] *= -1
    pcl[:, 2] += 0.0125
    for i in range(pcl.shape[0]):
        color = np.array(color_dict[default_color]) / 255
        # color = colormap_fn(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5-0.0125)
        xml_segments.append(xml_ball_segment.format(radius_map[cat], pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
    xml_segments.append(
        xml_tail.format(pcl[:, 2].min()))

    xml_content = str.join('', xml_segments)

    with open(file, 'w') as f:
        f.write(xml_content)
