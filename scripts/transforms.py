import argparse
import json
import math
import os
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from tqdm import tqdm

checkpoint = 'checkpoint'


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return float(fm)


def dump_text(path, params):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, params, fmt='%.8f')


def dump_flame(flame, frame_id, output):
    # params = ['exp', 'tex', 'sh', 'eyes', 'jaw']
    params = ['exp']
    for param in params:
        coeff = flame[param]
        coeff = coeff[0].flatten('F')
        dump_text(f'{output}/flame/{param}/{frame_id}.txt', coeff)


def dump_intrinsics(frame):
    data = {}
    payload = torch.load(frame)

    # Intrinsic
    h = payload['img_size'][0]
    w = payload['img_size'][1]

    oepncv = payload['opencv']
    K = oepncv['K'][0]

    # OpenCV
    cx = K[0, 2]
    cy = K[1, 2]
    fl_x = K[0, 0]
    fl_y = K[1, 1]
    angle_x = math.atan(w / (fl_x * 2)) * 2
    angle_y = math.atan(h / (fl_y * 2)) * 2
    fovx = angle_x * 180 / math.pi
    fovy = angle_y * 180 / math.pi

    data['w'] = w
    data['h'] = h
    data['cx'] = cx
    data['cy'] = cy
    data['fl_x'] = fl_x
    data['fl_y'] = fl_y
    data['camera_angle_x'] = angle_x
    data['camera_angle_y'] = angle_y
    data['x_fov'] = fovx
    data['y_fov'] = fovy

    data['integer_depth_scale'] = 1 / 1000

    return data


def create_input_images(src, dst):
    src = str(Path(src).parent)
    os.system(f'cp -r {src}/input/* {dst}')


def dump_frame(payload):
    frame, src, output = payload
    payload = torch.load(frame)
    frame_id = payload['frame_id']
    mesh_path = frame.replace('.frame', '.ply').replace(checkpoint, 'mesh')
    if not os.path.exists(mesh_path):
        return None

    trimesh.load(mesh_path, process=False).export(f'{output}/meshes/' + frame_id + '.obj')

    depth_path = frame.replace(checkpoint, 'depth').replace('.frame', '.png')
    if os.path.exists(depth_path):
        os.system(f'cp {depth_path} {output}/depth/{frame_id}.png')

    img = f'images/{frame_id}.png'
    depth = f'depth/{frame_id}.png'

    # Flame
    dump_flame(payload['flame'], frame_id, output)

    oepncv = payload['opencv']
    R = oepncv['R'][0]
    t = oepncv['t'][0]

    # Extrinsic
    w2c = np.eye(4)
    w2c[0:3, 0:3] = R
    w2c[0:3, 3] = t

    c2w = np.linalg.inv(w2c)

    data_frame = {
        'transform_matrix': c2w,
        'file_path': img,
        'mesh_path': f'meshes/{frame_id}.obj',
        'exp_path': f'flame/exp/{frame_id}.txt',
        'depth_path': depth,
        'seg_mask_path': depth.replace('depth', 'seg_mask')
    }

    return data_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, help='Input. Specify absolute path.', required=True)
    parser.add_argument('-o', type=str, help='Output. Specify absolute path.', required=True)
    parser.add_argument('-t', type=int, default=350, help='Number of testing images. From the end of the sequence')
    args = parser.parse_args()

    input = args.i
    output = args.o
    test_frames = args.t

    src = input + f'/{checkpoint}/'

    Path(f'{output}/meshes/').mkdir(parents=True, exist_ok=True)
    Path(f'{output}/flame/').mkdir(parents=True, exist_ok=True)
    Path(f'{output}/background/').mkdir(parents=True, exist_ok=True)
    Path(f'{output}/depth/').mkdir(parents=True, exist_ok=True)

    print('Scanning location ' + src + '/*.frame')

    frames = sorted(glob(src + '/*.frame'))
    data = dump_intrinsics(frames[0])
    data['frames'] = []

    os.system(f'cp {input}/canonical.obj {output}/canonical.obj')

    print(f'Processing {len(frames)} frames!')

    create_input_images(src, f'{output}/background/')

    with Pool(processes=8) as pool:
        params = [(frames[i], src, output) for i in range(len(frames))]
        for task in tqdm(pool.imap_unordered(dump_frame, params), total=len(frames)):
            if task is not None:
                data['frames'].append(task)

    for f in data["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()

    for key in data.keys():
        if "frames" in key:
            continue
        data[key] = float(data[key])

    print(f'Generated = ' + str(len(data["frames"])) + ' frames!')

    with open(f'{output}/transforms.json', 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)

    train = data["frames"][:-test_frames]
    test = data["frames"][-test_frames:]
    val = data["frames"][-20:]

    data["frames"] = train
    with open(f'{output}/transforms_train.json', 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)

    data["frames"] = test
    with open(f'{output}/transforms_test.json', 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)

    data["frames"] = val
    with open(f'{output}/transforms_val.json', 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)
