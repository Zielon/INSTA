import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm
import open3d as o3d


def transfer(vertices, faces, flame_vertices_mask, template, head_idxs, path):
    output = template.copy()
    output[head_idxs, :] = vertices
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(output)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.remove_vertices_by_mask(flame_vertices_mask)
    o3d.io.write_triangle_mesh(path, mesh)


def main(input):
    template = np.load('topology/average.npy')
    head_idxs = np.load('topology/head_idxs.npy')
    template_face_sparse = np.load('topology/template_face_sparse.npy')
    flame_vertices_mask = np.load('topology/flame_vertices_mask.npy')
    folder = Path(input, 'meshes')
    folder.mkdir(parents=True, exist_ok=True)
    mesh_paths = Path(input, 'meshes')

    canonical = trimesh.load(os.path.join(input, 'canonical.obj'), process=False)

    transfer(canonical.vertices, template_face_sparse, flame_vertices_mask, template, head_idxs, f'{input}/canonical.obj')

    for path in tqdm(sorted(glob(f'{mesh_paths}/*.obj'))):
        mesh = trimesh.load(path, process=False)
        name = Path(path).stem
        output_path = f'{folder}/{name}.obj'

        transfer(mesh.vertices, template_face_sparse, flame_vertices_mask, template, head_idxs, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer Simplified Mesh')
    parser.add_argument('-i', required=True, type=str, help='Tracking results')
    args = parser.parse_args()
    main(args.i)
