import json
import argparse
import os
import shutil
from pathlib import Path
import time
import numpy as np
import trimesh
from tqdm import tqdm


import numpy as np


def tri2tet(tris):
    v1 = tris[:, 0, :]
    v2 = tris[:, 1, :]
    v3 = tris[:, 2, :]

    e21 = v2 - v1
    e31 = v3 - v1
    n = np.cross(e21, e31)
    n = n / np.sqrt(np.linalg.norm(n, axis=1, keepdims=True))

    v4 = v1 + n

    return v1, v2, v3, v4


def gradient(v1, v2, v3, v4):
    n = v1.shape[0]

    R = np.stack([v2 - v1, v3 - v1, v4 - v1], axis=2)
    T = v1

    RT = np.eye(4)[None].repeat(n, axis=0).astype(np.float32)

    RT[:, :3, :3] = R
    RT[:, :3, 3] = T

    return RT

def calculate_tbn(mesh):
    tris = mesh[0][mesh[1]]
    tets = tri2tet(tris)
    R = gradient(*tets)[:, :3, :3]
    return R


def deformation_gradient(canon_tris, deform_tris):
    tet_canon = tri2tet(canon_tris)
    tet_def = tri2tet(deform_tris)

    RT_canon = gradient(*tet_canon)
    RT_def = gradient(*tet_def)

    to_local = np.linalg.inv(RT_canon)
    to_deform = RT_def

    return to_local, to_deform


def transform_vertices(vertices, faces, to_local, to_deform):
    tris = vertices[faces]  # Shape: (num_faces, 3, 3)
    homogeneous_tris = np.concatenate([tris, np.ones((*tris.shape[:2], 1))], axis=2)  # Shape: (num_faces, 3, 4)
    local_tris = np.einsum('fij,fkj->fki', to_local, homogeneous_tris)  # Shape: (num_faces, 3, 4)
    deformed_tris = np.einsum('fij,fkj->fki', to_deform, local_tris)  # Shape: (num_faces, 3, 4)
    transformed_tris = deformed_tris[:, :, :3] / deformed_tris[:, :, 3:4]

    transformed_vertices = np.zeros_like(vertices)
    counts = np.zeros((vertices.shape[0], 1))

    np.add.at(transformed_vertices, faces[:, 0], transformed_tris[:, 0])
    np.add.at(transformed_vertices, faces[:, 1], transformed_tris[:, 1])
    np.add.at(transformed_vertices, faces[:, 2], transformed_tris[:, 2])

    np.add.at(counts, faces[:, 0], 1)
    np.add.at(counts, faces[:, 1], 1)
    np.add.at(counts, faces[:, 2], 1)

    transformed_vertices = np.divide(transformed_vertices, counts, where=counts != 0)

    return transformed_vertices


def process(canon_target, canon_source, source_data, target_data, source, target):
    source_meshes = []
    source_frames = source_data["frames"]
    for frame in tqdm(source_frames):
        mesh_path = frame["mesh_path"]
        mesh = trimesh.load(f"./data/{source}/{mesh_path}", process=False)
        source_meshes.append(mesh)

    canon_tris = canon_source.vertices[canon_source.faces]
    target_canon = canon_target.vertices[canon_target.faces]

    mesh_dst = f"./data/{target}/transfer/meshes"
    os.system(f"rm -rf {mesh_dst}")
    Path(mesh_dst).mkdir(parents=True, exist_ok=True)

    expr_dst = f"./data/{target}/transfer/expr"
    os.system(f"rm -rf {expr_dst}")
    Path(expr_dst).mkdir(parents=True, exist_ok=True)

    target_frames = target_data["frames"]
    for i, src_mesh in tqdm(enumerate(source_meshes)):
        vertices = src_mesh.vertices
        faces = src_mesh.faces
        deform_tris = vertices[faces]
        to_local, to_deform = deformation_gradient(canon_tris, deform_tris)
        new_vertices = transform_vertices(canon_target.vertices, faces, to_local, to_deform)
        mesh_name = f"{str(i).zfill(5)}.obj"
        expr_name = f"{str(i).zfill(5)}.txt"
        trimesh.Trimesh(new_vertices, faces).export(f"{mesh_dst}/{mesh_name}")

        src_expr = source_frames[i]["exp_path"]
        os.system(f"cp ./data/{source}/{src_expr} {expr_dst}/{expr_name}")

        source_frames[i]["mesh_path"] = f"transfer/meshes/{mesh_name}"
        source_frames[i]["exp_path"] = f"transfer/expr/{expr_name}"
        # Copy rest
        source_frames[i]["depth_path"] = target_frames[i]["depth_path"]
        source_frames[i]["file_path"] = target_frames[i]["depth_path"]
        source_frames[i]["seg_mask_path"] = target_frames[i]["seg_mask_path"]

    return source_frames


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Update JSON file copy with n_sample.")
    parser.add_argument("--target", type=str, required=True, help="The target actor name used to locate the JSON file.")
    parser.add_argument("--source", type=str, required=True, help="The source actor name used to locate the JSON file.")
    args = parser.parse_args()
    
    print(f"Processing {args.source} to {args.target}")

    # Define JSON file paths
    source_test_path = f"./data/{args.source}/transforms_test.json"
    target_test_path = f"./data/{args.target}/transforms_test.json"
    
    canonical_target = trimesh.load(f"./data/{args.target}/canonical.obj", process=False)
    canonical_soruce = trimesh.load(f"./data/{args.source}/canonical.obj", process=False)

    with open(source_test_path, "r") as file:
        source_data = json.load(file)

    with open(target_test_path, "r") as file:
        target_data = json.load(file)

    source_data["frames"] = process(canonical_target, canonical_soruce, source_data, target_data, args.source, args.target)
    with open(f"./data/{args.target}/transforms_transfer.json", "w") as file:
        json.dump(source_data, file, indent=4)
    

if __name__ == "__main__":
    main()
