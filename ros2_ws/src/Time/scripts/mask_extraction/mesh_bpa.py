#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pcd", required=True)
    ap.add_argument("--out_mesh", required=True)
    ap.add_argument("--voxel", type=float, default=0.003)     # 3mm default
    ap.add_argument("--nb_neighbors", type=int, default=30)
    ap.add_argument("--std_ratio", type=float, default=2.0)
    ap.add_argument("--normal_k", type=int, default=30)
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.in_pcd)
    print("Loaded:", len(pcd.points), "points")

    # Clean + downsample
    pcd = pcd.voxel_down_sample(args.voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio)
    print("After clean:", len(pcd.points), "points")

    if len(pcd.points) < 200:
        raise RuntimeError("Too few points for meshing. Collect more scans or lower voxel.")

    # Normals (critical)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=args.normal_k))
    pcd.orient_normals_consistent_tangent_plane(50)

    # Ball Pivoting radii (set relative to voxel)
    r = args.voxel
    radii = o3d.utility.DoubleVector([2.0*r, 4.0*r, 8.0*r, 12.0*r])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)

    # Basic cleanup
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(args.out_mesh, mesh)
    print("Saved mesh:", args.out_mesh)
    print("Triangles:", np.asarray(mesh.triangles).shape[0])

if __name__ == "__main__":
    main()