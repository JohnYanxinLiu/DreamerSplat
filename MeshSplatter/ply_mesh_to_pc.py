import open3d as o3d
import trimesh
import numpy as np
import sys
import os

def ply_to_sparse_pc(ply_path, output_path=None, num_points=15000):
    """
    Converts a PLY mesh (with vertex colors) to a sparse point cloud PLY
    for Gaussian Splatting initialization.
    Automatically triangulates polygon faces.
    """
    if output_path is None:
        base = os.path.splitext(ply_path)[0]
        output_path = base + "_sparse_pc.ply"

    print(f"[INFO] Loading PLY mesh (triangulating if needed): {ply_path}")

    # Load with trimesh and triangulate
    mesh_trimesh = trimesh.load(ply_path, process=False)
    if not mesh_trimesh.is_watertight:
        print("[INFO] Mesh is not watertight; proceeding anyway.")
    
    # Convert to Open3D TriangleMesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_trimesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_trimesh.faces)

    # Copy vertex colors if they exist
    if hasattr(mesh_trimesh.visual, 'vertex_colors') and mesh_trimesh.visual.vertex_colors is not None:
        colors = np.array(mesh_trimesh.visual.vertex_colors[:, :3], dtype=np.float64) / 255.0
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    else:
        print("[WARN] Mesh has no vertex colors. Using white.")
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(np.ones((len(mesh_trimesh.vertices), 3)))

    o3d_mesh.compute_vertex_normals()

    # Sample points on mesh surface
    print(f"[INFO] Sampling {num_points} points on mesh surface...")
    pcd = o3d_mesh.sample_points_uniformly(num_points)
    points = np.asarray(pcd.points)

    # Assign nearest vertex color
    print("[INFO] Assigning vertex colors to sampled points...")
    vertex_pcd = o3d.geometry.PointCloud()
    vertex_pcd.points = o3d_mesh.vertices
    vertex_pcd.colors = o3d_mesh.vertex_colors
    kdtree = o3d.geometry.KDTreeFlann(vertex_pcd)

    colors = []
    for pt in points:
        _, idx, _ = kdtree.search_knn_vector_3d(pt, 1)
        colors.append(np.asarray(vertex_pcd.colors)[idx[0]])
    colors = np.array(colors)

    # Write sparse point cloud PLY (ASCII format for Gaussian Splatting)
    print(f"[INFO] Writing sparse point cloud PLY: {output_path}")
    with open(output_path, "w") as f:
        N = len(points)
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uint8 red\n")
        f.write("property uint8 green\n")
        f.write("property uint8 blue\n")
        f.write("end_header\n")
        
        # Write vertex data
        for i in range(N):
            x, y, z = points[i]
            r, g, b = (colors[i] * 255).astype(np.uint8)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    
    print(f"[INFO] Conversion complete! Wrote {N} points to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ply_to_splat.py <input.ply> [output.ply] [num_points]")
        print("Example: python ply_to_splat.py mesh.ply sparse_pc.ply 15000")
        sys.exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2] if len(sys.argv) > 2 else None
    num_points = int(sys.argv[3]) if len(sys.argv) > 3 else 15000
    
    ply_to_sparse_pc(infile, outfile, num_points)
