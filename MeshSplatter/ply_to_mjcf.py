import numpy as np
from tqdm import tqdm
import os
import struct
from collections import defaultdict

"""
Convert PLY mesh to MJCF with color-based submeshes.
Each unique color becomes a separate mesh with its own material.
"""

def read_ply(filename):
    print(f"Reading PLY file: {filename}")
    with open(filename, 'rb') as f:
        # Read header
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header.append(line)
            if line == "end_header":
                break
        
        # Extract counts
        vertex_count = next(int(line.split()[-1]) for line in header if line.startswith("element vertex"))
        face_count   = next(int(line.split()[-1]) for line in header if line.startswith("element face"))
        print(f"  Vertices: {vertex_count}, Faces: {face_count}")

        # Read vertex data (x, y, z, nx, ny, nz, r, g, b)
        vertex_dtype = np.dtype([
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        print("  Reading vertices...")
        vertices = np.fromfile(f, dtype=vertex_dtype, count=vertex_count)

        # Read face data
        print("  Reading faces...")
        faces = []
        for _ in tqdm(range(face_count), desc="Loading faces"):
            n = np.fromfile(f, dtype=np.uint8, count=1)[0]
            idx = np.fromfile(f, dtype=np.int32, count=n)
            faces.append(idx)
        faces = np.array(faces, dtype=object)

    return vertices, faces

def triangulate_face(face):
    """Convert a face to triangles using fan triangulation."""
    if len(face) < 3:
        return []
    if len(face) == 3:
        return [face]
    
    triangles = []
    v0 = face[0]
    for i in range(1, len(face) - 1):
        triangles.append([v0, face[i], face[i + 1]])
    return triangles

def compute_triangle_area(v0, v1, v2):
    """Compute the area of a triangle given three vertices."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = np.cross(edge1, edge2)
    return np.linalg.norm(cross) / 2.0

def is_degenerate_triangle(v0, v1, v2, epsilon=1e-12):
    """Check if a triangle is degenerate (zero area or invalid)."""
    if np.allclose(v0, v1, atol=epsilon) or np.allclose(v1, v2, atol=epsilon) or np.allclose(v0, v2, atol=epsilon):
        return True
    area = compute_triangle_area(v0, v1, v2)
    return area < epsilon

def is_mesh_volumetric(vertices, triangles, epsilon=1e-6):
    """
    Check if a mesh is volumetric (3D) or planar/degenerate.
    Returns True if the mesh spans 3D space, False if it's flat/coplanar.
    """
    if len(triangles) < 4:
        return False
    
    # Get all unique vertex positions
    used_vertex_indices = set()
    for tri in triangles:
        used_vertex_indices.update(tri)
    
    if len(used_vertex_indices) < 4:
        return False
    
    # Extract positions
    positions = np.array([[vertices[idx]['x'], vertices[idx]['y'], vertices[idx]['z']] 
                          for idx in used_vertex_indices])
    
    # Check if points are coplanar by computing their covariance matrix
    # and checking if one of the eigenvalues is near zero
    centered = positions - np.mean(positions, axis=0)
    cov = np.dot(centered.T, centered)
    eigenvalues = np.linalg.eigvalsh(cov)
    
    # If smallest eigenvalue is very small, points are coplanar
    min_eigenvalue = eigenvalues[0]
    max_eigenvalue = eigenvalues[-1]
    
    # Relative threshold: smallest eigenvalue should be at least 1e-6 of the largest
    if max_eigenvalue > 0:
        relative_threshold = max_eigenvalue * epsilon
        return min_eigenvalue > relative_threshold
    
    return False

def group_faces_by_color(vertices, faces):
    """Group faces by the color of their first vertex."""
    print("\nGrouping faces by color...")
    
    # Triangulate all faces first
    all_triangles = []
    degenerate_count = 0
    
    for face in tqdm(faces, desc="Triangulating"):
        triangles = triangulate_face(face)
        for tri in triangles:
            v0 = np.array([vertices[tri[0]]['x'], vertices[tri[0]]['y'], vertices[tri[0]]['z']])
            v1 = np.array([vertices[tri[1]]['x'], vertices[tri[1]]['y'], vertices[tri[1]]['z']])
            v2 = np.array([vertices[tri[2]]['x'], vertices[tri[2]]['y'], vertices[tri[2]]['z']])
            
            if not is_degenerate_triangle(v0, v1, v2):
                all_triangles.append(tri)
            else:
                degenerate_count += 1
    
    print(f"  Filtered out {degenerate_count} degenerate triangles")
    print(f"  Valid triangles: {len(all_triangles)}")
    
    # Group by color
    color_groups = defaultdict(list)
    for tri in tqdm(all_triangles, desc="Grouping by color"):
        # Use average color of triangle vertices
        colors = [
            (vertices[tri[0]]['red'], vertices[tri[0]]['green'], vertices[tri[0]]['blue']),
            (vertices[tri[1]]['red'], vertices[tri[1]]['green'], vertices[tri[1]]['blue']),
            (vertices[tri[2]]['red'], vertices[tri[2]]['green'], vertices[tri[2]]['blue'])
        ]
        avg_color = tuple(int(np.mean([c[i] for c in colors])) for i in range(3))
        color_groups[avg_color].append(tri)
    
    print(f"  Found {len(color_groups)} unique colors")
    return color_groups, all_triangles

def write_submesh_obj(vertices, triangles, obj_filename):
    """Write a submesh OBJ file for a specific set of triangles."""
    # Find unique vertices used by these triangles
    used_vertices = set()
    for tri in triangles:
        used_vertices.update(tri)
    used_vertices = sorted(used_vertices)
    
    # Create mapping from old to new indices
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
    
    with open(obj_filename, 'w') as f:
        # Write vertices
        for v_idx in used_vertices:
            v = vertices[v_idx]
            f.write(f"v {v['x']:.6f} {v['y']:.6f} {v['z']:.6f}\n")
        
        f.write("\n")
        
        # Write normals
        for v_idx in used_vertices:
            v = vertices[v_idx]
            f.write(f"vn {v['nx']:.6f} {v['ny']:.6f} {v['nz']:.6f}\n")
        
        f.write("\n")
        
        # Write faces with remapped indices
        for tri in triangles:
            new_tri = [vertex_map[idx] + 1 for idx in tri]  # +1 for OBJ 1-based indexing
            f.write(f"f {new_tri[0]}//{new_tri[0]} {new_tri[1]}//{new_tri[1]} {new_tri[2]}//{new_tri[2]}\n")
    
    return len(used_vertices), len(triangles)

def ply_to_mjcf_submeshes(input_ply, output_dir, scene_name="scene"):
    """
    Convert PLY to MJCF with separate submeshes per color.
    
    Args:
        input_ply: Input PLY file path
        output_dir: Output directory for meshes and XML
        scene_name: Name for the scene
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    meshes_dir = os.path.join(output_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)
    
    # Read PLY
    vertices, faces = read_ply(input_ply)
    
    # Group faces by color
    color_groups, all_triangles = group_faces_by_color(vertices, faces)
    
    # Write submeshes - First pass: identify valid color groups
    print("\nIdentifying valid submeshes...")
    valid_color_groups = {}
    invalid_triangles_by_color = {}
    small_count = 0
    degenerate_count = 0
    
    for color, triangles in tqdm(color_groups.items(), desc="Validating color groups"):
        # Find unique vertices used by these triangles
        used_vertices = set()
        for tri in triangles:
            used_vertices.update(tri)
        
        # Check if submesh is valid for MuJoCo
        if len(used_vertices) < 4:
            invalid_triangles_by_color[color] = triangles
            small_count += 1
            continue
        
        # Check if mesh is volumetric (not coplanar/degenerate)
        if not is_mesh_volumetric(vertices, triangles):
            invalid_triangles_by_color[color] = triangles
            degenerate_count += 1
            continue
        
        valid_color_groups[color] = triangles
    
    print(f"  Valid color groups: {len(valid_color_groups)}")
    print(f"  Invalid (small): {small_count}, Invalid (coplanar): {degenerate_count}")
    
    # Second pass: reassign invalid triangles to nearest valid color
    if invalid_triangles_by_color and valid_color_groups:
        print("\nReassigning invalid triangles to nearest valid colors...")
        valid_colors = np.array(list(valid_color_groups.keys()))
        
        for invalid_color, triangles in tqdm(invalid_triangles_by_color.items(), desc="Reassigning"):
            # Find nearest valid color (Euclidean distance in RGB space)
            invalid_rgb = np.array(invalid_color)
            distances = np.linalg.norm(valid_colors - invalid_rgb, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_color = tuple(valid_colors[nearest_idx])
            
            # Add triangles to the nearest valid color group
            valid_color_groups[nearest_color].extend(triangles)
    
    # Third pass: write all submeshes
    print("\nWriting submeshes...")
    submesh_info = []
    
    for idx, (color, triangles) in enumerate(tqdm(valid_color_groups.items(), desc="Writing submeshes")):
        mesh_name = f"mesh_{idx:04d}"
        obj_filename = os.path.join(meshes_dir, f"{mesh_name}.obj")
        
        num_verts, num_faces = write_submesh_obj(vertices, triangles, obj_filename)
        
        r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
        submesh_info.append({
            'name': mesh_name,
            'file': f"meshes/{mesh_name}.obj",
            'color': (r, g, b),
            'num_verts': num_verts,
            'num_faces': num_faces
        })
    
    # Write MJCF
    mjcf_filename = os.path.join(output_dir, f"{scene_name}.xml")
    print(f"\nWriting MJCF file: {mjcf_filename}")
    
    with open(mjcf_filename, 'w') as f:
        f.write('<mujoco>\n')
        f.write('  <asset>\n')
        
        # Define all meshes
        for info in submesh_info:
            f.write(f'    <mesh name="{info["name"]}" file="{info["file"]}" />\n')
        
        f.write('  </asset>\n\n')
        
        f.write('  <worldbody>\n')
        
        # Create geoms for each submesh with its color
        for info in submesh_info:
            r, g, b = info['color']
            f.write(f'    <geom name="{info["name"]}_geom" type="mesh" mesh="{info["name"]}" '
                   f'rgba="{r:.6f} {g:.6f} {b:.6f} 1" />\n')
        
        f.write('  </worldbody>\n')
        f.write('</mujoco>\n')
    
    # Print summary
    print("\n" + "="*70)
    print("Conversion complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  MJCF file: {mjcf_filename}")
    print(f"  Number of submeshes: {len(submesh_info)}")
    print(f"  Total triangles: {len(all_triangles)}")
    print(f"  Meshes directory: {meshes_dir}")
    print("="*70)
    
    # Print submesh statistics
    print(f"\nSubmesh details:")
    total_verts = sum(info['num_verts'] for info in submesh_info)
    total_faces = sum(info['num_faces'] for info in submesh_info)
    print(f"  Total vertices: {total_verts}")
    print(f"  Total faces: {total_faces}")
    print(f"  Average faces per submesh: {total_faces / len(submesh_info):.1f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ply_to_mjcf_submeshes.py input.ply [output_dir] [scene_name]")
        print("\nConverts PLY mesh to MJCF with color-based submeshes")
        print("Each unique color becomes a separate mesh with its own material")
        print("\nExamples:")
        print("  python ply_to_mjcf_submeshes.py scene.ply")
        print("  python ply_to_mjcf_submeshes.py scene.ply output_scene")
        print("  python ply_to_mjcf_submeshes.py scene.ply output_scene office")
        sys.exit(1)
    
    input_ply = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else input_ply.replace('.ply', '_mjcf')
    scene_name = sys.argv[3] if len(sys.argv) > 3 else "scene"
    
    ply_to_mjcf_submeshes(input_ply, output_dir, scene_name)
