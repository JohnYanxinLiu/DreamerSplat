import numpy as np
from tqdm import tqdm
import struct

def read_ply_mesh(filename):
    """
    Manually read PLY mesh file with vertex colors.
    """
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
        face_count = next(int(line.split()[-1]) for line in header if line.startswith("element face"))
        print(f"  Vertices: {vertex_count}, Faces: {face_count}")

        # Read vertex data (x, y, z, nx, ny, nz, r, g, b)
        vertex_dtype = np.dtype([
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        vertices = np.fromfile(f, dtype=vertex_dtype, count=vertex_count)

        # Read face data
        faces = []
        print("  Reading faces...")
        for _ in tqdm(range(face_count), desc="Loading faces"):
            n = np.fromfile(f, dtype=np.uint8, count=1)[0]
            idx = np.fromfile(f, dtype=np.int32, count=n)
            faces.append(idx)
        faces = np.array(faces, dtype=object)

    return vertices, faces

def build_vertex_neighbors(vertices, faces):
    """
    Build a neighbor list for each vertex based on face connectivity.
    """
    print("Building vertex neighbor graph...")
    neighbors = [set() for _ in range(len(vertices))]
    
    for face in tqdm(faces, desc="Building neighbors"):
        # Add all vertices in the face as neighbors to each other
        for i in range(len(face)):
            for j in range(len(face)):
                if i != j:
                    neighbors[face[i]].add(face[j])
    
    # Convert sets to lists
    neighbors = [list(n) for n in neighbors]
    print(f"  Average neighbors per vertex: {np.mean([len(n) for n in neighbors]):.1f}")
    return neighbors

def smooth_colors(vertices, neighbors, iterations=1):
    """
    Apply local color smoothing using neighbor averaging.
    """
    print(f"Smoothing colors ({iterations} iterations)...")
    
    colors = np.column_stack([vertices['red'], vertices['green'], vertices['blue']]).astype(np.float32)
    
    for it in range(iterations):
        new_colors = colors.copy()
        for i in tqdm(range(len(vertices)), desc=f"Iteration {it+1}/{iterations}"):
            if len(neighbors[i]) > 0:
                # Average with neighbors
                neighbor_colors = colors[list(neighbors[i])]
                new_colors[i] = 0.5 * colors[i] + 0.5 * neighbor_colors.mean(axis=0)
        colors = new_colors
    
    return colors.astype(np.uint8)

def quantize_colors(colors, buckets):
    """
    Quantize RGB colors into buckets.
    """
    print(f"Quantizing colors to {buckets} buckets per channel...")
    
    step = 256 // buckets
    colors_quant = (colors // step) * step
    colors_quant = np.clip(colors_quant, 0, 255)
    
    return colors_quant

def write_ply_mesh(filename, vertices, faces, new_colors):
    """
    Write PLY mesh file with new vertex colors.
    """
    print(f"Writing PLY file: {filename}")
    
    # Update vertex colors
    vertices_out = vertices.copy()
    vertices_out['red'] = new_colors[:, 0]
    vertices_out['green'] = new_colors[:, 1]
    vertices_out['blue'] = new_colors[:, 2]
    
    with open(filename, 'wb') as f:
        # Write header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {len(vertices)}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property float nx\n")
        f.write(b"property float ny\n")
        f.write(b"property float nz\n")
        f.write(b"property uchar red\n")
        f.write(b"property uchar green\n")
        f.write(b"property uchar blue\n")
        f.write(f"element face {len(faces)}\n".encode())
        f.write(b"property list uint8 int vertex_indices\n")
        f.write(b"end_header\n")
        
        # Write vertices
        print("  Writing vertices...")
        vertices_out.tofile(f)
        
        # Write faces
        print("  Writing faces...")
        for face in tqdm(faces, desc="Writing faces"):
            f.write(struct.pack('B', len(face)))
            f.write(struct.pack(f'{len(face)}i', *face))
    
    print(f"Saved: {filename}")

def quantize_ply(input_ply, output_ply, buckets, smooth_iterations=0):
    """
    Load a PLY mesh, optionally smooth, quantize vertex colors, and save.
    
    Args:
        input_ply: Input PLY file
        output_ply: Output PLY file
        buckets: Number of color buckets per channel (e.g., 8 = 512 total colors)
        smooth_iterations: Number of smoothing iterations (0 = no smoothing)
    """
    # Read PLY
    vertices, faces = read_ply_mesh(input_ply)
    
    # Get original colors
    colors_orig = np.column_stack([vertices['red'], vertices['green'], vertices['blue']])
    
    # Calculate statistics
    unique_before = len(np.unique(colors_orig.view([('', colors_orig.dtype)] * 3)))
    print(f"\nOriginal unique colors: {unique_before}")
    
    # Smooth if requested
    if smooth_iterations > 0:
        neighbors = build_vertex_neighbors(vertices, faces)
        colors = smooth_colors(vertices, neighbors, smooth_iterations)
    else:
        colors = colors_orig
    
    # Quantize
    colors_quant = quantize_colors(colors, buckets)
    
    # Calculate statistics after quantization
    unique_after = len(np.unique(colors_quant.view([('', colors_quant.dtype)] * 3)))
    print(f"Quantized unique colors: {unique_after}")
    print(f"Reduction: {(1 - unique_after/unique_before)*100:.1f}%")
    print(f"Total possible colors: {buckets**3}")
    
    # Write output
    write_ply_mesh(output_ply, vertices, faces, colors_quant)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python quantize_ply_colors.py input.ply output.ply [buckets] [smooth_iterations]")
        print("\nExamples:")
        print("  python quantize_ply_colors.py mesh.ply mesh_q.ply 8")
        print("  python quantize_ply_colors.py mesh.ply mesh_q.ply 16 2")
        print("\nBucket recommendations:")
        print("  8 = 512 colors (aggressive)")
        print("  16 = 4096 colors (moderate)")
        print("  32 = 32768 colors (conservative)")
        sys.exit(1)
    
    input_ply = sys.argv[1]
    output_ply = sys.argv[2]
    buckets = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    smooth_iterations = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    
    quantize_ply(input_ply, output_ply, buckets, smooth_iterations)
