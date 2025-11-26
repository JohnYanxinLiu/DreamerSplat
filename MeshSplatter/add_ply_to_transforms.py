#!/usr/bin/env python3
"""
Add ply_file_path to transforms.json for Gaussian Splatting initialization
"""
import json
import sys
from pathlib import Path

def add_ply_path(transforms_path, ply_filename="sparse_pc.ply"):
    """
    Add ply_file_path to transforms.json
    
    Args:
        transforms_path: Path to transforms.json file
        ply_filename: Name of the PLY file (relative to transforms.json directory)
    """
    transforms_path = Path(transforms_path)
    
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    
    data['ply_file_path'] = ply_filename
    
    with open(transforms_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Added ply_file_path: '{ply_filename}' to {transforms_path}")

# python add_ply_to_transforms.py <path/to/transforms.json> [ply_filename]
# Example:
#     python add_ply_to_transforms.py workspace/Replica-Dataset/office_0_dataset/transforms.json
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    
    transforms_path = sys.argv[1]
    ply_filename = sys.argv[2] if len(sys.argv) > 2 else "sparse_pc.ply"
    
    add_ply_path(transforms_path, ply_filename)
