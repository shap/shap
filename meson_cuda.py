"""
Helper script to check CUDA availability for meson.

This module provides utilities to detect CUDA installation
and generate appropriate meson options.
"""

import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

def find_in_path(name: str, path: Optional[str] = None) -> Optional[str]:
    """Find a file in a search path and return its full path."""
    if path is None:
        path = os.environ.get("PATH", "")
    
    for directory in path.split(os.pathsep):
        binary_path = os.path.join(directory, name)
        if os.path.exists(binary_path):
            return os.path.abspath(binary_path)
    return None

def get_cuda_path() -> Tuple[Optional[str], Optional[str]]:
    """Return a tuple with (base_cuda_directory, full_path_to_nvcc_compiler)."""
    nvcc_bin = "nvcc.exe" if sys.platform == "win32" else "nvcc"
    
    # Check environment variables first
    cuda_home = os.environ.get("CUDAHOME") or os.environ.get("CUDA_PATH")
    
    if cuda_home is None:
        # Search for nvcc in PATH
        found_nvcc = find_in_path(nvcc_bin)
        if found_nvcc is None:
            return None, None
        
        # Get CUDA home from nvcc location
        cuda_home = os.path.dirname(os.path.dirname(found_nvcc))
    
    # Verify cuda_home has include directory
    if not os.path.exists(os.path.join(cuda_home, "include")):
        cuda_home = "/usr/local/cuda"
    
    # Build full path to nvcc
    nvcc = os.path.join(cuda_home, "bin", nvcc_bin)
    if not os.path.exists(nvcc):
        cuda_home = "/usr/local/cuda"
        nvcc = os.path.join(cuda_home, "bin", nvcc_bin)
        if not os.path.exists(nvcc):
            return None, None
    
    return cuda_home, nvcc

def get_cuda_version(nvcc_path: str) -> Optional[str]:
    """Get CUDA version from nvcc."""
    try:
        result = subprocess.run(
            [nvcc_path, "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract version from output
        for line in result.stdout.splitlines():
            if "release" in line.lower():
                parts = line.split("release")
                if len(parts) >= 2:
                    version = parts[1].strip().split(",")[0].strip()
                    return version
        
        return None
    except (subprocess.SubprocessError, IndexError):
        return None

def has_cuda() -> bool:
    """Check if CUDA is available."""
    _, nvcc = get_cuda_path()
    return nvcc is not None

if __name__ == "__main__":
    if has_cuda():
        cuda_home, nvcc = get_cuda_path()
        version = get_cuda_version(nvcc)
        print(f"CUDA found: {cuda_home}")
        print(f"NVCC: {nvcc}")
        print(f"Version: {version}")
        sys.exit(0)
    else:
        print("CUDA not found")
        sys.exit(1)