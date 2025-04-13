#!/usr/bin/env python3
"""
SHAP Wheel Build Script.

This script builds both CUDA and non-CUDA wheels for SHAP.
"""

import os
import platform
import shutil
import subprocess
import sys
from typing import List

def run_command(cmd: List[str], description: str, env=None) -> bool:
    """Run a command with the given description."""
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {description} failed with code {e.returncode}")
        return False

def build_wheel(tag: str, with_cuda: bool) -> bool:
    """Build a wheel with the specified tag and CUDA option."""
    # Create a clean build environment
    build_dir = f"build_wheel_{tag}"
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)
    
    # Get the current directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Copy source files to build directory
    for item in os.listdir(root_dir):
        if item not in ["build", "dist", "__pycache__", "build_wheel_*", ".git"]:
            src_path = os.path.join(root_dir, item)
            dst_path = os.path.join(build_dir, item)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, symlinks=True)
            else:
                shutil.copy2(src_path, dst_path)
    
    # Create build environment
    build_env = os.environ.copy()
    if not with_cuda:
        # Disable CUDA for non-CUDA wheels
        build_env["MESON_ARGS"] = "-Dwith_cuda=false"
    
    # Add tag suffix for CUDA wheels
    if with_cuda:
        # Create a custom version file with CUDA tag
        with open(os.path.join(build_dir, "shap", "_version.py"), "r") as f:
            version_content = f.read()
        
        # Update version to include CUDA tag
        version_content = version_content.replace("__version__ = version = '", "__version__ = version = '")
        
        with open(os.path.join(build_dir, "shap", "_version.py"), "w") as f:
            f.write(version_content)
    
    # Build the wheel
    cmd = [
        sys.executable, "-m", "pip", "wheel",
        "--no-deps",
        "--wheel-dir", os.path.join(root_dir, "dist"),
        build_dir
    ]
    
    tag_description = "CUDA" if with_cuda else "non-CUDA"
    result = run_command(cmd, f"Building {tag_description} wheel", env=build_env)
    
    # Clean up
    shutil.rmtree(build_dir)
    
    return result

def build_wheels():
    """Build both CUDA and non-CUDA wheels."""
    # Ensure dist directory exists
    if not os.path.exists("dist"):
        os.makedirs("dist")
    
    # Build non-CUDA wheel first (always works)
    build_wheel("cpu", False)
    
    # Try to build CUDA wheel if CUDA is available
    cuda_check_successful = run_command(
        [sys.executable, "meson_cuda.py"],
        "Checking for CUDA"
    )
    
    if cuda_check_successful:
        build_wheel("cuda", True)
        print("\n✅ Successfully built both CPU and CUDA wheels!")
    else:
        print("\n⚠️ CUDA not available. Only built CPU wheel.")

if __name__ == "__main__":
    build_wheels()