#!/usr/bin/env python3
"""
SHAP Installation Script with CUDA support.

This script tries to build SHAP with CUDA support first, then falls back to CPU-only
if CUDA installation fails. It provides a smoother installation experience compared
to dealing with build errors directly.
"""

import os
import subprocess
import sys
from typing import List, Optional

def run_command(cmd: List[str], description: str, allow_fail: bool = False) -> bool:
    """Run a command and print its output."""
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode != 0:
        if not allow_fail:
            print(f"Error: {description} failed with code {result.returncode}")
        return False
    
    print(f"{description} completed successfully")
    return True

def check_cuda_available() -> bool:
    """Check if CUDA is available on the system."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            print("CUDA found:")
            print(result.stdout)
            return True
    except FileNotFoundError:
        pass
    
    print("CUDA not found")
    return False

def install_shap(with_cuda: bool = True) -> bool:
    """Install SHAP with or without CUDA support."""
    pip_cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    
    if not with_cuda:
        # Disable CUDA for the build
        pip_cmd += ["--config-settings=options=-Dwith_cuda=false"]
    
    description = "Installing SHAP with CUDA" if with_cuda else "Installing SHAP without CUDA"
    return run_command(pip_cmd, description)

def main():
    """Main installation function with fallback logic."""
    has_cuda = check_cuda_available()
    
    if has_cuda:
        print("Attempting to build SHAP with CUDA support...")
        if install_shap(with_cuda=True):
            print("\n✅ SHAP installed successfully with CUDA support!")
            return 0
        
        print("\n⚠️ CUDA build failed, falling back to CPU-only build...")
    else:
        print("CUDA not detected, building CPU-only version...")
    
    if install_shap(with_cuda=False):
        print("\n✅ SHAP installed successfully (CPU-only version)")
        return 0
    
    print("\n❌ Installation failed.")
    return 1

if __name__ == "__main__":
    sys.exit(main())