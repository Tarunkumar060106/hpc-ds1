#!/usr/bin/env python3
"""
GPU Setup Checker for HPC-DS1

Verifies that the GPU environment is properly configured for training.
Checks:
  - NVIDIA GPU drivers
  - CUDA installation
  - cuDNN installation
  - PyTorch with CUDA support
  - GPU VRAM
  - Model fitting on GPU
"""

import sys
import subprocess
from pathlib import Path


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_status(name, status, detail=""):
    """Print a status line."""
    symbol = "[OK]" if status else "[FAIL]"
    detail_str = f" - {detail}" if detail else ""
    print(f"{symbol} {name}{detail_str}")


def run_command(cmd):
    """Run a command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        return False, str(e)


def check_nvidia_driver():
    """Check NVIDIA GPU driver."""
    print_header("1. NVIDIA GPU Driver")

    success, output = run_command("nvidia-smi")

    if success:
        lines = output.split('\n')
        # Extract GPU name and driver version
        for line in lines:
            if 'NVIDIA-SMI' in line:
                print(line)
            elif 'Driver Version' in line:
                print_status("NVIDIA Driver", True, line.strip())
            elif 'CUDA Version' in line:
                cuda_version = line.split(':')[1].strip()
                print_status("CUDA Version (from nvidia-smi)", True, cuda_version)

        # Try to get GPU info
        success2, gpu_output = run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
        if success2:
            lines = gpu_output.split('\n')
            for i, line in enumerate(lines):
                parts = line.split(', ')
                if len(parts) >= 2:
                    gpu_name = parts[0].strip()
                    gpu_mem = parts[1].strip()
                    print_status(f"GPU {i}", True, f"{gpu_name} ({gpu_mem})")

        return True
    else:
        print_status("NVIDIA Driver", False, "nvidia-smi not found - GPU drivers not installed")
        return False


def check_cuda_installation():
    """Check CUDA toolkit installation."""
    print_header("2. CUDA Toolkit Installation")

    # Check common CUDA paths
    cuda_paths = [
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
        "C:\\Program Files\\NVIDIA\\CUDA",
        "/usr/local/cuda",
        "/opt/cuda",
    ]

    cuda_found = False
    for path in cuda_paths:
        cuda_bin = Path(path) / "bin"
        if cuda_bin.exists():
            print_status("CUDA Installation", True, f"Found at {path}")
            cuda_found = True
            break

    if not cuda_found:
        print_status("CUDA Installation", False, "Not found in standard locations")
        print("  Hint: Install CUDA from https://developer.nvidia.com/cuda-downloads")
        return False

    # Check nvcc compiler
    success, output = run_command("nvcc --version")
    if success:
        # Extract version
        for line in output.split('\n'):
            if 'release' in line:
                print_status("CUDA Compiler (nvcc)", True, line.strip())
                return True
    else:
        print_status("CUDA Compiler (nvcc)", False, "Not found in PATH")
        return False


def check_cudnn_installation():
    """Check cuDNN installation."""
    print_header("3. cuDNN (CUDA Deep Neural Network Library)")

    cudnn_paths = [
        "C:\\Program Files\\NVIDIA\\CUDNN",
        "C:\\cuda\\bin",
        "/usr/local/cuda/lib64",
        "/usr/local/cudnn/lib64",
    ]

    found = False
    for path in cudnn_paths:
        cudnn_path = Path(path)
        if cudnn_path.exists():
            # Look for cuDNN library files
            if any(f.name.startswith('cudnn') for f in cudnn_path.glob('*')):
                print_status("cuDNN Installation", True, f"Found in {path}")
                found = True
                break

    if not found:
        print_status("cuDNN Installation", False, "Not found in standard locations")
        print("  Hint: Install cuDNN from https://developer.nvidia.com/cudnn")
        print("         (Requires NVIDIA Developer account)")
        return False

    return True


def check_pytorch_installation():
    """Check PyTorch installation and CUDA support."""
    print_header("4. PyTorch Installation")

    try:
        import torch

        # Check PyTorch version
        print_status("PyTorch", True, f"Version {torch.__version__}")

        # Check CUDA availability
        if torch.cuda.is_available():
            print_status("CUDA Support", True, "PyTorch can use GPU")

            # Get device info
            device_count = torch.cuda.device_count()
            print_status("Number of GPUs", True, f"{device_count} GPU(s) detected")

            # Detailed GPU info
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print_status(f"GPU {i}", True, f"{gpu_name} ({gpu_mem:.1f} GB)")

            # Check current device
            print_status("Current Device", True, f"GPU {torch.cuda.current_device()}")

            # Check cuDNN
            if torch.backends.cudnn.enabled:
                print_status("cuDNN", True, f"Version {torch.backends.cudnn.version()}")
                print_status("cuDNN Enabled", True, "Yes")
            else:
                print_status("cuDNN Enabled", False, "No")

            return True
        else:
            print_status("CUDA Support", False, "PyTorch cannot access GPU")
            print("  Possible reasons:")
            print("    - NVIDIA drivers not installed")
            print("    - CUDA/cuDNN not installed")
            print("    - PyTorch installed for CPU only")
            print("\n  Solution: Reinstall PyTorch with CUDA support:")
            print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False

    except ImportError:
        print_status("PyTorch", False, "Not installed")
        print("\n  Install PyTorch:")
        print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False


def check_model_fits_on_gpu():
    """Check if HPC-DS1 model can fit on GPU."""
    print_header("5. Model VRAM Requirements")

    try:
        import torch

        if not torch.cuda.is_available():
            print_status("GPU VRAM Check", False, "GPU not available")
            return False

        # Estimate model VRAM requirements
        # HPC-DS1: ~2.5M parameters
        # Training requires: params + gradients + optimizer states + batch

        model_params = 2.5e6  # 2.5M parameters
        bytes_per_param = 4  # float32

        # Forward pass: model + batch
        forward_vram = model_params * bytes_per_param / (1024**3)  # GB

        # Backward pass: gradients + optimizer states (Adam: 2x params)
        backward_vram = (model_params * bytes_per_param * 3) / (1024**3)  # model + grad + optimizer

        # Batch data (assuming batch_size=32, ~512 floats per sample)
        batch_vram = (32 * 512 * 4) / (1024**3)

        total_vram_required = forward_vram + backward_vram + batch_vram

        # Get actual GPU VRAM
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        print_status("Model Parameters", True, f"2.5M (10.0 MB)")
        print_status("Model Weights (float32)", True, f"{forward_vram:.2f} GB")
        print_status("Gradients + Optimizer", True, f"{backward_vram:.2f} GB")
        print_status("Batch Data (batch_size=32)", True, f"{batch_vram:.3f} GB")
        print_status("Total VRAM Required (estimate)", True, f"{total_vram_required:.2f} GB")
        print_status("GPU VRAM Available", True, f"{gpu_vram:.1f} GB")

        # Check if it fits
        safety_margin = 0.9  # Use 90% of GPU VRAM for safety
        fits = total_vram_required < (gpu_vram * safety_margin)

        if fits:
            print_status("Fits on GPU", True, f"Yes (using {(total_vram_required/gpu_vram)*100:.1f}% of VRAM)")
            print("\n  Recommendation:")
            print(f"    - Training batch size: 32 (current estimate)")
            print(f"    - Safe batch size: 16-32 with safety margin")
            print(f"    - Memory utilization: {(total_vram_required/gpu_vram)*100:.1f}%")
            return True
        else:
            print_status("Fits on GPU", False, f"No - requires {total_vram_required:.2f}GB but only {gpu_vram:.1f}GB available")
            print("\n  Recommendation:")
            print(f"    - Use smaller batch size (8-16)")
            print(f"    - Or use CPU for training (slower but will work)")
            return False

    except ImportError:
        print_status("Model Check", False, "PyTorch not installed")
        return False


def check_training_setup():
    """Check if all training dependencies are available."""
    print_header("6. Training Dependencies")

    dependencies = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
    ]

    all_good = True
    for module_name, display_name in dependencies:
        try:
            mod = __import__(module_name)
            version = getattr(mod, '__version__', 'unknown')
            print_status(display_name, True, f"v{version}")
        except ImportError:
            print_status(display_name, False, f"Not installed")
            all_good = False

    return all_good


def main():
    """Run all GPU checks."""

    print("\n" + "="*70)
    print("  HPC-DS1: GPU SETUP VERIFICATION")
    print("="*70)
    print("\nThis script checks if your system is ready to train HPC-DS1 on GPU.")
    print("Required: NVIDIA GPU driver + CUDA + cuDNN + PyTorch")

    # Run all checks
    checks = [
        ("NVIDIA Driver", check_nvidia_driver),
        ("CUDA Installation", check_cuda_installation),
        ("cuDNN Installation", check_cudnn_installation),
        ("PyTorch", check_pytorch_installation),
        ("Model VRAM", check_model_fits_on_gpu),
        ("Dependencies", check_training_setup),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            results[name] = False

    # Summary
    print_header("SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nPassed: {passed}/{total} checks")
    print()

    for name, passed_check in results.items():
        symbol = "[OK]" if passed_check else "[FAIL]"
        print(f"  {symbol} {name}")

    # Recommendations
    print_header("RECOMMENDATIONS")

    if results["NVIDIA Driver"]:
        print("[OK] GPU driver found - good!")
    else:
        print("[ACTION] Install NVIDIA GPU drivers:")
        print("  https://www.nvidia.com/Download/driverDetails.aspx")
        print()

    if results["CUDA Installation"]:
        print("[OK] CUDA installed - good!")
    else:
        print("[ACTION] Install CUDA Toolkit:")
        print("  https://developer.nvidia.com/cuda-downloads")
        print()

    if results["cuDNN Installation"]:
        print("[OK] cuDNN installed - good!")
    else:
        print("[ACTION] Install cuDNN (requires free NVIDIA Developer account):")
        print("  https://developer.nvidia.com/cudnn")
        print()

    if results["PyTorch"]:
        print("[OK] PyTorch with CUDA support installed - ready to train!")
    else:
        print("[ACTION] Install PyTorch with CUDA 11.8 support:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print()

    if results["Model VRAM"]:
        print("[OK] GPU has enough VRAM for training - proceed with Phase 1!")
    else:
        print("[ACTION] GPU VRAM may be tight. Options:")
        print("  1. Reduce batch size to 16 or 8")
        print("  2. Use CPU for training (slower but works)")
        print("  3. Consider using a cloud GPU (AWS, GCP, Colab)")
        print()

    if all(results.values()):
        print("\n" + "="*70)
        print("  ALL CHECKS PASSED! You're ready for Phase 1")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print("  SOME CHECKS FAILED - Please address issues above")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
