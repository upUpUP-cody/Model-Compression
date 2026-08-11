"""
GPU 和 CUDA 环境检测脚本
"""
import sys


def check_gpu_environment():
    """检查 GPU 和 CUDA 环境"""

    print("=" * 70)
    print("GPU and CUDA Environment Check")
    print("=" * 70)
    print()

    # Check PyTorch
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            print()

            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"  - Total Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"  - Compute Capability: {props.major}.{props.minor}")
                print(f"  - Multi-Processor Count: {props.multi_processor_count}")
        else:
            print("\n[WARNING] CUDA is not available")
            print("Running on CPU only")

    except ImportError:
        print("[ERROR] PyTorch is not installed")
        print("Run: pip install torch torchvision torchaudio")
        return False

    print()
    print("=" * 70)

    # Test tensor creation
    print("\nTesting tensor operations...")
    try:
        # CPU tensor
        cpu_tensor = torch.randn(3, 3)
        print(f"[OK] CPU tensor created: shape {cpu_tensor.shape}")

        # GPU tensor (if available)
        if torch.cuda.is_available():
            gpu_tensor = torch.randn(3, 3).cuda()
            print(f"[OK] GPU tensor created: shape {gpu_tensor.shape}")
            print(f"[OK] GPU tensor device: {gpu_tensor.device}")
        else:
            print("[SKIP] GPU tensor test (CUDA not available)")

        print("\n[SUCCESS] Environment is ready for training!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Tensor operation failed: {e}")
        return False


if __name__ == "__main__":
    success = check_gpu_environment()
    sys.exit(0 if success else 1)
