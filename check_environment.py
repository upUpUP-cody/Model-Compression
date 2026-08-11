"""
环境验证脚本
检查所有依赖包是否正确安装
"""
import sys
import importlib.util


def check_package(package_name, import_name=None):
    """检查单个包是否安装"""
    if import_name is None:
        import_name = package_name

    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            return True, version
        return False, None
    except Exception as e:
        return False, str(e)


def main():
    """主函数：检查所有必需的包"""

    packages = {
        # Deep Learning
        'PyTorch': ('torch', '2.0.0+'),
        'TorchVision': ('torchvision', '0.15.0+'),
        'TorchAudio': ('torchaudio', '2.0.0+'),

        # NLP
        'Transformers': ('transformers', '4.30.0+'),
        'Datasets': ('datasets', '2.12.0+'),
        'Tokenizers': ('tokenizers', '0.13.0+'),

        # Scientific Computing
        'NumPy': ('numpy', '1.24.0+'),
        'Pandas': ('pandas', '2.0.0+'),
        'SciPy': ('scipy', '1.10.0+'),

        # Visualization
        'Matplotlib': ('matplotlib', '3.7.0+'),
        'Seaborn': ('seaborn', '0.12.0+'),
        'Plotly': ('plotly', '5.14.0+'),

        # Experiment Tracking
        'WandB': ('wandb', '0.15.0+'),
        'TensorBoard': ('tensorboard', '2.13.0+'),

        # Configuration
        'PyYAML': ('yaml', '6.0+'),
        'OmegaConf': ('omegaconf', '2.3.0+'),

        # Utilities
        'tqdm': ('tqdm', '4.65.0+'),
        'Rich': ('rich', '13.3.0+'),

        # Testing
        'pytest': ('pytest', '7.3.0+'),
        'pytest-cov': ('pytest_cov', '4.1.0+'),

        # Code Quality
        'Black': ('black', '23.3.0+'),
        'Flake8': ('flake8', '6.0.0+'),
        'isort': ('isort', '5.12.0+'),

        # Acceleration
        'Accelerate': ('accelerate', '0.20.0+'),
        'BitsAndBytes': ('bitsandbytes', '0.39.0+'),
    }

    print("=" * 70)
    print("Environment Verification")
    print("=" * 70)
    print(f"\nPython Version: {sys.version.split()[0]}")
    print()

    results = []
    for name, (import_name, expected_ver) in packages.items():
        installed, version = check_package(name, import_name)
        results.append((name, installed, version, expected_ver))

    # Display results
    print(f"{'Package':<20} {'Status':<10} {'Version':<15} {'Required':<15}")
    print("-" * 70)

    success_count = 0
    for name, installed, version, expected in results:
        if installed:
            status = "[OK]"
            success_count += 1
            ver_str = version if version != 'unknown' else 'installed'
        else:
            status = "[MISSING]"
            ver_str = "not found"

        print(f"{name:<20} {status:<10} {ver_str:<15} {expected:<15}")

    # Summary
    print()
    print("=" * 70)
    print(f"Summary: {success_count}/{len(packages)} packages installed")
    print("=" * 70)

    if success_count == len(packages):
        print("\n[SUCCESS] All dependencies are installed!")
        return 0
    else:
        print("\n[FAIL] Some dependencies are missing")
        print("Run: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
