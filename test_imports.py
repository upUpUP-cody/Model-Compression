"""
快速导入测试
验证关键包是否可以正常导入
"""
import sys


def quick_import_test():
    """快速测试关键包的导入"""

    print("=" * 70)
    print("Quick Import Test")
    print("=" * 70)
    print()

    critical_imports = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('transformers', 'Transformers'),
        ('datasets', 'Datasets'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('yaml', 'PyYAML'),
        ('omegaconf', 'OmegaConf'),
        ('tqdm', 'tqdm'),
        ('pytest', 'pytest'),
    ]

    success_count = 0
    failed = []

    for module_name, display_name in critical_imports:
        try:
            __import__(module_name)
            print(f"[OK] {display_name}")
            success_count += 1
        except ImportError as e:
            print(f"[FAIL] {display_name}: {e}")
            failed.append(display_name)

    print()
    print("=" * 70)
    print(f"Result: {success_count}/{len(critical_imports)} packages imported successfully")
    print("=" * 70)

    if failed:
        print(f"\nFailed imports: {', '.join(failed)}")
        return False
    else:
        print("\n[SUCCESS] All critical packages can be imported!")
        return True


if __name__ == "__main__":
    success = quick_import_test()
    sys.exit(0 if success else 1)
