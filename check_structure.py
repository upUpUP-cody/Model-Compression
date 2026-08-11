"""
项目结构验证脚本
检查所有必需的目录和文件是否存在
"""
import os
from pathlib import Path


def check_project_structure():
    """检查项目结构完整性"""

    required_dirs = [
        'src/models',
        'src/pruning',
        'src/recovery',
        'src/controller',
        'src/evaluation',
        'src/utils',
        'configs',
        'experiments',
        'tests',
        'data/mnist',
        'data/cifar10',
        'data/squad',
        'checkpoints',
        'logs',
        'results'
    ]

    required_files = [
        'src/__init__.py',
        'src/models/__init__.py',
        'src/pruning/__init__.py',
        'src/recovery/__init__.py',
        'src/controller/__init__.py',
        'src/evaluation/__init__.py',
        'src/utils/__init__.py',
        'tests/__init__.py',
        'experiments/__init__.py',
        'requirements.txt',
        'README.md',
        'PROJECT_PLAN.md',
        'ROADMAP.md'
    ]

    print("=" * 60)
    print("项目结构检查")
    print("=" * 60)

    # 检查目录
    print("\n[目录检查]")
    missing_dirs = []
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"[OK] {dir_path}")
        else:
            print(f"[MISS] {dir_path}")
            missing_dirs.append(dir_path)

    # 检查文件
    print("\n[文件检查]")
    missing_files = []
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"[OK] {file_path}")
        else:
            print(f"[MISS] {file_path}")
            missing_files.append(file_path)

    # 总结
    print("\n" + "=" * 60)
    print("检查总结")
    print("=" * 60)
    print(f"目录: {len(required_dirs) - len(missing_dirs)}/{len(required_dirs)} 完成")
    print(f"文件: {len(required_files) - len(missing_files)}/{len(required_files)} 完成")

    if not missing_dirs and not missing_files:
        print("\n[SUCCESS] Project structure is complete!")
        return True
    else:
        print("\n[FAIL] Project structure is incomplete")
        if missing_dirs:
            print(f"\nMissing directories: {missing_dirs}")
        if missing_files:
            print(f"Missing files: {missing_files}")
        return False


if __name__ == "__main__":
    check_project_structure()
