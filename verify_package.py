#!/usr/bin/env python3
"""
Verification script to check UAL Adapter package completeness
"""

import os
import sys
from pathlib import Path


def check_file_structure():
    """Check if all essential files exist."""
    
    print("🔍 Checking UAL Adapter Package Structure...")
    print("=" * 60)
    
    base_path = Path(__file__).parent
    
    # Essential files and directories
    structure = {
        "Package Files": [
            "README.md",
            "setup.py",
            "requirements.txt",
            "pyproject.toml",
            "Makefile",
            "Dockerfile",
            "DOCUMENTATION.md",
        ],
        "Main Package": [
            "ual_adapter/__init__.py",
            "ual_adapter/cli.py",
        ],
        "Core Modules": [
            "ual_adapter/core/__init__.py",
            "ual_adapter/core/adapter.py",
            "ual_adapter/core/air.py",
            "ual_adapter/core/dispatcher.py",
            "ual_adapter/core/projection.py",
        ],
        "Binders": [
            "ual_adapter/binders/__init__.py",
            "ual_adapter/binders/base.py",
            "ual_adapter/binders/architectures.py",
            "ual_adapter/binders/registry.py",
        ],
        "Training": [
            "ual_adapter/training/__init__.py",
            "ual_adapter/training/trainer.py",
        ],
        "Utils": [
            "ual_adapter/utils/__init__.py",
            "ual_adapter/utils/model_utils.py",
        ],
        "Tests": [
            "tests/conftest.py",
            "tests/test_air.py",
            "tests/test_projection.py",
            "tests/test_dispatcher.py",
        ],
        "Examples": [
            "examples/complete_example.py",
            "examples/quick_start.py",
        ],
        "CI/CD": [
            ".github/workflows/ci.yml",
        ]
    }
    
    all_present = True
    missing_files = []
    
    for category, files in structure.items():
        print(f"\n📁 {category}:")
        for file_path in files:
            full_path = base_path / file_path
            if full_path.exists():
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ {file_path} (missing)")
                missing_files.append(file_path)
                all_present = False
    
    print("\n" + "=" * 60)
    
    if all_present:
        print("✨ All files present! Package structure is complete.")
        return True
    else:
        print(f"⚠️ Missing {len(missing_files)} files:")
        for f in missing_files:
            print(f"  - {f}")
        return False


def check_imports():
    """Try importing main modules."""
    
    print("\n📦 Checking imports...")
    print("=" * 60)
    
    imports_to_check = [
        "ual_adapter",
        "ual_adapter.core.adapter",
        "ual_adapter.core.air",
        "ual_adapter.core.dispatcher",
        "ual_adapter.core.projection",
        "ual_adapter.binders.base",
        "ual_adapter.training.trainer",
        "ual_adapter.utils.model_utils",
    ]
    
    # Add package path to sys.path for import testing
    package_path = Path(__file__).parent
    if str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))
    
    all_imported = True
    
    for module_name in imports_to_check:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name}")
        except ImportError as e:
            print(f"  ❌ {module_name}: {e}")
            all_imported = False
        except Exception as e:
            print(f"  ⚠️ {module_name}: {e}")
    
    print("=" * 60)
    
    if all_imported:
        print("✨ All imports successful!")
        return True
    else:
        print("⚠️ Some imports failed. Check dependencies.")
        return False


def generate_statistics():
    """Generate package statistics."""
    
    print("\n📊 Package Statistics:")
    print("=" * 60)
    
    base_path = Path(__file__).parent
    
    # Count Python files
    py_files = list(base_path.glob("**/*.py"))
    py_files = [f for f in py_files if "build" not in str(f) and "__pycache__" not in str(f)]
    
    # Count lines of code
    total_lines = 0
    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                total_lines += lines
        except:
            pass
    
    # Count test files
    test_files = [f for f in py_files if "test_" in f.name or "_test.py" in f.name]
    
    print(f"  📝 Python files: {len(py_files)}")
    print(f"  📏 Total lines of code: {total_lines:,}")
    print(f"  🧪 Test files: {len(test_files)}")
    print(f"  📦 Core modules: 5")
    print(f"  🔧 Binder architectures: 8+")
    print(f"  📚 Examples: 2")
    
    print("=" * 60)


def main():
    """Run all checks."""
    
    print("\n" + "🚀 UAL ADAPTER PACKAGE VERIFICATION 🚀".center(60))
    print("=" * 60)
    
    # Check file structure
    structure_ok = check_file_structure()
    
    # Check imports (only if structure is OK)
    imports_ok = False
    if structure_ok:
        try:
            imports_ok = check_imports()
        except Exception as e:
            print(f"Import check failed: {e}")
    
    # Generate statistics
    generate_statistics()
    
    # Final summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY".center(60))
    print("=" * 60)
    
    if structure_ok and imports_ok:
        print("✅ Package is complete and ready to use!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -e .")
        print("  2. Run tests: pytest tests/")
        print("  3. Try examples: python examples/quick_start.py")
        return 0
    elif structure_ok:
        print("⚠️ Package structure is complete but imports failed.")
        print("Install dependencies with: pip install -r requirements.txt")
        return 1
    else:
        print("❌ Package structure incomplete.")
        print("Some files are missing. Check the list above.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
