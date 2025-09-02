#!/usr/bin/env python3
"""
Verify that all training data files are saved in plain text format
"""

import os
import json
from pathlib import Path


def verify_text_format(
    datasets_dir="/Users/hannojacobs/Desktop/LLM-Presentation/Datasets",
):
    """Verify all training files are in plain text format"""

    datasets_path = Path(datasets_dir)
    processed_path = datasets_path / "processed"

    if not processed_path.exists():
        print("❌ Processed directory not found")
        return False

    # Find all training text files
    txt_files = list(processed_path.glob("*.txt"))

    if not txt_files:
        print("❌ No training text files found")
        return False

    # Group files by size and format
    file_groups = {"nano": [], "mini": [], "full": [], "individual": []}

    for txt_file in txt_files:
        filename = txt_file.name
        if "_nano_" in filename:
            file_groups["nano"].append(txt_file)
        elif "_mini_" in filename:
            file_groups["mini"].append(txt_file)
        elif (
            "combined_" in filename
            and "_nano_" not in filename
            and "_mini_" not in filename
        ):
            file_groups["full"].append(txt_file)
        else:
            file_groups["individual"].append(txt_file)

    print(f"📊 Dataset Overview:")
    print(f"   Total files: {len(txt_files)}")
    print(f"   Nano (5 animals): {len(file_groups['nano'])} files")
    print(f"   Mini (50% animals): {len(file_groups['mini'])} files")
    print(f"   Full (all animals): {len(file_groups['full'])} files")
    print(f"   Individual files: {len(file_groups['individual'])} files")
    print("=" * 50)

    all_valid = True

    for txt_file in sorted(txt_files):
        print(f"\n📄 {txt_file.name}")

        try:
            # Read the file and check if it's valid text
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Basic checks
            if len(content) == 0:
                print("❌ File is empty")
                all_valid = False
                continue

            # Check for binary characters (very basic check)
            binary_chars = 0
            for char in content[:1000]:  # Check first 1000 chars
                if ord(char) < 32 and char not in "\n\r\t":
                    binary_chars += 1

            if binary_chars > 0:
                print(
                    f"⚠️  Found {binary_chars} potential binary characters in first 1000 chars"
                )

            # Show sample content
            lines = content.split("\n")
            print(f"✓ Total lines: {len(lines)}")
            print(f"✓ Total characters: {len(content):,}")
            print(f"✓ File size: {txt_file.stat().st_size:,} bytes")

            # Show first few lines as sample
            sample_lines = lines[:5]
            print("📝 Sample content:")
            for i, line in enumerate(sample_lines, 1):
                if line.strip():  # Only show non-empty lines
                    print(f"   {i}: {line[:100]}{'...' if len(line) > 100 else ''}")

        except Exception as e:
            print(f"❌ Error reading file: {e}")
            all_valid = False

    print("\n" + "=" * 50)
    if all_valid:
        print("✅ All training files are valid plain text format!")
        print("✅ Ready for LLM training")
    else:
        print("❌ Some files have issues - please check above")

    return all_valid


def show_file_structure(
    datasets_dir="/Users/hannojacobs/Desktop/LLM-Presentation/Datasets",
):
    """Show the current file structure"""

    datasets_path = Path(datasets_dir)

    print("\n📁 Dataset Directory Structure:")
    print("=" * 50)

    if datasets_path.exists():
        for file_path in sorted(datasets_path.rglob("*")):
            if file_path.is_file():
                relative_path = file_path.relative_to(datasets_path.parent)
                size = file_path.stat().st_size
                print("12")
    else:
        print("❌ Datasets directory not found")


def main():
    """Main verification function"""

    print("🔍 Verifying Plain Text Format for LLM Training Data")
    print("=" * 60)

    # Show file structure
    show_file_structure()

    # Verify text format
    success = verify_text_format()

    if success:
        print("\n🎉 All checks passed! Your training data is ready.")
        print("\n💡 Training data location: Datasets/processed/*.txt")
        print("💡 Use these .txt files directly in your LLM training pipeline")
    else:
        print("\n⚠️  Some issues found. Please review the output above.")


if __name__ == "__main__":
    main()
