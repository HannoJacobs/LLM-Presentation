#!/usr/bin/env python3
"""
Demonstration of how to use the plain text training data with an LLM
"""

import os
from pathlib import Path


def demonstrate_training_data_usage():
    """Show how to load and use the plain text training data"""

    print("🔧 Plain Text Training Data Usage Demo")
    print("=" * 50)

    # Path to your training data
    datasets_dir = "/Users/hannojacobs/Desktop/LLM-Presentation/Datasets"
    processed_dir = os.path.join(datasets_dir, "processed")

    # List available training files
    if os.path.exists(processed_dir):
        txt_files = [f for f in os.listdir(processed_dir) if f.endswith(".txt")]

        # Group files by size
        nano_files = [f for f in txt_files if "_nano_" in f]
        mini_files = [f for f in txt_files if "_mini_" in f]
        full_files = [
            f
            for f in txt_files
            if "combined_" in f and "_nano_" not in f and "_mini_" not in f
        ]
        individual_files = [
            f
            for f in txt_files
            if f not in nano_files and f not in mini_files and f not in full_files
        ]

        print(f"📂 Found {len(txt_files)} training data files:")
        print(f"   🧬 Nano (5 animals): {len(nano_files)} files")
        print(f"   📊 Mini (50% animals): {len(mini_files)} files")
        print(f"   🌍 Full (all animals): {len(full_files)} files")
        print(f"   📄 Individual files: {len(individual_files)} files")

        print("\n📋 Available sizes:")
        for size, files in [
            ("Nano", nano_files),
            ("Mini", mini_files),
            ("Full", full_files),
        ]:
            if files:
                print(f"   {size}: {files[0]}")
    else:
        print("❌ Processed directory not found")
        return

    print("\n" + "=" * 50)
    print("💡 How to use these files in your LLM training:")

    print(
        """
1. CHOOSE YOUR DATASET SIZE:
   ```python
   # Nano (5 animals) - for quick testing
   nano_file = 'Datasets/processed/combined_llm_training_data_basic_nano_2025-09-02.txt'

   # Mini (50% of animals) - for development
   mini_file = 'Datasets/processed/combined_llm_training_data_basic_mini_2025-09-02.txt'

   # Full (all animals) - for production training
   full_file = 'Datasets/processed/combined_llm_training_data_basic_full_2025-09-02.txt'
   ```

2. LOAD THE DATA:
   ```python
   with open(full_file, 'r', encoding='utf-8') as f:
       training_text = f.read()
   ```

3. SPLIT INTO TRAINING SAMPLES:
   ```python
   # Split by article separators
   articles = training_text.split('=' * 80)

   # Each article becomes a training sample
   training_samples = [article.strip() for article in articles if article.strip()]
   ```

4. TOKENIZE FOR YOUR MODEL:
   ```python
   # Use your tokenizer (example with transformers)
   from transformers import AutoTokenizer

   tokenizer = AutoTokenizer.from_pretrained('your-model-name')
   tokenized_data = tokenizer(training_samples, truncation=True, padding=True)
   ```

5. FORMAT FOR TRAINING:
   Each file contains plain text articles separated by:
   ==================================================

   Perfect for:
   • Pre-training on factual knowledge
   • Fine-tuning on specific domains
   • Creating custom knowledge bases
   """
    )

    print("\n" + "=" * 50)
    print("📋 Available Training Formats:")
    print("• basic: Simple animal name + article format")
    print("• with_summary: Includes Wikipedia summary")
    print("• qa_format: Question-Answer format")
    print("• instruction: Instruction-response format")

    print("\n🎯 Choose format based on your training objective:")
    print("• Basic format → General knowledge pre-training")
    print("• Q&A format → Conversational fine-tuning")
    print("• Instruction format → Instruction-tuned models")


def show_sample_training_data():
    """Show a sample of the training data"""

    sample_file = "/Users/hannojacobs/Desktop/LLM-Presentation/Datasets/processed/combined_llm_training_data_basic_2025-09-02.txt"

    if os.path.exists(sample_file):
        print("\n" + "=" * 50)
        print("📖 Sample Training Data (First Article):")
        print("=" * 50)

        with open(sample_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Get first article (before first separator)
        first_article = content.split("=" * 80)[0].strip()

        # Show first 500 characters
        print(first_article[:500] + "...")

        print("\n✂️  Articles are separated by 80 '=' characters")
        print("📏 Total training data size: Very large (90K+ characters per format)")


def main():
    """Main demonstration function"""

    demonstrate_training_data_usage()
    show_sample_training_data()

    print("\n" + "=" * 50)
    print("✅ Your training data is ready to use!")
    print("📍 Location: Datasets/processed/*.txt")
    print("📄 Format: Plain text (.txt) files")
    print("🔧 Compatible with any LLM training framework")


if __name__ == "__main__":
    main()
