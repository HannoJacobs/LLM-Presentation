#!/usr/bin/env python3
"""
Consolidated Data Pipeline for Animal Wikipedia Collection

This single file contains all functionality for:
- Scraping Wikipedia animal articles
- Processing and formatting text for LLM training
- Creating multiple dataset sizes (nano/mini/full)
- Generating various training formats
- Validating output format and statistics

Usage:
    python3 data_pipeline.py --test    # Quick test with 5 animals
    python3 data_pipeline.py           # Full collection (50 animals)
    python3 data_pipeline.py --animals 25  # Custom count
    python3 data_pipeline.py --demo    # Show usage demo
"""

import wikipediaapi
import json
import os
import re
from datetime import datetime
from pathlib import Path
import time
import random
import argparse
import sys


class AnimalWikiScraper:
    """Wikipedia scraper for animal articles"""

    def __init__(
        self, datasets_dir="/Users/hannojacobs/Desktop/LLM-Presentation/Datasets"
    ):
        self.wiki = wikipediaapi.Wikipedia(
            user_agent="AnimalDatasetScraper/1.0 (Educational LLM Training Project)",
            language="en",
        )
        self.datasets_dir = datasets_dir
        self.ensure_datasets_dir()

    def ensure_datasets_dir(self):
        """Ensure the datasets directory exists"""
        if not os.path.exists(self.datasets_dir):
            os.makedirs(self.datasets_dir)

    def get_animal_list(self):
        """Comprehensive list of animals to scrape"""
        animals = [
            # Mammals
            "Cat",
            "Dog",
            "Elephant",
            "Lion",
            "Tiger",
            "Giraffe",
            "Zebra",
            "Panda",
            "Koala",
            "Kangaroo",
            "Platypus",
            "Wombat",
            "Sloth",
            "Armadillo",
            "Hedgehog",
            "Fox",
            "Wolf",
            "Bear",
            "Raccoon",
            "Otter",
            "Seal",
            "Walrus",
            "Dolphin",
            "Whale",
            "Shark",
            "Octopus",
            "Squid",
            "Jellyfish",
            "Starfish",
            "Lobster",
            # Birds
            "Eagle",
            "Owl",
            "Penguin",
            "Ostrich",
            "Flamingo",
            "Peacock",
            "Parrot",
            "Hummingbird",
            "Woodpecker",
            "Swan",
            "Duck",
            "Goose",
            "Turkey",
            "Chicken",
            # Reptiles
            "Crocodile",
            "Alligator",
            "Snake",
            "Lizard",
            "Turtle",
            "Tortoise",
            "Chameleon",
            "Iguana",
            "Gecko",
            # Amphibians
            "Frog",
            "Toad",
            "Salamander",
            "Newt",
            # Fish
            "Goldfish",
            "Tuna",
            "Salmon",
            "Clownfish",
            "Piranha",
            "Swordfish",
            # Insects
            "Butterfly",
            "Bee",
            "Ant",
            "Spider",
            "Scorpion",
            "Dragonfly",
            "Ladybug",
            # Additional animals
            "Cheetah",
            "Leopard",
            "Jaguar",
            "Hyena",
            "Meerkat",
            "Lemur",
            "Monkey",
            "Gorilla",
            "Chimpanzee",
            "Orangutan",
            "Rabbit",
            "Hare",
            "Squirrel",
            "Mouse",
            "Rat",
            "Hamster",
            "Guinea pig",
            "Horse",
            "Cow",
            "Sheep",
            "Goat",
            "Pig",
            "Deer",
            "Moose",
            "Caribou",
            "Buffalo",
            "Camel",
            "Llama",
            "Alpaca",
        ]
        return animals

    def scrape_animal_page(self, animal_name):
        """Scrape a single animal's Wikipedia page"""
        try:
            page = self.wiki.page(animal_name)

            if not page.exists():
                print(f"❌ Page for {animal_name} does not exist")
                return None

            # Get the full text content
            content = page.text

            # Extract basic info
            data = {
                "animal_name": animal_name,
                "title": page.title,
                "url": page.fullurl,
                "content": content,
                "summary": page.summary[:500] if page.summary else "",
                "content_length": len(content),
                "scraped_at": datetime.now().isoformat(),
            }

            return data

        except Exception as e:
            print(f"❌ Error scraping {animal_name}: {str(e)}")
            return None

    def save_daily_dataset(self, scraped_data, date_str=None):
        """Save the scraped data to a daily JSON file"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        filename = f"animal_wiki_dataset_{date_str}.json"
        filepath = os.path.join(self.datasets_dir, filename)

        # Filter out None values (failed scrapes)
        valid_data = [item for item in scraped_data if item is not None]

        dataset = {
            "metadata": {
                "date_created": date_str,
                "total_animals": len(valid_data),
                "scraper_version": "1.0",
            },
            "animals": valid_data,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved {len(valid_data)} animal articles to {filepath}")
        return filepath

    def scrape_daily_animals(self, max_animals=None, delay_range=(1, 3)):
        """Scrape a daily set of animal Wikipedia pages"""
        animals = self.get_animal_list()

        if max_animals:
            # Randomly select animals to avoid always scraping the same ones
            animals = random.sample(animals, min(max_animals, len(animals)))

        scraped_data = []
        successful = 0

        print(f"🚀 Starting to scrape {len(animals)} animal pages...")

        for i, animal in enumerate(animals, 1):
            print(f"📄 Scraping {i}/{len(animals)}: {animal}")

            data = self.scrape_animal_page(animal)
            if data:
                scraped_data.append(data)
                successful += 1

            # Add random delay to be respectful to Wikipedia
            if i < len(animals):  # Don't delay after the last request
                delay = random.uniform(*delay_range)
                time.sleep(delay)

        print(f"✅ Successfully scraped {successful}/{len(animals)} animals")

        # Save the dataset
        filepath = self.save_daily_dataset(scraped_data)

        return scraped_data, filepath


class AnimalDataProcessor:
    """Process scraped data for LLM training"""

    def __init__(
        self, datasets_dir="/Users/hannojacobs/Desktop/LLM-Presentation/Datasets"
    ):
        self.datasets_dir = Path(datasets_dir)
        self.processed_dir = self.datasets_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)

    def load_dataset(self, dataset_path):
        """Load a scraped dataset from JSON file"""
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def clean_text(self, text):
        """Clean text for LLM training"""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r" +", " ", text)

        # Remove Wikipedia-specific formatting
        text = re.sub(r"\[\d+\]", "", text)  # Remove citation numbers [1], [2], etc.
        text = re.sub(r"==.*?==", "", text)  # Remove section headers
        text = re.sub(r"^\s*$", "", text, flags=re.MULTILINE)  # Remove empty lines

        return text.strip()

    def process_animal_data(self, animal_data):
        """Process a single animal's data for training"""
        content = self.clean_text(animal_data["content"])
        summary = self.clean_text(animal_data["summary"])

        # Create different training formats
        formats = {
            "basic": f"Animal: {animal_data['animal_name']}\n\n{content}",
            "with_summary": f"Animal: {animal_data['animal_name']}\nSummary: {summary}\n\n{content}",
            "qa_format": f"Question: Tell me about {animal_data['animal_name']}.\nAnswer: {content}",
            "instruction": f"Write an article about {animal_data['animal_name']}.\n\n{content}",
        }

        return {
            "animal_name": animal_data["animal_name"],
            "title": animal_data["title"],
            "url": animal_data["url"],
            "content_length": len(content),
            "scraped_at": animal_data["scraped_at"],
            "training_formats": formats,
        }

    def create_combined_training_data(self, format_type="basic", size="full"):
        """Create a combined training file from all processed datasets with different sizes"""
        processed_files = list(
            self.processed_dir.glob("processed_animal_dataset_*.json")
        )

        if not processed_files:
            print("No processed dataset files found. Run process_all_datasets() first.")
            return None

        all_training_texts = []
        total_animals = 0

        for processed_file in sorted(processed_files):
            with open(processed_file, "r", encoding="utf-8") as f:
                dataset = json.load(f)

            for animal in dataset["animals"]:
                all_training_texts.append(animal["training_formats"][format_type])
                total_animals += 1

        # Apply size filtering
        if size == "nano":
            # First 5 animals
            all_training_texts = all_training_texts[:5]
            total_animals = min(5, total_animals)
        elif size == "mini":
            # Half of all animals
            half_size = len(all_training_texts) // 2
            all_training_texts = all_training_texts[:half_size]
            total_animals = half_size
        # "full" uses all animals (default)

        # Save combined training file
        size_suffix = f"_{size}" if size != "full" else ""
        combined_filename = f"combined_llm_training_data_{format_type}{size_suffix}_{datetime.now().strftime('%Y-%m-%d')}.txt"
        combined_path = self.processed_dir / combined_filename

        with open(combined_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(all_training_texts):
                f.write(text)
                if i < len(all_training_texts) - 1:
                    f.write("\n\n" + "=" * 80 + "\n\n")

        size_name = {
            "nano": "Nano (5 animals)",
            "mini": "Mini (50% animals)",
            "full": "Full (all animals)",
        }[size]
        print(f"📦 Created {size_name} training file: {combined_path}")
        print(f"   📊 Total training samples: {total_animals}")

        # Calculate statistics
        total_chars = sum(len(text) for text in all_training_texts)
        avg_chars = total_chars / total_animals if total_animals > 0 else 0

        print(f"   📏 Total characters: {total_chars:,}")
        print(f"Average characters per sample: {avg_chars:.2f}")
        return combined_path

    def process_all_datasets(self):
        """Process all dataset files in the datasets directory"""
        dataset_files = list(self.datasets_dir.glob("animal_wiki_dataset_*.json"))

        if not dataset_files:
            print("❌ No dataset files found in the datasets directory.")
            return []

        processed_files = []
        for dataset_file in sorted(dataset_files):
            print(f"🔄 Processing dataset: {dataset_file}")
            processed_file = self.process_dataset_file(dataset_file)
            processed_files.append(processed_file)

        print(f"\n✅ Processed {len(processed_files)} dataset files.")
        return processed_files

    def process_dataset_file(self, dataset_path):
        """Process an entire dataset file"""
        dataset = self.load_dataset(dataset_path)
        processed_animals = []

        for animal in dataset["animals"]:
            processed = self.process_animal_data(animal)
            processed_animals.append(processed)

        # Save processed data
        date_str = dataset["metadata"]["date_created"]
        output_filename = f"processed_animal_dataset_{date_str}.json"
        output_path = self.processed_dir / output_filename

        processed_dataset = {
            "metadata": {
                "original_dataset": os.path.basename(dataset_path),
                "date_created": date_str,
                "total_animals": len(processed_animals),
                "processing_date": datetime.now().isoformat(),
                "processor_version": "1.0",
            },
            "animals": processed_animals,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_dataset, f, indent=2, ensure_ascii=False)

        print(
            f"💾 Processed {len(processed_animals)} animals and saved to {output_path}"
        )
        return output_path


class DataValidator:
    """Validate training data format and statistics"""

    def __init__(
        self, datasets_dir="/Users/hannojacobs/Desktop/LLM-Presentation/Datasets"
    ):
        self.datasets_dir = Path(datasets_dir)
        self.processed_dir = self.datasets_dir / "processed"

    def verify_text_format(self):
        """Verify all training files are in plain text format"""

        if not self.processed_dir.exists():
            print("❌ Processed directory not found")
            return False

        # Find all training text files
        txt_files = list(self.processed_dir.glob("*.txt"))

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

        print("📊 Dataset Overview:")
        print(f"   Total files: {len(txt_files)}")
        print(f"   🧬 Nano (5 animals): {len(file_groups['nano'])} files")
        print(f"   📊 Mini (50% animals): {len(file_groups['mini'])} files")
        print(f"   🌍 Full (all animals): {len(file_groups['full'])} files")
        print(f"   📄 Individual files: {len(file_groups['individual'])} files")
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


def run_data_pipeline(max_animals=50):
    """Run the complete data collection and processing pipeline"""

    print("=" * 60)
    print(
        f"🚀 Starting Animal Data Collection Pipeline - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("=" * 60)

    # Step 1: Scrape animal data
    print("\n1️⃣ 📊 SCRAPING PHASE")
    scraper = AnimalWikiScraper()

    scraped_data, dataset_path = scraper.scrape_daily_animals(max_animals=max_animals)

    if not scraped_data:
        print("❌ No data was scraped. Exiting.")
        return False

    print(f"✅ Scraped {len(scraped_data)} animal articles")

    # Step 2: Process the data for LLM training
    print("\n2️⃣ 🔄 PROCESSING PHASE")
    processor = AnimalDataProcessor()

    processed_path = processor.process_dataset_file(dataset_path)
    print(f"✅ Processed data saved to: {processed_path}")

    # Step 3: Create training files in different formats and sizes
    print("\n3️⃣ 📦 TRAINING DATA GENERATION")

    formats = ["basic", "with_summary", "qa_format", "instruction"]
    sizes = ["nano", "mini", "full"]

    training_files = []
    for format_type in formats:
        print(f"\n📝 {format_type.upper()} format:")

        # Create different sizes for this format
        for size in sizes:
            try:
                combined_file = processor.create_combined_training_data(
                    format_type=format_type, size=size
                )
                if combined_file:
                    training_files.append(combined_file)
                    print(f"    ✓ {size.upper()}: {os.path.basename(combined_file)}")
            except Exception as e:
                print(f"    ✗ Failed {size} {format_type}: {e}")

    # Step 4: Validation
    print("\n4️⃣ 🔍 VALIDATION PHASE")
    validator = DataValidator()
    validation_success = validator.verify_text_format()

    # Step 5: Summary
    print(f"\n5️⃣ 📊 PIPELINE SUMMARY")
    print("=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)
    print(f"📁 Dataset saved: {dataset_path}")
    print(f"📁 Processed data: {processed_path}")
    print(f"📦 Training files created: {len(training_files)}")
    print("📏 Dataset sizes: Nano (5), Mini (50%), Full (100%)")
    print("📝 Training formats: Basic, With Summary, Q&A, Instruction")
    if validation_success:
        print("✅ All validation checks passed!")
        print("🎯 Ready for LLM training!")
    else:
        print("⚠️  Some validation checks failed - please review output above")

    return True


def show_usage_demo():
    """Show how to use the generated training data"""

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


def main():
    """Main function with command line interface"""

    parser = argparse.ArgumentParser(
        description="Animal Wikipedia Data Collection Pipeline"
    )
    parser.add_argument(
        "--animals",
        type=int,
        default=50,
        help="Maximum number of animals to scrape (default: 50)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with 5 animals"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Show usage demo instead of running pipeline",
    )

    args = parser.parse_args()

    # Show demo if requested
    if args.demo:
        show_usage_demo()
        return

    # Adjust for test mode
    if args.test:
        args.animals = 5
        print("🧪 Running in TEST MODE (5 animals)")

    # Run the complete pipeline
    success = run_data_pipeline(max_animals=args.animals)

    if success:
        print("\n🎉 Data collection pipeline completed successfully!")
        print("📍 Training data location: Datasets/processed/*.txt")
        print("🔧 Use these plain text files directly in your LLM training!")
    else:
        print("\n❌ Data collection pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
