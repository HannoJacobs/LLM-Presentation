import json
import os
import re
from pathlib import Path
import pandas as pd
from datetime import datetime


class AnimalDataProcessor:
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

    def process_dataset_file(self, dataset_path):
        """Process an entire dataset file"""
        print(f"Processing dataset: {dataset_path}")

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

        print(f"Processed {len(processed_animals)} animals and saved to {output_path}")
        return output_path

    def create_training_file(self, processed_dataset_path, format_type="basic"):
        """Create a single training text file from processed data"""
        with open(processed_dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        training_texts = []
        for animal in dataset["animals"]:
            training_texts.append(animal["training_formats"][format_type])

        # Save training file
        date_str = dataset["metadata"]["date_created"]
        training_filename = f"llm_training_data_{format_type}_{date_str}.txt"
        training_path = self.processed_dir / training_filename

        with open(training_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(training_texts):
                f.write(text)
                if i < len(training_texts) - 1:
                    f.write("\n\n" + "=" * 80 + "\n\n")

        print(f"Created training file: {training_path}")
        print(f"Total training samples: {len(training_texts)}")

        # Calculate statistics
        total_chars = sum(len(text) for text in training_texts)
        avg_chars = total_chars / len(training_texts) if training_texts else 0

        print(f"Total characters: {total_chars:,}")
        print(f"Average characters per sample: {avg_chars:.2f}")
        return training_path

    def process_all_datasets(self):
        """Process all dataset files in the datasets directory"""
        dataset_files = list(self.datasets_dir.glob("animal_wiki_dataset_*.json"))

        if not dataset_files:
            print("No dataset files found in the datasets directory.")
            return []

        processed_files = []
        for dataset_file in sorted(dataset_files):
            processed_file = self.process_dataset_file(dataset_file)
            processed_files.append(processed_file)

        print(f"\nProcessed {len(processed_files)} dataset files.")
        return processed_files

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
        print(f"Created {size_name} training file: {combined_path}")
        print(f"Total training samples: {total_animals}")

        # Calculate statistics
        total_chars = sum(len(text) for text in all_training_texts)
        avg_chars = total_chars / total_animals if total_animals > 0 else 0

        print(f"Total characters: {total_chars:,}")
        print(f"Average characters per sample: {avg_chars:.2f}")
        return combined_path


def main():
    """Main function to process animal datasets for LLM training"""
    processor = AnimalDataProcessor()

    print("Starting data processing pipeline...")

    # Process all datasets
    processed_files = processor.process_all_datasets()

    if processed_files:
        # Create training files in different formats and sizes
        formats = ["basic", "with_summary", "qa_format", "instruction"]
        sizes = ["nano", "mini", "full"]

        for format_type in formats:
            print(f"\n{'='*60}")
            print(f"Creating {format_type.upper()} format training data...")

            for size in sizes:
                print(f"\n  📦 {size.upper()} size ({format_type}):")
                processor.create_combined_training_data(
                    format_type=format_type, size=size
                )

    print(f"\n{'='*60}")
    print("Data processing completed!")
    print(
        f"Created datasets in 3 sizes (nano/mini/full) × 4 formats = 12 training files"
    )


if __name__ == "__main__":
    main()
