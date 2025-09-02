#!/usr/bin/env python3
"""
Daily Animal Wikipedia Scraper

This script automates the daily collection of animal Wikipedia pages
for LLM training data.
"""

import sys
import os
from datetime import datetime
from dataset_scraper import AnimalWikiScraper
from data_processor import AnimalDataProcessor


def run_daily_pipeline(max_animals=50):
    """Run the complete daily data collection and processing pipeline"""

    print("=" * 60)
    print(
        f"Starting Daily Animal Data Collection - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("=" * 60)

    # Step 1: Scrape animal data
    print("\n1. Starting data collection...")
    scraper = AnimalWikiScraper()

    scraped_data, dataset_path = scraper.scrape_daily_animals(max_animals=max_animals)

    if not scraped_data:
        print("No data was scraped. Exiting.")
        return False

    print(f"✓ Scraped {len(scraped_data)} animal articles")

    # Step 2: Process the data for LLM training
    print("\n2. Processing data for LLM training...")
    processor = AnimalDataProcessor()

    processed_path = processor.process_dataset_file(dataset_path)
    print(f"✓ Processed data saved to: {processed_path}")

    # Step 3: Create training files in different formats and sizes
    print("\n3. Creating training data files...")
    formats = ["basic", "with_summary", "qa_format", "instruction"]
    sizes = ["nano", "mini", "full"]

    training_files = []
    for format_type in formats:
        print(f"\n  📝 {format_type.upper()} format:")

        # Create individual format file
        try:
            training_file = processor.create_training_file(
                processed_path, format_type=format_type
            )
            training_files.append(training_file)
            print(f"    ✓ Individual: {os.path.basename(training_file)}")
        except Exception as e:
            print(f"    ✗ Failed individual {format_type}: {e}")

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

    # Step 4: Summary
    print(f"\n4. Dataset creation complete!")
    print(f"   📊 Total training files created: {len(training_files)}")
    print(f"   📦 Sizes: Nano (5 animals), Mini (50%), Full (100%)")
    print(f"   📝 Formats: Basic, With Summary, Q&A, Instruction")

    print("\n" + "=" * 60)
    print("Daily pipeline completed successfully!")
    print("=" * 60)
    print(f"Dataset saved: {dataset_path}")
    print(f"Processed data: {processed_path}")
    print(f"Training files created: {len(training_files)}")

    return True


def main():
    """Main function with command line argument support"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Daily Animal Wikipedia Data Collection"
    )
    parser.add_argument(
        "--animals",
        type=int,
        default=50,
        help="Maximum number of animals to scrape (default: 50)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with only 5 animals"
    )

    args = parser.parse_args()

    # Adjust for test mode
    if args.test:
        args.animals = 5
        print("Running in TEST MODE (5 animals)")

    # Run the pipeline
    success = run_daily_pipeline(max_animals=args.animals)

    if success:
        print("\n🎉 Daily data collection completed successfully!")
        print(
            "You can now use the training data files in the Datasets/processed/ directory"
        )
    else:
        print("\n❌ Daily data collection failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
