import wikipediaapi
import json
import os
from datetime import datetime
import time
import random


class AnimalWikiScraper:
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
                print(f"Page for {animal_name} does not exist")
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
            print(f"Error scraping {animal_name}: {str(e)}")
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

        print(f"Saved {len(valid_data)} animal articles to {filepath}")
        return filepath

    def scrape_daily_animals(self, max_animals=None, delay_range=(1, 3)):
        """Scrape a daily set of animal Wikipedia pages"""
        animals = self.get_animal_list()

        if max_animals:
            # Randomly select animals to avoid always scraping the same ones
            animals = random.sample(animals, min(max_animals, len(animals)))

        scraped_data = []
        successful = 0

        print(f"Starting to scrape {len(animals)} animal pages...")

        for i, animal in enumerate(animals, 1):
            print(f"Scraping {i}/{len(animals)}: {animal}")

            data = self.scrape_animal_page(animal)
            if data:
                scraped_data.append(data)
                successful += 1

            # Add random delay to be respectful to Wikipedia
            if i < len(animals):  # Don't delay after the last request
                delay = random.uniform(*delay_range)
                time.sleep(delay)

        print(f"Successfully scraped {successful}/{len(animals)} animals")

        # Save the dataset
        filepath = self.save_daily_dataset(scraped_data)

        return scraped_data, filepath


def main():
    """Main function to run the daily scraper"""
    scraper = AnimalWikiScraper()

    # Scrape up to 50 animals per day (adjust as needed)
    scraped_data, filepath = scraper.scrape_daily_animals(max_animals=50)

    print(f"\nDataset saved to: {filepath}")
    print(f"Total animals scraped: {len(scraped_data)}")


def process_dataset_for_llm(dataset_path, output_dir=None):
    """Process scraped dataset for LLM training format"""
    import re

    if output_dir is None:
        output_dir = os.path.dirname(dataset_path)

    # Load the dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    processed_texts = []

    for animal in dataset["animals"]:
        content = animal["content"]

        # Basic text cleaning
        # Remove excessive whitespace
        content = re.sub(r"\n+", "\n", content)
        content = re.sub(r" +", " ", content)

        # Remove Wikipedia-specific formatting
        content = re.sub(r"\[\d+\]", "", content)  # Remove citation numbers
        content = re.sub(r"==.*?==", "", content)  # Remove section headers

        # Create training format
        training_text = f"Animal: {animal['animal_name']}\n\n{animal['summary']}\n\n{content.strip()}"

        processed_texts.append(training_text)

    # Save processed data
    output_filename = f"processed_{os.path.basename(dataset_path)}"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        for text in processed_texts:
            f.write(text + "\n\n" + "=" * 50 + "\n\n")

    print(
        f"Processed {len(processed_texts)} animal articles and saved to {output_path}"
    )
    return output_path


if __name__ == "__main__":
    main()
