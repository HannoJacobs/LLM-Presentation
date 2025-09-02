# Animal Wikipedia Data Collection for LLM Training

This system automatically collects Wikipedia articles about animals and processes them into training data for decoder-only large language models.

## Files Overview

### Core Scripts
- `src/dataset_scraper.py` - Main scraping functionality with AnimalWikiScraper class
- `src/data_processor.py` - Data processing and formatting for LLM training
- `src/daily_scrape.py` - Automated daily data collection pipeline

### Data Structure
```
Datasets/
├── animal_wiki_dataset_YYYY-MM-DD.json          # Raw scraped data
└── processed/
    ├── processed_animal_dataset_YYYY-MM-DD.json # Cleaned structured data
    ├── llm_training_data_*.txt                  # Individual format training files
    └── combined_llm_training_data_*.txt         # Combined training files
```

## Usage

### Daily Data Collection
Run the complete pipeline to scrape and process new animal data:

```bash
# Full daily collection (50 animals)
python3 src/daily_scrape.py

# Test mode (5 animals)
python3 src/daily_scrape.py --test

# Custom number of animals
python3 src/daily_scrape.py --animals 25
```

### Individual Components
```bash
# Just scrape data
python3 src/dataset_scraper.py

# Process existing data
python3 src/data_processor.py
```

## Data Formats

The system creates training data in multiple formats:

### Basic Format
```
Animal: Cat

[Full Wikipedia article content]
```

### With Summary Format
```
Animal: Cat
Summary: [Brief summary]

[Full Wikipedia article content]
```

### Q&A Format
```
Question: Tell me about Cat.

Answer: [Full Wikipedia article content]
```

### Instruction Format
```
Write an article about Cat.

[Full Wikipedia article content]
```

## Animal Categories

The scraper includes animals from these categories:
- **Mammals**: Cat, Dog, Elephant, Lion, Tiger, Giraffe, Panda, Koala, etc.
- **Birds**: Eagle, Owl, Penguin, Ostrich, Flamingo, Peacock, Parrot, etc.
- **Reptiles**: Crocodile, Snake, Turtle, Chameleon, Iguana, Gecko, etc.
- **Amphibians**: Frog, Toad, Salamander, Newt
- **Fish**: Goldfish, Tuna, Salmon, Clownfish, Piranha, etc.
- **Insects**: Butterfly, Bee, Ant, Spider, Scorpion, Dragonfly, etc.
- **Additional**: Cheetah, Leopard, Jaguar, Hyena, Meerkat, Monkey, etc.

## Features

- **Daily Automation**: Timestamped datasets for tracking data collection over time
- **Respectful Scraping**: Built-in delays to avoid overwhelming Wikipedia servers
- **Error Handling**: Graceful handling of missing pages or network issues
- **Multiple Formats**: Training data prepared in various formats for different LLM training approaches
- **Data Cleaning**: Automatic removal of Wikipedia formatting artifacts
- **Statistics**: Detailed reporting of data collection metrics

## Configuration

### Animal List
Edit `get_animal_list()` in `dataset_scraper.py` to modify the animals being scraped.

### Scraping Parameters
- **Max Animals**: Default 50 per day (configurable)
- **Delay Range**: 1-3 seconds between requests (configurable)
- **Random Selection**: Animals are randomly selected to avoid repetition

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `wikipedia-api` - For scraping Wikipedia content
- Standard libraries: `json`, `os`, `datetime`, `time`, `random`, `re`, `pathlib`

## Dataset Sizes

The system creates **3 different sizes** of training datasets:

### 🧬 Nano Dataset (5 animals)
- **Purpose**: Quick testing, development, debugging
- **Size**: First 5 animals only
- **Use case**: Fast iteration, testing your training pipeline
- **Files**: `*_nano_*.txt`

### 📊 Mini Dataset (50% animals)
- **Purpose**: Development, hyperparameter tuning
- **Size**: Half of all available animals
- **Use case**: Training experiments, validation
- **Files**: `*_mini_*.txt`

### 🌍 Full Dataset (all animals)
- **Purpose**: Production training
- **Size**: Complete animal collection
- **Use case**: Final model training
- **Files**: `*_full_*.txt` or `combined_*.txt`

## Plain Text Training Data Format

✅ **ALL TRAINING DATA IS SAVED IN PLAIN TEXT (.txt) FORMAT**

Your LLM training files are located in:
```
Datasets/processed/*.txt
```

### File Format Details:
- **Encoding**: UTF-8 (supports all languages)
- **Format**: Pure plain text, no binary data
- **Separators**: Articles separated by 80 "=" characters
- **Content**: Clean, readable text ready for tokenization

### Usage Examples:

```python
# Quick testing with Nano dataset
with open('Datasets/processed/combined_llm_training_data_basic_nano_2025-09-02.txt', 'r', encoding='utf-8') as f:
    nano_data = f.read()

# Development with Mini dataset
with open('Datasets/processed/combined_llm_training_data_basic_mini_2025-09-02.txt', 'r', encoding='utf-8') as f:
    mini_data = f.read()

# Production with Full dataset
with open('Datasets/processed/combined_llm_training_data_basic_full_2025-09-02.txt', 'r', encoding='utf-8') as f:
    full_data = f.read()

# Split any dataset into training samples
articles = full_data.split('=' * 80)
training_samples = [article.strip() for article in articles if article.strip()]
```

## Output Statistics

Recent test run statistics:
- **Animals Scraped**: 5 (test mode)
- **Nano Dataset**: 5 animals × 4 formats = 4 files
- **Mini Dataset**: ~3 animals × 4 formats = 4 files (50% of scraped)
- **Full Dataset**: 5 animals × 4 formats = 4 files
- **Total Characters**: ~90,000 per format
- **Average per Article**: ~18,000 characters
- **Processing Time**: ~1-2 minutes
- **Total Training Files**: 12 plain text files
- **File Format**: Plain text (.txt) - perfect for LLM training

### Expected Production Output:
- **Nano**: 5 animals, ~9,000 characters
- **Mini**: ~35 animals, ~630,000 characters
- **Full**: ~70 animals, ~1,260,000 characters

## Integration with LLM Training

The processed text files can be directly used for training decoder-only LLMs like GPT-style models. The data includes:

- Rich factual content about diverse animal species
- Consistent formatting across all entries
- Multiple training formats for different learning objectives
- Clean, preprocessed text ready for tokenization

## Future Enhancements

- Add more animal categories (marine life, prehistoric animals, etc.)
- Implement incremental updates (only scrape changed articles)
- Add data validation and quality checks
- Support for multiple languages
- Integration with LLM training pipelines
