# Data Gathering Scripts

This folder contains all the scripts for collecting, processing, and managing animal Wikipedia data for LLM training.

## 📁 Folder Structure

```
data_gathering/
├── dataset_scraper.py      # Main Wikipedia scraping functionality
├── data_processor.py       # Data processing and formatting for LLM training
├── daily_scrape.py         # Automated daily data collection pipeline
├── verify_text_format.py   # Verification script for plain text format
├── training_data_demo.py   # Demo script showing how to use the data
└── DATA_COLLECTION_README.md  # Detailed documentation
```

## 🚀 Quick Start

### Run Daily Data Collection
```bash
cd data_gathering
python3 daily_scrape.py --test    # Test mode (5 animals)
python3 daily_scrape.py           # Full collection (50 animals)
```

### Verify Data Format
```bash
cd data_gathering
python3 verify_text_format.py     # Check all training files
```

### View Usage Demo
```bash
cd data_gathering
python3 training_data_demo.py     # See how to use the data
```

## 📊 Dataset Sizes

The system creates 3 dataset sizes:
- **Nano**: 5 animals (quick testing)
- **Mini**: 50% of animals (development)
- **Full**: All animals (production)

Each size has 4 formats: Basic, With Summary, Q&A, Instruction.

## 📂 Output Location

Training data is saved to:
```
/Datasets/processed/*.txt
```

All files are in **plain text format** ready for LLM training.

## 🔧 Individual Scripts

### dataset_scraper.py
```python
from dataset_scraper import AnimalWikiScraper
scraper = AnimalWikiScraper()
scraper.scrape_daily_animals(max_animals=10)
```

### data_processor.py
```python
from data_processor import AnimalDataProcessor
processor = AnimalDataProcessor()
processor.process_all_datasets()
```

## 📋 Requirements

Make sure you have the required dependencies:
```bash
pip install wikipedia-api
```

## 📖 Documentation

See `DATA_COLLECTION_README.md` for comprehensive documentation including:
- Detailed usage examples
- Data format specifications
- Integration with LLM training pipelines
- Troubleshooting guide

## 🎯 Use Cases

- **Research**: Academic studies on LLM training data
- **Development**: Testing and prototyping
- **Production**: Building animal knowledge bases
- **Education**: Teaching data collection and processing

## 🔄 Automation

Set up cron jobs for automated daily data collection:
```bash
# Example: Run daily at 2 AM
0 2 * * * cd /path/to/data_gathering && python3 daily_scrape.py
```
