# Consolidated Data Pipeline

This folder contains a single comprehensive script for collecting, processing, and managing animal Wikipedia data for LLM training.

## 📁 Folder Structure

```
data_gathering/
├── data_pipeline.py        # 🔧 ALL-IN-ONE consolidated script
└── README.md              # This documentation
```

## 🚀 Quick Start

### Run Complete Data Pipeline
```bash
cd data_gathering
python3 data_pipeline.py
```

That's it! The script automatically creates all 3 dataset sizes with no arguments needed.

## 📊 What the Pipeline Does

The single `data_pipeline.py` script handles everything efficiently:

1. **📊 Single Scrape**: Collects all 30 animal Wikipedia articles once
2. **🔄 Processing**: Cleans and formats text for LLM training
3. **📦 Smart Generation**: Creates 3 dataset sizes from single scrape (no duplicates!)
4. **🔍 Validation**: Verifies plain text format and statistics
5. **📋 Results**: Ready-to-use training datasets

## 🎯 Dataset Sizes Created

- **🧬 Nano**: 5 animals (first 5 from curated list)
- **📊 Mini**: 10 animals (first 10 from curated list)
- **🌍 Full**: 30 animals (complete curated list)

## 📝 Training Formats

Each dataset size includes 3 training formats:
- **Basic**: `Animal: Cat\n\n[Article content]`
- **With Summary**: Includes Wikipedia summary
- **Q&A**: `Question: Tell me about Cat.\nAnswer: [Content]`

## 📂 Output Location

Training data is saved to:
```
/Datasets/processed/*.txt
```

**All files are in pure plain text format** ready for LLM training!

## 🔧 Usage Examples

### Quick Test (Nano)
```bash
python3 data_pipeline.py --test
```

### Mini Dataset (10 animals)
```bash
python3 data_pipeline.py --size mini
```

### Full Dataset (30 animals)
```bash
python3 data_pipeline.py --size full
```

### View Usage Instructions
```bash
python3 data_pipeline.py --demo
```

## 📋 Requirements

Make sure you have the required dependencies:
```bash
pip install wikipedia-api
```

## 📊 Expected Output

Recent test run creates:
- **9 training files** total
- **3 files per format** (nano, mini, full)
- **3 formats** (basic, with_summary, qa_format)
- **All in plain text** (.txt) format
- **~100K+ characters** per full dataset (30 animals)

## 🎯 Use Cases

- **🧪 Testing**: Nano datasets for quick validation
- **🔬 Research**: Mini datasets for experiments
- **🏭 Production**: Full datasets for model training
- **📚 Education**: Complete pipeline for learning

## 🔄 Automation

Set up cron jobs for automated data collection:
```bash
# Example: Run daily at 2 AM
0 2 * * * cd /path/to/data_gathering && python3 data_pipeline.py
```

## 📖 Script Documentation

The `data_pipeline.py` file contains:

- `AnimalWikiScraper` class: Handles Wikipedia scraping
- `AnimalDataProcessor` class: Processes and formats data
- `DataValidator` class: Validates output format
- `run_data_pipeline()`: Main pipeline execution
- `show_usage_demo()`: Usage instructions
- `__main__`: Command-line interface

All functionality is consolidated into one maintainable file!
