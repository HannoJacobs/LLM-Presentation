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
python3 data_pipeline.py --test    # Test mode (5 animals)
python3 data_pipeline.py           # Full collection (50 animals)
python3 data_pipeline.py --animals 25  # Custom count
```

### Show Usage Demo
```bash
python3 data_pipeline.py --demo    # See how to use the generated data
```

## 📊 What the Pipeline Does

The single `data_pipeline.py` script handles everything:

1. **📊 Scraping**: Collects animal Wikipedia articles
2. **🔄 Processing**: Cleans and formats text for LLM training
3. **📦 Generation**: Creates multiple dataset sizes and formats
4. **🔍 Validation**: Verifies plain text format and statistics
5. **📋 Demo**: Shows how to use the generated training data

## 🎯 Dataset Sizes Created

- **🧬 Nano**: 5 animals (perfect for testing/debugging)
- **📊 Mini**: 50% of animals (development/experiments)
- **🌍 Full**: All animals (production training)

## 📝 Training Formats

Each dataset size includes 4 training formats:
- **Basic**: `Animal: Cat\n\n[Article content]`
- **With Summary**: Includes Wikipedia summary
- **Q&A**: `Question: Tell me about Cat.\nAnswer: [Content]`
- **Instruction**: `Write an article about Cat.\n\n[Content]`

## 📂 Output Location

Training data is saved to:
```
/Datasets/processed/*.txt
```

**All files are in pure plain text format** ready for LLM training!

## 🔧 Usage Examples

### Quick Test
```bash
python3 data_pipeline.py --test
```

### Full Collection
```bash
python3 data_pipeline.py
```

### Custom Animal Count
```bash
python3 data_pipeline.py --animals 25
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
- **16 training files** total
- **4 files per format** (nano, mini, full, individual)
- **4 formats** (basic, with_summary, qa_format, instruction)
- **All in plain text** (.txt) format
- **115K+ characters** per full dataset

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
