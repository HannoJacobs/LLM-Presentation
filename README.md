# LLM-Presentation

A comprehensive project for LLM development featuring decoder-only architecture implementation and automated data collection for training.

## 📁 Project Structure

```
LLM-Presentation/
├── src/                          # Core application code
│   ├── decoder_only.py          # Main decoder-only LLM implementation
│   ├── mha.py                   # Multi-head attention implementation
│   ├── ollama_run.py            # Ollama integration
│   └── infer_decoder_only.py    # Inference utilities
├── data_gathering/              # Data collection and processing scripts
│   ├── dataset_scraper.py       # Wikipedia scraping functionality
│   ├── data_processor.py        # Data processing and formatting
│   ├── daily_scrape.py          # Automated daily data collection
│   ├── verify_text_format.py    # Data verification scripts
│   ├── training_data_demo.py    # Usage demonstrations
│   ├── README.md               # Data gathering documentation
│   └── DATA_COLLECTION_README.md # Comprehensive data collection docs
├── Datasets/                    # Generated training data
│   └── processed/              # Processed training files (.txt)
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Collect Training Data
```bash
cd data_gathering
python3 daily_scrape.py --test    # Quick test (5 animals)
python3 daily_scrape.py           # Full collection (50 animals)
```

### 3. Verify Data
```bash
cd data_gathering
python3 verify_text_format.py     # Check data format
```

### 4. Use in Training
```python
# Load training data
with open('Datasets/processed/combined_llm_training_data_basic_nano_2025-09-02.txt', 'r') as f:
    training_text = f.read()

# Split into samples
samples = training_text.split('=' * 80)
training_samples = [s.strip() for s in samples if s.strip()]
```

## 🧠 LLM Implementation

The `src/` folder contains a complete decoder-only transformer implementation:
- **decoder_only.py**: Main LLM architecture with positional encoding, multi-head attention, and feed-forward networks
- **mha.py**: Multi-head attention mechanism implementation
- **infer_decoder_only.py**: Text generation and inference utilities

## 📊 Data Collection

The `data_gathering/` folder provides automated Wikipedia data collection:
- **3 Dataset Sizes**: Nano (5), Mini (50%), Full (100%)
- **4 Training Formats**: Basic, With Summary, Q&A, Instruction
- **Plain Text Output**: Ready for any LLM tokenizer
- **Daily Automation**: Cron job ready for continuous data collection

### Training Data Formats
- **Basic**: "Animal: Cat\n\n[Article content]"
- **With Summary**: Includes Wikipedia summary
- **Q&A**: "Question: Tell me about Cat.\nAnswer: [Content]"
- **Instruction**: "Write an article about Cat.\n\n[Content]"

## 🔧 Usage Examples

### Run LLM Inference
```bash
cd src
python3 infer_decoder_only.py
```

### Collect More Data
```bash
cd data_gathering
python3 daily_scrape.py --animals 25  # Custom animal count
```

### Process Existing Data
```bash
cd data_gathering
python3 data_processor.py
```

## 📈 Training Data Statistics

Recent collection:
- **16 training files** created
- **3 dataset sizes** (nano/mini/full)
- **4 training formats** per size
- **115K+ characters** per full dataset
- **Plain text format** ready for tokenization

## 🎯 Project Goals

1. **Educational**: Learn LLM architecture and training data collection
2. **Practical**: Build working decoder-only transformer
3. **Scalable**: Automated data pipeline for continuous training
4. **Modular**: Clean separation of concerns (core code vs. utilities)

## 📚 Documentation

- **Data Collection**: `data_gathering/DATA_COLLECTION_README.md`
- **LLM Architecture**: See docstrings in `src/decoder_only.py`
- **Usage Examples**: `data_gathering/training_data_demo.py`

## 🤝 Contributing

1. Core LLM code → `src/`
2. Data collection → `data_gathering/`
3. Training data → `Datasets/`
4. Documentation → Update relevant README files