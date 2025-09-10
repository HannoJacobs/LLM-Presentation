# LLM-Presentation

A comprehensive project for LLM development featuring decoder-only architecture implementation and automated data collection for training.

## 🚀 Quickstart - Run Inference with Pre-trained Models

Get started immediately with our trained models! All models are available in the `models/` directory.

### Prerequisites
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Base Model Inference (General Text Generation)
```bash
source .venv/bin/activate && python3 src/infer_decoder_only.py
```
This loads the base model and generates text completions for animal-related prompts.

### 2. Fine-tuned Model Inference (Q&A)
```bash
# Edit infer_decoder_only.py to switch to fine-tuned model:
# Change BASE_MODEL = False or True based on if you want to use the trained base model or the fine tuned one
# Set MODEL_PATH = "models/finetune_nano_10_512_4_4_512_latest.pth"
source .venv/bin/activate && python3 src/infer_decoder_only.py
```
This loads the fine-tuned model optimized for answering animal-related questions.

### 3. Try Different Temperature Settings
```bash
# In infer_decoder_only.py, set DEMO_TEMP = True
source .venv/bin/activate && python3 src/infer_decoder_only.py
```
This demonstrates how temperature affects creativity:
- ❄️ Temperature 0.0: Deterministic responses
- 🌤️ Temperature 0.5: Balanced creativity
- 🌡️ Temperature 1.0: Moderate creativity
- 🔥 Temperature 2.0: High creativity

### Available Models
- **Base Model**: `models/base_nano_10_512_4_4_512.pth` - General text generation
- **Fine-tuned Model**: `models/finetune_nano_10_512_4_4_512_latest.pth` - Q&A optimized
- **Demo Model**: `models/finetune_nano_10_512_4_4_512_temp_demo.pth` - For temperature demos

## 📁 Project Structure

```
LLM-Presentation/
├── src/                          # Core application code
│   ├── decoder_only.py          # Main decoder-only LLM implementation with training/inference
│   ├── mha.py                   # Multi-head attention mechanism implementation
│   ├── infer_decoder_only.py    # Text generation and inference utilities for trained models
│   ├── finetune_decoder.py      # Fine-tuning script for Q&A optimization
│   ├── bertviz_tutorial.py      # BERT attention visualization tutorial using bertviz
│   ├── bertviz_tutorial.ipynb   # Jupyter notebook version of BERT visualization
│   ├── tokenization.ipynb       # Character vs word-level tokenization examples
│   ├── ollama_run.py            # Ollama API integration for external model testing
│   └── __pycache__/             # Python bytecode cache
├── data_gathering/              # Data collection and processing
│   ├── data_pipeline.py        # 🔧 ALL-IN-ONE consolidated script
│   └── README.md               # Data gathering documentation
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
python3 data_pipeline.py
```

The script automatically creates all 3 dataset sizes (nano, mini, full) with no arguments needed.

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

### Core Architecture (`decoder_only.py`)
The main decoder-only transformer implementation featuring:
- **Complete Transformer Architecture**: Multi-head attention, feed-forward networks, positional encoding
- **Training Pipeline**: Data loading, optimization, and checkpoint saving
- **Inference Engine**: Text generation with temperature sampling and top-k filtering
- **Custom Tokenization**: Character-level tokenizer with special tokens
- **Training Utilities**: Loss computation, validation, and progress tracking

### Multi-Head Attention (`mha.py`)
Standalone implementation of the attention mechanism:
- **Self-Attention**: Query-Key-Value attention computation
- **Multi-Head Processing**: Parallel attention heads with different projections
- **Scaled Dot-Product**: Efficient attention calculation with masking
- **Visualization Ready**: Compatible with attention visualization tools

### Inference & Generation (`infer_decoder_only.py`)
Text generation utilities for trained models:
- **Model Loading**: Load pre-trained checkpoints with vocabularies
- **Text Completion**: Generate continuations for given prompts
- **Temperature Control**: Adjustable creativity vs determinism
- **Top-K Sampling**: Quality control for generated text
- **Batch Processing**: Multiple prompt generation in one run

### Fine-tuning (`finetune_decoder.py`)
Specialized training for question-answering:
- **Q&A Dataset**: Custom dataset class for question-answer pairs
- **Masked Loss**: Ignore prompt tokens during training
- **Resume Training**: Load base models and fine-tune on specific tasks
- **Progress Tracking**: Training/validation metrics and checkpointing

### Educational Tutorials
- **`bertviz_tutorial.py`**: Interactive BERT attention visualization using bertviz library
- **`tokenization.ipynb`**: Jupyter notebook demonstrating character vs word-level tokenization
- **`ollama_run.py`**: Example integration with Ollama API for external model comparison

## 📊 Data Collection

The `data_gathering/data_pipeline.py` provides an efficient automated pipeline:
- **Single Script**: All-in-one scraping, processing, and validation
- **Smart Scraping**: Scrapes all 30 animals once, creates subsets (Wikipedia-friendly!)
- **3 Dataset Sizes**: Nano (5), Mini (10), Full (30 animals)
- **3 Training Formats**: Basic, With Summary, Q&A
- **Plain Text Output**: Ready for any LLM tokenizer
- **Built-in Validation**: Automatic format verification
- **Curated Animals**: Handpicked selection of 30 diverse animals

### Training Data Formats
- **Basic**: `Animal: Cat\n\n[Article content]`
- **With Summary**: Includes Wikipedia summary
- **Q&A**: `Question: Tell me about Cat.\nAnswer: [Content]`

## 🔧 Usage Examples

### Run LLM Inference
```bash
cd src
python3 infer_decoder_only.py
```

### Collect More Data
```bash
cd data_gathering
python3 data_pipeline.py  # Creates all 3 datasets automatically
```

## 📈 Training Data Statistics

Recent collection:
- **9 training files** created
- **3 dataset sizes** (nano: 5, mini: 10, full: 30 animals)
- **3 training formats** per size
- **~100K+ characters** per full dataset
- **Plain text format** ready for tokenization
- **30 curated animals** (handpicked selection)

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