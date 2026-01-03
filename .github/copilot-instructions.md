# Romanian Sentiment Analysis - ML Project

## Project Context
ML assignment: Binary sentiment classification (positive/negative) on Romanian reviews using RNN/LSTM. Dataset: `ro_sent` (~18k train, ~11k test samples).

## Architecture Overview
```
src/
├── config.py              # Hyperparameters, paths, constants
├── data/
│   ├── dataloader.py      # enhanced for tensor batching
│   ├── augmentations.py   
│   ├── download_and_split.py
│   └── vocab.py           # Vocabulary building, word→index mapping
├── preprocessing/
│   └── text.py            
├── embeddings/            
│   └── fasttext_loader.py # Load pretrained Romanian fastText
├── models/
│   ├── base.py            # Base class with common forward logic
│   ├── rnn.py             # Simple RNN variants
│   └── lstm.py            # LSTM variants (uni/bidirectional)
├── train/
│   ├── trainer.py         # Training loop with metrics tracking
│   └── callbacks.py       # Early stopping, checkpointing (optional)
├── evaluate/
│   ├── metrics.py         # Accuracy, F1, loss computation
│   └── visualize.py       # Confusion matrix, loss curves, comparison plots
└── run_experiment.py      # Main entry point with CLI args
```

## Required Components

### 1. Data Exploration & Visualization
- Class balance analysis with bar/count plots for train and test sets
- Text length distribution (words/characters) per sentiment class
- Most frequent words per class visualization
- Store plots in dedicated directory (e.g., `plots/` or `visualizations/`)

### 2. Text Preprocessing Pipeline
- **Data cleaning**: Remove special characters, normalize text, optionally remove stopwords
- **Tokenization**: Use `spacy` for Romanian tokenization (`ro_core_news_sm` model)
- **Embedding**: Use `fastText` pretrained Romanian embeddings
- **Padding**: Normalize sequences to fixed length(e.g., based on EDA length distribution).
- Handle unknown words with special tokens

### 3. Model Architectures
Implement and experiment with:
- **Simple RNN**: Vary number of layers, hidden state dimensions
- **LSTM**: Test unidirectional vs bidirectional, multiple layers, combined with linear layers
- Consider: Batch normalization, pooling layers, dropout for regularization

### 4. Data Augmentation
Implement techniques like:
- Random Swap/Delete/Insert operations
- Back-translation (Romanian → English → Romanian)
- Contextual word embeddings with Romanian BERT
- Reference: https://neptune.ai/blog/data-augmentation-nlp

### 5. Training & Evaluation
- Track training vs validation loss,accuracy and f1 curves
- Compare models with and without augmentation on same plots
- Generate confusion matrices for best configurations
- Create comparison table: rows = configurations, columns = metrics

## Key Technical Requirements

### Hyperparameters to Experiment With
- Optimizer choice (Adam, SGD, AdamW)
- Learning rate (consider schedulers)
- Batch size
- Number of epochs
- Regularization techniques (dropout, weight decay)

### Documentation Standards
Every architectural or optimization choice MUST be justified in the final report:
- What problem occurred during training
- Why a specific change was made
- Show before/after comparison plots for augmentation impact


## Key Patterns & Conventions

### Data Pipeline
- CSV columns: `text`, `label` (0=negative, 1=positive)
- Data paths: `data/raw/` (original), `data/processed/` (train/val/test splits)
- Use `TextCsvDataset` from `src/data/dataloader.py` with optional `transform` callable
- Preprocessing: `src/preprocessing/text.py::tokenize(text, use_spacy=True)` returns token list

### Text Processing (Romanian-specific)
```python
from src.preprocessing.text import clean_text, tokenize
# clean_text: lowercase, keeps [a-zăâîșț], removes special chars
# tokenize: uses spaCy ro_core_news_sm model (must be installed)
```

### Augmentation Functions
```python
from src.data.augmentations import random_swap, random_delete, random_insert
# All take List[str] tokens and return List[str]
```

### Training Command Pattern
```bash
python src/train.py --model lstm --layers 2 --hidden 128 --lr 0.001 --augment random_swap
```

### Evaluation
Report must include:
1. Architecture description + full hyperparameter configuration
2. Training/validation loss curves (same plot)
3. Training/validation metric curves (separate plot)
4. Performance comparison table
5. Confusion matrix for best model per architecture type
6. Text analysis of results: architecture influence, hyperparameter impact, per-class performance

## Important Notes
- All text must be in Romanian character encoding (UTF-8)
- Results must be reproducible - set random seeds
- Focus on justifying choices, not just reporting numbers
- Compare augmentation impact with side-by-side plots
- Final deliverable: PDF report with all visualizations and analysis
