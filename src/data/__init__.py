"""Data loading and processing utilities."""
from .dataloader import (
    Sample,
    EncodedSample,
    TextCsvDataset,
    collate_encoded,
    collate_raw,
    create_dataloaders,
)
from .vocab import Vocabulary, build_vocab_from_csv, build_vocab_from_texts
from .augmentations import (
    # Basic EDA operations
    random_swap,
    random_delete,
    random_crop,
    # FastText synonym replacement
    synonym_replacement_fasttext,
    # Back-translation
    back_translate,
    # BERT contextual replacement
    contextual_word_replacement,
    # Composite augmenter
    TextAugmenter,
    # Utilities
    get_romanian_stopwords,
)

__all__ = [
    # DataLoader
    "Sample",
    "EncodedSample",
    "TextCsvDataset",
    "collate_encoded",
    "collate_raw",
    "create_dataloaders",
    # Vocabulary
    "Vocabulary",
    "build_vocab_from_csv",
    "build_vocab_from_texts",
    # Augmentations - EDA
    "random_swap",
    "random_delete",
    "random_crop",
    # Augmentations - Advanced
    "synonym_replacement_fasttext",
    "get_fasttext_synonyms",
    "back_translate",
    "back_translate_tokens",
    "contextual_word_replacement",
    "TextAugmenter",
    # Utilities
    "get_romanian_stopwords",
]