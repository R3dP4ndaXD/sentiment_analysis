"""
Text augmentation utilities for Romanian sentiment analysis using nlpaug.

Implements multiple augmentation strategies:
1. Random operations: swap, delete, insert, crop
2. Synonym replacement using fastText/WordNet
3. Back-translation (Romanian ↔ English)
4. Contextual word replacement using Romanian BERT

Reference: https://neptune.ai/blog/data-augmentation-nlp

nlpaug Flow types:
- Sequential: Apply all augmenters in order
- Sometimes: Apply each augmenter with probability p
"""
from typing import List, Optional, Union
import random
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.flow as naf

# Download required NLTK resources for nlpaug
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Cache for expensive models
_FASTTEXT_MODEL_PATH = None
_BERT_MODEL_NAME = "dumitrescustefan/bert-base-romanian-cased-v1"

# =============================================================================
# Romanian Stopwords
# =============================================================================

_ROMANIAN_STOPWORDS_CACHED = None


def get_romanian_stopwords() -> set:
    """Get Romanian stopwords from spaCy model.
    
    Returns a comprehensive set of ~494 Romanian stopwords from spaCy's
    ro_core_news_sm model.
    """
    global _ROMANIAN_STOPWORDS_CACHED
    if _ROMANIAN_STOPWORDS_CACHED is None:
        import spacy
        nlp = spacy.load("ro_core_news_sm")
        _ROMANIAN_STOPWORDS_CACHED = nlp.Defaults.stop_words.copy()
    return _ROMANIAN_STOPWORDS_CACHED


# =============================================================================
# Basic Augmentation Functions (standalone, nlpaug-based)
# =============================================================================

def random_swap(tokens: List[str], n_swaps: int = 1) -> List[str]:
    """Randomly swap pairs of words.
    
    Args:
        tokens: List of tokens
        n_swaps: Number of swap operations
    
    Returns:
        Augmented token list
    """
    if len(tokens) < 2:
        return tokens
    text = " ".join(tokens)
    aug = naw.RandomWordAug(
        action="swap",
        aug_p=min(1.0, n_swaps / max(len(tokens), 1)),
        aug_min=1,
        aug_max=n_swaps,
    )
    result = aug.augment(text)
    return result[0].split() if isinstance(result, list) else result.split()


def random_delete(tokens: List[str], p: float = 0.1) -> List[str]:
    """Randomly delete words with probability p.
    
    Args:
        tokens: List of tokens
        p: Probability of deleting each token
    
    Returns:
        Augmented token list (at least one token preserved)
    """
    if len(tokens) <= 1:
        return tokens
    text = " ".join(tokens)
    aug = naw.RandomWordAug(
        action="delete",
        aug_p=p,
        aug_min=0,
        aug_max=max(1, int(len(tokens) * p * 2)),
    )
    result = aug.augment(text)
    result_tokens = result[0].split() if isinstance(result, list) else result.split()
    
    # Ensure at least one token
    return result_tokens if result_tokens else [random.choice(tokens)]

def random_crop(tokens: List[str], min_ratio: float = 0.7, max_ratio: float = 0.9) -> List[str]:
    """Randomly crop a contiguous portion of the text.
    
    Args:
        tokens: List of tokens
        min_ratio: Minimum ratio of tokens to keep
        max_ratio: Maximum ratio of tokens to keep
    
    Returns:
        Cropped token list
    """
    if len(tokens) <= 2:
        return tokens
    
    text = " ".join(tokens)
    aug = naw.RandomWordAug(
        action="crop",
        aug_p=1 - random.uniform(min_ratio, max_ratio),
        aug_min=1,
        aug_max=max(1, int(len(tokens) * (1 - min_ratio))),
    )
    result = aug.augment(text)
    result_tokens = result[0].split() if isinstance(result, list) else result.split()
    return result_tokens if result_tokens else tokens


# =============================================================================
# Synonym Replacement
# =============================================================================

def synonym_replacement_fasttext(
    tokens: List[str],
    n_replacements: int = 1,
    model_path: Optional[str] = None,
    stopwords: Optional[set] = None,
) -> List[str]:
    """Replace random words with fastText nearest neighbors (synonyms).
    
    Args:
        tokens: List of tokens
        n_replacements: Number of words to replace
        model_path: Path to fastText model
        stopwords: Set of words to skip
    
    Returns:
        Augmented token list
    """
    if not tokens:
        return tokens
    
    text = " ".join(tokens)
    stopwords_list = list(stopwords) if stopwords else get_romanian_stopwords()
    
    global _FASTTEXT_MODEL_PATH
    if model_path:
        _FASTTEXT_MODEL_PATH = model_path
    
    # Try to find fastText model
    if _FASTTEXT_MODEL_PATH is None:
        import os
        default_paths = [
            "cc.ro.300.bin",
            "/content/drive/MyDrive/ML_Sentiment_Analysis/cc.ro.300.bin",
            os.path.expanduser("~/cc.ro.300.bin"),
        ]
        for p in default_paths:
            if os.path.exists(p):
                _FASTTEXT_MODEL_PATH = p
                break
    
    if not _FASTTEXT_MODEL_PATH:
        raise ValueError(
            "FastText model path not set. Please provide a valid model path "
            "via the 'model_path' parameter or set the global _FASTTEXT_MODEL_PATH."
        )
    
    try:
        aug = naw.WordEmbsAug(
            model_type="fasttext",
            model_path=_FASTTEXT_MODEL_PATH,
            action="substitute",
            aug_p=min(1.0, n_replacements / max(len(tokens), 1)),
            aug_min=1,
            aug_max=n_replacements,
            stopwords=stopwords_list,
        )
        
        result = aug.augment(text)
        return result[0].split() if isinstance(result, list) else result.split()
    
    except Exception as e:
        print(f"Warning: FastText synonym augmentation failed: {e}")
        raise


def synonym_replacement_wordnet(
    tokens: List[str],
    n_replacements: int = 1,
    stopwords: Optional[set] = None,
) -> List[str]:
    """Replace random words with WordNet synonyms (Romanian via OMW).
    
    Args:
        tokens: List of tokens
        n_replacements: Number of words to replace
        stopwords: Set of words to skip
    
    Returns:
        Augmented token list
    """
    if not tokens:
        return tokens
    
    text = " ".join(tokens)
    stopwords_list = list(stopwords) if stopwords else list(get_romanian_stopwords())
    
    try:
        aug = naw.SynonymAug(
            aug_src="wordnet",
            lang="ron",  # Romanian
            aug_p=min(1.0, n_replacements / max(len(tokens), 1)),
            aug_min=1,
            aug_max=n_replacements,
            stopwords=stopwords_list,
        )
        result = aug.augment(text)
        return result[0].split() if isinstance(result, list) else result.split()
    
    except Exception as e:
        print(f"Warning: WordNet augmentation failed: {e}")
        return tokens


# =============================================================================
# Contextual Word Replacement using BERT
# =============================================================================

def contextual_word_replacement(
    tokens: List[str],
    n_replacements: int = 1,
    model_name: str = None,
    stopwords: Optional[set] = None,
    device: str = "cpu",
) -> List[str]:
    """Replace words using BERT's masked language model predictions.
    
    Args:
        tokens: List of tokens
        n_replacements: Number of words to replace
        model_name: HuggingFace model name
        stopwords: Words to skip
        device: Device to run on ('cpu' or 'cuda')
    
    Returns:
        Augmented token list
    """
    if not tokens or len(tokens) < 3:
        return tokens
    
    text = " ".join(tokens)
    model_name = model_name or _BERT_MODEL_NAME
    stopwords_list = list(stopwords) if stopwords else get_romanian_stopwords()
    
    try:
        aug = naw.ContextualWordEmbsAug(
            model_path=model_name,
            action="substitute",
            aug_p=min(1.0, n_replacements / max(len(tokens), 1)),
            aug_min=1,
            aug_max=n_replacements,
            stopwords=stopwords_list,
            device=device,
        )
        result = aug.augment(text)
        return result[0].split() if isinstance(result, list) else result.split()
    
    except Exception as e:
        print(f"Warning: Contextual augmentation failed: {e}")
        return tokens


def contextual_insert(
    tokens: List[str],
    n_inserts: int = 1,
    model_name: str = None,
    device: str = "cpu",
) -> List[str]:
    """Insert contextually appropriate words using BERT.
    
    Args:
        tokens: List of tokens
        n_inserts: Number of words to insert
        model_name: HuggingFace model name
        device: Device to run on
    
    Returns:
        Augmented token list
    """
    if not tokens:
        return tokens
    
    text = " ".join(tokens)
    model_name = model_name or _BERT_MODEL_NAME
    
    try:
        aug = naw.ContextualWordEmbsAug(
            model_path=model_name,
            action="insert",
            aug_p=min(1.0, n_inserts / max(len(tokens), 1)),
            aug_min=1,
            aug_max=n_inserts,
            device=device,
        )
        result = aug.augment(text)
        return result[0].split() if isinstance(result, list) else result.split()
    
    except Exception as e:
        print(f"Warning: Contextual insert failed: {e}")
        return tokens


# =============================================================================
# Back-Translation
# =============================================================================

def back_translate(
    tokens: List[str],
    from_model: str = "Helsinki-NLP/opus-mt-ro-en",
    to_model: str = "Helsinki-NLP/opus-mt-en-ro",
    device: str = "cpu",
) -> List[str]:
    """Augment via back-translation: Romanian → English → Romanian.
    
    Args:
        tokens: List of tokens
        from_model: Model for source→intermediate translation
        to_model: Model for intermediate→source translation
        device: Device to run on
    
    Returns:
        Augmented token list
    """
    if not tokens:
        return tokens
    
    text = " ".join(tokens)
    
    try:
        aug = naw.BackTranslationAug(
            from_model_name=from_model,
            to_model_name=to_model,
            device=device,
        )
        result = aug.augment(text)
        return result[0].split() if isinstance(result, list) else result.split()
    
    except Exception as e:
        print(f"Warning: Back-translation failed: {e}")
        return tokens

# =============================================================================
# Character-level Augmentation
# =============================================================================

def keyboard_typo(tokens: List[str], aug_p: float = 0.1) -> List[str]:
    """Simulate keyboard typos using nlpaug.
    
    Args:
        tokens: List of tokens
        aug_p: Probability of augmenting each character
    
    Returns:
        Augmented token list
    """
    if not tokens:
        return tokens
    
    text = " ".join(tokens)
    
    try:
        aug = nac.KeyboardAug(
            aug_char_p=aug_p,
            aug_word_p=0.3,
        )
        result = aug.augment(text)
        return result[0].split() if isinstance(result, list) else result.split()
    
    except Exception as e:
        print(f"Warning: Keyboard augmentation failed: {e}")
        return tokens


def ocr_error(tokens: List[str], aug_p: float = 0.1) -> List[str]:
    """Simulate OCR errors using nlpaug.
    
    Args:
        tokens: List of tokens
        aug_p: Probability of augmenting each character
    
    Returns:
        Augmented token list
    """
    if not tokens:
        return tokens
    
    text = " ".join(tokens)
    
    try:
        aug = nac.OcrAug(
            aug_char_p=aug_p,
            aug_word_p=0.3,
        )
        result = aug.augment(text)
        return result[0].split() if isinstance(result, list) else result.split()
    
    except Exception as e:
        print(f"Warning: OCR augmentation failed: {e}")
        return tokens


# =============================================================================
# Composite Augmentation with nlpaug Flows
# =============================================================================

class TextAugmenter:
    """Composite augmenter using nlpaug flows for smart chaining.
    
    Uses nlpaug's Sequential and Sometimes flows for controlled augmentation.
    
    Modes:
        - "sometimes": Apply each augmenter with probability p (default)
        - "sequential": Apply all augmenters in order
        - "one_of": Apply exactly one augmenter randomly
    
    Example:
        augmenter = TextAugmenter(
            strategies=["random_swap", "random_delete", "synonym_wordnet"],
            p=0.3,  # Probability for Sometimes flow
            mode="sometimes",
        )
        augmented = augmenter(tokens)
    """
    
    AVAILABLE_STRATEGIES = [
        "random_swap",      # Swap word pairs
        "random_delete",    # Delete words randomly
        "random_crop",      # Crop portion of text
        "random_split",     # Split words
        "synonym_wordnet",  # WordNet synonyms
        "synonym_fasttext", # FastText synonyms
        "contextual_substitute",  # BERT word replacement
        "contextual_insert",      # BERT word insertion
        "back_translation", # Back-translate via English
        "keyboard",         # Keyboard typos
        "ocr",              # OCR-like errors
    ]
    
    def __init__(
        self,
        strategies: List[str] = None,
        p: float = 0.3,
        mode: str = "sometimes",
        fasttext_path: Optional[str] = None,
        bert_model: Optional[str] = None,
        device: str = "cpu",
        aug_p: float = 0.1,
        n_ops: int = 2,
        stopwords: set = None,
    ):
        """
        Args:
            strategies: List of strategy names (see AVAILABLE_STRATEGIES)
            p: Probability of applying each augmentation (for 'sometimes' mode)
            mode: "sometimes", "sequential", or "one_of"
            fasttext_path: Path to fastText model
            bert_model: HuggingFace model name for BERT
            device: Device for BERT ('cpu' or 'cuda')
            aug_p: Probability parameter for individual augmenters
            n_ops: Max number of operations per augmenter
            stopwords: Set of stopwords to skip
        """
        self.strategies = strategies or ["random_swap", "random_delete"]
        self.p = p
        self.mode = mode
        self.fasttext_path = fasttext_path
        self.bert_model = bert_model or _BERT_MODEL_NAME
        self.device = device
        self.aug_p = aug_p
        self.n_ops = n_ops
        self.stopwords = list(stopwords) if stopwords else list(get_romanian_stopwords())
        
        # Set global fasttext path
        global _FASTTEXT_MODEL_PATH
        if fasttext_path:
            _FASTTEXT_MODEL_PATH = fasttext_path
        
        # Build augmenters
        self.augmenters = self._build_augmenters()
        self.flow = self._build_flow()
    
    def _build_augmenters(self) -> List:
        """Build list of nlpaug augmenters."""
        augmenters = []
        
        for strategy in self.strategies:
            try:
                aug = self._create_augmenter(strategy)
                if aug is not None:
                    augmenters.append(aug)
            except Exception as e:
                print(f"Warning: Failed to create augmenter for '{strategy}': {e}")
        
        return augmenters
    
    def _create_augmenter(self, strategy: str):
        """Create a single augmenter for given strategy."""
        if strategy == "random_swap":
            return naw.RandomWordAug(
                action="swap",
                aug_p=self.aug_p,
                aug_min=1,
                aug_max=self.n_ops,
            )
        
        elif strategy == "random_delete":
            return naw.RandomWordAug(
                action="delete",
                aug_p=self.aug_p,
                aug_min=0,
                aug_max=self.n_ops,
            )
        
        elif strategy == "random_crop":
            return naw.RandomWordAug(
                action="crop",
                aug_p=0.2,
                aug_min=1,
                aug_max=5,
            )
        
        elif strategy == "random_split":
            return naw.SplitAug(
                aug_p=self.aug_p,
                min_char=3,
            )
        
        elif strategy == "synonym_wordnet":
            return naw.SynonymAug(
                aug_src="wordnet",
                lang="ron",
                aug_p=self.aug_p,
                aug_min=1,
                aug_max=self.n_ops,
                stopwords=self.stopwords,
            )
        
        elif strategy == "synonym_fasttext":
            if not _FASTTEXT_MODEL_PATH:
                raise ValueError(
                    "FastText model path not set."
                )
            return naw.WordEmbsAug(
                model_type="fasttext",
                model_path=_FASTTEXT_MODEL_PATH,
                action="substitute",
                aug_p=self.aug_p,
                aug_min=1,
                aug_max=self.n_ops,
                stopwords=self.stopwords,
            )

        elif strategy == "contextual_substitute":
            return naw.ContextualWordEmbsAug(
                model_path=self.bert_model,
                action="substitute",
                aug_p=self.aug_p,
                aug_min=1,
                aug_max=self.n_ops,
                stopwords=self.stopwords,
                device=self.device,
            )
        
        elif strategy == "contextual_insert":
            return naw.ContextualWordEmbsAug(
                model_path=self.bert_model,
                action="insert",
                aug_p=self.aug_p,
                aug_min=1,
                aug_max=self.n_ops,
                device=self.device,
            )
        
        elif strategy == "back_translation":
            return naw.BackTranslationAug(
                from_model_name="Helsinki-NLP/opus-mt-ro-en",
                to_model_name="Helsinki-NLP/opus-mt-en-ro",
                device=self.device,
            )
        
        elif strategy == "keyboard":
            return nac.KeyboardAug(
                aug_char_p=self.aug_p,
                aug_word_p=0.3,
            )
        
        elif strategy == "ocr":
            return nac.OcrAug(
                aug_char_p=self.aug_p,
                aug_word_p=0.3,
            )
        
        else:
            print(f"Warning: Unknown strategy '{strategy}'. Available: {self.AVAILABLE_STRATEGIES}")
            return None
    
    def _build_flow(self):
        """Build nlpaug flow based on mode."""
        if not self.augmenters:
            return None
        
        if self.mode == "sequential":
            # Apply all augmenters in sequence
            return naf.Sequential(self.augmenters)
        
        elif self.mode == "sometimes":
            # Apply each augmenter with probability p
            return naf.Sometimes(self.augmenters, aug_p=self.p)
        
        elif self.mode == "one_of":
            # For one_of, we return None and handle it in __call__
            return None
        
        else:
            # Default to sometimes
            return naf.Sometimes(self.augmenters, aug_p=self.p)
    
    def __call__(self, tokens: List[str]) -> List[str]:
        """Apply augmentation flow.
        
        Args:
            tokens: Input token list
        
        Returns:
            Augmented token list
        """
        if not tokens:
            return tokens
        
        if not self.augmenters:
            return tokens
        
        text = " ".join(tokens)
        
        try:
            # Handle one_of mode: randomly select one augmenter
            if self.mode == "one_of":
                aug = random.choice(self.augmenters)
                result = aug.augment(text)
            elif self.flow is not None:
                result = self.flow.augment(text)
            else:
                return tokens
            
            result_tokens = result[0].split() if isinstance(result, list) else result.split()
            return result_tokens if result_tokens else tokens
        except Exception as e:
            print(f"Warning: Augmentation failed: {e}")
            return tokens
    
    def augment(self, tokens: List[str]) -> List[str]:
        """Alias for __call__ (compatibility)."""
        return self.__call__(tokens)
    
    def augment_batch(self, texts: List[str]) -> List[str]:
        """Augment a batch of texts efficiently.
        
        Args:
            texts: List of text strings
        
        Returns:
            List of augmented text strings
        """
        if self.flow is None:
            return texts
        
        try:
            return self.flow.augment(texts)
        except Exception as e:
            print(f"Warning: Batch augmentation failed: {e}")
            return texts


# =============================================================================
# Preset Augmentation Configurations
# =============================================================================

def get_eda_augmenter(aug_p: float = 0.1, fasttext_path: str = None) -> TextAugmenter:
    """Get standard EDA (Easy Data Augmentation) configuration.
    
    EDA includes: swap, delete, contextual_insert, synonym replacement
    Applied probabilistically (sometimes mode).
    """
    return TextAugmenter(
        strategies=["random_swap", "random_delete", "contextual_insert", "synonym_wordnet"],
        p=0.3,
        mode="sometimes",
        fasttext_path=fasttext_path,
        aug_p=aug_p,
        n_ops=2,
    )


def get_light_augmenter(aug_p: float = 0.1) -> TextAugmenter:
    """Get light augmentation (structural only, no semantic changes).
    
    Best for: experiments where you want minimal text changes.
    """
    return TextAugmenter(
        strategies=["random_swap", "random_delete"],
        p=0.3,
        mode="sometimes",
        aug_p=aug_p,
        n_ops=1,
    )


def get_semantic_augmenter(
    fasttext_path: str = None,
    bert_model: str = None,
    device: str = "cpu",
) -> TextAugmenter:
    """Get semantic augmentation (synonym + contextual).
    
    Uses 'one_of' mode to apply exactly ONE semantic change per sample.
    Best for: semantic diversity without compounding changes.
    """
    strategies = ["synonym_wordnet"]
    if fasttext_path:
        strategies.append("synonym_fasttext")
    if bert_model:
        strategies.append("contextual_substitute")
    
    return TextAugmenter(
        strategies=strategies,
        p=0.5,
        mode="one_of",  # Apply ONE semantic change
        fasttext_path=fasttext_path,
        bert_model=bert_model,
        device=device,
        aug_p=0.15,
        n_ops=2,
    )


def get_heavy_augmenter(
    fasttext_path: str = None,
    bert_model: str = None,
    device: str = "cpu",
) -> TextAugmenter:
    """Get heavy augmentation (multiple techniques).
    
    Warning: May cause semantic drift. Use with low p.
    Best for: generating diverse training examples.
    """
    return TextAugmenter(
        strategies=[
            "random_swap", "random_delete", "random_crop",
            "synonym_wordnet", "contextual_substitute",
        ],
        p=0.2,
        mode="sometimes",
        fasttext_path=fasttext_path,
        bert_model=bert_model,
        device=device,
        aug_p=0.1,
        n_ops=2,
    )


def get_noise_augmenter(aug_p: float = 0.05) -> TextAugmenter:
    """Get noise augmentation (typos, OCR errors).
    
    Best for: robustness to input noise.
    """
    return TextAugmenter(
        strategies=["keyboard", "ocr"],
        p=0.3,
        mode="one_of",
        aug_p=aug_p,
    )

