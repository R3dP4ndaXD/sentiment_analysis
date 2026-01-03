"""
Text augmentation utilities for Romanian sentiment analysis.

Implements multiple augmentation strategies:
1. Random operations: swap, delete, insert, shuffle
2. Synonym replacement using fastText nearest neighbors
3. Back-translation (Romanian ↔ English)
4. Contextual word replacement using Romanian BERT

Reference: https://neptune.ai/blog/data-augmentation-nlp
"""
from typing import List, Optional, Callable, Union
import random
import re

# Lazy imports for optional dependencies
_FASTTEXT_MODEL = None
_TRANSLATOR_RO_EN = None
_TRANSLATOR_EN_RO = None
_BERT_FILL_MASK = None


# =============================================================================
# Basic Random Operations (EDA - Easy Data Augmentation)
# =============================================================================

def random_swap(tokens: List[str], n_swaps: int = 1) -> List[str]:
    """Randomly swap pairs of words in the token list.
    
    Args:
        tokens: List of tokens
        n_swaps: Number of swap operations to perform
    
    Returns:
        Augmented token list
    """
    if len(tokens) < 2:
        return tokens
    tokens = tokens[:]
    for _ in range(n_swaps):
        i, j = random.sample(range(len(tokens)), 2)
        tokens[i], tokens[j] = tokens[j], tokens[i]
    return tokens


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
    result = [t for t in tokens if random.random() > p]
    # Ensure at least one token remains
    return result if result else [random.choice(tokens)]


def random_insert(
    tokens: List[str], 
    n_inserts: int = 1, 
    word_pool: Optional[List[str]] = None,
) -> List[str]:
    """Randomly insert words from the same sentence (or a word pool).
    
    Args:
        tokens: List of tokens
        n_inserts: Number of insertions
        word_pool: Optional list of words to insert from. If None, uses tokens from input.
    
    Returns:
        Augmented token list
    """
    if not tokens:
        return tokens
    tokens = tokens[:]
    pool = word_pool if word_pool else tokens
    for _ in range(n_inserts):
        word = random.choice(pool)
        pos = random.randint(0, len(tokens))
        tokens.insert(pos, word)
    return tokens


def random_shuffle_within_window(tokens: List[str], window_size: int = 3) -> List[str]:
    """Shuffle tokens within sliding windows to maintain local coherence.
    
    Args:
        tokens: List of tokens
        window_size: Size of window for local shuffling
    
    Returns:
        Augmented token list
    """
    if len(tokens) <= window_size:
        result = tokens[:]
        random.shuffle(result)
        return result
    
    result = []
    for i in range(0, len(tokens), window_size):
        window = tokens[i:i + window_size]
        random.shuffle(window)
        result.extend(window)
    return result


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
    
    keep_ratio = random.uniform(min_ratio, max_ratio)
    keep_len = max(1, int(len(tokens) * keep_ratio))
    
    max_start = len(tokens) - keep_len
    start = random.randint(0, max_start)
    
    return tokens[start:start + keep_len]


# =============================================================================
# Synonym Replacement using FastText
# =============================================================================

def _get_fasttext_model():
    """Lazy load fastText model."""
    global _FASTTEXT_MODEL
    if _FASTTEXT_MODEL is None:
        try:
            from ..embeddings.fasttext_loader import load_fasttext_model
            _FASTTEXT_MODEL = load_fasttext_model()
        except Exception as e:
            raise RuntimeError(f"Failed to load fastText for synonym augmentation: {e}")
    return _FASTTEXT_MODEL


def get_fasttext_synonyms(
    word: str, 
    top_k: int = 10,
    model=None,
) -> List[str]:
    """Get similar words using fastText nearest neighbors.
    
    Args:
        word: Word to find synonyms for
        top_k: Number of similar words to retrieve
        model: FastText model (uses cached if None)
    
    Returns:
        List of similar words
    """
    if model is None:
        model = _get_fasttext_model()
    
    neighbors = model.get_nearest_neighbors(word, k=top_k)
    # neighbors is list of (similarity, word) tuples
    return [w for _, w in neighbors]


def synonym_replacement_fasttext(
    tokens: List[str],
    n_replacements: int = 1,
    top_k: int = 5,
    model=None,
    stopwords: Optional[set] = None,
) -> List[str]:
    """Replace random words with fastText nearest neighbors (synonyms).
    
    Args:
        tokens: List of tokens
        n_replacements: Number of words to replace
        top_k: Number of candidate synonyms to sample from
        model: FastText model
        stopwords: Set of words to skip (don't replace)
    
    Returns:
        Augmented token list
    """
    if not tokens:
        return tokens
    
    if model is None:
        model = _get_fasttext_model()
    
    tokens = tokens[:]
    stopwords = stopwords or set()
    
    # Get indices of replaceable words (not stopwords, not too short)
    replaceable = [
        i for i, t in enumerate(tokens) 
        if t.lower() not in stopwords and len(t) > 2
    ]
    
    if not replaceable:
        return tokens
    
    # Sample positions to replace
    n_replace = min(n_replacements, len(replaceable))
    positions = random.sample(replaceable, n_replace)
    
    for pos in positions:
        word = tokens[pos]
        synonyms = get_fasttext_synonyms(word, top_k=top_k, model=model)
        if synonyms:
            # Filter out the original word and pick randomly
            synonyms = [s for s in synonyms if s.lower() != word.lower()]
            if synonyms:
                tokens[pos] = random.choice(synonyms)
    
    return tokens


# =============================================================================
# Back-Translation (Romanian ↔ English)
# =============================================================================

def _get_translators():
    """Lazy load translation models using transformers."""
    global _TRANSLATOR_RO_EN, _TRANSLATOR_EN_RO
    
    if _TRANSLATOR_RO_EN is None or _TRANSLATOR_EN_RO is None:
        try:
            from transformers import pipeline
            print("Loading translation models for back-translation...")
            print("  Loading Romanian → English...")
            _TRANSLATOR_RO_EN = pipeline(
                "translation", 
                model="Helsinki-NLP/opus-mt-ro-en",
                device=-1,  # -1 for CPU; 0 for GPU
            )
            print("  Loading English → Romanian...")
            _TRANSLATOR_EN_RO = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-en-ro",
                device=-1,
            )
            print("  Translation models loaded.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load translation models. Install transformers: pip install transformers\n"
                f"Error: {e}"
            )
    
    return _TRANSLATOR_RO_EN, _TRANSLATOR_EN_RO


def back_translate(
    text: str,
    ro_en_translator=None,
    en_ro_translator=None,
) -> str:
    """Augment text via back-translation: Romanian → English → Romanian.
    
    Args:
        text: Romanian text string
        ro_en_translator: Romanian to English translation pipeline
        en_ro_translator: English to Romanian translation pipeline
    
    Returns:
        Back-translated Romanian text
    """
    if ro_en_translator is None or en_ro_translator is None:
        ro_en_translator, en_ro_translator = _get_translators()
    
    # Romanian → English
    en_result = ro_en_translator(text, max_length=512)
    en_text = en_result[0]["translation_text"]
    
    # English → Romanian
    ro_result = en_ro_translator(en_text, max_length=512)
    ro_text = ro_result[0]["translation_text"]
    
    return ro_text


def back_translate_tokens(
    tokens: List[str],
    ro_en_translator=None,
    en_ro_translator=None,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> List[str]:
    """Back-translate a token list.
    
    Args:
        tokens: List of tokens
        ro_en_translator: Romanian → English translator
        en_ro_translator: English → Romanian translator  
        tokenizer: Function to re-tokenize the result
    
    Returns:
        Augmented token list
    """
    text = " ".join(tokens)
    translated = back_translate(text, ro_en_translator, en_ro_translator)
    
    if tokenizer:
        return tokenizer(translated)
    return translated.split()


# =============================================================================
# Contextual Word Replacement using Romanian BERT
# =============================================================================

def _get_bert_fill_mask():
    """Lazy load Romanian BERT for masked language modeling."""
    global _BERT_FILL_MASK
    
    if _BERT_FILL_MASK is None:
        try:
            from transformers import pipeline
            print("Loading Romanian BERT for contextual augmentation...")
            _BERT_FILL_MASK = pipeline(
                "fill-mask",
                model="dumitrescustefan/bert-base-romanian-cased-v1",
                device=-1,  # -1 for CPU; 0 for GPU
            )
            print("  Romanian BERT loaded.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Romanian BERT. Install transformers: pip install transformers\n"
                f"Error: {e}"
            )
    
    return _BERT_FILL_MASK


def contextual_word_replacement(
    tokens: List[str],
    n_replacements: int = 1,
    top_k: int = 5,
    fill_mask_pipeline=None,
    stopwords: Optional[set] = None,
) -> List[str]:
    """Replace words using BERT's masked language model predictions.
    
    This provides contextually appropriate replacements based on surrounding words.
    
    Args:
        tokens: List of tokens
        n_replacements: Number of words to replace
        top_k: Number of BERT predictions to sample from
        fill_mask_pipeline: Hugging Face fill-mask pipeline
        stopwords: Words to skip
    
    Returns:
        Augmented token list
    """
    if not tokens or len(tokens) < 3:
        return tokens
    
    if fill_mask_pipeline is None:
        fill_mask_pipeline = _get_bert_fill_mask()
    
    tokens = tokens[:]
    stopwords = stopwords or set()
    
    # Get replaceable positions
    replaceable = [
        i for i, t in enumerate(tokens)
        if t.lower() not in stopwords and len(t) > 2
    ]
    
    if not replaceable:
        return tokens
    
    n_replace = min(n_replacements, len(replaceable))
    positions = random.sample(replaceable, n_replace)
    
    for pos in positions:
        # Create masked sentence
        masked_tokens = tokens[:]
        masked_tokens[pos] = fill_mask_pipeline.tokenizer.mask_token
        masked_text = " ".join(masked_tokens)
        
        try:
            predictions = fill_mask_pipeline(masked_text, top_k=top_k)
            # predictions is list of dicts with 'token_str' and 'score'
            candidates = [p["token_str"].strip() for p in predictions]
            # Filter out original and empty
            candidates = [c for c in candidates if c and c.lower() != tokens[pos].lower()]
            
            if candidates:
                tokens[pos] = random.choice(candidates)
        except Exception:
            # Skip if BERT fails (e.g., sequence too long)
            pass
    
    return tokens


# =============================================================================
# Composite Augmentation
# =============================================================================

class TextAugmenter:
    """Composite augmenter that applies multiple augmentation strategies.
    
    Example:
        augmenter = TextAugmenter(
            strategies=["random_swap", "random_delete", "synonym"],
            p=0.5,
        )
        augmented = augmenter(tokens)
    """
    
    STRATEGY_MAP = {
        "random_swap": lambda tokens, **kw: random_swap(tokens, n_swaps=kw.get("n_swaps", 1)),
        "random_delete": lambda tokens, **kw: random_delete(tokens, p=kw.get("delete_p", 0.1)),
        "random_insert": lambda tokens, **kw: random_insert(tokens, n_inserts=kw.get("n_inserts", 1)),
        "random_shuffle": lambda tokens, **kw: random_shuffle_within_window(tokens, window_size=kw.get("window_size", 3)),
        "random_crop": lambda tokens, **kw: random_crop(tokens, min_ratio=kw.get("min_ratio", 0.7)),
        "synonym": lambda tokens, **kw: synonym_replacement_fasttext(
            tokens, 
            n_replacements=kw.get("n_replacements", 1),
            top_k=kw.get("top_k", 5),
            stopwords=kw.get("stopwords"),
        ),
        "contextual": lambda tokens, **kw: contextual_word_replacement(
            tokens,
            n_replacements=kw.get("n_replacements", 1),
            top_k=kw.get("top_k", 5),
            stopwords=kw.get("stopwords"),
        ),
    }
    
    def __init__(
        self,
        strategies: List[str] = None,
        p: float = 0.5,
        **kwargs,
    ):
        """
        Args:
            strategies: List of strategy names to apply. Options:
                - "random_swap": Swap word pairs
                - "random_delete": Delete words with probability
                - "random_insert": Insert random words
                - "random_shuffle": Shuffle within windows
                - "random_crop": Crop contiguous portion
                - "synonym": FastText synonym replacement
                - "contextual": BERT contextual replacement
            p: Probability of applying each strategy
            **kwargs: Additional arguments passed to strategies
        """
        self.strategies = strategies or ["random_swap", "random_delete"]
        self.p = p
        self.kwargs = kwargs
        
        # Validate strategies
        for s in self.strategies:
            if s not in self.STRATEGY_MAP:
                raise ValueError(f"Unknown strategy: {s}. Available: {list(self.STRATEGY_MAP.keys())}")
    
    def __call__(self, tokens: List[str]) -> List[str]:
        """Apply augmentation strategies.
        
        Args:
            tokens: Input token list
        
        Returns:
            Augmented token list
        """
        result = tokens[:]
        
        for strategy in self.strategies:
            if random.random() < self.p:
                aug_fn = self.STRATEGY_MAP[strategy]
                result = aug_fn(result, **self.kwargs)
        
        return result


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

