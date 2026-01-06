from typing import List, Set, Optional
import re
import unicodedata
import spacy

_NLP = None

# Sentiment-important words that should NOT be removed as stopwords
# These are crucial for sentiment analysis even though they might be stopwords
SENTIMENT_KEEP_WORDS = {
    # Negations
    "nu", "n", "nici", "niciodată", "niciodata", "nicidecum", 
    "nicidecât", "nicidecat", "nicăieri", "nimic", "nimeni", "ba", "fără", "fara",
    "nicio", "niciunul", "niciuna",
    
    # Contrast
    "dar", "însă", "insa", "ci", "totuși", "totusi", "deși", "desi",
  
    # Positive sentiment words
    "bine", "bun", "buna", "bună", "excelent", "minunat", "perfect",
    "frumos", "grozav", "super", "genial", "fantastic", "extraordinar",
    "recomand", "recomandat", "multumit", "mulțumit", "mulțumesc", "multumesc", "bucur", "noroc", "vai", "placut", "plăcut",
    "calitate", "rapid", "profesionist", "ok", "okay",

    # Negative sentiment words  
    "rau", "rău", "prost", "groaznic", "oribil", "dezamagit", "dezamăgit",
    "nasol", "slab", "defect", "stricat", "problema", "problemă",
    "teapa", "țeapă", "frauda", "fraudă", "esec", "eșec",
    "niciodata", "niciodată", "deloc", "dezastru",

    # Other sentiment-relevant
    "da", "sigur", "clar", "evident", "posibil", "imposibil",

    # Intensifiers
    "foarte", "extrem", "incredibil", "absolut", "total", "complet", 
    "destul", "prea", "mult", "putin", "puțin", "cel", "cea", "cele",
    "mai", "măcar", "macar", "chiar", "decât", "decit", "doar", "tocmai", "exact",
}

# Romanian diacritic normalization mapping (legacy cedilla -> correct comma-below)
DIACRITIC_MAP = {
    "ş": "ș",  # U+015F -> U+0219
    "ţ": "ț",  # U+0163 -> U+021B
    "Ş": "Ș",  # U+015E -> U+0218
    "Ţ": "Ț",  # U+0162 -> U+021A
}
DIACRITIC_PATTERN = re.compile("|".join(DIACRITIC_MAP.keys()))


def normalize_romanian_diacritics(text: str) -> str:
    """Normalize Romanian diacritics to standard comma-below forms."""
    return DIACRITIC_PATTERN.sub(lambda m: DIACRITIC_MAP[m.group()], text)


def get_spacy_nlp(model: str = "ro_core_news_sm"):
    global _NLP
    if _NLP is not None:
        return _NLP

    try:
        _NLP = spacy.load(model)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load spaCy model '{model}'. Make sure to run: python -m spacy download {model}. Error: {e}"
        )
    return _NLP


def clean_text(s: str, keep_digits: bool = True, keep_punctuation: bool = False) -> str:
    """Clean and normalize Romanian text.
    
    Args:
        s: Input text string
        keep_digits: Whether to keep numeric digits (useful for ratings like "5 stele")
        keep_punctuation: Whether to keep basic punctuation marks
    
    Returns:
        Cleaned and normalized text
    """
    # Unicode normalization (NFKC: compatibility decomposition + canonical composition)
    s = unicodedata.normalize("NFKC", s)
    
    # Normalize Romanian diacritics (cedilla -> comma-below)
    s = normalize_romanian_diacritics(s)
    
    # Lowercase
    s = s.lower().strip()
    
    # Build allowed character set
    # Romanian letters: a-z + ă, â, î, ș, ț
    allowed = r"a-zăâîșț "
    if keep_digits:
        allowed += r"0-9"
    if keep_punctuation:
        allowed += r".,!?;:\-'\"()"
    
    # Remove disallowed characters
    s = re.sub(f"[^{allowed}]", " ", s)
    
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    
    return s


def tokenize(
    text: str, 
    use_spacy: bool = True, 
    keep_digits: bool = True,
    keep_punctuation: bool = False,
    remove_stopwords: bool = False,
    keep_sentiment_words: bool = True,
) -> List[str]:
    """Tokenize Romanian text.
    
    Args:
        text: Input text string
        use_spacy: Use spaCy tokenizer (recommended) or simple whitespace split
        keep_digits: Pass through to clean_text
        keep_punctuation: Pass through to clean_text
        remove_stopwords: Remove Romanian stopwords (default False)
        keep_sentiment_words: Keep sentiment-important words even if stopwords (default True)
    
    Returns:
        List of tokens
    """
    t = clean_text(text, keep_digits=keep_digits, keep_punctuation=keep_punctuation)
    if use_spacy:
        nlp = get_spacy_nlp()
        doc = nlp(t)
        tokens = [tok.text for tok in doc if not tok.is_space]
    else:
        tokens = t.split()
    
    # Optionally remove stopwords while keeping sentiment-important words
    if remove_stopwords:
        tokens = filter_stopwords(tokens, keep_sentiment_words=keep_sentiment_words)
    
    return tokens


def get_stopwords() -> Set[str]:
    """Get Romanian stopwords from spaCy."""
    nlp = get_spacy_nlp()
    return nlp.Defaults.stop_words.copy()


def filter_stopwords(
    tokens: List[str],
    keep_sentiment_words: bool = True,
    custom_keep: Optional[Set[str]] = None,
) -> List[str]:
    """Filter stopwords from token list, keeping sentiment-important words.
    
    Args:
        tokens: List of tokens
        keep_sentiment_words: Keep words from SENTIMENT_KEEP_WORDS
        custom_keep: Additional words to keep
    
    Returns:
        Filtered token list
    """
    stopwords = get_stopwords()
    
    # Words to keep even if they are stopwords
    keep_words = set()
    if keep_sentiment_words:
        keep_words.update(SENTIMENT_KEEP_WORDS)
    if custom_keep:
        keep_words.update(custom_keep)
    
    # Filter: remove stopwords unless they're in keep_words
    return [
        tok for tok in tokens 
        if tok not in stopwords or tok in keep_words
    ]


def detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)
