from typing import List
import re
import unicodedata
import spacy

_NLP = None

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
) -> List[str]:
    """Tokenize Romanian text.
    
    Args:
        text: Input text string
        use_spacy: Use spaCy tokenizer (recommended) or simple whitespace split
        keep_digits: Pass through to clean_text
        keep_punctuation: Pass through to clean_text
    
    Returns:
        List of tokens
    """
    t = clean_text(text, keep_digits=keep_digits, keep_punctuation=keep_punctuation)
    if use_spacy:
        nlp = get_spacy_nlp()
        doc = nlp(t)
        return [tok.text for tok in doc if not tok.is_space]
    # Fallback simple split
    return t.split()


def detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)
