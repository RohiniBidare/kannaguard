"""
Text preprocessing - MUST match exactly what was used during model training,
otherwise predictions will be less accurate (the model expects cleaned text
in the same format it was trained on).
"""

import re
import unicodedata
import emoji


def preprocess_kannada_english_text(text: str) -> str:
    """Cleans code-mixed Kannada/English comments - same logic used in training."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text