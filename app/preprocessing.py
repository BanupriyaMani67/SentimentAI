"""Text cleaning shared by training AND the API.

Using ONE function in both places guarantees the API transforms text
exactly the way the training script did.
"""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_NOT_LETTERS = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, remove URLs/@mentions/punctuation/numbers, expand n't to ' not'."""
    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = text.replace("\u2019", "'")          # curly apostrophe -> straight
    text = re.sub(r"n't\b", " not", text)       # "don't" -> "do not" (keeps negation)
    text = _NOT_LETTERS.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
