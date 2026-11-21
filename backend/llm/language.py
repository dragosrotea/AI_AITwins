"""Conservatively admit English text for the spoken alert path."""

from functools import cache
from threading import Lock

from langdetect import DetectorFactory, LangDetectException, detect_langs

from .templates import english_templates

DetectorFactory.seed = 0
_DETECTOR_LOCK = Lock()
_SHORT_QUESTIONS = {"are you okay?", "are you safe?", "can you focus?", "can you hear me?"}


@cache
def _known_english() -> set[str]:
    return _SHORT_QUESTIONS | {
        text.casefold() for variants in english_templates().values() for text, _ in variants
    }


def is_english(text: str) -> bool:
    if text.strip().casefold() in _known_english():
        return True
    try:
        # langdetect lazily loads shared profiles on the first call.
        with _DETECTOR_LOCK:
            languages = detect_langs(text)
    except LangDetectException:
        return False
    return bool(languages and languages[0].lang == "en" and languages[0].prob >= 0.85)
