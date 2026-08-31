"""
Sentence Blocklist — učitava i primenjuje blocklist na retrieval rezultate.

Blocklist fajl: config/sentence_blocklist.txt
Format: jedna rečenica po liniji, # za komentare, prazni redovi se ignorišu.
Matching: case-insensitive contains — rečenica iz baze se blokira ako
          SADRŽI bilo koji tekst iz blockliste.

Fajl se učitava jednom pri startu. Reload: pozvati reload_blocklist().
"""

import os
from pathlib import Path
from typing import List, Set

_BLOCKLIST_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "sentence_blocklist.txt"

# Module-level cache
_blocklist: List[str] = []
_loaded: bool = False


def _load() -> List[str]:
    """Parse blocklist file → list of lowercase patterns."""
    path = os.environ.get("SENTENCE_BLOCKLIST_PATH", str(_BLOCKLIST_PATH))
    patterns: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                patterns.append(line.lower())
        print(f"INFO: Loaded {len(patterns)} sentence blocklist patterns from {path}", flush=True)
    except FileNotFoundError:
        print(f"WARNING: Sentence blocklist not found at {path} — no filtering applied", flush=True)
    return patterns


def get_blocklist() -> List[str]:
    """Return cached blocklist patterns (loads on first call)."""
    global _blocklist, _loaded
    if not _loaded:
        _blocklist = _load()
        _loaded = True
    return _blocklist


def reload_blocklist() -> int:
    """Force-reload blocklist from disk. Returns pattern count."""
    global _blocklist, _loaded
    _blocklist = _load()
    _loaded = True
    return len(_blocklist)


def is_blocked(sentence: str) -> bool:
    """Check if a sentence matches any blocklist pattern (case-insensitive contains)."""
    patterns = get_blocklist()
    if not patterns:
        return False
    sentence_lower = sentence.lower()
    return any(p in sentence_lower for p in patterns)


def filter_weaviate_objects(objects: list, text_property: str = "text") -> list:
    """
    Filter a list of Weaviate result objects, removing blocked sentences.
    
    Args:
        objects: list of Weaviate query result objects
        text_property: name of the property containing sentence text
    
    Returns:
        filtered list with blocked sentences removed
    """
    patterns = get_blocklist()
    if not patterns:
        return objects

    filtered = []
    blocked_count = 0
    for obj in objects:
        text = (obj.properties.get(text_property) or "").lower()
        if any(p in text for p in patterns):
            blocked_count += 1
        else:
            filtered.append(obj)

    if blocked_count > 0:
        print(f"DEBUG: Blocklist filtered out {blocked_count}/{len(objects)} sentences "
              f"({len(filtered)} remaining)", flush=True)

    return filtered


def build_weaviate_blocklist_filter(text_property: str = "text"):
    """
    Build a Weaviate Filter that excludes blocklisted sentences at the DB level.
    
    Uses Filter.by_property(text_property).not_equal(pattern) for each exact
    blocklist pattern. This is much faster than post-filtering in Python because
    Weaviate skips blocked objects during search (no over-fetch needed).
    
    Returns:
        A Weaviate Filter object, or None if blocklist is empty.
    """
    from weaviate.classes.query import Filter
    
    patterns = get_blocklist()
    if not patterns:
        return None
    
    # Rebuild original-case patterns from the file (not_equal is case-sensitive in Weaviate)
    path = os.environ.get("SENTENCE_BLOCKLIST_PATH", str(_BLOCKLIST_PATH))
    original_patterns = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                original_patterns.append(line)
    except FileNotFoundError:
        return None
    
    if not original_patterns:
        return None
    
    # Chain not_equal filters with AND (all must pass)
    combined = None
    for pattern in original_patterns:
        f = Filter.by_property(text_property).not_equal(pattern)
        combined = (combined & f) if combined is not None else f
    
    print(f"DEBUG: Weaviate blocklist filter: {len(original_patterns)} not_equal patterns on '{text_property}'", flush=True)
    return combined
