"""
oneweaviate property names and Diplo-compatible metadata helpers.

Centralizes the Documents / Paragraphs / Sentences field map so retrievers
and catalogs do not scatter string literals.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple
from urllib.parse import unquote, urlparse

from app.ai.ai_services.label_weights import POST_TYPE_TO_LABEL

# --- Collection-level property names ---
TEXT = "text"
EMBEDDING_TEXT = "embedding_text"
SOURCE_URL = "source_url"
SOURCE_SITE = "source_site"
SOURCE_TYPE = "source_type"
DOCUMENT_TITLE = "document_title"
FULL_TEXT = "full_text"
HEADING_PATH = "heading_path"
PUBLISH_DATE = "publish_date"
PERSONS = "persons"
COUNTRIES = "countries"
ACCESS_GROUPS = "access_groups"
COPYRIGHT = "copyright"
PARENT_DOCUMENT_HASH = "parent_document_hash"
DOCUMENT_HASH = "document_hash"
PARENT_PARAGRAPH_ID = "parent_paragraph_id"
PARAGRAPH_ID = "paragraph_id"
SENTENCE_ID = "sentence_id"
SENTENCE_INDEX = "sentence_index"
PARAGRAPH_INDEX = "paragraph_index"

# Legacy Diplo base names → oneweaviate collections
_LEGACY_COLLECTION_MAP = {
    "DiploDocument": "Documents",
    "DiploDocument_contextual": "Documents",
    "DiploParagraph": "Paragraphs",
    "DiploParagraph_contextual": "Paragraphs",
    "DiploChunk": "Sentences",
    "DiploChunk_contextual": "Sentences",
    "DiploTurn": "Paragraphs",
    "DiploTurn_contextual": "Paragraphs",
    "DiploHeading": "Paragraphs",
}


def map_collection_name(name: str) -> str:
    """Map legacy Diplo / env names to oneweaviate collection names."""
    if not name:
        return name
    if name in _LEGACY_COLLECTION_MAP:
        return _LEGACY_COLLECTION_MAP[name]
    # Strip accidental _contextual suffix
    if name.endswith("_contextual"):
        base = name[: -len("_contextual")]
        return _LEGACY_COLLECTION_MAP.get(base, name)
    return name


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


_HASH_RE = re.compile(r"^[0-9a-fA-F_-]{16,}$")
_FILE_EXTS = (
    ".pdf", ".html", ".htm", ".docx", ".doc", ".txt", ".md", ".epub", ".pptx",
)


def _fully_unquote(value: str) -> str:
    """Unquote percent-encoding up to 3 times (ingest often double-encodes)."""
    text = (value or "").strip()
    for _ in range(3):
        nxt = unquote(text)
        if nxt == text:
            break
        text = nxt
    return text


def _looks_like_url(value: str) -> bool:
    text = (value or "").strip()
    if text.startswith("http://") or text.startswith("https://"):
        return True
    decoded = _fully_unquote(text)
    return decoded.startswith("http://") or decoded.startswith("https://")


def _looks_like_hash(value: str) -> bool:
    text = (value or "").strip()
    return bool(text) and bool(_HASH_RE.fullmatch(text))


def _strip_dump_ext(url: str) -> str:
    """Drop the ingest dump extension: the live page has no ``.txt`` / ``.md``."""
    base, sep, query = url.partition("?")
    if base.endswith("/.txt") or base.endswith("/.md"):
        base = base.rsplit("/", 1)[0]
    else:
        for ext in (".txt", ".md"):
            if base.endswith(ext):
                base = base[: -len(ext)]
                break
    return base + sep + query


def recover_url_from_text(value: str) -> str:
    """If *value* is a (possibly encoded) http(s) URL, return the decoded URL."""
    decoded = _fully_unquote(value)
    if decoded.startswith("http://") or decoded.startswith("https://"):
        # Ingest titles sometimes decode "%20" into a real space mid-path.
        decoded = decoded.replace(" ", "%20").rstrip(").,;")
        return _strip_dump_ext(decoded)
    return ""


def human_title_from_url(url: str) -> str:
    """Filename / last path segment from a URL, cleaned for display."""
    if not url:
        return ""
    path = urlparse(url).path.rstrip("/")
    name = path.split("/")[-1] if path else ""
    name = _fully_unquote(name)
    lower = name.lower()
    for ext in _FILE_EXTS:
        if lower.endswith(ext):
            name = name[: -len(ext)]
            break
    name = name.replace("_", " ").replace("%20", " ")
    name = re.sub(r"\s+", " ", name).strip(" -_.")
    if _looks_like_hash(name.replace(" ", "")):
        return ""
    return name


_HASH_PREFIX_RE = re.compile(r"^[0-9a-f]{16,64}[_-]+", re.IGNORECASE)


def _clean_display_title(value: str) -> str:
    """Strip ingest artifacts from a filename-style title (hash prefix, extension)."""
    name = _HASH_PREFIX_RE.sub("", (value or "").strip())
    lower = name.lower()
    for ext in _FILE_EXTS:
        if lower.endswith(ext):
            name = name[: -len(ext)]
            break
    return re.sub(r"\s+", " ", name).strip(" -_.") or (value or "").strip()


def _heading_display(heading: str) -> str:
    """First meaningful heading_path segment (not a URL or hash)."""
    if not heading or heading == "general":
        return ""
    first = heading.split(">")[0].strip()
    if first and not _looks_like_url(first) and not _looks_like_hash(first):
        return first
    leaf = heading_leaf(heading)
    if leaf and not _looks_like_url(leaf) and not _looks_like_hash(leaf):
        return leaf
    return ""


def raw_document_title(props: Dict[str, Any]) -> str:
    return _as_str(
        props.get(DOCUMENT_TITLE)
        or props.get("h1")
        or props.get("name")
        or props.get("title")
        or ""
    ).strip()


def doc_url(props: Dict[str, Any]) -> str:
    """Best-effort URL: source_url, else URL recovered from encoded document_title."""
    url = _as_str(
        props.get(SOURCE_URL) or props.get("url") or props.get("link") or ""
    ).strip()
    if url and url.startswith("http"):
        return _strip_dump_ext(url)
    recovered = recover_url_from_text(url) if url else ""
    if recovered:
        return recovered
    return recover_url_from_text(raw_document_title(props))


def doc_title(props: Dict[str, Any]) -> str:
    """Human-readable title for source cards (never a raw encoded URL or hash)."""
    heading = _heading_display(section_key(props))
    raw = raw_document_title(props)
    url = doc_url(props)

    if raw and not _looks_like_url(raw) and not _looks_like_hash(raw):
        return _clean_display_title(raw)
    if heading:
        return heading
    from_url = human_title_from_url(url)
    if from_url:
        return from_url
    if raw:
        decoded = _fully_unquote(raw)
        if decoded and not _looks_like_url(decoded) and not _looks_like_hash(decoded):
            return decoded
    return "Untitled"


def present_source_fields(props: Dict[str, Any]) -> Tuple[str, str]:
    """Return (display_title, url) for frontend source cards."""
    return doc_title(props), doc_url(props)


def doc_date(props: Dict[str, Any]) -> str:
    return _as_str(props.get(PUBLISH_DATE) or props.get("date") or "")


def section_key(props: Dict[str, Any]) -> str:
    return _as_str(
        props.get(HEADING_PATH)
        or props.get("section")
        or props.get("last_h_title")
        or "general"
    ).strip() or "general"


def source_type(props: Dict[str, Any]) -> str:
    return _as_str(props.get(SOURCE_TYPE) or props.get("post_type") or "").strip().lower()


def label_from_props(props: Dict[str, Any]) -> str:
    st = source_type(props)
    if not st:
        return "Post"
    return POST_TYPE_TO_LABEL.get(st, st.capitalize())


def group_key(props: Dict[str, Any]) -> str:
    """Stable group id when source_url may be null: prefer hash, else url, else title."""
    doc_hash = _as_str(props.get(PARENT_DOCUMENT_HASH) or props.get(DOCUMENT_HASH) or "").strip()
    if doc_hash:
        return f"hash:{doc_hash}"
    url = doc_url(props)
    if url:
        return f"url:{url}"
    title = doc_title(props)
    if title:
        return f"title:{title}"
    return "unknown"


def sentence_text(props: Dict[str, Any]) -> str:
    return _as_str(props.get(TEXT) or props.get("sentence") or "")


def normalize_props(props: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy with Diplo-compatible aliases for downstream code."""
    out = dict(props or {})
    url = doc_url(out)
    title = doc_title(out)
    date = doc_date(out)
    section = section_key(out)
    st = source_type(out)
    sent = sentence_text(out)

    out.setdefault("link", url)
    out.setdefault("url", url)
    out.setdefault("h1", title)
    out.setdefault("name", title)
    out.setdefault("title", title)
    out.setdefault("date", date)
    out.setdefault("section", section)
    out.setdefault("last_h_title", section)
    out.setdefault("post_type", st)
    out.setdefault("sentence", sent)
    out.setdefault("site", _as_str(out.get(SOURCE_SITE) or ""))
    if PERSONS in out and "person" not in out:
        out["person"] = out.get(PERSONS)
    if COUNTRIES in out and "country" not in out:
        countries = out.get(COUNTRIES)
        if isinstance(countries, list) and countries:
            out["country"] = countries[0]
        elif countries:
            out["country"] = countries
    return out


def heading_leaf(heading_path: Optional[str]) -> str:
    """Last segment of ``a > b > c`` style heading_path."""
    if not heading_path:
        return ""
    parts = [p.strip() for p in str(heading_path).split(">") if p.strip()]
    return parts[-1] if parts else ""
