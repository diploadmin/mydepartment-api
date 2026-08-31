"""
Shared utilities for retriever strategies (oneweaviate schema).

Extracts common logic used across SentenceFirstRetriever, CombinedRetriever,
and TwoPhaseRetriever:
- URL / document-hash grouping from paragraph search results
- Score aggregation with weighted top-N
- Section selection via embedding similarity
- Text cleaning for deep-link highlighting
"""

import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import weaviate
from weaviate.classes.query import Filter as WvFilter

from app.core.collections import DOCUMENT, PARAGRAPH
from app.core.config import (
    USE_CONTEXTUAL_COLLECTIONS,
    DEFAULT_ACCESS_GROUPS,
    EXCLUDE_COPYRIGHT,
)
from app.core.retrieval_context import get_param
from app.core.weaviate_props import (
    SOURCE_URL,
    SOURCE_SITE,
    SOURCE_TYPE,
    DOCUMENT_TITLE,
    HEADING_PATH,
    PUBLISH_DATE,
    PERSONS,
    COUNTRIES,
    ACCESS_GROUPS,
    COPYRIGHT,
    PARENT_DOCUMENT_HASH,
    DOCUMENT_HASH,
    PARENT_PARAGRAPH_ID,
    PARAGRAPH_ID,
    TEXT,
    map_collection_name,
    doc_url,
    doc_title,
    doc_date,
    section_key,
    source_type,
    label_from_props,
    group_key,
    heading_leaf,
    normalize_props,
)

_parent_hash_cache: Dict[str, Optional[str]] = {}
_city_org_warned = False


def combine_filters(*parts: Optional[WvFilter]) -> Optional[WvFilter]:
    """AND-combine filters, skipping None."""
    combined: Optional[WvFilter] = None
    for part in parts:
        if part is None:
            continue
        combined = part if combined is None else (combined & part)
    return combined


def visibility_filter() -> Optional[WvFilter]:
    """oneweaviate has no visibility property — no-op."""
    return None


def post_types_filter(post_types: Optional[List[str]]) -> Optional[WvFilter]:
    """Whitelist filter on ``source_type`` (legacy request field: post_types)."""
    if not post_types:
        return None

    normalized = [pt.strip().lower() for pt in post_types if pt and pt.strip()]
    if not normalized:
        return None

    return WvFilter.by_property(SOURCE_TYPE).contains_any(normalized)


def _merge_filter_values(
    single_value: Optional[str] = None,
    multi_values: Optional[list] = None,
) -> list[str]:
    """Merge single + multi filter values, deduplicated (order preserved)."""
    values: list[str] = []
    if multi_values:
        for item in multi_values:
            if item is not None and str(item).strip():
                values.append(str(item).strip())
    if single_value and str(single_value).strip():
        values.append(str(single_value).strip())
    return list(dict.fromkeys(values))


def scalar_text_filter(
    property_name: str,
    single_value: Optional[str] = None,
    multi_values: Optional[list] = None,
) -> Optional[WvFilter]:
    """Build an OR filter on a TEXT or TEXT[] property via ``contains_any``."""
    unique_values = _merge_filter_values(single_value, multi_values)
    if not unique_values:
        return None
    return WvFilter.by_property(property_name).contains_any(unique_values)


def person_filter(person_name: Optional[str] = None, person_names: Optional[list] = None) -> Optional[WvFilter]:
    """Build an include filter on the ``persons`` property (TEXT[])."""
    return scalar_text_filter(PERSONS, person_name, person_names)


def person_filter_from_params() -> Optional[WvFilter]:
    """Resolve person filter from per-request retrieval overrides."""
    return scalar_text_filter_from_params(PERSONS, "person_filter_name", "person_filter_names")


def city_filter_from_params() -> Optional[WvFilter]:
    """City filter unsupported on oneweaviate — no-op."""
    global _city_org_warned
    values = merged_param_values("city_filter_name", "city_filter_names")
    if values and not _city_org_warned:
        print(
            f"WARNING: city_filter ignored (oneweaviate has no city property): {values}",
            flush=True,
        )
        _city_org_warned = True
    return None


def country_filter_from_params() -> Optional[WvFilter]:
    """Filter on ``countries`` TEXT[] (no people-only gate)."""
    return scalar_text_filter_from_params(
        COUNTRIES, "country_filter_name", "country_filter_names"
    )


def organisation_filter_from_params() -> Optional[WvFilter]:
    """Organisation filter unsupported on oneweaviate — no-op."""
    global _city_org_warned
    values = merged_param_values("organisation_filter_name", "organisation_filter_names")
    if values and not _city_org_warned:
        print(
            f"WARNING: organisation_filter ignored "
            f"(oneweaviate has no organisation property): {values}",
            flush=True,
        )
        _city_org_warned = True
    return None


def resolve_access_groups() -> list[str]:
    """Access groups from request, else DEFAULT_ACCESS_GROUPS from .env."""
    from_request = merged_param_values("access_group_name", "access_group_names")
    if from_request:
        return from_request
    return [str(g).strip() for g in (DEFAULT_ACCESS_GROUPS or []) if str(g).strip()]


def access_groups_filter(groups: Optional[List[str]] = None) -> Optional[WvFilter]:
    """Whitelist filter on ``access_groups`` TEXT[] via ``contains_any``."""
    values = groups if groups is not None else resolve_access_groups()
    if not values:
        return None
    return WvFilter.by_property(ACCESS_GROUPS).contains_any(values)


def access_groups_filter_from_params() -> Optional[WvFilter]:
    """Resolve access_groups filter (request override or .env default)."""
    return access_groups_filter()


def copyright_filter() -> Optional[WvFilter]:
    """Exclude copyrighted objects (copyright=1) when EXCLUDE_COPYRIGHT is enabled.

    oneweaviate: INT ``copyright`` — 0 = free to use, 1 = under copyright.
    """
    enabled = bool(get_param("exclude_copyright", EXCLUDE_COPYRIGHT))
    if not enabled:
        return None
    return WvFilter.by_property(COPYRIGHT).not_equal(1)


def scalar_text_filter_from_params(
    property_name: str,
    single_param: str,
    multi_param: str,
) -> Optional[WvFilter]:
    """Resolve a scalar/TEXT[] filter from per-request retrieval overrides."""
    return scalar_text_filter(
        property_name,
        single_value=get_param(single_param, None),
        multi_values=get_param(multi_param, None),
    )


def merged_param_values(single_param: str, multi_param: str) -> list[str]:
    """Return merged active values for a single+multi filter param pair."""
    return _merge_filter_values(
        get_param(single_param, None),
        get_param(multi_param, None),
    )


_CONTENT_FILTER_SPECS = (
    ("person", person_filter_from_params, "person_filter_name", "person_filter_names"),
    ("city", city_filter_from_params, "city_filter_name", "city_filter_names"),
    ("country", country_filter_from_params, "country_filter_name", "country_filter_names"),
    ("organisation", organisation_filter_from_params, "organisation_filter_name", "organisation_filter_names"),
)


def _build_custom_filter_piece(field: str, operator: str, value_text: str) -> Optional[WvFilter]:
    if not field or not value_text:
        return None
    # Map legacy Diplo field names used by WP custom filters
    field_map = {
        "person": PERSONS,
        "link": SOURCE_URL,
        "site": SOURCE_SITE,
        "post_type": SOURCE_TYPE,
        "name": DOCUMENT_TITLE,
        "h1": DOCUMENT_TITLE,
        "date": PUBLISH_DATE,
        "section": HEADING_PATH,
        "last_h_title": HEADING_PATH,
        "sentence": TEXT,
        "country": COUNTRIES,
        "access_groups": ACCESS_GROUPS,
    }
    field = field_map.get(field, field)
    op = (operator or "equal").strip()
    if op.lower() == "like":
        pattern = value_text if "*" in value_text else f"*{value_text}*"
        return WvFilter.by_property(field).like(pattern)
    return WvFilter.by_property(field).equal(value_text)


def _piece_from_filter_item(item: dict, target_collection: str) -> tuple[Optional[WvFilter], Optional[str]]:
    """Return (weaviate_filter, debug_label) for one filter item, or (None, None)."""
    if not isinstance(item, dict):
        return None, None
    collection = map_collection_name((item.get("collection") or "").strip())
    target = map_collection_name(target_collection)
    if collection != target:
        return None, None
    path = item.get("path") or []
    field = path[0] if path else None
    value_text = (item.get("valueText") or "").strip()
    operator = item.get("operator") or "Equal"
    piece = _build_custom_filter_piece(field, operator, value_text)
    if piece is None:
        return None, None
    return piece, f"{field} {operator} {value_text!r}"


def _combine_filters(parts: list[WvFilter], how: str) -> Optional[WvFilter]:
    if not parts:
        return None
    out = parts[0]
    join_or = (how or "and").strip().lower() == "or"
    for part in parts[1:]:
        out = out | part if join_or else out & part
    return out


def custom_filters_for_collection(target_collection: str) -> Optional[WvFilter]:
    """Build Weaviate filter from per-request custom_filters for a target collection."""
    raw = get_param("custom_filters", None)
    if not raw or not isinstance(raw, dict):
        return None

    top_combine = (raw.get("combine") or "or").strip().lower()
    if top_combine not in ("and", "or"):
        top_combine = "or"

    groups = raw.get("groups")
    if isinstance(groups, list) and groups:
        group_filters: list[WvFilter] = []
        active_labels: list[str] = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            group_combine = (group.get("combine") or "and").strip().lower()
            if group_combine not in ("and", "or"):
                group_combine = "and"
            items = group.get("filters") or []
            pieces: list[WvFilter] = []
            labels: list[str] = []
            for item in items:
                piece, label = _piece_from_filter_item(item, target_collection)
                if piece is None:
                    continue
                pieces.append(piece)
                if label:
                    labels.append(label)
            group_f = _combine_filters(pieces, group_combine)
            if group_f is None:
                continue
            group_filters.append(group_f)
            if len(labels) == 1:
                active_labels.append(labels[0])
            else:
                joiner = f" {group_combine.upper()} "
                active_labels.append(f"({joiner.join(labels)})")

        combined = _combine_filters(group_filters, top_combine)
        if combined is not None:
            joiner = f" {top_combine.upper()} "
            print(
                f"DEBUG: custom_filters ACTIVE on {target_collection}: {joiner.join(active_labels)}",
                flush=True,
            )
        return combined

    items = raw.get("filters") or []
    if not items:
        return None

    pieces: list[WvFilter] = []
    active: list[str] = []
    for item in items:
        piece, label = _piece_from_filter_item(item, target_collection)
        if piece is None:
            continue
        pieces.append(piece)
        if label:
            active.append(label)

    combined = _combine_filters(pieces, top_combine)
    if combined is not None:
        joiner = f" {top_combine.upper()} "
        print(
            f"DEBUG: custom_filters ACTIVE on {target_collection}: {joiner.join(active)}",
            flush=True,
        )
    return combined


def append_content_filters(
    base_filter: Optional[WvFilter],
    target_collection: Optional[str] = None,
) -> Optional[WvFilter]:
    """Append person/country and optional custom filters."""
    for label, builder, single_key, multi_key in _CONTENT_FILTER_SPECS:
        field_filter = builder()
        if field_filter is not None:
            base_filter = combine_filters(base_filter, field_filter)
            values = merged_param_values(single_key, multi_key)
            print(f"DEBUG: {label}_filter ACTIVE for {values}", flush=True)

    if target_collection:
        custom_filter = custom_filters_for_collection(target_collection)
        if custom_filter is not None:
            base_filter = combine_filters(base_filter, custom_filter)

    return base_filter


def get_content_filter_debug_info() -> dict[str, dict]:
    """Active content filters for debug responses."""
    info: dict[str, dict] = {}
    for label, _, single_key, multi_key in _CONTENT_FILTER_SPECS:
        values = merged_param_values(single_key, multi_key)
        if values:
            entry: dict = {"names": values, "status": "active"}
            if label in ("city", "organisation"):
                entry["status"] = "ignored"
                entry["reason"] = "unsupported_on_oneweaviate"
            info[f"{label}_filter"] = entry
    access_groups = resolve_access_groups()
    if access_groups:
        from_request = bool(merged_param_values("access_group_name", "access_group_names"))
        info["access_groups_filter"] = {
            "names": access_groups,
            "status": "active",
            "source": "request" if from_request else "default_env",
        }
    exclude_copyright = bool(get_param("exclude_copyright", EXCLUDE_COPYRIGHT))
    info["copyright_filter"] = {
        "status": "active" if exclude_copyright else "disabled",
        "exclude_copyright": exclude_copyright,
        "rule": "copyright != 1" if exclude_copyright else None,
    }
    custom_filters = get_param("custom_filters")
    if custom_filters:
        info["custom_filters"] = custom_filters
    return info


def resolve_collection_name(base_name: str) -> str:
    """Resolve collection name for oneweaviate (no ``_contextual`` suffix)."""
    # USE_CONTEXTUAL_COLLECTIONS is intentionally ignored on this fork.
    _ = get_param("use_contextual", USE_CONTEXTUAL_COLLECTIONS)
    return map_collection_name(base_name)


def is_using_contextual() -> bool:
    """Contextual suffix toggle — always False on oneweaviate fork."""
    return False


def resolve_parent_document_hash(
    client: weaviate.WeaviateClient,
    parent_name: str,
    site_name: Optional[str] = None,
) -> Optional[str]:
    """Resolve a parent document title to its document_hash in Documents."""
    cache_key = f"{parent_name}||{site_name}" if site_name else parent_name
    if cache_key in _parent_hash_cache:
        return _parent_hash_cache[cache_key]

    try:
        doc_coll = client.collections.get(DOCUMENT)
        name_filter = WvFilter.by_property(DOCUMENT_TITLE).equal(parent_name)
        if site_name:
            name_filter = name_filter & WvFilter.by_property(SOURCE_SITE).equal(site_name)
        result = doc_coll.query.fetch_objects(
            filters=name_filter,
            limit=50,
            return_properties=[DOCUMENT_HASH, DOCUMENT_TITLE],
        )

        target = parent_name.strip().lower()
        exact = None
        first = None
        for obj in result.objects:
            props = obj.properties
            if first is None:
                first = props.get(DOCUMENT_HASH)
            obj_name = (props.get(DOCUMENT_TITLE) or "").strip().lower()
            if obj_name == target:
                exact = props.get(DOCUMENT_HASH)
                break

        doc_hash = exact or first
        if doc_hash:
            _parent_hash_cache[cache_key] = doc_hash
            match_type = "exact" if exact else "fuzzy (first token-match)"
            print(
                f"DEBUG: Resolved parent_filter_name '{parent_name}' (site={site_name}) "
                f"→ hash={doc_hash} [{match_type}]",
                flush=True,
            )
            return doc_hash
    except Exception as e:
        print(f"DEBUG: Failed to resolve parent_filter_name '{parent_name}': {e}", flush=True)

    _parent_hash_cache[cache_key] = None
    return None


def build_base_retrieval_filter(
    client: weaviate.WeaviateClient,
    target_collection: str,
) -> Optional[WvFilter]:
    """Shared site / parent / source_type / access_groups / copyright / content filters."""
    parts: list[Optional[WvFilter]] = [visibility_filter(), copyright_filter()]
    if parts[1] is not None:
        print("DEBUG: copyright filter ACTIVE (exclude copyright=1)", flush=True)
    site_name = get_param("site_filter_name", None)
    parent_name = get_param("parent_filter_name", None)

    if parent_name:
        parent_hash = resolve_parent_document_hash(client, parent_name, site_name=site_name)
        if parent_hash:
            parts.append(WvFilter.by_property(PARENT_DOCUMENT_HASH).equal(parent_hash))
            print(f"DEBUG: parent_filter ACTIVE for '{parent_name}' → {parent_hash}", flush=True)
        else:
            print(
                f"WARNING: parent_filter_name '{parent_name}' not found in {DOCUMENT}, "
                f"filter skipped",
                flush=True,
            )

    if site_name:
        parts.append(WvFilter.by_property(SOURCE_SITE).equal(site_name))
        print(f"DEBUG: site_filter ACTIVE for '{site_name}'", flush=True)

    pt_filter = post_types_filter(get_param("post_types", None))
    if pt_filter is not None:
        parts.append(pt_filter)
        print(f"DEBUG: post_types/source_type filter ACTIVE for {get_param('post_types', None)}", flush=True)

    access_groups = resolve_access_groups()
    ag_filter = access_groups_filter(access_groups)
    if ag_filter is not None:
        parts.append(ag_filter)
        from_request = bool(merged_param_values("access_group_name", "access_group_names"))
        src = "request" if from_request else "DEFAULT_ACCESS_GROUPS"
        print(f"DEBUG: access_groups filter ACTIVE for {access_groups} (source={src})", flush=True)

    return append_content_filters(combine_filters(*parts), target_collection)


def fetch_paragraph_texts_by_ids(
    client: weaviate.WeaviateClient,
    paragraph_ids: Iterable[str],
) -> Dict[str, str]:
    """Batch-fetch Paragraph ``text`` by ``paragraph_id``."""
    ids = [pid for pid in dict.fromkeys(paragraph_ids) if pid]
    if not ids:
        return {}

    out: Dict[str, str] = {}
    coll = client.collections.get(PARAGRAPH)
    # Weaviate contains_any has practical limits; chunk requests.
    chunk_size = 50
    for i in range(0, len(ids), chunk_size):
        batch = ids[i : i + chunk_size]
        try:
            result = coll.query.fetch_objects(
                filters=WvFilter.by_property(PARAGRAPH_ID).contains_any(batch),
                limit=len(batch),
                return_properties=[PARAGRAPH_ID, TEXT],
            )
            for obj in result.objects:
                props = obj.properties or {}
                pid = props.get(PARAGRAPH_ID)
                text = props.get(TEXT) or ""
                if pid and text:
                    out[str(pid)] = text
        except Exception as e:
            print(f"DEBUG: paragraph fetch batch failed: {e}", flush=True)
    return out


# Weights for top-N score aggregation across retrievers
AGGREGATE_WEIGHTS = [1.0, 0.5, 0.25, 0.15, 0.1]

MAX_SECTION_LEN = 3500


def clean_sentence_text(text: str) -> str:
    """Clean sentence text for deep link highlighting."""
    text = text.replace("\\'", "'").replace('\\"', '"').replace("\\n", " ").replace("\\r", "")
    text = re.sub(r"([.!?;:])([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"('(?:s|t|d|re|ve|ll|m))([a-z])", r"\1 \2", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


def aggregate_scores(scores: List[float], weights: List[float] = None) -> float:
    """Weighted aggregation of top-N sorted scores."""
    if not scores:
        return 0.0
    w = weights or AGGREGATE_WEIGHTS
    sorted_scores = sorted(scores, reverse=True)
    return sum(s * wt for s, wt in zip(sorted_scores, w))


def group_paragraphs_by_url(objects, *, url_prop: str = SOURCE_URL) -> Dict[str, dict]:
    """Group Weaviate paragraph search results by document key.

    When ``source_url`` is empty, groups by ``parent_document_hash``.
    Returns dict: group_id -> {paragraphs, sections, props, post_type, label, ...}
    """
    url_groups: Dict[str, dict] = {}

    for obj in objects:
        props = normalize_props(obj.properties or {})
        score = obj.metadata.score if obj.metadata else 0.0
        text = props.get(TEXT) or props.get("text") or ""
        if not text:
            continue

        url = props.get(url_prop) or doc_url(props)
        gkey = group_key(props)
        section = section_key(props)
        post_type = source_type(props)
        label = label_from_props(props)

        if gkey not in url_groups:
            url_groups[gkey] = {
                "paragraphs": [],
                "sections": {},
                "props": props,
                "post_type": post_type,
                "label": label,
                "url": url,
                "group_key": gkey,
            }

        para_entry = {
            "text": text,
            "score": score,
            "h2": heading_leaf(section),
            "section": section,
            "last_h_title": heading_leaf(section) or section,
            "paragraph_index": props.get("paragraph_index") or 0,
            PARENT_DOCUMENT_HASH: props.get(PARENT_DOCUMENT_HASH) or "",
        }
        url_groups[gkey]["paragraphs"].append(para_entry)

        if section not in url_groups[gkey]["sections"]:
            url_groups[gkey]["sections"][section] = []
        url_groups[gkey]["sections"][section].append(para_entry)

    return url_groups


def compute_url_aggregate_scores(url_groups: Dict[str, dict]) -> None:
    """Calculate aggregate scores per URL group in-place."""
    for url, group in url_groups.items():
        paras = group["paragraphs"]
        if paras:
            scores = sorted([p["score"] for p in paras], reverse=True)
            group["aggregate_score"] = aggregate_scores(scores)
            group["match_count"] = len(paras)
            group["best_para"] = max(paras, key=lambda p: p["score"])
        else:
            group["aggregate_score"] = 0.0
            group["match_count"] = 0
            group["best_para"] = {
                "text": "",
                "score": 0,
                "h2": "",
                "section": "general",
                "last_h_title": "",
            }

        best_section_name = "general"
        best_section_score = 0
        for sec_name, sec_paras in group["sections"].items():
            sec_score = sum(p["score"] for p in sec_paras)
            if sec_score > best_section_score:
                best_section_score = sec_score
                best_section_name = sec_name
        group["best_section"] = best_section_name


def select_best_section(
    section_paras: Dict[str, list],
    query_vector: List[float],
    embeddings: Embeddings,
    default_section: str = "general",
) -> str:
    """Select the most relevant section via embedding similarity of section titles."""
    eligible = []
    for sec_name, sec_ps in section_paras.items():
        sec_text = "\n\n".join(p["text"] for p in sec_ps if p["text"])
        if len(sec_text) < 200:
            continue
        title = (sec_ps[0].get("last_h_title") or heading_leaf(sec_name) or "").strip()
        if not title:
            title = sec_text[:150]
        eligible.append((sec_name, title))

    if len(eligible) <= 1:
        return eligible[0][0] if eligible else default_section

    try:
        titles = [t for _, t in eligible]
        title_vectors = embeddings.embed_documents(titles)

        q_norm = sum(x * x for x in query_vector) ** 0.5
        best_idx = 0
        best_sim = -1.0
        for i, tv in enumerate(title_vectors):
            dot = sum(a * b for a, b in zip(query_vector, tv))
            t_norm = sum(x * x for x in tv) ** 0.5
            sim = dot / (q_norm * t_norm) if q_norm and t_norm else 0.0
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        return eligible[best_idx][0]
    except Exception as e:
        print(f"DEBUG: Section embedding similarity failed: {e}", flush=True)
        return default_section


def build_section_content(
    section_paras: Dict[str, list],
    target_section: str,
    max_len: int = MAX_SECTION_LEN,
) -> Tuple[str, list]:
    """Join paragraphs of target section, truncate if needed."""
    target_paras = section_paras.get(target_section, [])
    for sec_name in section_paras:
        section_paras[sec_name].sort(key=lambda p: p.get("paragraph_index", 0))

    section_text = "\n\n".join(p["text"] for p in target_paras if p["text"])
    if len(section_text) > max_len:
        section_text = section_text[:max_len] + "..."

    return section_text, target_paras


def fetch_paragraphs_for_group(
    client: weaviate.WeaviateClient,
    group: dict,
    *,
    limit: int = 100,
) -> list:
    """Fetch paragraphs for a group by source_url or parent_document_hash."""
    props = group.get("props") or {}
    url = group.get("url") or doc_url(props)
    doc_hash = props.get(PARENT_DOCUMENT_HASH) or ""
    coll = client.collections.get(PARAGRAPH)

    filt = None
    if url:
        filt = WvFilter.by_property(SOURCE_URL).equal(url)
    elif doc_hash:
        filt = WvFilter.by_property(PARENT_DOCUMENT_HASH).equal(doc_hash)
    else:
        return []

    result = coll.query.fetch_objects(
        filters=filt,
        limit=limit,
        return_properties=[
            TEXT,
            HEADING_PATH,
            PARAGRAPH_ID,
            "paragraph_index",
            DOCUMENT_TITLE,
            SOURCE_URL,
            PARENT_DOCUMENT_HASH,
        ],
    )
    out = []
    for obj in result.objects:
        p = normalize_props(obj.properties or {})
        text = p.get(TEXT) or ""
        if not text:
            continue
        section = section_key(p)
        out.append(
            {
                "text": text,
                "h2": heading_leaf(section),
                "section": section,
                "paragraph_index": p.get("paragraph_index") or 0,
                "last_h_title": heading_leaf(section) or section,
            }
        )
    return out


def build_paragraph_document(
    url: str,
    group: dict,
    section_text: str,
    target_section: str,
    target_paras: list,
    fetched_count: int,
    source: str = "paragraph_search",
) -> Document:
    """Build a Document from aggregated paragraph search results."""
    props = normalize_props(group["props"])
    hybrid_best = group["best_para"]
    content = section_text if section_text else hybrid_best.get("text", "")
    display_url = url or group.get("url") or doc_url(props)

    target_section_title = ""
    for tp in target_paras:
        lht = tp.get("last_h_title", "") or heading_leaf(tp.get("section", ""))
        if lht:
            target_section_title = lht
            break

    best = target_paras[0] if target_paras else hybrid_best
    from app.services.doc_title_service import display_title

    title = display_title(props)

    metadata = {
        "title": title,
        "url": display_url,
        "date": doc_date(props),
        "label": group["label"],
        "post_type": group["post_type"],
        "h1": title,
        "h2": best.get("h2", ""),
        "h3": "",
        "section": target_section,
        "strong_heading": "",
        "parent_document_hash": props.get(PARENT_DOCUMENT_HASH) or "",
        "_index": group.get("_index", resolve_collection_name(PARAGRAPH)),
        "_best_sentence": clean_sentence_text((hybrid_best.get("text", "") or "")[:200]),
        "_best_sentence_score": hybrid_best.get("score", 0),
        "_aggregate_score": group["aggregate_score"],
        "_matched_paragraphs": group["match_count"],
        "_fetched_paragraphs": fetched_count,
        "_section": target_section,
        "_section_title": target_section_title,
        "_num_sections": len(group.get("sections", {})),
        "_force_boost": group.get("_force_boost", 1.0),
        "_force_reason": group.get("_force_reason", ""),
        "_label_weight": 1.0,
        "_final_score": group["aggregate_score"],
        "_source": source,
    }

    return Document(page_content=content, metadata=metadata)
