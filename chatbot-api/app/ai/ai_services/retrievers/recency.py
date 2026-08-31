"""Optional post-reranker recency boost (default off unless use_recency_boost=true)."""

import math
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

from app.core.config import (
    USE_RECENCY_BOOST,
    RECENCY_BOOST_MAX_AGE_DAYS,
    RECENCY_HALF_LIFE_DAYS,
    RECENCY_MAX_BOOST,
)
from app.core.retrieval_context import get_param


def parse_publish_date(value: Any) -> Optional[date]:
    """Parse chunk/document publish date from Weaviate or metadata strings."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date() if value.tzinfo else value.date()
    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    text = str(value).strip()
    if not text:
        return None

    normalized = text.replace("Z", "+00:00")
    try:
        if "T" in normalized or "+" in normalized or normalized.count("-") > 2:
            dt = datetime.fromisoformat(normalized)
            return dt.astimezone(timezone.utc).date() if dt.tzinfo else dt.date()
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y"):
        try:
            return datetime.strptime(text[:19], fmt).date()
        except ValueError:
            continue
    return None


def recency_multiplier(
    publish_date: Any,
    *,
    max_age_days: int = RECENCY_BOOST_MAX_AGE_DAYS,
    half_life_days: int = RECENCY_HALF_LIFE_DAYS,
    max_boost: float = RECENCY_MAX_BOOST,
    reference_date: Optional[date] = None,
) -> float:
    """
    Soft recency boost multiplier (always >= 1.0).

    Posts older than max_age_days get 1.0 (no boost, not penalized).
    """
    parsed = parse_publish_date(publish_date)
    if parsed is None:
        return 1.0

    today = reference_date or datetime.now(timezone.utc).date()
    age_days = max(0, (today - parsed).days)
    if age_days > max_age_days:
        return 1.0

    if half_life_days <= 0 or max_boost <= 0:
        return 1.0

    return 1.0 + max_boost * math.exp(-age_days / half_life_days)


def get_recency_multiplier_for_metadata(metadata: Dict[str, Any]) -> float:
    """Resolve recency multiplier from request overrides and document metadata."""
    if not get_param("use_recency_boost", USE_RECENCY_BOOST):
        return 1.0

    max_age = int(get_param("recency_boost_max_age_days", RECENCY_BOOST_MAX_AGE_DAYS))
    half_life = int(get_param("recency_half_life_days", RECENCY_HALF_LIFE_DAYS))
    max_boost = float(get_param("recency_max_boost", RECENCY_MAX_BOOST))
    publish_date = metadata.get("date") or metadata.get("publish_date")

    return recency_multiplier(
        publish_date,
        max_age_days=max_age,
        half_life_days=half_life,
        max_boost=max_boost,
    )
