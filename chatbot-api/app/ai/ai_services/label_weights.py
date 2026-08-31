"""
Profile-based label weights for re-scoring search results.

Each profile has its own label weights that adjust the relevance ranking
based on user type (General, Student, Diplomat, etc.).
"""

DEFAULT_LABEL_WEIGHT = 1.0

PROFILE_LABEL_WEIGHTS = {
    "general": {
        "Topic": 1.50,
        "Course": 1.42,
        "Blog": 1.33,
        "Event": 1.25,
        "Resources": 1.17,
        "Technologies": 1.17,
        "Actor": 1.08,
        "Updates": 1.04,
        "Diplonews": 1.04,
        "Newsletter": 1.04,
        "Post": 1.04,
        "People": 1.00,
    },
    "student": {
        "Course": 1.50,
        "Topic": 1.42,
        "Blog": 1.33,
        "Resources": 1.25,
        "Technologies": 1.25,
        "Actor": 1.17,
        "Event": 1.13,
        "Updates": 1.08,
        "Diplonews": 1.04,
        "Newsletter": 1.04,
        "Post": 1.04,
        "People": 1.00,
    },
    "diplomat": {
        "Topic": 1.50,
        "Blog": 1.42,
        "Updates": 1.33,
        "Resources": 1.25,
        "Technologies": 1.25,
        "Course": 1.17,
        "Actor": 1.08,
        "Event": 1.08,
        "Diplonews": 1.04,
        "Newsletter": 1.04,
        "Post": 1.04,
        "People": 1.00,
    },
    "researcher": {
        "Topic": 1.50,
        "Blog": 1.42,
        "Resources": 1.33,
        "Technologies": 1.33,
        "Updates": 1.25,
        "Course": 1.17,
        "Actor": 1.08,
        "Event": 1.08,
        "Diplonews": 1.04,
        "Newsletter": 1.04,
        "Post": 1.04,
        "People": 1.00,
    },
    "historian": {
        "History": 1.50,
        "Topic": 1.50,
        "Blog": 1.42,
        "Resources": 1.33,
        "Technologies": 1.33,
        "Updates": 1.25,
        "Course": 1.17,
        "Actor": 1.08,
        "Event": 1.08,
        "Diplonews": 1.04,
        "Newsletter": 1.04,
        "Post": 1.04,
        "People": 1.00,
    },
    "philosopher": {
        "Topic": 1.50,
        "Blog": 1.42,
        "Resources": 1.33,
        "Technologies": 1.33,
        "Updates": 1.25,
        "Course": 1.17,
        "Actor": 1.08,
        "Event": 1.08,
        "Diplonews": 1.04,
        "Newsletter": 1.04,
        "Post": 1.04,
        "People": 1.00,
    },
    "contrarian": {
        "Topic": 1.50,
        "Blog": 1.42,
        "Resources": 1.33,
        "Technologies": 1.33,
        "Updates": 1.25,
        "Course": 1.17,
        "Actor": 1.08,
        "Event": 1.08,
        "Diplonews": 1.04,
        "Newsletter": 1.04,
        "Post": 1.04,
        "People": 1.00,
    },
    "journalist": {
        "Updates": 1.50,
        "Blog": 1.42,
        "Topic": 1.33,
        "Resources": 1.25,
        "Technologies": 1.25,
        "Course": 1.17,
        "Actor": 1.08,
        "Event": 1.08,
        "Diplonews": 1.08,
        "Newsletter": 1.04,
        "Post": 1.04,
        "People": 1.00,
    },
}


POST_TYPE_TO_LABEL = {
    "updates": "Updates",
    "event": "Event",
    "blog": "Blog",
    "diplonews": "Diplonews",
    "people": "People",
    "course": "Course",
    "actor": "Actor",
    "newsletter": "Newsletter",
    "technologies": "Technologies",
    "post": "Post",
    "topic": "Topic",
    "resource": "Resources",
    "histories": "History",
}


def get_label_weights(user_type: str) -> dict:
    """Get label weights for a specific profile. Falls back to 'general' if not found."""
    profile = user_type.lower() if user_type else "general"
    return PROFILE_LABEL_WEIGHTS.get(profile, PROFILE_LABEL_WEIGHTS["general"])
