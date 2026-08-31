#!/usr/bin/env python3
"""
Ingest ALL WordPress topics via the chunking API.

Usage:
    .venv/bin/python3 scripts/ingest_topics.py
    .venv/bin/python3 scripts/ingest_topics.py --site diplomacy.edu
    .venv/bin/python3 scripts/ingest_topics.py --skip-existing
    .venv/bin/python3 scripts/ingest_topics.py --topic digital-diplomacy
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/opt/humainism_ai_chatbot_api")
os.chdir("/opt/humainism_ai_chatbot_api")

from app.services.ingest_service import ingest_single_topic, ingest_all_topics


def main():
    parser = argparse.ArgumentParser(description="Ingest WordPress topics via chunking API")
    parser.add_argument("--site", default="diplomacy.edu", help="Site to ingest")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip topics already in Weaviate (incremental mode)",
    )
    parser.add_argument(
        "--topic", type=str, default=None,
        help="Ingest a single topic by slug (e.g. 'digital-diplomacy')",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  TOPIC INGEST (contextual)")
    print("=" * 60)
    print(f"  Site: {args.site}")
    print(f"  Mode: {'single: ' + args.topic if args.topic else 'incremental' if args.skip_existing else 'full reingest'}")

    if args.topic:
        result = ingest_single_topic(topic_slug=args.topic, site=args.site)
    else:
        result = ingest_all_topics(site=args.site, skip_existing=args.skip_existing)

    print(f"\n{'=' * 60}\n  DONE\n{'=' * 60}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
