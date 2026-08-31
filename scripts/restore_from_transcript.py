"""Rebuild oneweaviate-adapted source files from the recorded agent transcript.

Some files in this tree were overwritten with the chatbot-api-dev versions.
The prior session's transcript holds the exact Write / StrReplace payloads, so
replaying them in order reproduces the adapted sources.
"""

import json
import shutil
import sys
from pathlib import Path

TRANSCRIPT = Path(
    "/root/.cursor/projects/opt-dev-mydepartment-api-dev/agent-transcripts/"
    "b0486a4e-c547-4fa8-9176-07e7d67593cc/b0486a4e-c547-4fa8-9176-07e7d67593cc.jsonl"
)

TARGETS = [
    "app/core/collections.py",
    "app/ai/ai_services/retrievers/utils.py",
    "app/ai/ai_services/retrievers/twophase.py",
    "app/ai/ai_services/retrievers/label_rescoring.py",
    "app/core/config.py",
    "app/schemas/chat_schema.py",
    "app/services/person_catalog_service.py",
    "app/services/metadata_catalog_service.py",
]

ROOT = Path("/opt/dev/mydepartment-api-dev/chatbot-api")


def collect_ops():
    """Ordered (target, tool_name, input) tuples for the files we restore."""
    ops = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "tool_use" and node.get("name") in ("Write", "StrReplace"):
                inp = node.get("input") or {}
                path = str(inp.get("path") or "")
                for target in TARGETS:
                    if path.endswith(target):
                        ops.append((target, node["name"], inp))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    with TRANSCRIPT.open() as handle:
        for line in handle:
            try:
                walk(json.loads(line))
            except json.JSONDecodeError:
                continue
    return ops


def main(apply: bool, only: list[str]) -> int:
    ops = [op for op in collect_ops() if not only or any(o in op[0] for o in only)]
    if not ops:
        print("No recorded operations found — aborting.")
        return 1

    per_target = {}
    for target, name, inp in ops:
        per_target.setdefault(target, []).append((name, inp))

    failures = []
    for target, target_ops in per_target.items():
        dest = ROOT / target
        text = dest.read_text() if dest.exists() else ""
        print(f"\n=== {target} ({len(target_ops)} ops, {len(text)} bytes on disk)")

        for idx, (name, inp) in enumerate(target_ops, 1):
            if name == "Write":
                text = inp.get("contents") or ""
                print(f"  {idx}. Write → {len(text)} bytes")
                continue

            old = inp.get("old_string") or ""
            new = inp.get("new_string") or ""
            count = text.count(old)
            if count == 1 or (count > 1 and inp.get("replace_all")):
                text = text.replace(old, new)
                print(f"  {idx}. StrReplace ok ({count} match)")
            elif new and new in text:
                print(f"  {idx}. StrReplace already applied — skipped")
            else:
                snippet = old.strip().splitlines()[0][:70] if old.strip() else "<empty>"
                print(f"  {idx}. StrReplace FAILED ({count} matches): {snippet}")
                failures.append((target, idx, snippet))

        if apply:
            if dest.exists():
                shutil.copy2(dest, dest.with_suffix(dest.suffix + ".reverted.bak"))
            dest.write_text(text)
            print(f"  written → {dest} ({len(text)} bytes)")

    if failures:
        print(f"\n{len(failures)} operation(s) could not be applied:")
        for target, idx, snippet in failures:
            print(f"  {target} op {idx}: {snippet}")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(main(apply="--apply" in sys.argv, only=args))
