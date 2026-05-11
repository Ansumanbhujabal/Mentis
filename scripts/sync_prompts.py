"""Push in-repo Jinja prompt templates into Langfuse Prompt Registry.

For each file matching prompts/{name}.{version}.j2:
  - Create or update the prompt in Langfuse with `name`
  - Set version label = "production" so PromptRegistry.get() picks it up at runtime
  - Tag with the template type for filtering in Langfuse UI

Run: uv run python scripts/sync_prompts.py
Requires LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY + LANGFUSE_HOST in environment.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
FILE_RE = re.compile(r"^(?P<name>.+)\.(?P<version>v\d+)\.j2$")

# Tags per prompt name — for filtering and dashboard grouping in Langfuse UI
TAGS = {
    "query_planner":        ["mentis", "stage:plan",     "kind:text"],
    "section_synthesizer":  ["mentis", "stage:synth",    "kind:text", "citation-grounded"],
    "executive_summary":    ["mentis", "stage:assemble", "kind:text"],
    "safety_reframe":       ["mentis", "stage:safety",   "kind:reframe"],
}


def main() -> int:
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        print("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set in environment", file=sys.stderr)
        return 1

    lf = Langfuse()

    files = sorted(PROMPTS_DIR.glob("*.j2"))
    if not files:
        print(f"No .j2 files in {PROMPTS_DIR}", file=sys.stderr)
        return 1

    print(f"Syncing {len(files)} prompt(s) from {PROMPTS_DIR} -> {os.environ.get('LANGFUSE_HOST')}\n")

    for path in files:
        m = FILE_RE.match(path.name)
        if not m:
            print(f"  - skip {path.name} (does not match {{name}}.v{{N}}.j2)")
            continue
        name = m.group("name")
        version = m.group("version")
        text = path.read_text()
        tags = TAGS.get(name, ["mentis"])

        try:
            created = lf.create_prompt(
                name=name,
                prompt=text,
                labels=["production"],
                tags=tags,
                type="text",
            )
            print(f"  OK {name} ({version}, {len(text)} chars) -> langfuse v{created.version} tags={tags}")
        except Exception as e:
            print(f"  FAIL {name}: {e!r}")
            continue

    lf.flush()
    print("\nDone. Verify at https://us.cloud.langfuse.com -> Prompts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
