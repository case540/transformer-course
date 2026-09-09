"""Fast structural and syntax checks; does not train models."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
files = sorted((ROOT / "notebooks").glob("*.ipynb"))
assert len(files) == 3, f"Expected three notebooks, found {len(files)}"


def source_text(cell):
    """Jupyter may serialize cell source as one string or a list of lines."""
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source

for path in files:
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    markdown = "\n".join(source_text(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    for phrase in ("Answer key", "Exercise", "References", "test"):
        assert phrase.lower() in markdown.lower(), f"{path.name}: missing {phrase}"
    exercises = sum("exercise" in c.get("metadata", {}).get("tags", []) for c in nb["cells"])
    checks = sum("check" in c.get("metadata", {}).get("tags", []) for c in nb["cells"])
    assert exercises >= 5 and checks == exercises
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code" or "skip-validation" in cell.get("metadata", {}).get("tags", []):
            continue
        # Learners are encouraged to add scratch cells. Validate generated lesson
        # cells while allowing an in-progress scratch cell to contain incomplete code.
        if not str(cell.get("id", "")).startswith("cell-"):
            continue
        try:
            ast.parse(source_text(cell))
        except SyntaxError as exc:
            raise AssertionError(f"{path.name} cell {index} has invalid Python: {exc}") from exc
    print(f"PASS {path.name}: {len(nb['cells'])} cells, {exercises} exercises")
