"""Fast structural and syntax checks; does not train models."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
files = sorted((ROOT / "notebooks").glob("*.ipynb"))
assert len(files) == 2, f"Expected two notebooks, found {len(files)}"

for path in files:
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    markdown = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "markdown")
    for phrase in ("Answer key", "Exercise", "References", "TensorBoard", "test"):
        assert phrase.lower() in markdown.lower(), f"{path.name}: missing {phrase}"
    exercises = sum("exercise" in c.get("metadata", {}).get("tags", []) for c in nb["cells"])
    checks = sum("check" in c.get("metadata", {}).get("tags", []) for c in nb["cells"])
    assert exercises >= 5 and checks == exercises
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code" or "skip-validation" in cell.get("metadata", {}).get("tags", []):
            continue
        try:
            ast.parse(cell["source"])
        except SyntaxError as exc:
            raise AssertionError(f"{path.name} cell {index} has invalid Python: {exc}") from exc
    print(f"PASS {path.name}: {len(nb['cells'])} cells, {exercises} exercises")
