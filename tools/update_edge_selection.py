import json
from pathlib import Path


def update_methods_in_notebook(nb_path: Path) -> bool:
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    changed = False

    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        src = cell.get("source", [])
        # Normalize to list of lines
        if isinstance(src, str):
            lines = src.splitlines(True)
        else:
            lines = list(src)

        text = "".join(lines)
        if "Determine which methods to use" not in text:
            continue

        # Current pattern has a gated-branch with only GATE+AS and an else with ES+AS.
        # We want ES and AS in ALL variants; for gated methods also include GATE.
        before = (
            "        # Determine which methods to use\n"
            "        if model_type in ['static', 'adaptive', 'halting'] and gate_values is not None:\n"
            "            methods = [('GATE', selector.gate_selection), ('AS', selector.aggregation_selection)]\n"
            "        else:\n"
            "            methods = [('ES', selector.edge_selection), ('AS', selector.aggregation_selection)]\n"
        )

        after = (
            "        # Determine which methods to use\n"
            "        # Always evaluate ES and AS; add GATE when available\n"
            "        methods = [('ES', selector.edge_selection), ('AS', selector.aggregation_selection)]\n"
            "        if model_type in ['static', 'adaptive', 'halting'] and gate_values is not None:\n"
            "            methods = [('GATE', selector.gate_selection)] + methods\n"
        )

        if before in text and after not in text:
            text = text.replace(before, after)
            # Reassign preserving list-of-lines structure
            new_lines = text.splitlines(True)
            cell["source"] = new_lines
            changed = True

    if changed:
        nb_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


if __name__ == "__main__":
    nb = Path("kaggle_notebooks/physionet-adaptive-gating-complete.ipynb")
    if not nb.exists():
        raise SystemExit("Notebook not found: " + str(nb))
    ok = update_methods_in_notebook(nb)
    print("Updated:" if ok else "No changes needed.")

