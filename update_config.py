"""
Quick script to update training configuration to be much faster
"""
import json
from pathlib import Path

# Update notebook config
notebook_path = Path('physionet_training.ipynb')

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and update config cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'EXPERIMENT_CONFIG' in source and 'max_subjects' in source:
            # Update the source
            new_source = source.replace("'epochs': 50", "'epochs': 10")
            new_source = new_source.replace("'max_subjects': 20", "'max_subjects': 5")
            new_source = new_source.replace("'n_folds': 3", "'n_folds': 2")

            cell['source'] = new_source.split('\n')
            print("Updated config cell:")
            print("  - epochs: 50 -> 10")
            print("  - max_subjects: 20 -> 5")
            print("  - n_folds: 3 -> 2")
            break

# Save updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"\nNotebook updated: {notebook_path}")
print("\nRECOMMENDED: Restart Jupyter kernel and re-run from the top")
