import json
import os

nb_path = "Signals/notebooks/7_complete_methods_comparison.ipynb"
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}\n")

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])
        print(f"Cell {i} (CODE, lines {len(cell['source'])}):")
        print(source_text[:200] if len(source_text) > 200 else source_text)
        print("\n" + "="*80 + "\n")
