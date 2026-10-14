"""Execute the four teaching notebooks and refresh their final Markdown summaries."""
import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
results = {}
for path in sorted((ROOT / 'notebooks').glob('0[1-4]_*.ipynb')):
    nb = nbformat.read(path, as_version=4)
    nb.cells.append(nbformat.v4.new_code_cell('print("SUMMARY_JSON:" + json.dumps(summary))'))
    NotebookClient(nb, timeout=600, resources={'metadata': {'path': str(ROOT)}}).execute()
    capture = nb.cells.pop()
    raw = ''.join(o.get('text', '') for o in capture.outputs)
    s = json.loads(raw.split('SUMMARY_JSON:', 1)[1])
    m = s['metrics']
    nb.cells[-1].source = (f"## Final summary\n\n**Model:** {s['model']}  \n"
        f"**Test MAE:** {m['mae']:,.2f} liters/day  \n**Test RMSE:** {m['rmse']:,.2f} liters/day  \n"
        f"**Test R²:** {m['r2']:.6f}  \n**Most important features:** {s['top_features']}  \n"
        f"**Main strength:** {s['strength']}  \n**Main limitation:** {s['limitation']}\n\n"
        "This saved snapshot reflects the execution above; after manual reruns, the preceding dynamic summary is authoritative.")
    assert all(o.output_type != 'error' for cell in nb.cells if cell.cell_type == 'code' for o in cell.outputs)
    assert all(cell.execution_count is not None for cell in nb.cells if cell.cell_type == 'code')
    plots = sum('image/png' in o.get('data', {}) for cell in nb.cells if cell.cell_type == 'code' for o in cell.outputs)
    assert plots >= 2, f'Missing rendered plots: {path}'
    nbformat.validate(nb)
    nbformat.write(nb, path)
    results[path.name] = s | {'rendered_plots': plots, 'execution': 'PASS'}
    print(path.name, m, 'plots:', plots, flush=True)
(ROOT / 'reports/metrics/notebook_execution.json').write_text(json.dumps(results, indent=2))
