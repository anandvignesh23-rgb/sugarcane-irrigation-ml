"""Create a self-contained HTML report from saved notebook outputs; no retraining."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def text(value):
    return ''.join(value) if isinstance(value, list) else value

def main():
    results = json.loads((ROOT / 'reports/metrics/notebook_execution.json').read_text())
    parts = ['''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Sugarcane irrigation — outputs of all four models</title>
<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#182333}h1,h2,h3{line-height:1.2}h2{border-top:3px solid #24756e;padding-top:24px;margin-top:48px}table{border-collapse:collapse;display:block;overflow:auto;margin:20px 0}th,td{padding:8px 12px;border:1px solid #dce3e8;text-align:left}th{background:#eef5f4}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:16px}a{color:#14685f}.output{overflow:auto;margin:20px 0}.note{background:#fff5dd;padding:18px} @media print{h2{break-before:page}img{break-inside:avoid}}</style>
<h1>Outputs of all four regression models</h1>
<p>AI-Based Irrigation Advisory System for Sugarcane. Results extracted from the executed notebooks; no values are simulated.</p>
<p><b>Data:</b> NIT Patna / SIDSS + mandatory NASA POWER. <b>Target:</b> drip_liters_per_day, configured unit liters/day. All models use the same data, features and 1,200-row fit / 300-row test split (seed 42).</p>
<p class="note">Target magnitudes require unit/provenance validation. NASA weather is constant across rows because dates are absent. Existing preprocessing clips inputs before splitting; identity and yield fields also limit deployment conclusions. R² is not accuracy.</p>
<h2>Model comparison</h2><table><tr><th>Model</th><th>Status</th><th>MAE (liters/day)</th><th>RMSE (liters/day)</th><th>R²</th></tr>''']
    for name, result in results.items():
        m = result['metrics']
        parts.append(f'<tr><td><a href="#{name}">{html.escape(result["model"])}</a></td><td>{result["execution"]}</td><td>{m["mae"]:,.2f}</td><td>{m["rmse"]:,.2f}</td><td>{m["r2"]:.6f}</td></tr>')
    parts.append('</table><p>XGBoost has the lowest observed test RMSE. This comparison does not establish performance on other farms or future observations.</p>')
    for name, result in results.items():
        nb = json.loads((ROOT / 'notebooks' / name).read_text())
        parts.append(f'<h2 id="{name}">{html.escape(result["model"])}</h2><p>Source: {name}</p>')
        for cell in nb['cells']:
            source = text(cell['source'])
            if cell['cell_type'] == 'markdown':
                if source.startswith('## '):
                    lines = source.split('\n', 1)
                    parts.append('<h3>'+html.escape(lines[0].lstrip('# '))+'</h3>')
                    if len(lines)>1: parts.append('<p>'+html.escape(lines[1]).replace('\n','<br>')+'</p>')
                continue
            for out in cell.get('outputs', []):
                assert out['output_type'] != 'error'
                data = out.get('data', {})
                if 'image/png' in data:
                    parts.append('<img alt="Model diagnostic or feature interpretation plot" src="data:image/png;base64,'+text(data['image/png']).replace('\n','')+'">')
                elif 'text/html' in data:
                    parts.append('<div class="output">'+text(data['text/html'])+'</div>')
                elif 'text/markdown' in data:
                    parts.append('<pre>'+html.escape(text(data['text/markdown']))+'</pre>')
                elif out['output_type'] == 'stream':
                    parts.append('<pre>'+html.escape(text(out['text']))+'</pre>')
                elif 'text/plain' in data:
                    parts.append('<pre>'+html.escape(text(data['text/plain']))+'</pre>')
    parts.append('</html>')
    output = ROOT / 'reports/model_outputs.html'
    output.write_text('\n'.join(parts), encoding='utf-8')
    print(output)

if __name__ == '__main__':
    main()
