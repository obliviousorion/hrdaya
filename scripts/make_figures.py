"""Generate dependency-free SVG pipeline/confusion assets from a completed run."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def figures(run=None, root=ROOT):
    if run is None:
        runs = sorted((root / "results/baseline").glob("*/details.json"))
        if not runs:
            raise ValueError("Run make baseline first; detailed run evidence is needed")
        run = runs[-1].parent
    d = json.loads((Path(run) / "details.json").read_text())
    target = root / "report/figures"
    target.mkdir(parents=True, exist_ok=True)
    pipeline = '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="160" viewBox="0 0 1200 160">'
    labels = ["ECG-ID raw", "0.5–40 Hz / beats", "AC + DCT(32)", "Scale / SVM", "Rank-1 / cosine EER"]
    for i, label in enumerate(labels):
        x = i * 240
        pipeline += f'<rect x="{x+5}" y="40" width="220" height="70" fill="#e4eef9" stroke="#234"/>'
        pipeline += f'<text x="{x+115}" y="80" text-anchor="middle" font-size="16">{label}</text>'
        if i < 4:
            pipeline += f'<text x="{x+228}" y="80">→</text>'
    (target / "pipeline.svg").write_text(pipeline + '</svg>')
    matrix = d["confusion_matrix"]
    n = len(matrix)
    maximum = max(map(max, matrix)) or 1
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="950" viewBox="0 0 900 950">'
    svg += '<rect width="900" height="950" fill="white"/><text x="450" y="28" text-anchor="middle">Recording-disjoint confusion: true (rows), predicted (columns)</text>'
    cell = 760/n
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            shade = 255 - int(230 * value/maximum)
            svg += f'<rect x="{70+j*cell:.2f}" y="{55+i*cell:.2f}" width="{cell:.2f}" height="{cell:.2f}" fill="rgb({shade},{shade},255)"><title>{d["classes"][i]} → {d["classes"][j]}: {value}</title></rect>'
    svg += f'<text x="70" y="850">{n} identities; cell counts 0–{maximum}; dark blue = more beats</text>'
    svg += '<text x="70" y="880">Order: classes in run details.json. Source: '+Path(run).parent.name+'/'+Path(run).name+'</text></svg>'
    (target / "confusion.svg").write_text(svg)
    print(f"Figures written to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    figures(parser.parse_args().run)
