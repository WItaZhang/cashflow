"""Render paper-style method figures without reading any consumer data."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from architecture import build_figure as architecture
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BUILDERS = {
    "architecture": architecture,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/figures/paper.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Cashflow Credit Risk",
        "Author": "WItaZhang",
        "Subject": "Soft-routing LightGBM mixture of experts",
        "CreationDate": None,
        "ModDate": None,
    }
    with PdfPages(output / config["pdf_filename"], metadata=metadata) as pdf:
        for name in config["figures"]:
            fig = BUILDERS[name]()
            svg = output / f"{name}.svg"
            fig.savefig(svg, metadata={"Date": None})
            # Matplotlib adds spaces at line ends inside SVG path attributes.
            svg.write_text(
                "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            fig.savefig(output / f"{name}.png", dpi=config["raster_dpi"])
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Rendered {name}: SVG + PNG + PDF page")
    print(f"Vector figure collection: {output / config['pdf_filename']}")


if __name__ == "__main__":
    main()
