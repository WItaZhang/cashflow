"""Inference graph: shared features, learned routing, and weighted expert scores."""

from __future__ import annotations

from theme import BLUE, MUTED, PALE_BLUE, arrow, box, canvas, junction, line, summation, text

from cashflow_ml.features.sets import FEATURE_SETS


def build_figure():
    fig, ax = canvas()

    # Input schemas, not example records. Balance is a required model input.
    box(ax, 2, 20, 17, 6, "Transactions", size=10.6)
    box(ax, 2, 30, 17, 6, "Balance", size=10.6)
    line(ax, [(19, 23), (22, 23), (22, 33), (19, 33)])
    junction(ax, 22, 28)
    arrow(ax, [(22, 28), (26, 28)])

    count = len(FEATURE_SETS["cashflow_56"])
    box(ax, 26, 21.5, 23, 13, "Cashflow feature\nextraction", size=11.2)
    text(ax, 37.5, 39.4, f"{count} features", size=10, color=MUTED)
    arrow(ax, [(49, 28), (55, 28)])
    text(ax, 52, 25.4, r"$\mathbf{x}$", size=13)

    # One feature vector feeds all four models. Explicit dots denote fan-out.
    line(ax, [(55, 6.5), (55, 37)])
    for y in (6.5, 19, 28, 37):
        junction(ax, 55, y)
    arrow(ax, [(55, 6.5), (62, 6.5)])
    box(ax, 62, 3, 23, 7, "LightGBM router", color=BLUE, fill=PALE_BLUE, size=11)

    text(ax, 73.5, 13.7, "LightGBM experts", size=10.2, color=MUTED)
    for index, y in enumerate((19, 28, 37), start=1):
        box(ax, 62, y - 3, 23, 6, rf"C0{index} expert $f_{index}$", size=11)
        arrow(ax, [(55, y), (62, y)])
        end = {1: (111.5, 25.65), 2: (110.4, 28), 3: (111.5, 30.35)}[index]
        arrow(ax, [(85, y), (98, y), end])
        text(ax, 92, y - 2.25, rf"$p_{index}(\mathbf{{x}})$", size=11.5)

    # The router contributes weights, never an additional risk prediction.
    arrow(ax, [(85, 6.5), (113.5, 6.5), (113.5, 24.9)], color=BLUE, width=1.15)
    text(ax, 99.5, 3.7, r"$\mathbf{r}(\mathbf{x})$", size=12, color=BLUE)
    text(ax, 116, 15, "Routing\nweights", size=10, color=BLUE, ha="left")
    summation(ax, 113.5, 28, radius=3.1)
    text(ax, 113.5, 34.8, "Weighted sum", size=10.5)
    text(ax, 113.5, 39.5, r"$\hat{p}=\sum_{c=1}^{3}r_c\,p_c$", size=12)

    arrow(ax, [(116.6, 28), (128.5, 28)])
    text(ax, 123, 25.4, r"$\hat{p}$", size=13)
    text(ax, 135.5, 28, "Credit-risk\nscore", size=11.5)
    return fig
