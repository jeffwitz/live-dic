"""Draw the actual 17-pixel crack-tip supports used by the geometry regression."""

from pathlib import Path

import numpy as np
from lucas_kanade_icgn.roi import prepare_roi, unpack_support

mask = np.ones((17, 17), bool)
mask[:14, 7] = False
panels = []
for truncation in ("none", "visibility"):
    plan = prepare_roi(mask, 17, 17, policy="partial", truncation=truncation)
    panels.append(unpack_support(plan.support_words[0], 17))
parts = [
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 670 355" role="img">',
    "<title>Fixed connected support and optional visibility truncation</title>",
    (
        "<desc>A one-pixel slit ends inside a 17 by 17 square. Connectivity retains "
        "pixels across the tip. Visibility removes the pixels hidden from the centre.</desc>"
    ),
    '<rect width="670" height="355" fill="white"/>',
    '<g font-family="sans-serif" font-size="15" fill="#18334a">',
]
for offset, support, title in zip(
    (35, 370), panels, ("Connected: 275 pixels", "Visibility: 189 pixels"), strict=True
):
    parts.append(f'<text x="{offset}" y="25">{title}</text>')
    for y in range(17):
        for x in range(17):
            color = (
                "#233745"
                if not mask[y, x]
                else "#489aba"
                if support[y, x]
                else "#e6d3ae"
            )
            parts.append(
                f'<rect x="{offset + 15 * x}" y="{40 + 15 * y}" width="14" height="14" '
                f'fill="{color}"/>'
            )
    parts.append(f'<circle cx="{offset + 127}" cy="167" r="5" fill="#cb3040"/>')
parts.extend(
    [
        '<text x="35" y="325">Blue: retained; sand: removed; dark: slit; red: centre.</text>',
        '<text x="35" y="347">Preparation only; nominal grid and centre remain unchanged.</text>',
        "</g></svg>",
    ]
)
output = Path(__file__).resolve().parents[3] / "docs/_static/partial-supports.svg"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(parts), encoding="utf-8")
print(output)
