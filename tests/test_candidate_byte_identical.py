"""Semantic regression net for the historical Candidate/Fill fixtures.

The snapshots protect appearance and meaning, not incidental SVG spelling.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from vectormark import Options, idealize
from vectormark.color import mean_delta_e
from tests._render import disk, paint, render_svg, ssim
from tests.test_acceptance_annulus import _ring
from tests.test_acceptance_occlusion import _synthetic_mastercard
from tests.test_acceptance_gradient import _linear_img, _radial_img
from tests.test_acceptance_smooth_gradient import _smooth_linear_rect, _rotate_img

GOLDEN = Path(__file__).parent / "fixtures" / "golden"
DAIKONIC = Path(__file__).parent / "fixtures" / "daikonic" / "source.png"

# Historical references establish the accepted per-fixture source baseline.
# The direct-reference budgets allow sub-pixel outline evolution while staying
# far tighter than ordinary product-level fidelity gates.
REFERENCE_SSIM_FLOOR = 0.995
REFERENCE_DELTA_E_CEILING = 0.005
SOURCE_SSIM_DRIFT_BUDGET = 0.005
SOURCE_DELTA_E_DRIFT_BUDGET = 0.005


def _daikonic_icon():
    return np.asarray(Image.open(DAIKONIC).convert("RGB"), dtype=np.uint8)[:392]


def _ring_disk():
    h, w = 160, 200
    return paint([(_ring(70, 70, 45, 25, h, w), (37, 99, 235)),
                  (disk(135, 70, 38, h, w), (220, 30, 30))], h, w)


def _linear_grad():
    return _linear_img(200, 260, (40, 100), (220, 100),
                       [(0.0, (37, 99, 235)), (1.0, (219, 39, 119))])


def _radial_grad():
    return _radial_img(220, 220, (110, 110), 90,
                       [(0.0, (125, 211, 252)), (1.0, (29, 78, 216))])


def _rectified_grad():
    base = _smooth_linear_rect(160, 240, 40, 200, (85, 145, 225), (70, 125, 210))
    return _rotate_img(base, 30)


# (name, image factory, Options) — covers every source path x both modes + rectified.
CASES = [
    ("daikonic", _daikonic_icon, Options()),
    ("daikonic_flatten", _daikonic_icon, Options(flatten=True)),
    ("mastercard", lambda: _synthetic_mastercard()[0], Options()),
    ("mastercard_flatten", lambda: _synthetic_mastercard()[0], Options(flatten=True)),
    ("ring_disk", _ring_disk, Options()),
    ("ring_disk_flatten", _ring_disk, Options(flatten=True)),
    ("linear_grad", _linear_grad, Options()),
    ("linear_grad_flatten", _linear_grad, Options(flatten=True)),
    ("radial_grad", _radial_grad, Options()),
    ("rectified_grad", _rectified_grad, Options()),
    ("rectified_grad_flatten", _rectified_grad, Options(flatten=True)),
]


def _case(name):
    return next((f, o) for n, f, o in CASES if n == name)


@lru_cache(maxsize=None)
def _render(name):
    factory, opts = _case(name)
    return idealize(factory(), options=opts)


def _mask_topology(mask):
    """Count meaningful connected components and every enclosed hole component."""
    labels, count = ndimage.label(mask)
    components = [labels == i for i in range(1, count + 1)]
    components = [component for component in components if component.sum() >= 4]
    holes = 0
    for component in components:
        enclosed = ndimage.binary_fill_holes(component) & ~component
        hole_labels, hole_count = ndimage.label(enclosed)
        holes += sum((hole_labels == i).sum() >= 4 for i in range(1, hole_count + 1))
    return len(components), int(holes)


def _visible_paint_topology(raster, reference=None):
    """Renderer-normalized visible paint ownership and topology.

    The reference render supplies paint anchors, so CSS, inheritance, clone
    encoding, color spelling, and gradient definition details are irrelevant.
    """
    reference = raster if reference is None else reference
    ref_rgb = reference.astype(np.int16)
    rgb = raster.astype(np.int16)
    foreground = np.max(np.abs(ref_rgb - 255), axis=2) > 16
    colors, counts = np.unique(reference[foreground].reshape(-1, 3), axis=0, return_counts=True)
    dominant = counts >= max(32, int(foreground.sum() * 0.02))
    colors, counts = colors[dominant], counts[dominant]
    anchors = colors[np.argsort(counts)[::-1][:16]].astype(np.int16)
    if not len(anchors):
        return {"foreground": (0, 0), "paints": (), "unexpected": 0}
    distances = np.max(np.abs(rgb[:, :, None, :] - anchors[None, None, :, :]), axis=3)
    owner = np.argmin(distances, axis=2)
    supported = np.min(distances, axis=2) <= 32
    candidate_foreground = np.max(np.abs(rgb - 255), axis=2) > 16
    paint_topology = tuple(
        _mask_topology(candidate_foreground & supported & (owner == i))
        for i in range(len(anchors))
    )
    unexpected = int((candidate_foreground & (np.min(distances, axis=2) > 32)).sum())
    return {
        "foreground": _mask_topology(candidate_foreground),
        "paints": paint_topology,
        "unexpected": unexpected,
    }


def _semantic_issues(candidate_svg, reference_svg, source):
    """Compare fidelity, paint, topology, and gradient relationships."""
    height, width = source.shape[:2]
    candidate = render_svg(candidate_svg, width, height)
    reference = render_svg(reference_svg, width, height)
    issues = []

    if ssim(candidate, reference) < REFERENCE_SSIM_FLOOR:
        issues.append("reference SSIM below budget")
    if mean_delta_e(candidate, reference) > REFERENCE_DELTA_E_CEILING:
        issues.append("reference mean Delta-E above budget")
    if ssim(candidate, source) < ssim(reference, source) - SOURCE_SSIM_DRIFT_BUDGET:
        issues.append("source SSIM regressed beyond the historical budget")
    if mean_delta_e(candidate, source) > mean_delta_e(reference, source) + SOURCE_DELTA_E_DRIFT_BUDGET:
        issues.append("source mean Delta-E regressed beyond the historical budget")
    reference_topology = _visible_paint_topology(reference)
    candidate_topology = _visible_paint_topology(candidate, reference)
    if candidate_topology["foreground"] != reference_topology["foreground"]:
        issues.append("visible foreground topology changed")
    if candidate_topology["paints"] != reference_topology["paints"]:
        issues.append("visible paint topology changed")
    unexpected_budget = max(4, int(np.prod(reference.shape[:2]) * 0.001))
    if abs(candidate_topology["unexpected"] - reference_topology["unexpected"]) > unexpected_budget:
        issues.append("unexpected visible paint changed")
    return issues


@pytest.mark.parametrize("name", [c[0] for c in CASES])
def test_candidate_semantics_match_historical_reference(name):
    factory, _ = _case(name)
    golden = (GOLDEN / f"{name}.svg").read_text()
    assert not _semantic_issues(_render(name), golden, factory()), name


@pytest.mark.parametrize(
    ("candidate_body", "expected_issue"),
    [
        ('<circle fill="#123456" cx="20" cy="20" r="10"/>'
         '<path fill="#ABCDEF" fill-rule="evenodd" '
         'd="M40 20a10 10 0 1 0 20 0a10 10 0 1 0-20 0 '
         'M46 20a4 4 0 1 0 8 0a4 4 0 1 0-8 0"/>',
         "visible paint topology changed"),
        ('<circle fill="#654321" cx="20" cy="20" r="10"/>'
         '<circle fill="#123456" cx="80" cy="20" r="10"/>'
         '<path fill="#ABCDEF" fill-rule="evenodd" '
         'd="M40 20a10 10 0 1 0 20 0a10 10 0 1 0-20 0 '
         'M46 20a4 4 0 1 0 8 0a4 4 0 1 0-8 0"/>',
         "unexpected visible paint changed"),
        ('<circle fill="#123456" cx="20" cy="20" r="16"/>'
         '<circle fill="#123456" cx="80" cy="20" r="10"/>'
         '<path fill="#ABCDEF" fill-rule="evenodd" '
         'd="M40 20a10 10 0 1 0 20 0a10 10 0 1 0-20 0 '
         'M46 20a4 4 0 1 0 8 0a4 4 0 1 0-8 0"/>',
         "reference SSIM below budget"),
        ('<circle fill="#123456" cx="20" cy="20" r="10"/>'
         '<circle fill="#123456" cx="80" cy="20" r="10"/>'
         '<circle fill="#ABCDEF" cx="50" cy="20" r="10"/>',
         "visible paint topology changed"),
    ],
    ids=["lost-instance", "wrong-fill", "distorted-outline", "missing-hole"],
)
def test_semantic_comparison_negative_controls(candidate_body, expected_issue):
    # The third shape is an even-odd ring.  Encoding the same outer circle as a
    # filled circle is the deliberately missing-hole control.
    reference_body = ('<circle fill="#123456" cx="20" cy="20" r="10"/>'
                      '<circle fill="#123456" cx="80" cy="20" r="10"/>'
                      '<path fill="#ABCDEF" fill-rule="evenodd" '
                      'd="M40 20a10 10 0 1 0 20 0a10 10 0 1 0-20 0 '
                      'M46 20a4 4 0 1 0 8 0a4 4 0 1 0-8 0"/>')
    wrapper = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="40">{}</svg>'
    reference = wrapper.format(reference_body)
    candidate = wrapper.format(candidate_body)
    source = render_svg(reference, 100, 40)
    assert expected_issue in _semantic_issues(candidate, reference, source)


@pytest.mark.parametrize(
    ("reference_body", "candidate_body"),
    [
        ('<circle fill="#123456" cx="20" cy="20" r="10"/>',
         '<path fill="rgb(18,52,86)" d="M10 20a10 10 0 1 0 20 0a10 10 0 1 0-20 0"/>'),
        ('<defs><circle id="dot" cx="20" cy="20" r="10"/></defs>'
         '<g fill="#123456"><use href="#dot"/><use href="#dot" x="40"/></g>',
         '<circle fill="#123456" cx="20" cy="20" r="10"/>'
         '<circle fill="#123456" cx="60" cy="20" r="10"/>'),
        ('<style>.paint{fill:#123456}</style><circle class="paint" cx="20" cy="20" r="10"/>',
         '<g style="fill:rgb(18, 52, 86)"><circle cx="20" cy="20" r="10"/></g>'),
        ('<defs><linearGradient id="g"><stop offset="0" stop-color="#000"/>'
         '<stop offset="1" stop-color="#fff"/></linearGradient></defs>'
         '<rect width="100" height="40" fill="url(#g)"/>',
         '<defs><linearGradient id="other"><stop offset="0" stop-color="rgb(0,0,0)"/>'
         '<stop offset=".5" stop-color="#808080"/><stop offset="1" stop-color="#fff"/>'
         '</linearGradient></defs><rect width="100" height="40" fill="url(#other)"/>'),
    ],
    ids=["primitive-path-paint-spelling", "clone-expanded", "inherited-css", "gradient-redundant-stop"],
)
def test_semantic_comparison_accepts_equivalent_svg_encodings(reference_body, candidate_body):
    wrapper = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="40">{}</svg>'
    reference = wrapper.format(reference_body)
    candidate = wrapper.format(candidate_body)
    source = render_svg(reference, 100, 40)
    assert not _semantic_issues(candidate, reference, source)


def test_visible_topology_counts_each_enclosed_hole_component():
    two_holes = np.zeros((30, 50), dtype=bool)
    two_holes[3:27, 3:47] = True
    two_holes[8:14, 10:16] = False
    two_holes[16:22, 32:38] = False
    one_hole = two_holes.copy()
    one_hole[16:22, 32:38] = True
    assert _mask_topology(two_holes) == (1, 2)
    assert _mask_topology(one_hole) == (1, 1)


def test_paint_membership_does_not_assign_navy_antialiasing_to_teal():
    wrapper = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50">{}</svg>'
    reference = render_svg(
        wrapper.format(
            '<circle fill="#062337" cx="25" cy="25" r="20"/>'
            '<circle fill="#3EA89E" cx="70" cy="25" r="12"/>'
        ),
        100,
        50,
    )
    topology = _visible_paint_topology(reference)
    assert sorted(topology["paints"]) == [(1, 0), (1, 0)]

    extra_teal = render_svg(
        wrapper.format(
            '<circle fill="#062337" cx="25" cy="25" r="20"/>'
            '<circle fill="#3EA89E" cx="70" cy="25" r="12"/>'
            '<circle fill="#3EA89E" cx="94" cy="8" r="3"/>'
        ),
        100,
        50,
    )
    assert sorted(_visible_paint_topology(extra_teal, reference)["paints"]) == [(1, 0), (2, 0)]


def test_serialization_is_deterministic_independently_of_historical_spelling():
    factory, opts = _case("mastercard")
    assert idealize(factory(), options=opts) == _render("mastercard")
