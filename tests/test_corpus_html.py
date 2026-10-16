import numpy as np
from PIL import Image

import scripts.generate_corpus_html as corpus_html
from scripts.generate_corpus_html import CorpusEntry, generate_corpus_html
from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark import Options
from vectormark.optimizer import trace as trace_module
from vectormark.optimizer.vector_region import VectorRegion, to_polygon


def _tiny_disk():
    img = np.full((48, 48, 3), 255, np.uint8)
    yy, xx = np.ogrid[:48, :48]
    img[((yy - 24) ** 2 + (xx - 24) ** 2) <= 12 * 12] = (200, 30, 30)
    return img


def test_generate_corpus_html_writes_render_svg_and_diagnostics(tmp_path):
    index = generate_corpus_html(
        tmp_path / "corpus",
        [
            CorpusEntry(
                "tiny_disk",
                "optimizer",
                _tiny_disk,
                Options(optimizer=True),
            )
        ],
    )

    html = index.read_text()
    svg = (index.parent / "svg" / "optimizer-tiny_disk.svg").read_text()
    png = index.parent / "rendered-input" / "optimizer-tiny_disk.png"
    raw = index.parent / "raw" / "optimizer-tiny_disk.svg.txt"
    diagnostics = index.parent / "diagnostics" / "optimizer-tiny_disk.json"
    diagnostics_summary = index.parent / "diagnostic-summary" / "optimizer-tiny_disk.html"
    options = index.parent / "options" / "optimizer-tiny_disk.json"

    assert "optimizer / tiny_disk" in html
    assert '<img src="rendered-input/optimizer-tiny_disk.png"' in html
    assert '<object data="svg/optimizer-tiny_disk.svg"' in html
    assert "<summary>SVG</summary>" in html
    assert "<summary>Diagnostics</summary>" in html
    assert 'src="raw/optimizer-tiny_disk.svg.txt"' in html
    assert 'src="diagnostic-summary/optimizer-tiny_disk.html"' in html
    assert 'src="diagnostics/optimizer-tiny_disk.json"' in html
    assert 'src="options/optimizer-tiny_disk.json"' in html
    assert png.exists()
    assert svg.startswith("<svg ")
    assert raw.read_text() == svg
    assert '"elements"' in diagnostics.read_text()
    assert "Optimizer Objects" in diagnostics_summary.read_text()
    assert "<td>circle</td>" in diagnostics_summary.read_text()
    assert '"optimizer": true' in options.read_text()
    assert '"epsilon": 1.0' in options.read_text()
    assert '"max_error": 1.0' in options.read_text()


def test_generate_corpus_html_can_opt_into_cubic_paths(tmp_path):
    index = generate_corpus_html(
        tmp_path / "corpus",
        [CorpusEntry("tiny_disk", "optimizer", _tiny_disk, Options(optimizer=True))],
        cubic_paths=True,
    )

    options = index.parent / "options" / "optimizer-tiny_disk.json"

    assert '"cubic_paths": true' in options.read_text()


def test_generate_corpus_html_can_use_mcp_trace_drawing_auto_workflow(tmp_path):
    output = tmp_path / "corpus"
    index = generate_corpus_html(
        output,
        [CorpusEntry("tiny_disk", "current", _tiny_disk, Options())],
        workflow="trace_auto",
    )

    svg = (output / "svg" / "trace_auto-tiny_disk.svg").read_text()
    diagnostics = (output / "diagnostics" / "trace_auto-tiny_disk.json").read_text()
    options = (output / "options" / "trace_auto-tiny_disk.json").read_text()

    assert "trace_auto / tiny_disk" in index.read_text()
    assert svg.startswith("<svg ")
    assert '"workflow": "trace_drawing_auto"' in diagnostics
    assert '"refine": "auto"' in options
    assert any((output / "cache").glob("drawing-trace-tiny_disk-*.pkl"))


def test_trace_auto_cache_reuses_trace_roots_but_reruns_auto_refinement(tmp_path, monkeypatch):
    output = tmp_path / "corpus"
    entry = CorpusEntry("tiny_disk", "current", _tiny_disk, Options())
    trace_calls = 0
    refine_calls = 0
    retained_evidence = []
    original_trace = corpus_html._trace_result
    original_refine = corpus_html.auto_refine

    def trace_once(*args, **kwargs):
        nonlocal trace_calls
        trace_calls += 1
        return original_trace(*args, **kwargs)

    def refine_each_run(*args, **kwargs):
        nonlocal refine_calls
        refine_calls += 1
        retained_evidence.append(args[0].root_path_evidence)
        return original_refine(*args, **kwargs)

    monkeypatch.setattr(corpus_html, "_trace_result", trace_once)
    monkeypatch.setattr(corpus_html, "auto_refine", refine_each_run)

    generate_corpus_html(output, [entry], workflow="trace_auto")
    generate_corpus_html(output, [entry], workflow="trace_auto")

    assert trace_calls == 1
    assert refine_calls == 2
    assert retained_evidence[0]
    assert retained_evidence[1] == retained_evidence[0]
    assert retained_evidence[1][0] is not retained_evidence[0][0]


def test_generate_corpus_html_can_override_corpus_tolerances(tmp_path):
    index = generate_corpus_html(
        tmp_path / "corpus",
        [CorpusEntry("tiny_disk", "optimizer", _tiny_disk, Options(optimizer=True))],
        epsilon=0.75,
        max_error=0.5,
    )

    options = (index.parent / "options" / "optimizer-tiny_disk.json").read_text()

    assert '"epsilon": 0.75' in options
    assert '"max_error": 0.5' in options


def test_generate_corpus_html_reuses_optimizer_trace_cache(tmp_path, monkeypatch):
    trace_calls = 0

    def _fake_trace_regions(arr, opt):
        nonlocal trace_calls
        trace_calls += 1
        shape = Shape("circle", {"cx": 24.0, "cy": 24.0, "r": 12.0})
        region = VectorRegion(
            0,
            shape,
            FlatFill("#c81e1e"),
            0,
            footprint=to_polygon(shape),
            raster=np.any(arr != 255, axis=2),
        )
        return [region], {0: region.raster}

    monkeypatch.setattr(trace_module, "trace_regions", _fake_trace_regions)
    output = tmp_path / "corpus"
    entries = [CorpusEntry("tiny_disk", "optimizer", _tiny_disk, Options(optimizer=True))]

    generate_corpus_html(output, entries)
    generate_corpus_html(output, entries, only=["tiny_disk"])

    assert trace_calls == 1
    assert any((output / "cache").glob("trace-tiny_disk-*.pkl"))


def test_generate_corpus_html_retraces_when_trace_options_change(tmp_path, monkeypatch):
    trace_calls = 0

    def _fake_trace_regions(arr, opt):
        nonlocal trace_calls
        trace_calls += 1
        shape = Shape("circle", {"cx": 24.0, "cy": 24.0, "r": 12.0})
        region = VectorRegion(
            0,
            shape,
            FlatFill("#c81e1e"),
            0,
            footprint=to_polygon(shape),
            raster=np.any(arr != 255, axis=2),
        )
        return [region], {0: region.raster}

    monkeypatch.setattr(trace_module, "trace_regions", _fake_trace_regions)
    output = tmp_path / "corpus"
    entries = [CorpusEntry("tiny_disk", "optimizer", _tiny_disk, Options(optimizer=True))]

    generate_corpus_html(output, entries)
    generate_corpus_html(output, entries, epsilon=0.75)

    assert trace_calls == 2


def test_corpus_image_factory_composites_alpha_on_white(tmp_path):
    rgba = np.zeros((3, 3, 4), dtype=np.uint8)
    rgba[1, 1] = (200, 30, 30, 255)
    path = tmp_path / "transparent.png"
    Image.fromarray(rgba, "RGBA").save(path)

    image = corpus_html._image_factory(path)()

    assert image.shape == (3, 3, 3)
    assert tuple(image[0, 0]) == (255, 255, 255)
    assert tuple(image[1, 1]) == (200, 30, 30)


def test_source_manifest_entries_download_only_rendered_items(tmp_path):
    source_a = tmp_path / "a.png"
    source_b = tmp_path / "b.png"
    Image.fromarray(_tiny_disk()).save(source_a)
    Image.fromarray(_tiny_disk()).save(source_b)
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        """
        {
          "entries": [
            {"name": "a", "url": "%s"},
            {"name": "b", "url": "%s"}
          ]
        }
        """ % (source_a.as_uri(), source_b.as_uri())
    )
    cache = tmp_path / "cache"
    entries = corpus_html.source_manifest_entries(manifest, cache)

    index = generate_corpus_html(tmp_path / "corpus", entries, only=["a"])

    assert len(entries) == 4
    assert (cache / "a.png").exists()
    assert not (cache / "b.png").exists()
    assert "current / a" in index.read_text()
    assert "current / b" in index.read_text()


def test_default_entries_use_corpus_input_and_include_vbird(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus" / "input"
    legacy = tmp_path / "scratch" / "real-logos"
    corpus.mkdir(parents=True)
    legacy.mkdir(parents=True)
    Image.fromarray(_tiny_disk()).save(corpus / "icloud.png")
    Image.fromarray(_tiny_disk()).save(corpus / "vbird.png")
    Image.fromarray(_tiny_disk()).save(corpus / "cmp_icloud.png")
    Image.fromarray(_tiny_disk()).save(legacy / "legacy_only.png")

    monkeypatch.setattr(corpus_html, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(corpus_html, "DEFAULT_CORPUS_INPUT", corpus)
    monkeypatch.setattr(corpus_html, "LEGACY_CORPUS_INPUT", legacy)

    entries = corpus_html.default_entries()

    assert [entry.name for entry in entries].count("icloud") == 2
    assert [entry.name for entry in entries].count("vbird") == 2
    assert "legacy_only" not in {entry.name for entry in entries}
    assert "cmp_icloud" not in {entry.name for entry in entries}
    assert {entry.mode for entry in entries if entry.name == "vbird"} == {"current", "optimizer"}


def test_generate_corpus_html_only_updates_sidecars_when_manifest_is_unchanged(tmp_path):
    output = tmp_path / "corpus"
    entries = [
        CorpusEntry("a", "current", _tiny_disk, Options()),
        CorpusEntry("b", "current", _tiny_disk, Options()),
    ]
    index = generate_corpus_html(output, entries)
    original_index = index.read_text()
    marker = "<!-- stable index marker -->"
    index.write_text(original_index + marker)

    generate_corpus_html(output, entries, only=["a"])

    assert index.read_text().endswith(marker)
    assert (output / "svg" / "current-a.svg").exists()
    assert (output / "svg" / "current-b.svg").exists()


def test_generate_corpus_html_writes_index_before_rendering_entries(tmp_path):
    output = tmp_path / "corpus"

    def _assert_index_exists():
        assert (output / "index.html").exists()
        return _tiny_disk()

    generate_corpus_html(
        output,
        [CorpusEntry("a", "current", _assert_index_exists, Options())],
    )


def test_corpus_output_does_not_write_generated_inputs_into_source_input_dir(tmp_path):
    corpus = tmp_path / "corpus"
    source = corpus / "input"
    source.mkdir(parents=True)
    Image.fromarray(_tiny_disk()).save(source / "appstore.png")

    generate_corpus_html(corpus, corpus_html.corpus_entries(source), only=["appstore"])

    assert (corpus / "index.html").exists()
    assert (corpus / "rendered-input" / "current-appstore.png").exists()
    assert not (source / "current-appstore.png").exists()
    assert not (source / "optimizer-appstore.png").exists()
