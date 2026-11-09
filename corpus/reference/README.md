# Corpus references

Store independent source SVGs and reviewed reconstructions here, named after
their corpus entry. Each SVG has a JSON provenance/review record. Generated
vectorEyes output belongs in `corpus/svg/`, never here. Corpus refreshes must not
overwrite references.

- `burger_king.svg`: upstream SVG approved by the user on 2026-09-08.
- `daikonic.svg`: existing geometric target from the original project fixtures;
  retained as a candidate pending review against the current corpus raster.
- Other brand SVGs: downloaded references approved by the user on 2026-09-08. Wikimedia
  Commons source credit and license metadata are retained in each JSON record;
  a hosted SVG is not automatically an official original or an exact match.

Open [the review gallery](index.html) for linked raster/reference pairs. See
`sources.json` for the acquisition manifest and custom/unresolved entries.
This newly assembled manifest does not reconstruct the missing original
`corpus/sources.json` raster-download manifest.

To collect missing references and regenerate the gallery, from the repository:

```sh
PYTHONPATH=src python scripts/download_corpus_references.py
```

The collector reuses the corpus byte downloader, refuses to replace existing
SVGs/review records, and rejects embedded raster or active content. Download
failures are shown in the gallery and cause a nonzero exit. It does not approve
references, trace images, update the main corpus report, or change test fixtures.

Approval of the artwork is separate from verification of its coordinate
registration against the corpus PNG. No reference has that registration
verified yet. These files are not yet wired into automated corpus scoring.

The original Daikonic test fixture remains in place for existing tests. Its hash
and location are recorded in the provenance file; no test inputs were changed.
