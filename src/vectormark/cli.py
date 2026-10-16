"""`vectormark` CLI."""

from __future__ import annotations

import argparse
import sys

from .pipeline import Options, idealize


def _floats(text: str) -> tuple[float, ...]:
    try:
        vals = tuple(float(t) for t in text.split(",") if t.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}")
    if not vals:
        raise argparse.ArgumentTypeError("at least one value required")
    return vals


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vectormark", description="Idealize a logo raster into SVG.")
    ap.add_argument("input", help="input raster (PNG/JPG)")
    ap.add_argument("-o", "--output", help="output .svg (default: stdout)")
    ap.add_argument("--epsilon", type=float, default=1.5, help="fit tolerance in px")
    ap.add_argument("--max-error", type=float, default=1.0, help="Bézier fit tolerance in px")
    ap.add_argument("--colors", type=int, default=16, help="max palette colours")
    ap.add_argument("--flatten", action="store_true", help="flatten primitives to paths")
    ap.add_argument("--no-symmetry", action="store_true", help="disable symmetry detection")
    ap.add_argument(
        "--corner-normalize",
        action="store_true",
        help="canonicalize recognized rounded corners to one endpoint-preserving quadratic",
    )
    ap.add_argument("--variants", action="store_true",
                    help="write an epsilon × max_error matrix of variants to a directory")
    ap.add_argument("--axes", action="store_true",
                    help="overlay detected symmetry axes on the contact-sheet tiles (with --variants)")
    ap.add_argument("--out-dir", help="output directory for --variants (default: ./<stem>-variants)")
    ap.add_argument("--epsilons", type=_floats, help="--variants epsilon axis, e.g. 0.5,1.5,3")
    ap.add_argument("--max-errors", type=_floats, help="--variants max_error axis, e.g. 0.5,1,2.5")
    args = ap.parse_args(argv)

    if args.axes and not args.variants:
        print("note: --axes only applies in --variants mode; ignoring it", file=sys.stderr)

    base = Options(
        max_sampling_colors=args.colors,
        flatten=args.flatten,
        no_symmetry=args.no_symmetry,
        corner_normalize=args.corner_normalize,
    )

    if args.variants:
        if args.epsilon != 1.5 or args.max_error != 1.0:
            print("note: --epsilon/--max-error are ignored in --variants mode "
                  "(the matrix sets them; use --epsilons/--max-errors)", file=sys.stderr)
        from pathlib import Path

        from .variants import (
            DEFAULT_EPSILONS, DEFAULT_MAX_ERRORS, compose_contact_sheet,
            generate_variants, write_variant_set,
        )

        epsilons = args.epsilons or DEFAULT_EPSILONS
        max_errors = args.max_errors or DEFAULT_MAX_ERRORS
        out_dir = Path(args.out_dir) if args.out_dir else Path(f"{Path(args.input).stem}-variants")

        variants = generate_variants(args.input, epsilons=epsilons, max_errors=max_errors, base=base)
        write_variant_set(variants, out_dir, source=args.input)
        png = compose_contact_sheet(variants, epsilons=epsilons, max_errors=max_errors, draw_axes=args.axes)
        if png is not None:
            (out_dir / "contact-sheet.png").write_bytes(png)
        else:
            print("contact sheet skipped (install vectormark[scoring] to render it)", file=sys.stderr)
        print(f"{len(variants)} variants -> {out_dir}"
              f"{' (+contact-sheet.png)' if png is not None else ''}")
        return 0

    svg = idealize(args.input, options=Options(
        epsilon=args.epsilon, max_error=args.max_error, max_sampling_colors=args.colors,
        flatten=args.flatten, no_symmetry=args.no_symmetry,
        corner_normalize=args.corner_normalize,
    ))
    if args.output:
        with open(args.output, "w") as f:
            f.write(svg)
    else:
        sys.stdout.write(svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
